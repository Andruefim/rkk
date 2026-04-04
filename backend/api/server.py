"""
server.py — FastAPI для Singleton AGI Humanoid (Фаза 11 + 12).

Фаза 12 новые endpoints:
  POST /vision/enable              — включить SlotAttention visual cortex
  POST /vision/disable             — вернуться к ручным переменным
  GET  /vision/slots               — слоты + attention masks (base64)
  GET  /vision/status              — статус кортекса
  GET  /vision/attn_frame?slot_idx — PyBullet frame с overlay маской

Установить для Фазы 12: pip install opencv-python scipy
"""
from __future__ import annotations
import asyncio
import json
import os
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.simulation import Simulation
from engine.rag_seeder import RAGSeeder, HARDCODED_SEEDS

app = FastAPI(title="RKK Singleton AGI Humanoid v12")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sim:    Simulation | None = None
_seeder: RAGSeeder  | None = None
_rag_status = {"last_run": None, "results": [], "running": False}


def get_sim() -> Simulation:
    global _sim
    if _sim is None:
        _sim = Simulation(
            device_str=os.environ.get("RKK_DEVICE", "cuda"),
            start_world="humanoid",
        )
    return _sim

def get_seeder() -> RAGSeeder:
    global _seeder
    if _seeder is None:
        _seeder = RAGSeeder(
            llm_url="http://localhost:11434/api/generate",
            llm_model="qwen3.5:4b"
        )
    return _seeder


# ── Health / State ────────────────────────────────────────────────────────────
def _hardware_label() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple Silicon MPS"
    return "CPU"


@app.get("/health")
def health():
    sim = get_sim()
    return {
        "status":        "ok",
        "singleton":     True,
        "device":        str(sim.device),
        "gpu":           _hardware_label(),
        "current_world": sim.current_world,
        "gnn_d":         sim.agent.graph._d,
        "fallen":        sim._fall_count,
        "visual_mode":   sim._visual_mode,
    }

@app.get("/state")
def state():
    return get_sim().public_state()

@app.post("/step")
def step():
    return get_sim().tick_step()


# ── Camera ────────────────────────────────────────────────────────────────────
@app.get("/camera/frame")
def camera_frame(view: str | None = Query(default=None)):
    frame = get_sim().get_camera_frame(view=view)
    if frame is None:
        return JSONResponse({"frame": None, "available": False})
    return JSONResponse({"frame": frame, "available": True, "view": view})


# ── Full scene ────────────────────────────────────────────────────────────────
@app.get("/scene")
def full_scene():
    sim = get_sim()
    fn  = getattr(sim.agent.env, "get_full_scene", None)
    if callable(fn):
        scene = fn()
    else:
        scene = {
            "skeleton": getattr(sim.agent.env, "get_joint_positions_world", lambda:[])(),
            "cubes":    getattr(sim.agent.env, "get_cube_positions", lambda:[])(),
            "target":   getattr(sim.agent.env, "get_target", lambda:{"x":0,"y":0,"z":0.9})(),
            "fallen":   sim._fall_count > 0,
        }
    return scene


# ── World switching ───────────────────────────────────────────────────────────
class WorldSwitchRequest(BaseModel):
    world: str

@app.post("/world/switch")
def world_switch(req: WorldSwitchRequest):
    return get_sim().switch_world(req.world)

@app.get("/world/list")
def world_list():
    from engine.simulation import WORLDS
    return {"worlds": WORLDS, "current": get_sim().current_world}


# ── Variables / Seeds ─────────────────────────────────────────────────────────
@app.get("/variables/{agent_id}")
def get_variables(agent_id: int):
    ctx = get_sim().agent_seed_context(agent_id)
    return ctx or {"error": "invalid"}

class SeedEdge(BaseModel):
    from_:  str
    to:     str
    weight: float = 0.3
    alpha:  float = 0.05

class SeedRequest(BaseModel):
    agent_id: int = 0
    edges:    list[SeedEdge]
    source:   str = "manual"

@app.post("/inject-seeds")
def inject_seeds(req: SeedRequest):
    result = get_sim().inject_seeds(agent_id=req.agent_id,
                                     edges=[e.model_dump() for e in req.edges])
    result["source"] = req.source
    return result


# ── Bootstrap ─────────────────────────────────────────────────────────────────
@app.post("/bootstrap/humanoid")
def bootstrap_humanoid():
    from engine.environment_humanoid import humanoid_hardcoded_seeds
    seeds  = humanoid_hardcoded_seeds()
    result = get_sim().inject_seeds(agent_id=0, edges=seeds)
    return {"source": "humanoid_hardcoded", **result}

@app.post("/bootstrap/robot")
def bootstrap_robot():
    try:
        from engine.environment_robot import robot_hardcoded_seeds
        seeds = robot_hardcoded_seeds()
    except ImportError:
        seeds = HARDCODED_SEEDS.get(get_sim().current_world, [])
    return get_sim().inject_seeds(agent_id=0, edges=seeds)

class LLMBootstrapRequest(BaseModel):
    world:     str | None = None
    llm_model: str        = "qwen3.5:4b"
    llm_url:   str        = "http://localhost:11434/api/generate"

@app.post("/bootstrap/llm")
async def bootstrap_llm(req: LLMBootstrapRequest):
    sim  = get_sim()
    ctx  = sim.agent_seed_context(0)
    if not ctx:
        return {"error": "no context"}
    preset = req.world or sim.current_world
    vars_  = ctx["variables"]
    seeder = RAGSeeder(llm_url=req.llm_url, llm_model=req.llm_model)
    try:
        hyps  = await seeder.generate(env_preset=preset, available_vars=vars_, max_hypotheses=8)
        edges = [h.to_dict() for h in hyps] if hyps else HARDCODED_SEEDS.get(preset, [])
        res   = sim.inject_seeds(agent_id=0, edges=edges)
        return {"source": "llm", "preset": preset, **res}
    except Exception as e:
        return {"error": str(e)}


# ── RAG ───────────────────────────────────────────────────────────────────────
@app.post("/rag/auto-seed-all")
async def rag_auto_seed_all():
    global _rag_status
    sim, seeder = get_sim(), get_seeder()
    ctx = sim.agent_seed_context(0)
    if not ctx:
        return {"error": "no context"}

    preset = ctx["preset"]
    _rag_status["running"] = True
    try:
        hyps   = await seeder.generate(env_preset=preset, available_vars=ctx["variables"], max_hypotheses=6)
        edges  = [h.to_dict() for h in hyps] if hyps else HARDCODED_SEEDS.get(preset, [])
        source = hyps[0].source if hyps else "hardcoded"
    except Exception:
        edges, source = HARDCODED_SEEDS.get(preset, []), "hardcoded"
    finally:
        _rag_status["running"] = False

    res = sim.inject_seeds(agent_id=0, edges=edges)
    _rag_status.update({"last_run": sim.tick,
                          "results": [{"agent":"Nova","preset":preset,
                                        "injected":res.get("injected",0),"source":source}]})
    return {"status": "ok", "results": _rag_status["results"]}

@app.get("/rag/status")
def rag_status():
    return _rag_status

@app.get("/demon/stats")
def demon_stats():
    return get_sim().demon.snapshot


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 12: VISUAL CORTEX ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class VisionEnableRequest(BaseModel):
    n_slots: int = 8
    mode:    str = "hybrid"   # "hybrid" (слоты + моторы) | "visual" (только слоты)

@app.post("/vision/enable")
def vision_enable(req: VisionEnableRequest):
    """
    Включаем Causal Visual Cortex (Фаза 12).
    Требует: pip install opencv-python scipy Pillow
    """
    sim = get_sim()
    result = sim.enable_visual(n_slots=req.n_slots, mode=req.mode)
    return result

@app.post("/vision/disable")
def vision_disable():
    """Возвращаемся к ручным переменным."""
    return get_sim().disable_visual()

@app.get("/vision/status")
def vision_status():
    """Статус Visual Cortex."""
    sim = get_sim()
    return {
        "visual_mode":  sim._visual_mode,
        "vision_ticks": sim._vision_ticks,
        "n_slots":      sim._visual_env.n_slots if sim._visual_env else 0,
        "gnn_d":        sim.agent.graph._d,
        "cortex":       sim._visual_env.cortex.snapshot() if sim._visual_env else None,
    }

@app.get("/vision/slots")
def vision_slots():
    """
    Текущие данные Visual Cortex:
      frame:       base64 JPEG
      masks:       list[base64 JPEG] — миниатюры масок (48×48), UI масштабирует
      slot_values: list[float]
      variability: list[float] — насколько активен слот
      active_slots: int
      cortex:      dict — stats
    """
    sim = get_sim()
    return sim.get_vision_state()

@app.get("/vision/attn_frame")
def vision_attn_frame(slot_idx: int = Query(default=0)):
    """PyBullet frame с наложенной attention mask конкретного слота."""
    sim   = get_sim()
    state = sim.get_vision_state()
    if not state.get("visual_mode"):
        return JSONResponse({"available": False, "reason": "visual mode disabled"})

    frame  = state.get("frame")
    masks  = state.get("masks", [])
    if not frame or slot_idx >= len(masks):
        return JSONResponse({"available": False, "reason": "no data"})

    try:
        import base64, numpy as np
        from io import BytesIO
        from PIL import Image as PILImage

        frame_bytes = base64.b64decode(frame)
        frame_img   = PILImage.open(BytesIO(frame_bytes)).convert("RGBA")
        W, H        = frame_img.size

        mask_bytes = base64.b64decode(masks[slot_idx])
        mask_img   = PILImage.open(BytesIO(mask_bytes)).convert("L")
        mask_img   = mask_img.resize((W, H), PILImage.BILINEAR)
        mask_np    = np.array(mask_img, dtype=np.float32) / 255.0

        SLOT_COLORS = [
            (255, 80,  80),   (80,  200, 255), (80,  255, 100), (255, 200, 80),
            (200, 80,  255),  (255, 140, 80),  (80,  255, 220), (180, 180, 255),
        ]
        color = SLOT_COLORS[slot_idx % len(SLOT_COLORS)]
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[:, :, 0] = color[0]
        overlay[:, :, 1] = color[1]
        overlay[:, :, 2] = color[2]
        overlay[:, :, 3] = (mask_np * 160).astype(np.uint8)

        composite = PILImage.alpha_composite(frame_img, PILImage.fromarray(overlay, "RGBA"))
        buf = BytesIO()
        composite.convert("RGB").save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({"available": True, "frame": b64, "slot_idx": slot_idx})
    except Exception as e:
        return JSONResponse({"available": False, "error": str(e)})


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/causal-stream")
async def causal_stream(websocket: WebSocket):
    await websocket.accept()
    sim   = get_sim()
    speed = 1
    print(f"[WS] Humanoid+Vision Singleton connected. d={sim.agent.graph._d}")

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(raw)
                cmd = msg.get("cmd")
                if cmd == "set_speed":
                    speed = int(msg.get("value", 1))
                elif cmd == "reset":
                    global _sim
                    _sim = Simulation(
                        device_str=os.environ.get("RKK_DEVICE", "cuda"),
                        start_world="humanoid",
                    )
                    sim  = get_sim()
                elif cmd == "inject_seeds":
                    sim.inject_seeds(agent_id=0, edges=msg.get("edges", []))
                elif cmd == "switch_world":
                    sim.switch_world(msg.get("world", "humanoid"))
                elif cmd == "bootstrap_humanoid":
                    from engine.environment_humanoid import humanoid_hardcoded_seeds
                    sim.inject_seeds(agent_id=0, edges=humanoid_hardcoded_seeds())
                elif cmd == "rag_auto":
                    ctx = sim.agent_seed_context(0)
                    if ctx:
                        edges = HARDCODED_SEEDS.get(ctx["preset"], [])
                        if edges:
                            sim.inject_seeds(agent_id=0, edges=edges)
                # Фаза 12: visual cortex commands
                elif cmd == "vision_enable":
                    n = int(msg.get("n_slots", 8))
                    mode = msg.get("mode", "hybrid")
                    sim.enable_visual(n_slots=n, mode=mode)
                elif cmd == "vision_disable":
                    sim.disable_visual()
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            data = None
            for _ in range(max(1, speed)):
                data = sim.tick_step()

            await websocket.send_json(data)
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, log_level="info")