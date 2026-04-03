"""
server_humanoid.py — FastAPI для Singleton AGI Humanoid (Фаза 11).

Новые endpoints:
  GET  /camera/frame?view=diag|side|front|top  — JPEG из PyBullet
  GET  /scene           — skeleton + cubes + target + fallen (JSON)
  POST /world/switch    — переключить мир
  GET  /world/list      — доступные миры
  POST /bootstrap/humanoid — сиды биомеханики
  POST /bootstrap/llm   — LLM гипотезы
"""
from __future__ import annotations
import asyncio
import json
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.simulation import Simulation
from engine.rag_seeder import RAGSeeder, HARDCODED_SEEDS

app = FastAPI(title="RKK Singleton Humanoid AGI")
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
        _sim = Simulation(device_str="cuda", start_world="humanoid")
    return _sim

def get_seeder() -> RAGSeeder:
    global _seeder
    if _seeder is None:
        _seeder = RAGSeeder(
            llm_url="http://localhost:11434/api/generate",
            llm_model="qwen3.5:4b"
        )
    return _seeder


@app.get("/health")
def health():
    sim = get_sim()
    return {
        "status":        "ok",
        "singleton":     True,
        "device":        str(sim.device),
        "gpu":           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "current_world": sim.current_world,
        "gnn_d":         sim.agent.graph._d,
        "fallen":        sim._fall_count,
    }

@app.get("/state")
def state():
    return get_sim().public_state()

@app.post("/step")
def step():
    return get_sim().tick_step()


# ── Camera ────────────────────────────────────────────────────────────────────
@app.get("/camera/frame")
def camera_frame(view: str = Query(default="diag")):
    """JPEG кадр из PyBullet. view = diag | side | front | top"""
    frame = get_sim().get_camera_frame(view=view)
    if frame is None:
        return JSONResponse({"frame": None, "available": False})
    return JSONResponse({"frame": frame, "available": True, "view": view})


# ── Full scene ────────────────────────────────────────────────────────────────
@app.get("/scene")
def full_scene():
    """Skeleton + cubes + target + fallen status — всё за один запрос."""
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


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/causal-stream")
async def causal_stream(websocket: WebSocket):
    await websocket.accept()
    sim   = get_sim()
    speed = 1
    print(f"[WS] Humanoid Singleton connected. d={sim.agent.graph._d}")

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
                    _sim = Simulation(device_str="cuda", start_world="humanoid")
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