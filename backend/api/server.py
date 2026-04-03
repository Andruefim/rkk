"""
server_singleton.py — FastAPI для Singleton AGI (Фаза 11).

Новые endpoints:
  GET  /camera/frame          — base64 PNG кадр из PyBullet робота
  POST /world/switch          — переключить среду без сброса агента
  GET  /world/list            — доступные миры
  GET  /skeleton              — позиции суставов для Three.js
  POST /bootstrap/robot       — сиды для робота (hardcoded + optional LLM)
  POST /bootstrap/llm         — LLM генерирует гипотезы для текущего мира

Всё остальное совместимо с предыдущим server.py:
  POST /inject-seeds
  POST /rag/auto-seed-all
  GET  /rag/status
  GET  /demon/stats
  GET  /variables/{agent_id}
  WS   /ws/causal-stream
"""
from __future__ import annotations
import asyncio
import json
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.simulation import Simulation
from engine.rag_seeder import RAGSeeder, HARDCODED_SEEDS

app = FastAPI(title="RKK Singleton AGI")
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

_rag_status: dict = {"last_run": None, "results": [], "running": False}


def get_sim() -> Simulation:
    global _sim
    if _sim is None:
        _sim = Simulation(device_str="cuda", start_world="robot")
    return _sim

def get_seeder() -> RAGSeeder:
    global _seeder
    if _seeder is None:
        _seeder = RAGSeeder(
            llm_url="http://localhost:11434/api/generate",
            llm_model="gemma4:e4b"
        )
    return _seeder


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    sim = get_sim()
    return {
        "status":       "ok",
        "singleton":    True,
        "device":       str(sim.device),
        "gpu":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "current_world":sim.current_world,
        "gnn_d":        sim.agent.graph._d,
    }

@app.get("/state")
def state():
    return get_sim().public_state()

@app.post("/step")
def step():
    return get_sim().tick_step()


# ── Camera ────────────────────────────────────────────────────────────────────
@app.get("/camera/frame")
def camera_frame():
    """Base64 PNG кадр из PyBullet. Используется UI для overlay."""
    frame = get_sim().get_camera_frame()
    if frame is None:
        return JSONResponse({"frame": None, "available": False})
    return JSONResponse({"frame": frame, "available": True})

@app.get("/skeleton")
def skeleton():
    """Позиции суставов + цель для Three.js скелетона."""
    sim = get_sim()
    return {
        "joints": sim.get_robot_skeleton() or [],
        "target": sim.get_robot_target() or {},
        "world":  sim.current_world,
    }


# ── World switching ───────────────────────────────────────────────────────────
class WorldSwitchRequest(BaseModel):
    world: str

@app.post("/world/switch")
def world_switch(req: WorldSwitchRequest):
    """Переключаем мир без сброса агента."""
    return get_sim().switch_world(req.world)

@app.get("/world/list")
def world_list():
    from engine.simulation import WORLDS
    return {"worlds": WORLDS, "current": get_sim().current_world}


# ── Variables / Seeds ─────────────────────────────────────────────────────────
@app.get("/variables/{agent_id}")
def get_variables(agent_id: int):
    ctx = get_sim().agent_seed_context(agent_id)
    if ctx is None:
        return {"error": "invalid"}
    return ctx

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
    result = get_sim().inject_seeds(
        agent_id=req.agent_id,
        edges=[e.model_dump() for e in req.edges]
    )
    result["source"] = req.source
    return result


# ── Bootstrap ─────────────────────────────────────────────────────────────────
@app.post("/bootstrap/robot")
def bootstrap_robot():
    """Инжектируем hardcoded seeds для робота (кинематика суставов)."""
    from engine.environment_robot import robot_hardcoded_seeds
    seeds  = robot_hardcoded_seeds()
    result = get_sim().inject_seeds(agent_id=0, edges=seeds)
    return {"source": "robot_hardcoded", **result}


class LLMBootstrapRequest(BaseModel):
    world:      str | None = None
    llm_model:  str        = "qwen3.5:4b"
    llm_url:    str        = "http://localhost:11434/api/generate"

@app.post("/bootstrap/llm")
async def bootstrap_llm(req: LLMBootstrapRequest):
    """
    LLM генерирует начальные гипотезы для текущего мира.
    Это 'cultural memory' — агент получает знания без опыта.
    """
    sim  = get_sim()
    ctx  = sim.agent_seed_context(0)
    if not ctx:
        return {"error": "no context"}

    preset = req.world or sim.current_world
    vars_  = ctx["variables"]

    seeder = RAGSeeder(llm_url=req.llm_url, llm_model=req.llm_model)
    try:
        hypotheses = await seeder.generate(
            env_preset=preset,
            available_vars=vars_,
            max_hypotheses=8,
        )
        if hypotheses:
            edges = [h.to_dict() for h in hypotheses]
        else:
            edges = HARDCODED_SEEDS.get(preset, [])

        result = sim.inject_seeds(agent_id=0, edges=edges)
        return {"source": "llm", "preset": preset, **result}
    except Exception as e:
        return {"error": str(e)}


# ── RAG ───────────────────────────────────────────────────────────────────────
@app.post("/rag/auto-seed-all")
async def rag_auto_seed_all():
    global _rag_status
    sim    = get_sim()
    seeder = get_seeder()
    ctx    = sim.agent_seed_context(0)
    if not ctx:
        return {"error": "no context"}

    preset = ctx["preset"]
    vars_  = ctx["variables"]

    _rag_status["running"] = True
    try:
        hypotheses = await seeder.generate(
            env_preset=preset, available_vars=vars_, max_hypotheses=6
        )
        edges  = [h.to_dict() for h in hypotheses] if hypotheses else HARDCODED_SEEDS.get(preset, [])
        source = hypotheses[0].source if hypotheses else "hardcoded"
    except Exception as e:
        edges  = HARDCODED_SEEDS.get(preset, [])
        source = "hardcoded"
    finally:
        _rag_status["running"] = False

    result = sim.inject_seeds(agent_id=0, edges=edges)
    _rag_status["last_run"] = sim.tick
    _rag_status["results"]  = [{"agent": "Nova", "preset": preset,
                                  "injected": result.get("injected", 0), "source": source}]
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
    print(f"[WS] Singleton connected. Device: {sim.device}")

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
                    _sim = Simulation(device_str="cuda", start_world="robot")
                    sim  = get_sim()

                elif cmd == "inject_seeds":
                    sim.inject_seeds(
                        agent_id=0,
                        edges=msg.get("edges", [])
                    )

                elif cmd == "switch_world":
                    world = msg.get("world", "robot")
                    sim.switch_world(world)

                elif cmd == "rag_auto":
                    ctx = sim.agent_seed_context(0)
                    if ctx:
                        preset = ctx["preset"]
                        edges  = HARDCODED_SEEDS.get(preset, [])
                        if edges:
                            sim.inject_seeds(agent_id=0, edges=edges)

                elif cmd == "bootstrap_robot":
                    from engine.environment_robot import robot_hardcoded_seeds
                    sim.inject_seeds(agent_id=0, edges=robot_hardcoded_seeds())

            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            state_data = None
            for _ in range(max(1, speed)):
                state_data = sim.tick_step()

            await websocket.send_json(state_data)
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, log_level="info")