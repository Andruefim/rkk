"""
server_v2.py — FastAPI с Value Layer и seed injection.

Новые endpoints:
  POST /inject-seeds  — загружаем LLM/RAG text priors в агента
  GET  /value-layer   — статистика Value Layer всех агентов
"""
from __future__ import annotations
import asyncio
import json
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.simulation import Simulation

app = FastAPI(title="RKK v5 Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sim: Simulation | None = None

def get_sim() -> Simulation:
    global _sim
    if _sim is None:
        _sim = Simulation(device_str="cuda")
    return _sim


# ── REST ──────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":   "ok",
        "device":   str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

@app.get("/state")
def state():
    return get_sim()._snapshot({})

@app.post("/step")
def step():
    return get_sim().tick_step()

@app.get("/graph/{agent_id}")
def graph(agent_id: int):
    sim = get_sim()
    if agent_id >= len(sim.agents):
        return {"error": "invalid agent_id"}
    return sim.agents[agent_id].graph.to_dict()

@app.get("/value-layer")
def value_layer_stats():
    sim = get_sim()
    return {
        a.name: a.value_layer.snapshot()
        for a in sim.agents
    }


# ── LLM/RAG Seed Injection ───────────────────────────────────────────────────
class SeedEdge(BaseModel):
    from_:  str
    to:     str
    weight: float = 0.3
    alpha:  float = 0.05

class SeedRequest(BaseModel):
    agent_id: int
    edges:    list[SeedEdge]
    source:   str = "manual"   # "llm", "rag", "manual"

@app.post("/inject-seeds")
def inject_seeds(req: SeedRequest):
    """
    Загружаем text priors от LLM/RAG в агента.

    Пример:
    {
      "agent_id": 0,
      "source": "rag",
      "edges": [
        {"from_": "Temp", "to": "Pressure", "weight": 0.8},
        {"from_": "Temp", "to": "Energy",   "weight": 0.7}
      ]
    }

    Все рёбра загружаются с alpha=0.05 (будут выжжены если неверны).
    """
    sim    = get_sim()
    result = sim.inject_seeds(
        agent_id=req.agent_id,
        edges=[e.model_dump() for e in req.edges]
    )
    result["source"] = req.source
    return result


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/causal-stream")
async def causal_stream(websocket: WebSocket):
    await websocket.accept()
    sim   = get_sim()
    speed = 1
    print(f"[WS] Connected. Device: {sim.device}")

    try:
        while True:
            # Команды от клиента
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(raw)
                cmd = msg.get("cmd")
                if cmd == "set_speed":
                    speed = int(msg.get("value", 1))
                elif cmd == "reset":
                    global _sim
                    _sim = Simulation(device_str="cuda")
                    sim  = get_sim()
                elif cmd == "inject_seeds":
                    # Инъекция прямо через WebSocket
                    sim.inject_seeds(
                        agent_id=int(msg.get("agent_id", 0)),
                        edges=msg.get("edges", [])
                    )
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            state = None
            for _ in range(max(1, speed)):
                state = sim.tick_step()

            await websocket.send_json(state)
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, log_level="info")