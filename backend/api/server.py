"""
server_v3.py — FastAPI с RAG pipeline, агрессивным Demon и auto-seed.

Новые endpoints:
  POST /rag/generate       — генерируем seeds из Wikipedia для агента
  POST /rag/auto-seed-all  — автоматически сидируем всех агентов при старте
  GET  /demon/stats        — статистика атак Демона
  GET  /rag/status         — статус последних RAG операций
  GET  /variables/{agent}  — список переменных агента (для UI seeds panel)
"""
from __future__ import annotations
import asyncio
import json
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.simulation import Simulation
from engine.rag_seeder import RAGSeeder, HARDCODED_SEEDS

app = FastAPI(title="RKK v5 Backend")
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
_seeder: RAGSeeder | None  = None

# RAG статус (для UI)
_rag_status: dict = {
    "last_run": None,
    "results":  [],
    "running":  False,
}

ENV_PRESETS = ["physics", "chemistry", "logic"]
AGENT_NAMES = ["Nova", "Aether", "Lyra"]


def get_sim() -> Simulation:
    global _sim
    if _sim is None:
        _sim = Simulation(device_str="cuda")
    return _sim

def get_seeder() -> RAGSeeder:
    global _seeder
    if _seeder is None:
        # По умолчанию без LLM (только Wikipedia + regex)
        # Для LLM: RAGSeeder(llm_url="http://localhost:11434/api/generate", llm_model="qwen2.5:3b")
        _seeder = RAGSeeder(llm_url="http://localhost:11434/api/generate", llm_model="qwen3.5:9b")
    return _seeder


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
    return {a.name: a.value_layer.snapshot(sim.tick) for a in sim.agents}

@app.get("/demon/stats")
def demon_stats():
    return get_sim().demon.snapshot


# ── Variable discovery (для seeds UI) ────────────────────────────────────────
@app.get("/variables/{agent_id}")
def get_variables(agent_id: int):
    """Возвращает список переменных агента для корректного заполнения seeds."""
    sim = get_sim()
    if agent_id >= len(sim.agents):
        return {"error": "invalid agent_id"}
    agent = sim.agents[agent_id]
    return {
        "agent_id":   agent_id,
        "agent_name": agent.name,
        "env_preset": agent.env.preset,
        "variables":  list(agent.graph.nodes.keys()),
    }


# ── Manual seed injection ─────────────────────────────────────────────────────
class SeedEdge(BaseModel):
    from_:  str
    to:     str
    weight: float = 0.3
    alpha:  float = 0.05

class SeedRequest(BaseModel):
    agent_id: int
    edges:    list[SeedEdge]
    source:   str = "manual"

@app.post("/inject-seeds")
def inject_seeds(req: SeedRequest):
    sim    = get_sim()
    result = sim.inject_seeds(
        agent_id=req.agent_id,
        edges=[e.model_dump() for e in req.edges]
    )
    result["source"] = req.source
    return result


# ── RAG endpoints ─────────────────────────────────────────────────────────────
class RAGRequest(BaseModel):
    agent_id: int
    topic:    str | None = None    # если None — автоматически из preset
    max_hypotheses: int = 8
    use_llm:  bool = False
    llm_url:  str | None = None
    llm_model: str = "qwen2.5:3b"

@app.post("/rag/generate")
async def rag_generate(req: RAGRequest):
    """
    Генерируем seeds из Wikipedia для агента и сразу инжектируем.
    """
    global _rag_status
    sim = get_sim()

    if req.agent_id >= len(sim.agents):
        return {"error": "invalid agent_id"}

    agent  = sim.agents[req.agent_id]
    preset = agent.env.preset
    vars_  = list(agent.graph.nodes.keys())

    # Настраиваем seeder
    seeder = RAGSeeder(
        llm_url=req.llm_url if req.use_llm else None,
        llm_model=req.llm_model,
    )

    _rag_status["running"] = True
    try:
        hypotheses = await seeder.generate(
            env_preset=preset,
            available_vars=vars_,
            max_hypotheses=req.max_hypotheses,
        )

        if not hypotheses:
            # Fallback на hardcoded seeds
            edges = HARDCODED_SEEDS.get(preset, [])
            source = "hardcoded"
        else:
            edges  = [h.to_dict() for h in hypotheses]
            source = hypotheses[0].source if hypotheses else "rag"

        # Инжектируем
        inject_result = sim.inject_seeds(agent_id=req.agent_id, edges=edges)
        inject_result["source"] = source
        inject_result["hypotheses_count"] = len(hypotheses)

        _rag_status["last_run"] = sim.tick
        _rag_status["results"].append({
            "agent":  AGENT_NAMES[req.agent_id],
            "preset": preset,
            "n":      inject_result["injected"],
            "source": source,
        })
        if len(_rag_status["results"]) > 10:
            _rag_status["results"].pop(0)

        return inject_result

    finally:
        _rag_status["running"] = False


@app.post("/rag/auto-seed-all")
async def rag_auto_seed_all():
    """
    Автоматически сидируем всех трёх агентов при старте.
    Использует hardcoded seeds как fallback если Wikipedia недоступна.
    """
    global _rag_status
    sim    = get_sim()
    seeder = get_seeder()

    _rag_status["running"] = True
    results = []

    for i, agent in enumerate(sim.agents):
        preset = agent.env.preset
        vars_  = list(agent.graph.nodes.keys())

        try:
            hypotheses = await seeder.generate(
                env_preset=preset,
                available_vars=vars_,
                max_hypotheses=6,
            )
            edges  = [h.to_dict() for h in hypotheses] if hypotheses else HARDCODED_SEEDS.get(preset, [])
            source = hypotheses[0].source if hypotheses else "hardcoded"
        except Exception as e:
            edges  = HARDCODED_SEEDS.get(preset, [])
            source = "hardcoded"
            print(f"[RAG] Fallback to hardcoded for {agent.name}: {e}")

        result = sim.inject_seeds(agent_id=i, edges=edges)
        results.append({
            "agent":    agent.name,
            "preset":   preset,
            "injected": result.get("injected", 0),
            "source":   source,
        })

    _rag_status["running"]  = False
    _rag_status["last_run"] = sim.tick
    _rag_status["results"]  = results

    return {"status": "ok", "results": results}


@app.get("/rag/status")
def rag_status():
    return _rag_status


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/causal-stream")
async def causal_stream(websocket: WebSocket):
    await websocket.accept()
    sim   = get_sim()
    speed = 1
    print(f"[WS] Connected. Device: {sim.device}")

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
                    _sim = Simulation(device_str="cuda")
                    sim  = get_sim()
                elif cmd == "inject_seeds":
                    sim.inject_seeds(
                        agent_id=int(msg.get("agent_id", 0)),
                        edges=msg.get("edges", [])
                    )
                elif cmd == "rag_auto":
                    # Быстрый хардкодный сид через WS
                    for i, agent in enumerate(sim.agents):
                        preset = agent.env.preset
                        edges  = HARDCODED_SEEDS.get(preset, [])
                        if edges:
                            sim.inject_seeds(agent_id=i, edges=edges)
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