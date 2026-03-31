"""
server.py — FastAPI WebSocket сервер для RKK v5.

Улучшение 1 (WebSocket вместо HTTP fetch):
  /ws/causal-stream → стримит дельты графа в реальном времени
  Three.js интерполирует изменения на 60 FPS

Запуск (Windows PowerShell):
  cd backend
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations
import asyncio
import json
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engine.simulation import Simulation

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RKK v5 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton simulation ───────────────────────────────────────────────────────
# Windows: создаём один раз в main process
_sim: Simulation | None = None

def get_sim() -> Simulation:
    global _sim
    if _sim is None:
        _sim = Simulation(device_str="cuda")
    return _sim


# ── REST endpoints (для отладки) ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "device":  str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

@app.get("/state")
def state():
    """Однократный снапшот без тика."""
    sim = get_sim()
    return sim._snapshot({})

@app.post("/step")
def step():
    """Один тик (для тестирования без WebSocket)."""
    return get_sim().tick_step()

@app.get("/graph/{agent_id}")
def graph(agent_id: int):
    sim = get_sim()
    if agent_id >= len(sim.agents):
        return {"error": "invalid agent_id"}
    return sim.agents[agent_id].graph.to_dict()


# ── WebSocket: Causal Stream ──────────────────────────────────────────────────
@app.websocket("/ws/causal-stream")
async def causal_stream(websocket: WebSocket):
    """
    Стримит состояние симуляции на фронт.
    
    Протокол:
      Server → Client: StreamFrame (JSON)
      Client → Server: {"cmd": "set_speed", "value": 4}
                       {"cmd": "reset"}
    """
    await websocket.accept()
    sim   = get_sim()
    speed = 1   # тиков на итерацию (клиент может изменить)

    print(f"[WS] Client connected. Device: {sim.device}")

    try:
        while True:
            # Слушаем команды клиента (non-blocking)
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(raw)
                if msg.get("cmd") == "set_speed":
                    speed = int(msg.get("value", 1))
                elif msg.get("cmd") == "reset":
                    _sim = Simulation(device_str="cuda")
                    sim  = get_sim()
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            # Выполняем N тиков симуляции
            state = None
            for _ in range(max(1, speed)):
                state = sim.tick_step()

            # Отправляем состояние
            # graph_deltas — только изменившиеся рёбра (трафик ↓)
            await websocket.send_json(state)

            # 20 Hz: Python считает физику, Three.js интерполирует визуал
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
# Windows: ОБЯЗАТЕЛЬНО через if __name__ == "__main__"
# (multiprocessing требует spawn, не fork)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True только при разработке
        workers=1,      # Windows: только 1 worker с GPU
    )
