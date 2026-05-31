"""
server.py — FastAPI для Singleton AGI Humanoid (Фаза 11 + 12).

Фаза 12 новые endpoints:
  POST /vision/enable              — включить SlotAttention visual cortex
  POST /vision/disable             — вернуться к ручным переменным
  GET  /vision/slots               — слоты + attention masks (base64)
  GET  /vision/status              — статус кортекса
  GET  /vision/attn_frame?slot_idx — PyBullet frame с overlay маской
Авто при старте (lifespan): опционально enable visual (SlotAttention).
  RKK_SKIP_AUTO_VISION=1 — не включать зрение.
  RKK_AUTO_VISION_N_SLOTS, RKK_AUTO_VISION_MODE (hybrid|visual)


Установить для Фазы 12: pip install opencv-python scipy
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import torch

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass
from contextlib import asynccontextmanager
import traceback
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.core.constants import agent_loop_hz_from_env
from engine.json_util import sanitize_for_json
from engine.simulation import Simulation

_sim: Simulation | None = None
_causal_stream_ws: WebSocket | None = None
_causal_stream_gen: int = 0


def _ws_conn_active(websocket: WebSocket, conn_gen: int) -> bool:
    return _causal_stream_ws is websocket and _causal_stream_gen == conn_gen


async def _ws_send_json(websocket: WebSocket, conn_gen: int, payload: dict) -> None:
    if not _ws_conn_active(websocket, conn_gen):
        raise WebSocketDisconnect(code=1000)
    try:
        await websocket.send_json(payload)
    except RuntimeError as e:
        msg = str(e).lower()
        if "send" in msg and "close" in msg:
            raise WebSocketDisconnect(code=1000) from e
        raise


def _ws_hello_payload(sim: Simulation) -> dict:
    """Лёгкий кадр сразу после accept — UI не ждёт полный public_state()."""
    return {
        "tick": int(getattr(sim, "tick", 0)),
        "phase": int(getattr(sim, "phase", 1)),
        "entropy": 100.0,
        "singleton": True,
        "_ws_hello": True,
        "agents": [],
        "events": [],
        "graph_deltas": {},
    }


def get_sim() -> Simulation:
    global _sim
    if _sim is None:
        _sim = Simulation(
            device_str=os.environ.get("RKK_DEVICE", "cuda"),
            start_world="humanoid",
        )
    return _sim


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _agent_loop_hz() -> float:
    try:
        return max(0.0, float(os.environ.get("RKK_AGENT_LOOP_HZ", "0")))
    except ValueError:
        return 0.0


async def _startup_auto_vision() -> None:
    """Опционально включить SlotAttention после старта API."""
    if _env_flag("RKK_SKIP_AUTO_VISION"):
        print("[RKK] Auto vision skipped (RKK_SKIP_AUTO_VISION)")
        return
    try:
        sim = get_sim()
        if sim.current_world != "humanoid":
            return
        if sim._visual_mode:
            return
        try:
            n_slots = int(os.environ.get("RKK_AUTO_VISION_N_SLOTS", "8"))
        except ValueError:
            n_slots = 8
        mode = (os.environ.get("RKK_AUTO_VISION_MODE", "hybrid") or "hybrid").strip()
        out = sim.enable_visual(n_slots=n_slots, mode=mode)
        if out.get("error"):
            print(f"[RKK] Auto vision failed: {out.get('error')}")
            return
        print(
            f"[RKK] Auto vision ON: n_slots={out.get('n_slots')}, "
            f"mode={out.get('mode')}, gnn_d={out.get('gnn_d')}"
        )
    except Exception as e:
        print(f"[RKK] Auto vision error: {e}")


async def _startup_post_boot_pipeline() -> None:
    await _startup_auto_vision()


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    get_sim()._uvicorn_loop = asyncio.get_running_loop()
    asyncio.create_task(_startup_post_boot_pipeline())
    yield


app = FastAPI(title="RKK Singleton AGI Humanoid v12", lifespan=_app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    sleeping = False
    sleep_phase = ""
    sc = getattr(sim, "_sleep_ctrl", None)
    if sc is not None and sc.is_sleeping:
        sleeping = True
        sleep_phase = getattr(sc.current_phase(), "name", "") or ""
    from engine.features.simulation.snapshot import humanoid_curriculum_step

    cur_step, cur_stab = humanoid_curriculum_step(sim)
    return {
        "status":        "ok",
        "singleton":     True,
        "tick":          int(getattr(sim, "tick", 0)),
        "device":        str(sim.device),
        "gpu":           _hardware_label(),
        "current_world": sim.current_world,
        "gnn_d":         sim.agent.graph._d,
        "fallen":        sim._fall_count,
        "fall_count":    sim._fall_count,
        "visual_mode":   sim._visual_mode,
        "fixed_root":    sim._fixed_root_active,
        "curriculum_step": cur_step,
        "curriculum_stabilize_until": cur_stab,
        "sleeping":      sleeping,
        "sleep_phase":   sleep_phase,
    }

@app.get("/state")
def state():
    return sanitize_for_json(get_sim().public_state())


@app.get("/api/snapshot")
def api_snapshot():
    """Alias для UI-виджетов: кэш фонового тика (без повторного PyBullet snapshot)."""
    sim = get_sim()
    ps = sim.public_state()
    if not isinstance(ps, dict):
        ps = {}
    if not ps.get("_json_sanitized"):
        ps = sanitize_for_json(ps)
    out = dict(ps)
    out["world"] = out.get("current_world", "humanoid")
    return out


@app.get("/api/tick_profile")
def api_tick_profile():
    """Ranked per-feature tick timings (RKK_TICK_PROFILE)."""
    from engine.tick_profiler import profile_snapshot

    return sanitize_for_json(profile_snapshot())


@app.get("/api/agent/messages")
def api_agent_messages(last_n: int = Query(default=50, ge=1, le=200)):
    """Phase L: история речи агента для чата."""
    verbal = getattr(get_sim(), "_verbal", None)
    if verbal is None:
        return {"messages": [], "available": False}
    return sanitize_for_json({
        "messages": verbal.get_messages_for_ui(last_n=last_n),
        "available": True,
        "stats": verbal.snapshot(),
    })


@app.post("/api/agent/reply")
def api_agent_reply(body: dict | None = Body(default=None)):
    """Phase L: ответ человека на реплику агента."""
    b = body if isinstance(body, dict) else {}
    text = str(b.get("text", "")).strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    return get_sim().handle_human_reply(text)


@app.websocket("/api/ws/chat")
async def api_ws_chat(websocket: WebSocket):
    """Phase L: realtime чат с агентом."""
    await websocket.accept()
    sim = get_sim()
    sim._chat_ws_clients.append(websocket)
    try:
        verbal = getattr(sim, "_verbal", None)
        if verbal is not None:
            history = verbal.get_messages_for_ui(last_n=30)
            await websocket.send_text(
                json.dumps({"event": "history", "data": history}, ensure_ascii=False)
            )
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "reply":
                t = str(data.get("text", "")).strip()
                if t:
                    sim.handle_human_reply(t)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            sim._chat_ws_clients.remove(websocket)
        except ValueError:
            pass


@app.post("/step")
def step():
    return sanitize_for_json(get_sim().tick_step())


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
    if not isinstance(scene, dict):
        return scene
    out = dict(scene)
    out["lever"] = scene.get("lever", {"x": 0.5, "y": 0.45, "z": 0.05})
    out["fixed_root"] = sim._fixed_root_active
    return out


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
      slot_labels: list[dict] — Фаза 2: label, likely_phys, confidence по индексу
      slot_lexicon_tick, slot_lexicon_frame_hash — метаданные лексикона (grounding)
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


# ══════════════════════════════════════════════════════════════════════════════
# FIXED ROOT / CURRICULUM ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/fixed-root/enable")
async def fixed_root_enable():
    """
    Фиксируем базу гуманоида (curriculum step 1).
    variable_ids → FIXED_BASE_VARS (18 vars), GNN d уменьшается.
    Value Layer → for_fixed_root() (warmup=300, entropy_limit=2.0).
    Работает поверх visual mode.
    """
    return get_sim().enable_fixed_root()


@app.post("/fixed-root/disable")
async def fixed_root_disable():
    """
    Снимаем фиксацию (curriculum step 2: переход к ходьбе).
    variable_ids → VAR_NAMES (31 var), Value Layer → default с warmup.
    """
    return get_sim().disable_fixed_root()


@app.post("/humanoid/reset-stance")
def humanoid_reset_stance():
    """Reset humanoid to default standing pose (for walk/recovery testing)."""
    sim = get_sim()
    if sim.current_world != "humanoid":
        return {"ok": False, "error": "not_humanoid"}
    sim.disable_fixed_root()
    base = sim.agent.env
    for _ in range(8):
        nxt = getattr(base, "base_env", None)
        if nxt is None or nxt is base:
            break
        base = nxt
    fn = getattr(base, "reset_stance", None)
    if not callable(fn):
        return {"ok": False, "error": "no_reset_stance"}
    fn()
    fallen = bool(getattr(base, "is_fallen", lambda: False)())
    return {"ok": True, "fallen": fallen, "tick": sim.tick}


@app.get("/fixed-root/status")
def fixed_root_status():
    """Статус fixed_root mode и текущего Value Layer."""
    sim = get_sim()
    vl = sim.agent.value_layer
    return {
        "fixed_root":           sim._fixed_root_active,
        "gnn_d":                sim.agent.graph._d,
        "var_count":            len(sim.agent.graph.nodes),
        "block_rate":           round(vl.block_rate, 3),
        "vl_mode":              "fixed_root" if sim._fixed_root_active else "full",
        "vl_fixed_root_bounds": sim._fixed_root_active,
        "total_checked":        vl.total_checked,
        "total_blocked":        vl.total_blocked,
    }


class MemorySaveBody(BaseModel):
    path: str | None = None


@app.post("/memory/save")
def memory_save(
    body: MemorySaveBody | None = Body(default=None),
    path: str | None = Query(default=None),
):
    """Фаза 1: сохранить .rkk. Тело JSON {\"path\": \"...\"} или query ?path=…"""
    p = path or (body.path if body and body.path else None)
    return get_sim().memory_save(p)


@app.get("/memory/load")
def memory_load(path: str | None = Query(default=None)):
    """Фаза 1: загрузка памяти; частичное совмещение весов при смене d."""
    return get_sim().memory_load(path)


@app.get("/memory/status")
def memory_status():
    """Статус .rkk файла по умолчанию."""
    from engine.persistence import default_memory_path

    p = default_memory_path()
    if not p.is_file():
        return {"exists": False, "path": str(p.resolve())}
    st = p.stat()
    return {
        "exists": True,
        "path": str(p.resolve()),
        "size_kb": round(float(st.st_size) / 1024.0, 2),
        "mtime": float(st.st_mtime),
    }


@app.post("/sleep")
def force_sleep():
    """Phase K: начать цикл консолидации сна (fixed_root на время сна)."""
    sim = get_sim()
    sleep_ctrl = getattr(sim, "_sleep_ctrl", None)
    if sleep_ctrl is None:
        return JSONResponse({"error": "Sleep controller not available"}, status_code=503)
    if sleep_ctrl.is_sleeping:
        return {
            "error": "Already sleeping",
            "phase": sleep_ctrl.current_phase.name,
        }
    sim._sleep_attach_fixed_root()
    sleep_ctrl.begin_sleep(sim.tick, "manual", sim=sim)
    return {
        "ok": True,
        "tick": sim.tick,
        "reason": "manual",
        "message": "Sleep initiated. Will complete in ~200 ticks.",
    }


@app.get("/sleep/status")
def sleep_status():
    """Phase K: состояние SleepController."""
    sleep_ctrl = getattr(get_sim(), "_sleep_ctrl", None)
    if sleep_ctrl is None:
        return {"available": False}
    return sleep_ctrl.snapshot()


@app.get("/graph/frozen-edges")
def graph_frozen_edges():
    """Диагностика замороженных кинематических рёбер."""
    sim = get_sim()
    frozen = sorted(list(getattr(sim.agent.graph, "_frozen_edge_set", set()) or set()))
    return {
        "count": len(frozen),
        "edges": [{"from_": f, "to": t} for (f, t) in frozen],
        "mask_active": bool(frozen and sim.agent.graph._core is not None),
        "frozen_weight_target": float(getattr(sim.agent.graph, "FROZEN_EDGE_W", 0.0)),
    }


@app.get("/concepts/list")
def concepts_list():
    """Список концептов: Phase 1 proto-concepts + Phase 2 ConceptStore snapshot."""
    return get_sim().concepts_list_payload()


@app.get("/concepts/{cid}/subgraph")
def concept_subgraph(cid: str):
    """Фаза 1: узлы и рёбра одного proto-concept по id (например c0)."""
    return get_sim().concept_subgraph_payload(cid)


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/causal-stream")
async def causal_stream(websocket: WebSocket):
    global _causal_stream_ws, _causal_stream_gen
    _causal_stream_gen += 1
    conn_gen = _causal_stream_gen
    if _causal_stream_ws is not None:
        try:
            await _causal_stream_ws.close(code=1000, reason="replaced by new client")
        except Exception:
            pass
        _causal_stream_ws = None
    await websocket.accept()
    _causal_stream_ws = websocket
    sim   = get_sim()
    speed = 1
    agent_hz = agent_loop_hz_from_env()
    ws_period = 0.05 if agent_hz <= 0 else max(1.0 / agent_hz, 0.033)
    print(f"[WS] Humanoid+Vision Singleton connected. d={sim.agent.graph._d}")

    try:
        loop = asyncio.get_running_loop()
        sim._bg.ensure_rkk_agent_loop()
        await _ws_send_json(websocket, conn_gen, sanitize_for_json(_ws_hello_payload(sim)))

        for _ in range(80):
            if not _ws_conn_active(websocket, conn_gen):
                raise WebSocketDisconnect(code=1000)
            with sim._sim_step_lock:
                payload0 = sim._agent_step_response
            if payload0 is not None:
                await _ws_send_json(websocket, conn_gen, payload0)
                break
            await asyncio.sleep(0.025)

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
                    from engine.environment_humanoid import humanoid_hardcoded_seeds

                    sim.inject_seeds(agent_id=0, edges=humanoid_hardcoded_seeds())
                # Фаза 12: visual cortex commands
                elif cmd == "vision_enable":
                    n = int(msg.get("n_slots", 8))
                    mode = msg.get("mode", "hybrid")
                    sim.enable_visual(n_slots=n, mode=mode)
                elif cmd == "vision_disable":
                    sim.disable_visual()
                elif cmd == "fixed_root_enable":
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, sim.enable_fixed_root)
                elif cmd == "fixed_root_disable":
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, sim.disable_fixed_root)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            if not _ws_conn_active(websocket, conn_gen):
                raise WebSocketDisconnect(code=1000)

            try:
                if agent_hz > 0:
                    with sim._sim_step_lock:
                        payload = sim._agent_step_response
                    if payload is not None:
                        await _ws_send_json(websocket, conn_gen, payload)
                else:
                    def _run_ticks() -> dict:
                        out: dict | None = None
                        for _ in range(max(1, speed)):
                            out = sim.tick_step()
                        raw = out or {}
                        if isinstance(raw, dict) and raw.get("_json_sanitized"):
                            return raw
                        return sanitize_for_json(raw)

                    payload = await loop.run_in_executor(None, _run_ticks)
                    if not _ws_conn_active(websocket, conn_gen):
                        raise WebSocketDisconnect(code=1000)
                    await _ws_send_json(websocket, conn_gen, payload)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                # Один плохой тик / сериализация не должны ронять весь uvicorn.
                print(f"[WS] tick/send failed: {e}")
                traceback.print_exc()
                if not _ws_conn_active(websocket, conn_gen):
                    raise WebSocketDisconnect(code=1000) from e
                try:
                    await _ws_send_json(
                        websocket,
                        conn_gen,
                        {
                            "tick": getattr(sim, "tick", 0),
                            "phase": getattr(sim, "phase", 1),
                            "entropy": 100.0,
                            "agents": [],
                            "events": [
                                {
                                    "tick": getattr(sim, "tick", 0),
                                    "text": f"[WS error] {e!s}"[:200],
                                    "color": "#ff4444",
                                    "type": "error",
                                }
                            ],
                            "graph_deltas": {},
                            "singleton": True,
                            "_ws_recovery": True,
                        },
                    )
                except WebSocketDisconnect:
                    raise
                except Exception as send_exc:
                    print(f"[WS] recovery send failed: {send_exc}")
                await asyncio.sleep(0.15)

            await asyncio.sleep(ws_period)

    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        traceback.print_exc()
    finally:
        if _causal_stream_ws is websocket:
            _causal_stream_ws = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, log_level="info")