"""
server.py — FastAPI для Singleton AGI Humanoid (Фаза 11 + 12).

Фаза 12 новые endpoints:
  POST /vision/enable              — включить SlotAttention visual cortex
  POST /vision/disable             — вернуться к ручным переменным
  GET  /vision/slots               — слоты + attention masks (base64)
  GET  /vision/status              — статус кортекса
  GET  /vision/attn_frame?slot_idx — PyBullet frame с overlay маской
  POST /vision/vlm-label          — Фаза 2: VLM-лексикон слотов (Ollama chat + images)
  POST /teacher/refresh           — Фаза 3: LLM-учитель (правила S1 + VL overlay TTL)

Авто при старте (lifespan): humanoid_structured LLM → enable visual → один VLM → фаза 3 teacher.
  RKK_SKIP_AUTO_VISION=1, RKK_SKIP_AUTO_VLM_BOOTSTRAP=1 — отключить шаги.
  RKK_AUTO_VISION_N_SLOTS, RKK_AUTO_VISION_MODE (hybrid|visual)

Этап D: RKK_LLM_LOOP=1 — фоновые L2/L3 консультации Ollama в tick_step (см. engine/simulation.py, engine/llm_loop.py).

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
from typing import Literal
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from engine.ollama_env import get_ollama_generate_url, get_ollama_model
from engine.simulation import Simulation
from engine.rag_seeder import RAGSeeder, HARDCODED_SEEDS

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
            llm_url=get_ollama_generate_url(),
            llm_model=get_ollama_model(),
        )
    return _seeder


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _agent_loop_hz() -> float:
    try:
        return max(0.0, float(os.environ.get("RKK_AGENT_LOOP_HZ", "0")))
    except ValueError:
        return 0.0


async def _startup_humanoid_llm_bootstrap() -> None:
    """
    После старта сервера: humanoid_structured через Ollama (как POST /bootstrap/llm).
    Отключение: RKK_SKIP_AUTO_HUMANOID_LLM=1
    URL/модель: RKK_OLLAMA_URL, RKK_OLLAMA_MODEL (или OLLAMA_MODEL) — см. engine/ollama_env.py
    """
    skip = os.environ.get("RKK_SKIP_AUTO_HUMANOID_LLM", "").strip().lower()
    if skip in ("1", "true", "yes", "on"):
        print("[RKK] Auto humanoid_structured LLM skipped (RKK_SKIP_AUTO_HUMANOID_LLM)")
        return
    try:
        sim = get_sim()
        if sim.current_world != "humanoid":
            return
        ctx = sim.agent_seed_context(0)
        if not ctx:
            return
        llm_url = get_ollama_generate_url()
        model = get_ollama_model()
        try:
            max_h = int(os.environ.get("RKK_HUMANOID_LLM_MAX_HYPOTHESES", "28"))
        except ValueError:
            max_h = 28
        vars_ = ctx["variables"]
        seeder = RAGSeeder(llm_url=llm_url, llm_model=model)
        from engine.environment_humanoid import (
            humanoid_hardcoded_seeds,
            merge_humanoid_golden_with_llm_edges,
        )

        hyps = await seeder.generate_humanoid_structured(
            available_vars=vars_, max_hypotheses=max_h,
        )
        if hyps:
            edges = merge_humanoid_golden_with_llm_edges([h.to_dict() for h in hyps])
            src = "llm_humanoid_merged"
        else:
            edges = humanoid_hardcoded_seeds()
            src = "hardcoded_fallback"
        res = sim.inject_seeds(agent_id=0, edges=edges)
        print(
            f"[RKK] Auto humanoid_structured bootstrap: source={src}, "
            f"injected={res.get('injected', 0)}, skipped={len(res.get('skipped', []))}"
        )
    except Exception as e:
        print(f"[RKK] Auto humanoid_structured bootstrap error: {e}")


async def _startup_auto_vision_and_one_vlm() -> None:
    """
    После LLM-bootstrap: включить SlotAttention (hybrid) и один вызов VLM-лексикона.
    Повтор VLM намеренно не делаем — позже можно триггерить по «запутался»
    (высокий block_rate / fallen / стагнация фазы), см. simulation tick.

    RKK_SKIP_AUTO_VISION=1 — не включать зрение (и VLM не запустится).
    RKK_SKIP_AUTO_VLM_BOOTSTRAP=1 — зрение да, VLM один раз пропустить.
    RKK_AUTO_VLM_WEAK_EDGES=1 — как у ручного POST: слабые slot→phys.
    RKK_AUTO_VLM_TEXT_ONLY=1 — только текстовый режим (без картинок в chat).
    RKK_AUTO_VLM_MAX_MASKS — маски в chat (0 = только кадр при bootstrap, быстрее; по умолчанию 0).
    """
    # По умолчанию теперь ВКЛЮЧЕНО при старте, если не задано RKK_SKIP_AUTO_VISION=1
    if _env_flag("RKK_SKIP_AUTO_VISION"):
        print("[RKK] Auto vision skipped (RKK_SKIP_AUTO_VISION)")
        return
    try:
        sim = get_sim()
        if sim.current_world != "humanoid":
            print("[RKK] Auto vision: only humanoid world, skipping")
            return
        if sim._visual_mode:
            print("[RKK] Auto vision: already enabled, skipping enable_visual")
        else:
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

        if _env_flag("RKK_SKIP_AUTO_VLM_BOOTSTRAP"):
            print("[RKK] Auto VLM bootstrap skipped (RKK_SKIP_AUTO_VLM_BOOTSTRAP)")
            return

        # PyBullet / cortex warmup: первый getCameraImage после старта может быть пустым.
        await asyncio.sleep(3.0)
        if _agent_loop_hz() > 0:
            sim.advance_agent_steps(5)
        else:
            for _ in range(5):
                sim.tick_step()
        await asyncio.sleep(0.5)

        llm_url = get_ollama_generate_url()
        model = get_ollama_model()
        try:
            max_masks = int(os.environ.get("RKK_AUTO_VLM_MAX_MASKS", "0"))
        except ValueError:
            max_masks = 0

        vlm_out = await sim.vlm_label_slots(
            llm_url=llm_url,
            llm_model=model,
            max_mask_images=max_masks,
            text_only=_env_flag("RKK_AUTO_VLM_TEXT_ONLY"),
            inject_weak_edges=_env_flag("RKK_AUTO_VLM_WEAK_EDGES"),
        )
        if vlm_out.get("ok"):
            w = vlm_out.get("weak_edges_injected") or 0
            print(
                f"[RKK] Auto VLM bootstrap: mode={vlm_out.get('mode')}, "
                f"labels={vlm_out.get('n_slots_labeled')}, weak_edges={w}"
            )
            if vlm_out.get("warning"):
                print(f"[RKK] Auto VLM note: {vlm_out.get('warning')}")
        else:
            print(f"[RKK] Auto VLM bootstrap failed: {vlm_out.get('error', vlm_out)}")
    except Exception as e:
        print(f"[RKK] Auto vision/VLM error: {e}")


async def _startup_phase3_teacher() -> None:
    """
    Фаза 3 после VLM: один запрос LLM → правила для push_experience + дельты VL (TTL).
    RKK_SKIP_PHASE3_LLM=1 — пропуск.
    """
    if _env_flag("RKK_SKIP_PHASE3_LLM"):
        print("[RKK] Phase3 teacher skipped (RKK_SKIP_PHASE3_LLM)")
        return
    try:
        sim = get_sim()
        if sim.current_world != "humanoid":
            print("[RKK] Phase3 teacher: only humanoid, skipping")
            return
        # Сразу после VLM Ollama иногда отдаёт пустой response; короткая пауза + retry в fetch.
        await asyncio.sleep(2.5)
        out = await sim.refresh_phase3_teacher_llm()
        if out.get("ok"):
            print(
                f"[RKK] Phase3 teacher: rules={out.get('n_rules', 0)}, "
                f"vl_overlay={out.get('vl_overlay')}, ttl_tick={out.get('expires_at_tick')}"
            )
            ins = (out.get("insight") or "").strip()
            if ins:
                print(f"[RKK] Phase3 teacher insight:\n{ins}\n")
            if out.get("warning"):
                print(f"[RKK] Phase3 teacher note: {out.get('warning')}")
        else:
            print(f"[RKK] Phase3 teacher failed: {out.get('error')}")
    except Exception as e:
        print(f"[RKK] Phase3 teacher error: {e}")


async def _startup_post_boot_pipeline() -> None:
    """Порядок: LLM приоры → зрение → VLM → фаза 3 teacher (фон после yield сервера)."""
    await _startup_humanoid_llm_bootstrap()
    await _startup_auto_vision_and_one_vlm()
    await _startup_phase3_teacher()


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
    return {
        "status":        "ok",
        "singleton":     True,
        "device":        str(sim.device),
        "gpu":           _hardware_label(),
        "current_world": sim.current_world,
        "gnn_d":         sim.agent.graph._d,
        "fallen":        sim._fall_count,
        "fall_count":    sim._fall_count,
        "visual_mode":   sim._visual_mode,
        "fixed_root":    sim._fixed_root_active,
    }

@app.get("/state")
def state():
    return get_sim().public_state()


@app.get("/api/snapshot")
def api_snapshot():
    """Alias для UI-виджетов: тот же снимок + поле world."""
    sim = get_sim()
    ps = sim.public_state()
    ps["world"] = ps.get("current_world", "humanoid")
    return ps


@app.get("/api/agent/messages")
def api_agent_messages(last_n: int = Query(default=50, ge=1, le=200)):
    """Phase L: история речи агента для чата."""
    verbal = getattr(get_sim(), "_verbal", None)
    if verbal is None:
        return {"messages": [], "available": False}
    return {
        "messages": verbal.get_messages_for_ui(last_n=last_n),
        "available": True,
        "stats": verbal.snapshot(),
    }


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

class LLMBootstrapRequest(BaseModel):
    world:          str | None = None
    mode:           Literal["rag_wiki", "humanoid_structured"] = "humanoid_structured"
    max_hypotheses: int        = 28
    llm_model:      str        = Field(default_factory=get_ollama_model)
    llm_url:        str        = Field(default_factory=get_ollama_generate_url)

@app.post("/bootstrap/llm")
async def bootstrap_llm(req: LLMBootstrapRequest):
    sim  = get_sim()
    ctx  = sim.agent_seed_context(0)
    if not ctx:
        return {"error": "no context"}
    preset = req.world or sim.current_world
    vars_  = ctx["variables"]
    seeder = RAGSeeder(llm_url=req.llm_url, llm_model=req.llm_model)

    if req.mode == "humanoid_structured":
        # current_world остаётся humanoid и в visual/hybrid режиме
        if preset != "humanoid":
            return {"error": "humanoid_structured requires current_world humanoid"}
        if not req.llm_url:
            return {"error": "llm_url required for humanoid_structured"}
        try:
            from engine.environment_humanoid import (
                humanoid_hardcoded_seeds,
                merge_humanoid_golden_with_llm_edges,
            )

            hyps = await seeder.generate_humanoid_structured(
                available_vars=vars_,
                max_hypotheses=req.max_hypotheses,
            )
            if hyps:
                edges = merge_humanoid_golden_with_llm_edges([h.to_dict() for h in hyps])
                src = "llm_humanoid_merged"
            else:
                edges = humanoid_hardcoded_seeds()
                src = "hardcoded_fallback"
            res = sim.inject_seeds(agent_id=0, edges=edges)
            return {"source": src, "preset": preset, **res}
        except Exception as e:
            return {"error": str(e)}

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
        if preset == "humanoid" and seeder.llm_url:
            from engine.environment_humanoid import (
                humanoid_hardcoded_seeds,
                merge_humanoid_golden_with_llm_edges,
            )

            hyps = await seeder.generate_humanoid_structured(
                available_vars=ctx["variables"],
                max_hypotheses=28,
            )
            if hyps:
                edges = merge_humanoid_golden_with_llm_edges([h.to_dict() for h in hyps])
                source = "llm_humanoid_merged"
            else:
                edges = humanoid_hardcoded_seeds()
                source = "hardcoded_fallback"
        else:
            hyps = await seeder.generate(
                env_preset=preset, available_vars=ctx["variables"], max_hypotheses=6
            )
            edges = [h.to_dict() for h in hyps] if hyps else HARDCODED_SEEDS.get(preset, [])
            source = hyps[0].source if hyps else "hardcoded"
    except Exception:
        if preset == "humanoid":
            from engine.environment_humanoid import humanoid_hardcoded_seeds

            edges, source = humanoid_hardcoded_seeds(), "hardcoded_fallback"
        else:
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


class VisionVLMRequest(BaseModel):
    """Фаза 2: разметка slot_k через Ollama (/api/chat + images или текстовый fallback)."""
    llm_url: str = Field(default_factory=get_ollama_generate_url)
    llm_model: str = Field(default_factory=get_ollama_model)
    max_mask_images: int = 4
    text_only: bool = False
    inject_weak_edges: bool = False

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
      slot_lexicon_tick, slot_lexicon_frame_hash — метаданные последней разметки
      cortex:      dict — stats
    """
    sim = get_sim()
    return sim.get_vision_state()


@app.post("/vision/vlm-label")
async def vision_vlm_label(req: VisionVLMRequest):
    """
    Фаза 2: один вызов VLM (кадр + до K масок) → словарь меток на слотах.
    При ошибке vision API — автоматический текстовый fallback (числа слотов).
    inject_weak_edges: очень слабые рёбра slot→phys в граф (опционально).
    """
    sim = get_sim()
    return await sim.vlm_label_slots(
        llm_url=req.llm_url,
        llm_model=req.llm_model,
        max_mask_images=req.max_mask_images,
        text_only=req.text_only,
        inject_weak_edges=req.inject_weak_edges,
    )


# ── Фаза 3: виртуальный учитель ───────────────────────────────────────────────
@app.post("/teacher/refresh")
async def teacher_refresh():
    """
    Повторный запрос LLM: ig_rules (бонус к actual_ig при совпадении do(var)) +
    vl_overlay с TTL. teacher_weight всё равно затухает с RKK_TEACHER_T_MAX интервенций.
    """
    return await get_sim().refresh_phase3_teacher_llm()


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
    sim._sleep_prev_fixed_root = sim._fixed_root_active
    if not sim._fixed_root_active:
        sim.enable_fixed_root()
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