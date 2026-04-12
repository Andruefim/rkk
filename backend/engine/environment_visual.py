"""
environment_visual.py — Visual Environment Wrapper (Фаза 12).

Оборачивает любую среду (EnvironmentHumanoid, EnvironmentPyBullet и т.д.),
заменяя observe() на зрительное восприятие через SlotAttention.

Интерфейс агента не меняется:
  observe()    → dict[str, float]  но ключи теперь "slot_0"..."slot_K-1"
  intervene()  → dict[str, float]  физика работает как раньше
  variable_ids → ["slot_0"..."slot_K-1"]

Фаза 12: два режима
  VISUAL_ONLY  — только слоты (агент не видит ручных переменных)
  HYBRID       — слоты + ручные переменные (мягкий переход)

Causal loop:
  do(slot_i = v) → physical env.intervene() → render frame →
      → visual_cortex.encode() → new slot_values → GNN prediction error →
      → visual_cortex.train_on_prediction_error()

Fallback:
  Если кадр недоступен (fallback env) → используем ручные переменные
  as pseudo-slots (масштабируем до K значений).
"""
from __future__ import annotations

import os
import queue
import threading
import numpy as np
import torch
import base64
from typing import Callable

from engine.environment_humanoid import MOTOR_INTENT_VARS, MOTOR_OBSERVABLE_VARS
from engine.causal_vision import CausalVisualCortex, make_visual_cortex
from engine.slot_lexicon import frame_content_hash

# Рендер камеры для слотов: меньше пикселей → быстрее PyBullet + JPEG (превью /camera — отдельно)
VISION_PIPELINE_CAM_W = 288
VISION_PIPELINE_CAM_H = 216
VISION_PIPELINE_JPEG_Q = 72
# Полные маски для UI — не на каждый _refresh (дорого: 8× JPEG + upscale)
VISION_UI_MASK_EVERY = 3
# Полный камера+encode раз в N интервенций (между — старые слоты, свежие phys_*)
VISION_ENCODE_EVERY = 6
# Фоновый cortex.encode + JPEG (очередь maxsize=1); GPU с второго потока — только если устраивает драйвер
VISION_ASYNC_ENCODE = os.environ.get("RKK_VISION_ASYNC_ENCODE", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# Hybrid: суставы/поза + sandbox наблюдаемые переменные, чтобы L4 концепты
# могли коррелировать с lever_pin / floor_friction / target_dist и т.д.
_HYBRID_PHYS_KEYS = (
    "com_z", "torso_roll", "lknee", "rknee",
    "spine_yaw", "spine_pitch", "neck_yaw", "neck_pitch",
    "lshoulder", "rshoulder",
    "lever_pin", "target_dist",
    "floor_friction", "stack_height", "stability_score",
    *MOTOR_INTENT_VARS,
    *MOTOR_OBSERVABLE_VARS,
)

# do(phys_*) в hybrid: только реально управляемые моторные ключи;
# прочие phys_* — наблюдаемые (для графа/концептов), не прямые actuators.
_HYBRID_CONTROLLABLE_KEYS = {
    "com_z", "torso_roll", "lknee", "rknee",
    "spine_yaw", "spine_pitch", "neck_yaw", "neck_pitch",
    "lshoulder", "rshoulder",
    *MOTOR_INTENT_VARS,
}


# ─── Имена переменных ─────────────────────────────────────────────────────────
def slot_var_ids(n_slots: int) -> list[str]:
    return [f"slot_{k}" for k in range(n_slots)]


# ─── EnvironmentVisual ────────────────────────────────────────────────────────
class EnvironmentVisual:
    """
    Visual wrapper для любого environment.

    Использование:
      base_env = EnvironmentHumanoid(device=device)
      vis_env  = EnvironmentVisual(base_env, device, n_slots=8)

      # Теперь агент видит только слоты:
      obs = vis_env.observe()  # {"slot_0": 0.52, "slot_1": 0.31, ...}
      vis_env.intervene("slot_3", 0.7)  # физическое действие через base_env

    Slot→action mapping:
      Слот с наибольшей вариабельностью маппится на соответствующий
      joint переменной base_env. В hybrid-режиме можно делать do() по имени сустава.
    """

    PRESET = "visual"

    def __init__(
        self,
        base_env,
        device:  torch.device,
        n_slots: int   = 8,
        mode:    str   = "hybrid",   # "hybrid" = слоты + моторные phys_* (рекомендуется); "visual" = только слоты
    ):
        self.base_env    = base_env
        self.device      = device
        self.n_slots     = n_slots
        self.mode        = mode
        self.preset      = f"visual({base_env.preset})"
        self.n_interventions = 0

        # Инициализируем visual cortex (все веса на том же device, что и GNN)
        self.cortex = make_visual_cortex(device, n_slots=n_slots)
        self.cortex.to(device)
        _p = next(self.cortex.parameters())
        print(
            f"[VisualEnv] mode={mode}, slots={n_slots}, base={base_env.preset}, "
            f"cortex_device={_p.device}"
        )

        # Кэш последнего кадра и слотов
        self._last_frame:    np.ndarray | None    = None
        self._last_slots:    torch.Tensor | None  = None   # (K,)
        self._last_slot_vecs: torch.Tensor | None = None   # (K,D)
        self._last_attn:     torch.Tensor | None  = None   # (K,H',W')

        # Тяжёлая визуализация для UI: считаем один раз при encode, не на каждый WS-тик
        self._cached_frame_b64: str | None = None
        self._cached_masks_b64: list[str]  = []

        # GNN prediction для текущего шага (заполняется агентом)
        self._gnn_predicted: torch.Tensor | None  = None

        # Поколение «визуального» снимка: кэш base_env.observe() в hybrid между вызовами observe()
        self._vision_generation = 0
        self._hybrid_phys_obs: dict[str, float] | None = None
        self._hybrid_phys_gen = -1
        self._refresh_index = 0
        self._vision_stride_counter = 0

        # Фаза 2: VLM-лексикон по индексу слота (после Hungarian; ключи slot_0…)
        self._slot_lexicon: dict[str, dict] = {}
        self._slot_lexicon_tick: int = -1
        self._slot_lexicon_frame_hash: str = ""

        self._async_encode = VISION_ASYNC_ENCODE
        self._encode_queue: queue.Queue[tuple[int, np.ndarray] | None] | None = None
        self._encode_thread: threading.Thread | None = None
        self._encode_ui_lock = threading.Lock()
        if self._async_encode:
            self._encode_queue = queue.Queue(maxsize=1)
            self._encode_thread = threading.Thread(
                target=self._encode_worker, daemon=True, name="RKK-vision-encode"
            )
            self._encode_thread.start()

        # Начальная инициализация (первый проход синхронно, чтобы слоты были заданы)
        self._refresh(run_encode=True, force_sync=True)

    # ── Frame acquisition ─────────────────────────────────────────────────────
    def _get_raw_frame(self) -> np.ndarray | None:
        """Получаем numpy RGB frame из base_env."""
        # PyBullet env: get_frame_base64 → декодируем
        fn = getattr(self.base_env, "get_frame_base64", None)
        if callable(fn):
            b64 = None
            try:
                b64 = fn(
                    None,
                    width=VISION_PIPELINE_CAM_W,
                    height=VISION_PIPELINE_CAM_H,
                    jpeg_quality=VISION_PIPELINE_JPEG_Q,
                )
            except TypeError:
                b64 = fn(None)
            if b64:
                import base64 as _b64
                from io import BytesIO
                try:
                    from PIL import Image as PILImage
                    raw  = _b64.b64decode(b64)
                    img  = PILImage.open(BytesIO(raw)).convert("RGB")
                    return np.array(img, dtype=np.uint8)
                except Exception:
                    pass
        return None

    def _encode_frame_jpeg_only(self, frame: np.ndarray, quality: int = 65) -> str | None:
        import base64 as _b64
        from io import BytesIO
        try:
            from PIL import Image as PILImage
            buf = BytesIO()
            PILImage.fromarray(frame).save(
                buf, format="JPEG", quality=quality, optimize=True
            )
            return _b64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    def _encode_worker(self) -> None:
        q = self._encode_queue
        if q is None:
            return
        while True:
            item = q.get()
            if item is None:
                break
            _gen, frame = item
            try:
                vals, vecs, attn = self.cortex.encode(frame)
                self._last_slots = vals
                self._last_slot_vecs = vecs
                self._last_attn = attn
                self._cached_frame_b64 = self._encode_frame_jpeg_only(frame, quality=68)
                with self._encode_ui_lock:
                    self._refresh_index += 1
                    ri = self._refresh_index
                    if ri % VISION_UI_MASK_EVERY == 0:
                        _, self._cached_masks_b64 = self._encode_frame_and_masks_for_ui(
                            frame, attn
                        )
                    elif not self._cached_masks_b64:
                        _, self._cached_masks_b64 = self._encode_frame_and_masks_for_ui(
                            frame, attn
                        )
            except Exception:
                pass

    def _refresh(self, run_encode: bool = True, force_sync: bool = False) -> None:
        """
        run_encode=True: камера + cortex.encode + UI-кэш.
        False: только смена поколения hybrid (свежий base_env.observe), слоты с прошлого encode.
        force_sync: игнорировать фоновый encode (первый кадр, смена fixed_root).
        """
        self._vision_generation += 1
        self._hybrid_phys_gen = -1
        if not run_encode:
            return

        frame = self._get_raw_frame()
        self._last_frame = frame

        use_async = (
            self._async_encode
            and self._encode_queue is not None
            and not force_sync
        )
        if use_async and frame is not None:
            try:
                self._encode_queue.put_nowait(
                    (self._vision_generation, np.ascontiguousarray(frame))
                )
            except queue.Full:
                try:
                    _ = self._encode_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._encode_queue.put_nowait(
                        (self._vision_generation, np.ascontiguousarray(frame))
                    )
                except queue.Full:
                    pass
            return

        self._refresh_index += 1

        if frame is not None:
            vals, vecs, attn = self.cortex.encode(frame)
            self._last_slots     = vals    # (K,) float in (0,1)
            self._last_slot_vecs = vecs    # (K,D)
            self._last_attn      = attn    # (K,H',W')
            self._cached_frame_b64 = self._encode_frame_jpeg_only(frame, quality=68)
            if self._refresh_index % VISION_UI_MASK_EVERY == 0:
                _, self._cached_masks_b64 = self._encode_frame_and_masks_for_ui(frame, attn)
            elif not self._cached_masks_b64:
                _, self._cached_masks_b64 = self._encode_frame_and_masks_for_ui(frame, attn)
        else:
            self._last_attn = None
            self._cached_frame_b64 = None
            self._cached_masks_b64 = []
            # Fallback: берём ручные переменные base_env → псевдо-слоты
            raw_obs = self.base_env.observe()
            vals    = list(raw_obs.values())
            K       = self.n_slots
            # Интерполируем N ручных переменных в K слотов
            indices = np.linspace(0, len(vals)-1, K)
            pseudo  = np.array([vals[int(round(i))] for i in indices], dtype=np.float32)
            self._last_slots = torch.from_numpy(pseudo).to(self.device)

    def _encode_frame_and_masks_for_ui(
        self, frame: np.ndarray, attn: torch.Tensor
    ) -> tuple[str | None, list[str]]:
        """Один JPEG кадра + base64 PNG масок — вызывать только после encode."""
        import base64 as _b64
        from io import BytesIO
        frame_b64: str | None = None
        try:
            from PIL import Image as PILImage
            buf = BytesIO()
            PILImage.fromarray(frame).save(buf, format="JPEG", quality=65, optimize=True)
            frame_b64 = _b64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass
        try:
            masks = self.cortex.get_slot_masks_base64(attn)
        except Exception:
            masks = []
        return frame_b64, masks

    def _hybrid_phys_snapshot(self) -> dict[str, float]:
        if self._hybrid_phys_gen != self._vision_generation:
            self._hybrid_phys_obs = self.base_env.observe()
            self._hybrid_phys_gen = self._vision_generation
        if self._hybrid_phys_obs is None:
            self._hybrid_phys_obs = self.base_env.observe()
        return self._hybrid_phys_obs

    # ── Observe ───────────────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        if self._last_slots is None:
            self._refresh(run_encode=True, force_sync=True)
        slots = self._last_slots
        obs = {f"slot_{k}": float(slots[k].item()) for k in range(self.n_slots)}

        # Hybrid mode: один вызов base_env.observe() на поколение кадра (много обращений за тик)
        if self.mode == "hybrid":
            raw = self._hybrid_phys_snapshot()
            for key in _HYBRID_PHYS_KEYS:
                if key in raw:
                    obs[f"phys_{key}"] = raw[key]
        return obs

    @property
    def variables(self) -> dict[str, float]:
        return self.observe()

    @property
    def variable_ids(self) -> list[str]:
        ids = slot_var_ids(self.n_slots)
        if self.mode == "hybrid":
            raw = self._hybrid_phys_snapshot()
            for key in _HYBRID_PHYS_KEYS:
                if key in raw:
                    ids.append(f"phys_{key}")
        return ids

    # ── do() ──────────────────────────────────────────────────────────────────
    def intervene(
        self, variable: str, value: float, *, count_intervention: bool = True
    ) -> dict[str, float]:
        """
        Интервенция через физику; полный камера+encode — раз в VISION_ENCODE_EVERY шагов.

        Slot → physical action mapping:
          "slot_K" → маппим на ближайший joint base_env по вариабельности
        count_intervention=False — Phase B skills: не увеличивать счётчик интервенций.
        """
        if count_intervention:
            self.n_interventions += 1
        frame_before = self._last_frame

        def _base_iv(v: str, x: float) -> None:
            try:
                self.base_env.intervene(v, x, count_intervention=count_intervention)
            except TypeError:
                self.base_env.intervene(v, x)

        # Маппинг слота на физическое действие
        if variable.startswith("slot_"):
            slot_idx = int(variable.split("_")[1])
            phys_var = self._slot_to_physical(slot_idx, value)
            if phys_var is not None:
                _base_iv(phys_var, value)
            else:
                # Нет маппинга — шагаем физику без воздействия
                step_fn = getattr(self.base_env, "_sim", None)
                if step_fn is not None:
                    step_fn.step(8)
        elif variable.startswith("phys_"):
            # Hybrid mode: прямое воздействие только на управляемые моторные phys_*.
            # Наблюдаемые sandbox-переменные остаются read-only.
            phys_key = variable[5:]
            if phys_key in _HYBRID_CONTROLLABLE_KEYS:
                _base_iv(phys_key, value)
            else:
                step_fn = getattr(self.base_env, "_sim", None)
                if step_fn is not None:
                    step_fn.step(8)
        else:
            _base_iv(variable, value)

        self._vision_stride_counter += 1
        run_encode = (self._vision_stride_counter % VISION_ENCODE_EVERY == 1)
        self._refresh(run_encode=run_encode)

        # Predictive coding только когда был свежий кадр/slots
        if (run_encode and frame_before is not None and self._last_frame is not None
                and self._gnn_predicted is not None):
            try:
                loss = self.cortex.train_on_prediction_error(
                    frame_before,
                    self._last_frame,
                    self._gnn_predicted[:self.n_slots],
                )
            except Exception:
                pass

        return self.observe()

    # ── Slot → Physical action ────────────────────────────────────────────────
    def _slot_to_physical(self, slot_idx: int, value: float) -> str | None:
        """
        Маппим slot_idx на физическую переменную base_env.
        Используем round-robin по переменным ног/рук.
        """
        # Приоритет: сначала суставы (каузально богаче)
        base_vars = getattr(self.base_env, "variable_ids", [])
        joint_vars = [v for v in base_vars
                      if any(k in v for k in
                             ["hip","knee","ankle","shoulder","elbow","j0","j1","j2"])]
        if not joint_vars:
            return None
        return joint_vars[slot_idx % len(joint_vars)]

    # ── GT для discovery rate ─────────────────────────────────────────────────
    def discovery_rate(self, agent_edges: list[dict]) -> float:
        """
        В визуальном режиме discovery rate = % активных слотов
        (слотов с достаточной вариабельностью).
        """
        variability = self.cortex.slot_variability()
        active      = float((variability > 0.02).mean())
        # Дополнительно: насколько связан граф (есть рёбра между слотами)
        if agent_edges:
            edge_rate = min(len(agent_edges) / (self.n_slots * 2), 1.0)
            return 0.5 * active + 0.5 * edge_rate
        return active * 0.5

    def gt_edges(self) -> list[dict]:
        """Нет GT в визуальном режиме — агент всё открывает сам."""
        return []

    # ── Visual cortex helpers ─────────────────────────────────────────────────
    def set_gnn_prediction(self, predicted: torch.Tensor):
        """Агент сообщает что GNN предсказал → для predictive coding."""
        self._gnn_predicted = predicted.detach()

    def set_slot_lexicon(
        self,
        labels: dict[str, dict],
        tick: int,
        frame_b64: str | None = None,
    ) -> None:
        """Сохраняем последнюю VLM-разметку (метки по slot_k, не меняя порядок слотов).

        Phase M: после полного батча VLM вызывается simulation.vlm_label_slots →
        _phase_m_sync_from_vision(). In-place правки лексикона делаются в visual_grounding.
        """
        self._slot_lexicon = dict(labels)
        self._slot_lexicon_tick = tick
        self._slot_lexicon_frame_hash = frame_content_hash(frame_b64)

    def clear_slot_lexicon(self) -> None:
        self._slot_lexicon = {}
        self._slot_lexicon_tick = -1
        self._slot_lexicon_frame_hash = ""

    def set_fixed_root(self, enabled: bool) -> None:
        """
        Прокидываем fixed_root в base_env и сбрасываем кэш hybrid phys / лексикона,
        чтобы variable_ids совпали с новым набором суставов базы.
        """
        fn = getattr(self.base_env, "set_fixed_root", None)
        if callable(fn):
            fn(enabled)
        self._hybrid_phys_gen = -1
        self._hybrid_phys_obs = None
        self.clear_slot_lexicon()
        self._refresh(run_encode=True, force_sync=True)

    def get_slot_visualization(self) -> dict:
        """Данные для UI: кадр + slot masks (кэш из _refresh, без PIL/cv2 на каждый тик)."""
        result: dict = {
            "frame":      self._cached_frame_b64,
            "masks":      list(self._cached_masks_b64),
            "slot_values": [],
            "variability": [],
            "active_slots": 0,
        }

        if self._last_slots is not None:
            result["slot_values"] = [round(float(v), 4)
                                     for v in self._last_slots.cpu().numpy()]

        var = self.cortex.slot_variability()
        result["variability"]   = [round(float(v), 4) for v in var]
        result["active_slots"]  = int((var > 0.02).sum())

        # Фаза 2: подписи слотов (параллельно индексу 0…K-1)
        lex = self._slot_lexicon
        slot_labels: list[dict] = []
        for k in range(self.n_slots):
            sid = f"slot_{k}"
            e = lex.get(sid)
            if e:
                slot_labels.append(
                    {
                        "slot_id": sid,
                        "label": e.get("label"),
                        "likely_phys": list(e.get("likely_phys") or []),
                        "confidence": round(float(e.get("confidence", 0)), 3),
                    }
                )
            else:
                slot_labels.append(
                    {
                        "slot_id": sid,
                        "label": None,
                        "likely_phys": [],
                        "confidence": 0.0,
                    }
                )
        result["slot_labels"] = slot_labels
        result["slot_lexicon_tick"] = self._slot_lexicon_tick
        result["slot_lexicon_frame_hash"] = self._slot_lexicon_frame_hash
        return result

    def get_frame_base64(self, view: str | None = None) -> str | None:
        fn = getattr(self.base_env, "get_frame_base64", None)
        return fn(None) if callable(fn) else None

    def get_full_scene(self) -> dict:
        fn = getattr(self.base_env, "get_full_scene", None)
        scene = fn() if callable(fn) else {}
        scene["visual_cortex"] = self.cortex.snapshot()
        return scene

    def get_joint_positions_world(self):
        fn = getattr(self.base_env, "get_joint_positions_world", None)
        return fn() if callable(fn) else []

    def get_cube_positions(self):
        fn = getattr(self.base_env, "get_cube_positions", None)
        return fn() if callable(fn) else []

    def get_target(self):
        fn = getattr(self.base_env, "get_target", None)
        return fn() if callable(fn) else {"x": 0, "y": 0, "z": 0.9}

    def is_fallen(self) -> bool:
        fn = getattr(self.base_env, "is_fallen", None)
        return fn() if callable(fn) else False

    def reset_stance(self) -> None:
        fn = getattr(self.base_env, "reset_stance", None)
        if callable(fn):
            fn()
        self._vision_stride_counter = 0
        self.clear_slot_lexicon()
        self._refresh(run_encode=True, force_sync=True)

    # ── Seeds для visual env ─────────────────────────────────────────────────
    def hardcoded_seeds(self) -> list[dict]:
        """Инициализируем парные связи между соседними слотами."""
        seeds = []
        K = self.n_slots
        for i in range(K - 1):
            seeds.append({
                "from_": f"slot_{i}",
                "to":    f"slot_{i+1}",
                "weight": 0.1,
                "alpha":  0.05,
            })
        # Циклическая связь
        seeds.append({
            "from_": f"slot_{K-1}",
            "to":    "slot_0",
            "weight": 0.08,
            "alpha":  0.05,
        })
        return seeds
