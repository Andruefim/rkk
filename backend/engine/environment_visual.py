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

import numpy as np
import torch
import base64
from typing import Callable

from engine.causal_vision import CausalVisualCortex, make_visual_cortex


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

        # Инициализируем visual cortex
        self.cortex = make_visual_cortex(device, n_slots=n_slots)
        print(f"[VisualEnv] mode={mode}, slots={n_slots}, base={base_env.preset}")

        # Кэш последнего кадра и слотов
        self._last_frame:    np.ndarray | None    = None
        self._last_slots:    torch.Tensor | None  = None   # (K,)
        self._last_slot_vecs: torch.Tensor | None = None   # (K,D)
        self._last_attn:     torch.Tensor | None  = None   # (K,H',W')

        # GNN prediction для текущего шага (заполняется агентом)
        self._gnn_predicted: torch.Tensor | None  = None

        # Начальная инициализация
        self._refresh()

    # ── Frame acquisition ─────────────────────────────────────────────────────
    def _get_raw_frame(self) -> np.ndarray | None:
        """Получаем numpy RGB frame из base_env."""
        # PyBullet env: get_frame_base64 → декодируем
        fn = getattr(self.base_env, "get_frame_base64", None)
        if callable(fn):
            b64 = None
            for view in ("ego", "diag"):
                b64 = fn(view)
                if b64:
                    break
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

    def _refresh(self):
        """Обновляем слоты из нового кадра."""
        frame = self._get_raw_frame()
        self._last_frame = frame

        if frame is not None:
            vals, vecs, attn = self.cortex.encode(frame)
            self._last_slots     = vals    # (K,) float in (0,1)
            self._last_slot_vecs = vecs    # (K,D)
            self._last_attn      = attn    # (K,H',W')
        else:
            # Fallback: берём ручные переменные base_env → псевдо-слоты
            raw_obs = self.base_env.observe()
            vals    = list(raw_obs.values())
            K       = self.n_slots
            # Интерполируем N ручных переменных в K слотов
            indices = np.linspace(0, len(vals)-1, K)
            pseudo  = np.array([vals[int(round(i))] for i in indices], dtype=np.float32)
            self._last_slots = torch.from_numpy(pseudo).to(self.device)

    # ── Observe ───────────────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        if self._last_slots is None:
            self._refresh()
        slots = self._last_slots
        obs = {f"slot_{k}": float(slots[k].item()) for k in range(self.n_slots)}

        # Hybrid mode: добавляем несколько ключевых ручных переменных
        if self.mode == "hybrid":
            raw = self.base_env.observe()
            for key in ["com_z", "torso_roll", "lknee", "rknee"]:
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
            raw = self.base_env.observe()
            for key in ["com_z", "torso_roll", "lknee", "rknee"]:
                if key in raw:
                    ids.append(f"phys_{key}")
        return ids

    # ── do() ──────────────────────────────────────────────────────────────────
    def intervene(self, variable: str, value: float) -> dict[str, float]:
        """
        Интервенция через физику + re-encode нового кадра.

        Slot → physical action mapping:
          "slot_K" → маппим на ближайший joint base_env по вариабельности
        """
        self.n_interventions += 1
        frame_before = self._last_frame

        # Маппинг слота на физическое действие
        if variable.startswith("slot_"):
            slot_idx = int(variable.split("_")[1])
            phys_var = self._slot_to_physical(slot_idx, value)
            if phys_var is not None:
                self.base_env.intervene(phys_var, value)
            else:
                # Нет маппинга — шагаем физику без воздействия
                step_fn = getattr(self.base_env, "_sim", None)
                if step_fn is not None:
                    step_fn.step(8)
        elif variable.startswith("phys_"):
            # Hybrid mode: прямое воздействие на физическую переменную
            phys_key = variable[5:]
            self.base_env.intervene(phys_key, value)
        else:
            self.base_env.intervene(variable, value)

        # Рендерим новый кадр и кодируем
        self._refresh()

        # Predictive coding: обучаем зрение на prediction error
        if (frame_before is not None and self._last_frame is not None
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

    def get_slot_visualization(self) -> dict:
        """Данные для UI: кадр + slot masks."""
        result: dict = {
            "frame":      None,
            "masks":      [],
            "slot_values": [],
            "variability": [],
            "active_slots": 0,
        }
        if self._last_frame is not None:
            import base64 as _b64
            from io import BytesIO
            try:
                from PIL import Image as PILImage
                img = PILImage.fromarray(self._last_frame)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=75)
                result["frame"] = _b64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass

        if self._last_attn is not None:
            result["masks"] = self.cortex.get_slot_masks_base64(self._last_attn)

        if self._last_slots is not None:
            result["slot_values"] = [round(float(v), 4)
                                     for v in self._last_slots.cpu().numpy()]

        var = self.cortex.slot_variability()
        result["variability"]   = [round(float(v), 4) for v in var]
        result["active_slots"]  = int((var > 0.02).sum())
        return result

    def get_frame_base64(self, view: str = "diag") -> str | None:
        fn = getattr(self.base_env, "get_frame_base64", None)
        return fn(view) if callable(fn) else None

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
        self._refresh()

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
