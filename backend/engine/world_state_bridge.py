"""
world_state_bridge.py — Единый мост «мир ↔ семантика» (AGI-направление).

Идея:
  - Один вектор состояния (GNN snapshot) — источник истины для inner voice.
  - Каждый тик после зрения (Phase M): собираем **якорные** семантические метки
    (bodily из InnerVoice + визуальные через явное отображение в SemanticConceptStore).
  - В буфер кладём переход (s_t, a_t, s_{t+1}, labels_{t+1}): метки относятся к
    состоянию **после** шага — предиктивное заземление (thought(s_t) → концепты мира после динамики),
    а не голая корреляция картинка↔текст.

Сон (Phase K REM): grounded_sleep_consolidate() — distillation samples в InnerVoice
с теми же метками, что наблюдались в симуляции.

RKK_WORLD_BRIDGE_ENABLED=1
RKK_WORLD_BRIDGE_MAX=512
RKK_WORLD_BRIDGE_FIXED_ROOT=0   — не писать буфер при fixed_root (только живое тело)
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.concept_store import CONCEPT_DEFS


def world_bridge_enabled() -> bool:
    return os.environ.get("RKK_WORLD_BRIDGE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _env_int(key: str, default: int, *, min_val: int = 1) -> int:
    try:
        return max(min_val, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# Явные мосты: визуальный концепт → один или несколько имён из SemanticConceptStore.
# Только существующие имена из CONCEPT_DEFS (проверка в рантайме по name_to_idx).
VISUAL_TO_SEMANTIC: dict[str, list[str]] = {
    # Риск / сложный рельеф
    "RAMP_DETECTED": ["HIGH_FALL_RISK", "LOSING_BALANCE"],
    "INCLINED_SURFACE": ["HIGH_FALL_RISK"],
    "STAIRS_DETECTED": ["HIGH_FALL_RISK", "LOW_STABILITY"],
    "BEAM_DETECTED": ["HIGH_FALL_RISK", "CRITICAL_STABILITY"],
    "NARROW_OBJECT": ["HIGH_FALL_RISK"],
    # Препятствия / навигация
    "OBJECT_BLOCKING_PATH": ["INTENT_SLOW_DOWN", "LOSING_BALANCE"],
    "OBJECT_VERY_CLOSE": ["INTENT_SLOW_DOWN"],
    "CLUTTERED_SCENE": ["LOSING_BALANCE", "LOW_STABILITY"],
    # Свободный путь
    "CLEAR_PATH_AHEAD": ["EXPLOITING_KNOWN", "LOW_CURIOSITY"],
    "OPEN_SCENE": ["EXPLOITING_KNOWN"],
    # Объекты для взаимодействия
    "BALL_DETECTED": ["INTENT_EXPLORE", "LEARNING_OPPORTUNITY"],
    "CONTAINER_DETECTED": ["INTENT_EXPLORE"],
    "OBJECT_MOVING": ["LEARNING_OPPORTUNITY", "HIGH_CURIOSITY"],
    # Новизна
    "FIRST_TIME_SEEING": ["LEARNING_OPPORTUNITY", "HIGH_CURIOSITY"],
    "UNEXPECTED_OBJECT": ["LEARNING_OPPORTUNITY", "HIGH_CURIOSITY"],
}


_VALID_SEMANTIC = {name for name, _, _ in CONCEPT_DEFS}


def _graph_state_vec(sim) -> list[float]:
    g = sim.agent.graph
    node_ids = list(g._node_ids) if hasattr(g, "_node_ids") else list(g.nodes.keys())
    return [float(g.nodes.get(n, 0.5)) for n in node_ids]


def collect_grounded_semantic_labels(sim) -> list[str]:
    """
    Bodily (inner voice) + визуальные концепты, отфильтрованные/смапленные в SemanticConceptStore.
    Порядок: сначала телесные (выше приоритет в train_step), затем мосты из зрения.
    """
    iv = getattr(sim, "_inner_voice", None)
    if iv is None:
        return []

    store = iv.concept_store
    out: list[str] = []
    seen: set[str] = set()
    iv_thr = float(getattr(store, "activation_threshold", 0.55))

    for name, score in iv.get_active_concepts():
        if name in store.name_to_idx and name not in seen and float(score) >= iv_thr:
            out.append(name)
            seen.add(name)
        if len(out) >= 8:
            return out[:8]

    vis_thr = _env_float("RKK_WORLD_BRIDGE_VISUAL_CONF", 0.32)
    vv = getattr(sim, "_visual_voice", None)
    if vv is not None:
        for name, score in vv.get_active_concepts():
            if float(score) < vis_thr:
                continue
            for sem in VISUAL_TO_SEMANTIC.get(name, []):
                if sem in store.name_to_idx and sem not in seen and sem in _VALID_SEMANTIC:
                    out.append(sem)
                    seen.add(sem)
                if len(out) >= 8:
                    return out[:8]

    # Эвристика по устойчивости (наблюдаемая физика, не корреляция с пикселями)
    try:
        obs = dict(sim.agent.env.observe())
        ps = float(
            obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
        )
        cz = float(obs.get("com_z", obs.get("phys_com_z", 0.8)))
        if ps < 0.32 and "LOW_STABILITY" in store.name_to_idx and "LOW_STABILITY" not in seen:
            out.append("LOW_STABILITY")
            seen.add("LOW_STABILITY")
        if cz < 0.45 and "HIGH_FALL_RISK" in store.name_to_idx and "HIGH_FALL_RISK" not in seen:
            out.append("HIGH_FALL_RISK")
            seen.add("HIGH_FALL_RISK")
    except Exception:
        pass

    return out[:8]


@dataclass
class WorldTransition:
    tick: int
    state_before: list[float]
    state_after: list[float]
    action_var: str | None
    action_val: float | None
    labels_after: list[str] = field(default_factory=list)
    state_delta_l2: float = 0.0


class WorldStateBridge:
    """
    Буфер переходов для офлайн консолидации. Запись — в конце тика, после Phase M.
    """

    def __init__(self) -> None:
        self._maxlen = _env_int("RKK_WORLD_BRIDGE_MAX", 512)
        self._transitions: deque[WorldTransition] = deque(maxlen=self._maxlen)
        self._prev_state: list[float] | None = None
        self._prev_tick: int = -1
        self._record_fixed_root = os.environ.get(
            "RKK_WORLD_BRIDGE_FIXED_ROOT", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._min_delta = _env_float("RKK_WORLD_BRIDGE_MIN_DELTA", 0.0)

    def on_tick(self, sim) -> None:
        if not world_bridge_enabled():
            return
        iv = getattr(sim, "_inner_voice", None)
        if iv is None:
            return
        if getattr(sim, "current_world", "") != "humanoid":
            return
        if getattr(sim, "_fixed_root_active", False) and not self._record_fixed_root:
            self._prev_state = None
            return

        state_vec = _graph_state_vec(sim)
        if not state_vec:
            return

        tick = int(getattr(sim, "tick", 0))
        labels = collect_grounded_semantic_labels(sim)

        action = getattr(sim, "_last_action_for_memory", None)
        action_var = action[0] if action else None
        action_val = float(action[1]) if action and len(action) > 1 else None

        # GNN размерность может вырасти (нейрогенез и т.д.) — несопоставимые векторы
        if self._prev_state is not None and len(self._prev_state) != len(state_vec):
            self._prev_state = list(state_vec)
            self._prev_tick = tick
            return

        if self._prev_state is not None and self._prev_tick >= 0 and tick == self._prev_tick + 1:
            a = np.asarray(self._prev_state, dtype=np.float64)
            b = np.asarray(state_vec, dtype=np.float64)
            d = float(np.linalg.norm(a - b))
            if d >= self._min_delta or len(labels) > 0:
                self._transitions.append(
                    WorldTransition(
                        tick=tick,
                        state_before=list(self._prev_state),
                        state_after=list(state_vec),
                        action_var=action_var,
                        action_val=action_val,
                        labels_after=list(labels),
                        state_delta_l2=d,
                    )
                )

        self._prev_state = list(state_vec)
        self._prev_tick = tick

    def snapshot(self) -> dict[str, Any]:
        last = self._transitions[-1] if self._transitions else None
        return {
            "enabled": world_bridge_enabled(),
            "buffer_len": len(self._transitions),
            "maxlen": self._maxlen,
            "last_tick": last.tick if last else None,
            "last_delta_l2": round(last.state_delta_l2, 5) if last else None,
            "last_n_labels": (last.labels_after[:5] if last else []),
        }

    def recent_for_sleep(self, n: int = 64) -> list[WorldTransition]:
        if not self._transitions:
            return []
        return list(self._transitions)[-n:]


def grounded_sleep_consolidate(sim) -> dict[str, Any]:
    """
    Вызывается из Phase K REM: предиктивное заземление inner voice на буфере симуляции.
    """
    bridge = getattr(sim, "_world_bridge", None)
    iv = getattr(sim, "_inner_voice", None)
    if bridge is None or iv is None:
        return {"ok": False, "reason": "no_bridge_or_inner_voice"}

    transitions = bridge.recent_for_sleep(
        _env_int("RKK_WORLD_BRIDGE_SLEEP_BATCH", 64)
    )
    pushed = 0
    for tr in transitions:
        if not tr.labels_after:
            continue
        iv.push_distill_sample(tr.state_before, tr.labels_after)
        pushed += 1

    train_steps = _env_int("RKK_WORLD_BRIDGE_SLEEP_TRAIN_STEPS", 8)
    losses: list[float] = []
    for _ in range(train_steps):
        loss = iv.train_step()
        if loss is not None:
            losses.append(loss)

    return {
        "ok": True,
        "transitions_available": len(transitions),
        "samples_pushed": pushed,
        "train_steps": train_steps,
        "loss_last": round(losses[-1], 6) if losses else None,
    }
