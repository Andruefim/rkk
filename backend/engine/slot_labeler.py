"""
slot_labeler.py — Phase M: SlotLabeler.

Мост между SlotAttention (числовые векторы) и ConceptStore (язык).

Источники (после замены keyword/position-хуков на neural path в engine.neural_lang_integration):
  A. Slot vector — embedding слота → NeuralConceptProjector (в патче process_slots)
  B. Motion / scene / novelty — как раньше (без keyword text→concept и без фиксированной сетки position→spatial)

Выход:
  activated_concepts: list[tuple[str, float]] — (concept_name, confidence)
  → передаётся в InnerVoiceController
  → инжектируется в GNN как concept_visual_* узлы
  → используется в verbal_action.py для world-aware высказываний

Интеграция:
  slot_labeler = SlotLabeler()
  # При каждом VLM update:
  concepts = slot_labeler.process_slots(vlm_labels, slot_positions, slot_vectors)
  inner_voice_ctrl.push_visual_concepts(concepts)

RKK_SLOT_LABEL_ENABLED=1
RKK_SLOT_LABEL_EVERY=30          — тиков между обновлениями
RKK_SLOT_LABEL_MIN_CONF=0.35     — минимальная уверенность концепта
RKK_SLOT_LABEL_MAX_ACTIVE=8      — максимум активных визуальных концептов
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.visual_concepts import VISUAL_CONCEPT_NAMES


def slot_label_enabled() -> bool:
    return os.environ.get("RKK_SLOT_LABEL_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ── Slot state ─────────────────────────────────────────────────────────────────
@dataclass
class SlotState:
    """Current state of one attention slot."""
    slot_idx: int
    label: str                    # VLM text label e.g. "green sphere"
    position_2d: tuple[float, float]  # normalized [0,1] x,y in image
    vector: list[float]           # raw slot embedding
    confidence: float             # slot attention weight
    timestamp: float = field(default_factory=time.time)
    prev_position: tuple[float, float] | None = None  # for motion detection

    def position_delta(self) -> tuple[float, float]:
        if self.prev_position is None:
            return (0.0, 0.0)
        return (
            self.position_2d[0] - self.prev_position[0],
            self.position_2d[1] - self.prev_position[1],
        )

    def is_moving(self, threshold: float = 0.02) -> bool:
        dx, dy = self.position_delta()
        return (dx**2 + dy**2) ** 0.5 > threshold


# ── Motion concepts ────────────────────────────────────────────────────────────
def motion_to_concepts(slot: SlotState) -> list[tuple[str, float]]:
    """Detect motion concepts from slot position delta."""
    if slot.prev_position is None:
        return [("OBJECT_STATIC", 0.6)]
    dx, dy = slot.position_delta()
    speed = (dx**2 + dy**2) ** 0.5
    if speed > 0.05:
        concepts = [("OBJECT_MOVING", 0.85), ("SLOT_CHANGED", 0.7)]
        if "sphere" in slot.label.lower() or "ball" in slot.label.lower():
            concepts.append(("ROLLING_OBJECT", 0.7))
        return concepts
    return [("OBJECT_STATIC", 0.6), ("SCENE_STABLE", 0.4)]


# ── Scene-level concepts ───────────────────────────────────────────────────────
def scene_level_concepts(slots: list[SlotState]) -> list[tuple[str, float]]:
    """Derive scene-wide concepts from all active slots."""
    active = [s for s in slots if s.confidence > 0.15]
    n = len(active)
    concepts: list[tuple[str, float]] = []

    if n == 0:
        concepts.append(("OPEN_SCENE", 0.9))
        concepts.append(("CLEAR_PATH_AHEAD", 0.8))
        concepts.append(("EXPLORING", 0.5))
    elif n == 1:
        concepts.append(("SINGLE_OBJECT", 0.8))
        concepts.append(("OPEN_SCENE", 0.6))
    elif n <= 3:
        concepts.append(("MULTIPLE_OBJECTS", 0.7))
    else:
        concepts.append(("MULTIPLE_OBJECTS", 0.9))
        concepts.append(("CLUTTERED_SCENE", 0.65))

    # Check if any object is ahead/blocking
    ahead_slots = [s for s in active if 0.3 < s.position_2d[0] < 0.7]
    if not ahead_slots:
        concepts.append(("CLEAR_PATH_AHEAD", 0.75))
    elif any(s.position_2d[1] > 0.6 for s in ahead_slots):
        concepts.append(("OBJECT_BLOCKING_PATH", 0.70))

    # Scene symmetry
    left_slots = [s for s in active if s.position_2d[0] < 0.4]
    right_slots = [s for s in active if s.position_2d[0] > 0.6]
    if abs(len(left_slots) - len(right_slots)) <= 1 and n >= 2:
        concepts.append(("SYMMETRIC_SCENE", 0.55))
    elif len(left_slots) > len(right_slots) + 1:
        concepts.append(("ASYMMETRIC_SCENE", 0.55))
        concepts.append(("OBJECT_LEFT", 0.5))
    elif len(right_slots) > len(left_slots) + 1:
        concepts.append(("ASYMMETRIC_SCENE", 0.55))
        concepts.append(("OBJECT_RIGHT", 0.5))

    return concepts


# ── Novelty detection ──────────────────────────────────────────────────────────
class VisualNoveltyDetector:
    """
    Tracks slot vector history to detect novelty.
    High novelty → HIGH_VISUAL_SURPRISE → curiosity in InnerVoiceNet.
    """

    def __init__(self, memory_size: int = 20):
        self._slot_history: deque[list[list[float]]] = deque(maxlen=memory_size)
        self._mean_change: float = 0.0

    def update(self, slot_vectors: list[list[float]]) -> float:
        """Returns novelty score [0,1]."""
        if not self._slot_history or not slot_vectors:
            self._slot_history.append(slot_vectors)
            return 0.5

        prev = self._slot_history[-1]
        if len(prev) != len(slot_vectors):
            self._slot_history.append(slot_vectors)
            return 0.8  # slot count change = high novelty

        # Mean cosine distance between current and previous slot vectors
        changes = []
        for v1, v2 in zip(prev, slot_vectors):
            if len(v1) == len(v2) and len(v1) > 0:
                a = np.array(v1, dtype=float)
                b = np.array(v2, dtype=float)
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    cos_sim = float(np.dot(a, b) / (na * nb))
                    changes.append(1.0 - cos_sim)

        if not changes:
            return 0.3

        novelty = float(np.mean(changes))
        alpha = 0.1
        self._mean_change = (1 - alpha) * self._mean_change + alpha * novelty
        self._slot_history.append(slot_vectors)

        return float(np.clip(novelty / max(self._mean_change + 0.01, 0.01) - 0.5, 0.0, 1.0))

    def get_novelty_concepts(self, novelty: float) -> list[tuple[str, float]]:
        if novelty > 0.7:
            return [("HIGH_VISUAL_SURPRISE", novelty), ("FIRST_TIME_SEEING", novelty * 0.8)]
        elif novelty > 0.4:
            return [("SLOT_CHANGED", novelty), ("CHANGED_SINCE_LAST", novelty * 0.7)]
        else:
            return [("LOW_VISUAL_SURPRISE", 0.6), ("SAME_AS_LAST", 0.5)]


# ── Main SlotLabeler ───────────────────────────────────────────────────────────
class SlotLabeler:
    """
    Central controller for visual concept extraction.

    Интеграция в simulation.py или causal_vision.py:

      labeler = SlotLabeler()

      # При VLM update (каждые RKK_GROUNDING_EVERY тиков):
      visual_concepts = labeler.process_slots(
          vlm_labels={"slot_0": "large green sphere", "slot_3": "ramp surface"},
          slot_positions={"slot_0": (0.5, 0.6), "slot_3": (0.3, 0.4)},
          slot_vectors={"slot_0": [...], "slot_3": [...]},
          slot_confidences={"slot_0": 0.85, "slot_3": 0.72},
      )
      # visual_concepts: [("GREEN_OBJECT", 0.9), ("RAMP_DETECTED", 0.8), ...]

      # Передать в InnerVoiceController:
      inner_voice_ctrl.push_visual_concepts(visual_concepts)
    """

    def __init__(self):
        self._novelty = VisualNoveltyDetector()
        self._every = _env_int("RKK_SLOT_LABEL_EVERY", 30)
        self._min_conf = _env_float("RKK_SLOT_LABEL_MIN_CONF", 0.35)
        self._max_active = _env_int("RKK_SLOT_LABEL_MAX_ACTIVE", 8)
        self._last_tick = -999
        self._last_concepts: list[tuple[str, float]] = []
        self._slot_states: dict[str, SlotState] = {}
        self.total_updates: int = 0

    def process_slots(
        self,
        vlm_labels: dict[str, str],
        slot_positions: dict[str, tuple[float, float]],
        slot_vectors: dict[str, list[float]],
        slot_confidences: dict[str, float] | None = None,
        tick: int = 0,
    ) -> list[tuple[str, float]]:
        """
        Main entry point. Returns activated visual concepts sorted by confidence.
        """
        if not slot_label_enabled():
            return []

        if (tick - self._last_tick) < self._every and tick > 0:
            return self._last_concepts

        self._last_tick = tick
        confs = slot_confidences or {}
        all_concepts: dict[str, float] = {}

        # Update slot states
        slot_list: list[SlotState] = []
        for slot_id, label in vlm_labels.items():
            prev = self._slot_states.get(slot_id)
            pos = slot_positions.get(slot_id, (0.5, 0.5))
            vec = slot_vectors.get(slot_id, [])
            conf = float(confs.get(slot_id, 0.6))

            state = SlotState(
                slot_idx=int(slot_id.split("_")[-1]) if "_" in slot_id else 0,
                label=label,
                position_2d=pos,
                vector=vec,
                confidence=conf,
                prev_position=prev.position_2d if prev else None,
            )
            self._slot_states[slot_id] = state
            slot_list.append(state)

        # A–B: VLM text → concept и position → spatial перенесены в NeuralConceptProjector /
        #      InterventionalSpatialMemory (engine.neural_lang_integration).

        # C. Motion concepts
        for state in slot_list:
            for cname, score in motion_to_concepts(state):
                all_concepts[cname] = max(all_concepts.get(cname, 0.0), score * 0.8)

        # D. Scene-level
        for cname, score in scene_level_concepts(slot_list):
            all_concepts[cname] = max(all_concepts.get(cname, 0.0), score)

        # E. Novelty
        all_vecs = [s.vector for s in slot_list if s.vector]
        novelty = self._novelty.update(all_vecs)
        for cname, score in self._novelty.get_novelty_concepts(novelty):
            all_concepts[cname] = max(all_concepts.get(cname, 0.0), score)

        # Filter and sort
        filtered = [
            (k, v) for k, v in all_concepts.items()
            if v >= self._min_conf and k in VISUAL_CONCEPT_NAMES
        ]
        filtered.sort(key=lambda x: -x[1])
        result = filtered[:self._max_active]

        self._last_concepts = result
        self.total_updates += 1
        return result

    def get_current_concepts(self) -> list[tuple[str, float]]:
        return self._last_concepts

    def get_world_description(self) -> str:
        """One-line natural language description of current visual scene."""
        concepts = [c for c, _ in self._last_concepts]
        if not concepts:
            return ""

        parts = []

        # Object description
        if "RAMP_DETECTED" in concepts:
            parts.append("рампа")
        elif "STAIRS_DETECTED" in concepts:
            parts.append("ступени")
        elif "BEAM_DETECTED" in concepts:
            parts.append("балка")
        elif "SPHERE_DETECTED" in concepts or "BALL_DETECTED" in concepts:
            color = ""
            if "GREEN_OBJECT" in concepts: color = "зелёный "
            elif "BLUE_OBJECT" in concepts: color = "синий "
            elif "RED_OBJECT" in concepts: color = "красный "
            parts.append(f"{color}шар")
        elif "CUBE_DETECTED" in concepts:
            parts.append("куб")
        elif "PLATFORM_DETECTED" in concepts:
            parts.append("платформа")
        elif "SENSOR_PLATE_VISIBLE" in concepts:
            parts.append("сенсорная плита")
        elif "MULTIPLE_OBJECTS" in concepts:
            parts.append("несколько объектов")

        # Position
        if "OBJECT_LEFT" in concepts:
            parts.append("слева")
        elif "OBJECT_RIGHT" in concepts:
            parts.append("справа")
        elif "OBJECT_AHEAD" in concepts or "OBJECT_BLOCKING_PATH" in concepts:
            parts.append("впереди")

        # Distance
        if "OBJECT_VERY_CLOSE" in concepts:
            parts.append("(вплотную)")
        elif "OBJECT_CLOSE" in concepts:
            parts.append("(близко)")
        elif "OBJECT_FAR" in concepts:
            parts.append("(далеко)")

        # Scene state
        if not parts:
            if "OPEN_SCENE" in concepts or "CLEAR_PATH_AHEAD" in concepts:
                return "Путь открыт."
            elif "WALL_NEARBY" in concepts:
                return "Близко стена."

        if parts:
            return " ".join(parts).capitalize() + "."
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": slot_label_enabled(),
            "total_updates": self.total_updates,
            "active_concepts": self._last_concepts[:6],
            "n_slots_tracked": len(self._slot_states),
            "world_description": self.get_world_description(),
        }
