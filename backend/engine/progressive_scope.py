"""
progressive_scope.py — Progressive Variable Scope for AGI learning.

Instead of exposing all variables for intervention from tick 0,
progressively expands the intervention scope as the agent masters
each phase.

This is NOT hardcoded behavior — it controls WHICH variables the agent
can experiment with, not WHAT it does with them. The agent still
discovers causal structure autonomously within each scope.

Key principle: "master a tractable subspace before expanding."
This mirrors biological development (head control → rolling → crawling →
walking) and is fundamental to sample-efficient learning.

Design for generality (AGI):
  Phase 0: Only high-level actuators (intent_*) — the agent's "will"
  Phase 1: + core body joints (legs, spine) — direct body control
  Phase 2: + all joints (arms, head) — full motor repertoire
  Phase 3: + perceptual variables (slots) — visuo-motor integration

Observations are ALWAYS fully visible — scope only restricts interventions.
The GNN can learn correlations with ALL variables, it just can't
experiment with all of them yet.

Mastery criterion: general homeostatic quality over a time window,
not task-specific (works for any embodied agent).

Config:
  RKK_PROGRESSIVE_SCOPE=1                  — master switch
  RKK_SCOPE_MASTERY_WINDOW=80              — ticks to evaluate mastery
  RKK_SCOPE_MASTERY_THRESHOLD=0.55         — quality threshold to advance
  RKK_SCOPE_MIN_TICKS_PER_PHASE=300        — minimum ticks before advancing
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────
def progressive_scope_enabled() -> bool:
    return os.environ.get("RKK_PROGRESSIVE_SCOPE", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ei(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ── Variable classification ──────────────────────────────────────────────────
def _classify_variable(name: str) -> int:
    """
    Classify a variable into a phase group based on its name.
    Returns the minimum phase at which this variable becomes available
    for intervention.

    This uses naming conventions, not hardcoded lists, so it generalizes
    to any environment that follows the same conventions.
    """
    s = str(name).lower()

    # Phase 0: High-level motor intent (the agent's "will")
    if s.startswith("intent_") or s.startswith("phys_intent_"):
        return 0

    # Read-only observable variables — never intervened on
    read_only_prefixes = (
        "posture_", "com_", "foot_contact", "torso_",
        "gait_", "phys_posture", "phys_com", "phys_foot",
        "phys_torso", "phys_gait", "l1_", "mc_",
        "support_", "phys_support",
    )
    for pfx in read_only_prefixes:
        if s.startswith(pfx):
            return -1  # -1 = always observable, never intervened

    # Phase 1: Core body (legs + spine) — balance and locomotion substrate
    leg_keys = ("hip", "knee", "ankle")
    spine_keys = ("spine_", "torso_pitch")
    if any(k in s for k in leg_keys) or any(k in s for k in spine_keys):
        return 1

    # Phase 2: Extended body (arms, head) — manipulation, gaze
    arm_keys = ("shoulder", "elbow", "wrist")
    head_keys = ("neck_",)
    if any(k in s for k in arm_keys) or any(k in s for k in head_keys):
        return 2

    # Phase 3: Perceptual/abstract variables (visual slots, etc.)
    if s.startswith("slot_") or s.startswith("phys_slot"):
        return 3

    # Default: available from phase 1
    return 1


# ── Progressive Scope ────────────────────────────────────────────────────────
class ProgressiveScope:
    """
    Manages progressive expansion of the agent's intervention scope.

    The agent can OBSERVE all variables from the start, but can only
    INTERVENE on variables in its current phase. As it masters each
    phase, the scope expands to include more variables.

    This dramatically reduces the effective dimensionality of the
    exploration problem (e.g., 143D → 8D initially).
    """

    def __init__(self):
        self._phase: int = 0
        self._max_phase: int = 3
        self._phase_ticks: int = 0
        self._total_ticks: int = 0

        # Mastery tracking
        win_size = _ei("RKK_SCOPE_MASTERY_WINDOW", 80)
        self._quality_window: deque[float] = deque(maxlen=win_size)
        self._fallen_window: deque[bool] = deque(maxlen=win_size)

        # Variable classification cache
        self._var_phases: dict[str, int] = {}

        self._phase_history: list[dict[str, Any]] = []
        self._logged_phase = -1

    @property
    def phase(self) -> int:
        return self._phase

    def classify_vars(self, var_ids: list[str]) -> None:
        """Classify all variables on first encounter (or when vars change)."""
        for v in var_ids:
            if v not in self._var_phases:
                self._var_phases[v] = _classify_variable(v)

    def get_intervention_filter(self, all_var_ids: list[str]) -> set[str]:
        """
        Return the set of variable names the agent CAN intervene on
        in the current phase.

        Variables with phase <= current_phase are included.
        Variables with phase == -1 (read-only) are excluded.
        """
        if not progressive_scope_enabled():
            return set(all_var_ids)

        self.classify_vars(all_var_ids)
        allowed = set()
        for v in all_var_ids:
            vp = self._var_phases.get(v, 1)
            if vp == -1:
                continue  # read-only observable
            if vp <= self._phase:
                allowed.add(v)
        return allowed

    def tick(
        self,
        is_fallen: bool,
        posture: float,
        quality: float | None = None,
    ) -> bool:
        """
        Update mastery estimate. Returns True if phase advanced.

        Args:
            is_fallen: whether the agent is in a catastrophic state
            posture: current posture stability [0,1]
            quality: optional trajectory quality from TrajectoryCollector
        """
        if not progressive_scope_enabled():
            return False

        self._phase_ticks += 1
        self._total_ticks += 1

        # Quality signal: composite of upright + stability
        q = quality if quality is not None else (
            (0.0 if is_fallen else 0.6) + posture * 0.4
        )
        self._quality_window.append(float(np.clip(q, 0.0, 1.0)))
        self._fallen_window.append(bool(is_fallen))

        if self._phase >= self._max_phase:
            return False

        # Check mastery criteria
        min_ticks = _ei("RKK_SCOPE_MIN_TICKS_PER_PHASE", 300)
        threshold = _ef("RKK_SCOPE_MASTERY_THRESHOLD", 0.55)
        min_window_fill = max(20, len(self._quality_window) // 2)

        if self._phase_ticks < min_ticks:
            return False
        if len(self._quality_window) < min_window_fill:
            return False

        mastery = float(np.mean(list(self._quality_window)))
        fallen_rate = sum(1 for f in self._fallen_window if f) / len(self._fallen_window)

        # Advance if quality is high AND fall rate is low
        if mastery >= threshold and fallen_rate < 0.3:
            return self._advance_phase(mastery, fallen_rate)
        return False

    def _advance_phase(self, mastery: float, fallen_rate: float) -> bool:
        old_phase = self._phase
        self._phase = min(self._phase + 1, self._max_phase)
        self._phase_ticks = 0
        self._quality_window.clear()
        self._fallen_window.clear()

        record = {
            "from_phase": old_phase,
            "to_phase": self._phase,
            "tick": self._total_ticks,
            "mastery": round(mastery, 4),
            "fallen_rate": round(fallen_rate, 4),
        }
        self._phase_history.append(record)

        # Count variables in new scope
        n_new = sum(1 for v, p in self._var_phases.items()
                    if p == self._phase)
        print(
            f"[ProgressiveScope] Phase {old_phase} → {self._phase} "
            f"(mastery={mastery:.3f}, fallen={fallen_rate:.3f}, "
            f"+{n_new} vars, tick={self._total_ticks})"
        )
        return True

    def scope_summary(self, all_var_ids: list[str]) -> dict[str, int]:
        """Count variables per phase for diagnostics."""
        self.classify_vars(all_var_ids)
        counts: dict[str, int] = {}
        for phase in range(-1, self._max_phase + 1):
            label = f"phase_{phase}" if phase >= 0 else "read_only"
            counts[label] = sum(1 for v in all_var_ids
                                if self._var_phases.get(v, 1) == phase)
        return counts

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": progressive_scope_enabled(),
            "phase": self._phase,
            "max_phase": self._max_phase,
            "phase_ticks": self._phase_ticks,
            "total_ticks": self._total_ticks,
            "mastery": round(
                float(np.mean(list(self._quality_window)))
                if self._quality_window else 0.0, 4
            ),
            "fallen_rate": round(
                sum(1 for f in self._fallen_window if f) /
                max(1, len(self._fallen_window)), 4
            ) if self._fallen_window else 0.0,
            "n_vars_in_scope": sum(
                1 for p in self._var_phases.values()
                if 0 <= p <= self._phase
            ),
            "n_vars_total": len(self._var_phases),
            "phase_history": list(self._phase_history[-5:]),
        }
