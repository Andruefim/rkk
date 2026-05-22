"""Recovery LLM dispatch / replan timing helpers."""
from __future__ import annotations

import os

from engine.system2.controller import (
    System2Controller,
    _recovery_llm_delay_first_dispatch_ticks,
    _recovery_replan_min_interval_ticks,
)


def test_faster_defaults():
    assert _recovery_llm_delay_first_dispatch_ticks() == 16
    assert _recovery_replan_min_interval_ticks() == 20


def test_mid_schedule_replan_when_stagnant(monkeypatch):
    monkeypatch.setenv("RKK_S2_RECOVERY_REPLAN_MID_SCHEDULE", "1")
    monkeypatch.setenv("RKK_S2_RECOVERY_STAGNATION_REPLAN_TICKS", "30")
    c = System2Controller()
    c._s2_override_active = True
    c._s2_override_start_tick = 1000
    c._recovery_schedule_anchor_tick = 1000
    c._recovery_steps = [{"ticks": 80, "intent_deltas": {"intent_torso_forward": 0.1}}]
    c._recovery_cumulative = [80]
    c._override_start_obs_f = {"com_z": 0.12, "posture_stability": 0.0}
    c._recovery_best_com_z = 0.12
    c._recovery_ticks_since_com_z_gain = 35
    obs = {"com_z": 0.11, "posture_stability": 0.0}
    assert c._recovery_mid_schedule_replan_wanted(1050, obs)


def test_obs0_reanchor_on_floor(monkeypatch):
    c = System2Controller()
    c._override_start_obs_f = {"com_z": 0.45, "posture_stability": 0.98}
    c._maybe_refresh_override_obs0(
        {"com_z": 0.10, "posture_stability": 0.0}, fallen=True
    )
    assert c._obs_com_z(c._override_start_obs_f) < 0.15
