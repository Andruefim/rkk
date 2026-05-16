"""CPG during fixed_root curriculum + progressive scope mastery logging."""
from __future__ import annotations

import os

import pytest

from engine.wm_locomotion_gating import locomotion_wm_scales
from engine.core.constants import cpg_during_fixed_root_enabled
from engine.progressive_scope import ProgressiveScope


def test_cpg_during_fixed_root_follows_locomotion_by_default(monkeypatch):
    monkeypatch.delenv("RKK_CPG_DURING_FIXED_ROOT", raising=False)
    monkeypatch.setenv("RKK_LOCOMOTION_CPG", "1")
    assert cpg_during_fixed_root_enabled() is True
    monkeypatch.setenv("RKK_LOCOMOTION_CPG", "0")
    assert cpg_during_fixed_root_enabled() is False


def test_locomotion_wm_scales_low_reward_dampens_int(monkeypatch):
    monkeypatch.setenv("RKK_WM_LOCO_GATING", "1")
    monkeypatch.setenv("RKK_WM_LOCO_REWARD_EMA_MIN", "0.45")
    monkeypatch.setenv("RKK_WM_L_INT_LOCO_MULT", "0.35")
    int_s, rec_s, lr_s = locomotion_wm_scales(0.2, True)
    assert int_s < 0.25
    assert rec_s < 1.0
    assert lr_s < 1.0
    int_hi, _, _ = locomotion_wm_scales(0.55, True)
    assert int_hi > int_s


def test_progressive_scope_mastery_quality_property():
    ps = ProgressiveScope()
    ps._quality_window.append(0.8)
    ps._quality_window.append(0.6)
    assert abs(ps.mastery_quality - 0.7) < 1e-6
