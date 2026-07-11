"""Shoulder clamp, CPG bilateral gate, and phase-reset unit tests."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from engine.cpg_locomotion import LocomotionController
from engine.features.humanoid.pybullet_humanoid import clamp_shoulder_real_pos


def test_clamp_shoulder_real_pos_within_limits(monkeypatch):
    monkeypatch.setenv("RKK_SHOULDER_MIN_RAD", "-0.8")
    monkeypatch.setenv("RKK_SHOULDER_MAX_RAD", "1.1")
    assert clamp_shoulder_real_pos(-2.0) == pytest.approx(-0.8)
    assert clamp_shoulder_real_pos(1.8) == pytest.approx(1.1)
    assert clamp_shoulder_real_pos(0.4) == pytest.approx(0.4)


def test_bilateral_scale_lower_when_striding():
    """Bilateral recovery terms scale by (1 - walk_gate); striding lowers the scale."""

    def _bilateral_scale(stride_n: float, recover_n: float, low_z: float) -> float:
        gscale = float(np.clip(max(stride_n, 0.42 * recover_n + 0.48 * low_z) * 1.8, 0.0, 1.0))
        walk_blend = float(np.clip(max(stride_n, 0.5 * recover_n + 0.45 * low_z), 0.0, 1.0))
        walk_gate = float(np.clip(walk_blend * (0.35 + 0.65 * gscale), 0.0, 1.0))
        rec_gate = float(np.clip(max(recover_n, low_z * 0.9), 0.0, 1.0))
        if rec_gate > 0.35:
            walk_gate *= float(1.0 - 0.85 * rec_gate)
        return 1.0 - walk_gate

    stand_recover = _bilateral_scale(0.0, 0.8, 0.85)
    walk = _bilateral_scale(0.4, 0.0, 0.0)
    assert walk < stand_recover
    assert walk < 0.75


def test_reset_cpg_phases_antiphase():
    lc = LocomotionController(torch.device("cpu"))
    lc.cpg._phase[:] = 1.2
    lc.reset_cpg_phases()
    phi_l = float(lc.cpg._phase[0].item())
    phi_r = float(lc.cpg._phase[1].item())
    diff = abs((phi_r - phi_l) % (2 * math.pi) - math.pi)
    assert diff == pytest.approx(0.0, abs=1e-5)
    phi_kl = float(lc.cpg._phase[2].item())
    phi_kr = float(lc.cpg._phase[3].item())
    knee_diff = abs((phi_kr - phi_kl) % (2 * math.pi) - math.pi)
    assert knee_diff == pytest.approx(0.0, abs=1e-5)


def test_phase_bias_clamped_after_training():
    lc = LocomotionController(torch.device("cpu"))
    with torch.no_grad():
        lc.cpg.phase_bias[0, 1] = 4.0
        lc.cpg.phase_bias[1, 0] = -4.0
    lc._clamp_phase_bias_pairs()
    assert float(lc.cpg.phase_bias[0, 1].item()) <= math.pi + 0.35 + 1e-5
    assert float(lc.cpg.phase_bias[0, 1].item()) >= math.pi - 0.35 - 1e-5
    assert float(lc.cpg.phase_bias[1, 0].item()) >= -math.pi - 0.35 - 1e-5
    assert float(lc.cpg.phase_bias[1, 0].item()) <= -math.pi + 0.35 + 1e-5
