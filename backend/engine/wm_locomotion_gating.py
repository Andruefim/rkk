"""WM train-step scaling while CPG drives legs (avoid l_int blow-up)."""
from __future__ import annotations

import os

import numpy as np


def wm_locomotion_gating_enabled() -> bool:
    return os.environ.get("RKK_WM_LOCO_GATING", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def locomotion_wm_scales(
    reward_ema: float,
    cpg_active: bool,
    *,
    fallen: bool = False,
) -> tuple[float, float, float]:
    """
    Returns (int_scale, rec_scale, lr_scale) each in [floor, 1].
    When reward_ema reaches RKK_WM_LOCO_REWARD_EMA_MIN, scales → 1.
    """
    if not cpg_active or not wm_locomotion_gating_enabled():
        return 1.0, 1.0, 1.0
    try:
        thr = float(os.environ.get("RKK_WM_LOCO_REWARD_EMA_MIN", "0.45"))
    except ValueError:
        thr = 0.45
    thr = max(0.15, thr)
    try:
        floor = float(os.environ.get("RKK_WM_LOCO_TRAIN_SCALE_FLOOR", "0.25"))
    except ValueError:
        floor = 0.25
    floor = float(np.clip(floor, 0.05, 0.85))
    try:
        int_floor = float(os.environ.get("RKK_WM_LOCO_INT_FLOOR", "0.08"))
    except ValueError:
        int_floor = 0.08
    int_floor = float(np.clip(int_floor, 0.02, 0.35))
    t = float(np.clip(float(reward_ema) / thr, 0.0, 1.0))
    base = floor + (1.0 - floor) * t
    try:
        int_mult = float(os.environ.get("RKK_WM_L_INT_LOCO_MULT", "0.15"))
    except ValueError:
        int_mult = 0.15
    int_mult = float(np.clip(int_mult, 0.02, 1.0))
    int_scale = max(int_floor, min(1.0, base * int_mult))
    if fallen or float(reward_ema) < 0.12:
        int_scale = min(int_scale, int_floor)
        base = min(base, floor * 0.65)
    rec_scale = base
    lr_scale = base
    return int_scale, rec_scale, lr_scale
