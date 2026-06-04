"""
Track A Phase 0: post fixed_root release — alpha_trust decay, WM LR boost, ensemble entropy.
"""
from __future__ import annotations

import os
from typing import Any

import torch


def _env_float(key: str, default: str) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return float(default)


def _env_int(key: str, default: str) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return int(default)


def post_fr_alpha_decay() -> float:
    return max(0.0, min(1.0, _env_float("RKK_POST_FR_ALPHA_DECAY", "0.40")))


def post_fr_wm_lr_mult() -> float:
    return max(1.0, _env_float("RKK_POST_FR_WM_LR_MULT", "2.50"))


def post_fr_wm_lr_ticks() -> int:
    return max(0, _env_int("RKK_POST_FR_WM_LR_TICKS", "450"))


def post_fr_ensemble_entropy_boost() -> float:
    return max(0.0, _env_float("RKK_POST_FR_ENSEMBLE_ENT_BOOST", "0.25"))


def _edge_is_post_fr_target(from_: str, to: str) -> bool:
    s = f"{from_} {to}".lower()
    if "intent_" in s or s.startswith("phys_intent"):
        return True
    if any(
        p in s
        for p in (
            "posture",
            "support_bias",
            "support_leg",
            "foot_contact",
            "phys_support",
            "phys_posture",
        )
    ):
        return True
    leg = ("hip", "knee", "ankle", "spine", "torso_pitch")
    if any(k in s for k in leg):
        return True
    return False


def apply_post_fr_alpha_decay(graph: Any) -> int:
    """Decay alpha_trust on motor/posture/support edges once per release."""
    decay = post_fr_alpha_decay()
    if decay <= 0.0:
        return 0
    n = 0
    for edge in graph.edges:
        if not _edge_is_post_fr_target(edge.from_, edge.to):
            continue
        edge.alpha_trust = max(0.02, float(edge.alpha_trust) - decay)
        n += 1
    if n:
        graph._invalidate_cache()
    return n


def apply_post_fr_ensemble_entropy_boost(graph: Any) -> float | None:
    """
    Flatten ensemble posterior slightly after FR release (higher entropy → more EIG).
    Returns entropy after boost, or None if no ensemble.
    """
    ens = getattr(graph, "_ensemble", None)
    if ens is None:
        graph._maybe_init_ensemble()
        ens = getattr(graph, "_ensemble", None)
    if ens is None:
        return None
    boost = post_fr_ensemble_entropy_boost()
    if boost <= 0.0:
        return ens.entropy()
    n = int(ens.n)
    with torch.no_grad():
        uniform = torch.log(torch.ones(n, device=ens.log_weights.device) / n)
        ens.log_weights.add_(uniform * boost)
    return ens.entropy()


def post_fr_wm_lr_scale(sim: Any) -> float:
    """WM lr multiplier while within RKK_POST_FR_WM_LR_TICKS after release."""
    t0 = int(getattr(sim, "_post_fr_last_release_tick", -1))
    if t0 < 0:
        return 1.0
    tick = int(getattr(sim, "tick", 0))
    window = post_fr_wm_lr_ticks()
    if window <= 0 or tick < t0 or tick > t0 + window:
        return 1.0
    return post_fr_wm_lr_mult()


def post_fr_wm_lr_active(sim: Any) -> bool:
    return post_fr_wm_lr_scale(sim) > 1.001
