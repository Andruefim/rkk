"""
Anthropomorphic gait helpers — alternating support, coupling clamps.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def alternating_support_from_swings(
    swing_l: float,
    swing_r: float,
    *,
    amp: float | None = None,
) -> tuple[float, float]:
    """High support on stance leg; swing leg unloads (humanoid alternation)."""
    a = amp if amp is not None else _ef("RKK_LOCOMOTE_SUPPORT_AMP", 0.22)
    sup_l = float(np.clip(0.5 + a * (float(swing_r) - float(swing_l)), 0.28, 0.78))
    sup_r = float(np.clip(0.5 + a * (float(swing_l) - float(swing_r)), 0.28, 0.78))
    return sup_l, sup_r


def locomote_macro_active(sim: Any) -> bool:
    ic = getattr(sim, "_intention_state", None)
    if ic is not None:
        hint = str(getattr(ic, "macro_hint", "") or "").strip().upper()
        if hint in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            return True
    s2 = getattr(sim, "_system2_last", None) or {}
    if isinstance(s2, dict):
        macro = str(s2.get("macro") or "").strip().upper()
        if macro in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            return True
    return False


def clamp_locomote_gait_intents(sim: Any) -> dict[str, float]:
    """Cap coupling sync; keep stride floor during LOCOMOTE."""
    if not locomote_macro_active(sim):
        return {}
    agent = sim.agent
    nodes = agent.graph.nodes
    base = getattr(agent.env, "base_env", None) or agent.env
    ms = getattr(base, "_motor_state", None)
    if not isinstance(ms, dict):
        return {}

    coupling = _ef("RKK_NS_LOCOMOTE_COUPLING", 0.78)
    coupling_min = _ef("RKK_NS_LOCOMOTE_COUPLING_MIN", 0.72)
    stride_floor = _ef("RKK_NS_LOCOMOTE_STRIDE", 0.64)
    torso_floor = _ef("RKK_NS_LOCOMOTE_TORSO", 0.58)

    out: dict[str, float] = {}
    cur_c = float(ms.get("intent_gait_coupling", nodes.get("intent_gait_coupling", 0.5)))
    c_val = float(np.clip(cur_c, coupling_min, coupling))
    ms["intent_gait_coupling"] = c_val
    out["intent_gait_coupling"] = c_val

    s_val = float(np.clip(max(float(ms.get("intent_stride", 0.5)), stride_floor), 0.05, 0.95))
    ms["intent_stride"] = s_val
    out["intent_stride"] = s_val

    t_val = float(
        np.clip(max(float(ms.get("intent_torso_forward", 0.5)), torso_floor), 0.05, 0.95)
    )
    ms["intent_torso_forward"] = t_val
    out["intent_torso_forward"] = t_val

    for ck, tv in out.items():
        if ck in nodes:
            nodes[ck] = tv
        phys = f"phys_{ck}"
        if phys in nodes:
            nodes[phys] = tv

    arb = getattr(sim, "_motor_arbiter", None)
    if arb is not None:
        arb.register_from_dict("gait", out, precision=0.85)
    return out


def apply_alternating_support_from_cpg(sim: Any) -> dict[str, float]:
    """Write phase-derived L/R support into motor_state + graph."""
    if not locomote_macro_active(sim):
        return {}
    lc = getattr(sim, "_locomotion_controller", None)
    if lc is None:
        return {}
    sync = getattr(lc, "_last_cpg_sync", None) or {}
    swing_l = float(sync.get("swing_l", 0.0))
    swing_r = float(sync.get("swing_r", 0.0))

    agent = sim.agent
    base = getattr(agent.env, "base_env", None) or agent.env
    ms = getattr(base, "_motor_state", None)
    if not isinstance(ms, dict):
        return {}
    stride = float(ms.get("intent_stride", agent.graph.nodes.get("intent_stride", 0.5)))
    if stride < 0.54:
        return {}

    alt_l, alt_r = alternating_support_from_swings(swing_l, swing_r)
    blend = _ef("RKK_LOCOMOTE_SUPPORT_BLEND", 0.88)
    cur_l = float(ms.get("intent_support_left", 0.5))
    cur_r = float(ms.get("intent_support_right", 0.5))
    sup_l = float(np.clip((1.0 - blend) * cur_l + blend * alt_l, 0.22, 0.82))
    sup_r = float(np.clip((1.0 - blend) * cur_r + blend * alt_r, 0.22, 0.82))

    ms["intent_support_left"] = sup_l
    ms["intent_support_right"] = sup_r
    nodes = agent.graph.nodes
    nodes["intent_support_left"] = sup_l
    nodes["intent_support_right"] = sup_r
    if "phys_intent_support_left" in nodes:
        nodes["phys_intent_support_left"] = sup_l
    if "phys_intent_support_right" in nodes:
        nodes["phys_intent_support_right"] = sup_r

    phi_l = float(sync.get("phi_l", 0.0))
    phi_r = float(sync.get("phi_r", 0.0))
    if phi_l or phi_r:
        nodes["gait_phase_l"] = float(np.clip(0.5 + 0.5 * np.sin(phi_l), 0.0, 1.0))
        nodes["gait_phase_r"] = float(np.clip(0.5 + 0.5 * np.sin(phi_r), 0.0, 1.0))

    return {"intent_support_left": sup_l, "intent_support_right": sup_r}
