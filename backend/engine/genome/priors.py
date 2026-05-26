"""
genome/priors.py — Innate biological priors ("DNA") for the humanoid.

Like mammalian neonates who can stand within hours of birth, these priors
encode the minimum viable motor programs and causal knowledge that the
agent starts life with. Everything else is learned online.

Three layers:
  1. CAUSAL_PRIORS  — strong directed edges in the GNN (alpha_trust=0.8+)
  2. REFLEX_TABLE   — fast reactive mappings (observation → intent delta)
  3. STAND_PROGRAM  — motor sequence to get upright from the ground
  4. WALK_PROGRAM   — cyclic CPG-style intent keyframes for bipedal gait

These are NOT hardcoded controllers — they are initial conditions that
the online learning system can modify, override, or extend.
"""
from __future__ import annotations

import os

import numpy as np

# ─── Layer 1: Causal Priors (strong GNN edges) ──────────────────────────────
# Each entry: (from_node, to_node, weight, alpha_trust)
# weight > 0 = positive causal influence; < 0 = negative
# alpha_trust = confidence (0-1); higher = slower to override by learning
#
# These encode what a newborn "knows" about its body:
# - Legs support the body (hip/knee → com_z)
# - Leaning forward → moving forward (torso_pitch → com_x)
# - Falling = low com_z + high torso angles
# - Feet touching ground = stability

CAUSAL_PRIORS: list[dict] = [
    # === Gravity / support ===
    {"from": "lhip",  "to": "com_z", "weight": 0.35, "alpha": 0.85},
    {"from": "rhip",  "to": "com_z", "weight": 0.35, "alpha": 0.85},
    {"from": "lknee", "to": "com_z", "weight": 0.40, "alpha": 0.85},
    {"from": "rknee", "to": "com_z", "weight": 0.40, "alpha": 0.85},
    {"from": "lankle", "to": "com_z", "weight": 0.20, "alpha": 0.80},
    {"from": "rankle", "to": "com_z", "weight": 0.20, "alpha": 0.80},

    # === Locomotion intent → legs (alternating hips = anthropomorphic gait) ===
    {"from": "intent_stride", "to": "lhip",  "weight": 0.45, "alpha": 0.75},
    {"from": "intent_stride", "to": "rhip",  "weight": -0.45, "alpha": 0.75},
    {"from": "intent_stride", "to": "com_x", "weight": 0.30, "alpha": 0.70},
    {"from": "intent_stride", "to": "lknee", "weight": -0.22, "alpha": 0.72},
    {"from": "intent_stride", "to": "rknee", "weight": 0.22, "alpha": 0.72},

    # === Support/balance ===
    {"from": "intent_support_left",  "to": "lhip",  "weight": 0.30, "alpha": 0.75},
    {"from": "intent_support_right", "to": "rhip",  "weight": 0.30, "alpha": 0.75},
    {"from": "intent_support_left",  "to": "support_bias", "weight": 0.35, "alpha": 0.78},
    {"from": "intent_support_right", "to": "support_bias", "weight": -0.35, "alpha": 0.78},
    {"from": "intent_stop_recover", "to": "lknee", "weight": -0.35, "alpha": 0.80},
    {"from": "intent_stop_recover", "to": "rknee", "weight": -0.35, "alpha": 0.80},

    # === Torso → posture (upright bias; excessive pitch hurts stability) ===
    {"from": "intent_torso_forward", "to": "spine_pitch", "weight": 0.35, "alpha": 0.75},
    {"from": "intent_torso_forward", "to": "com_x", "weight": 0.22, "alpha": 0.70},
    {"from": "torso_pitch", "to": "posture_stability", "weight": -0.50, "alpha": 0.85},
    {"from": "torso_roll",  "to": "posture_stability", "weight": -0.50, "alpha": 0.85},
    {"from": "com_z", "to": "posture_stability", "weight": 0.60, "alpha": 0.90},

    # === Gait coupling (CPG-like rhythm in graph) ===
    {"from": "intent_gait_coupling", "to": "gait_phase_l", "weight": 0.35, "alpha": 0.70},
    {"from": "intent_gait_coupling", "to": "gait_phase_r", "weight": 0.35, "alpha": 0.70},
    {"from": "gait_phase_l", "to": "lhip",  "weight": 0.30, "alpha": 0.70},
    {"from": "gait_phase_r", "to": "rhip",  "weight": 0.30, "alpha": 0.70},
    {"from": "gait_phase_l", "to": "lknee", "weight": -0.25, "alpha": 0.70},
    {"from": "gait_phase_r", "to": "rknee", "weight": -0.25, "alpha": 0.70},
    {"from": "gait_phase_l", "to": "gait_phase_r", "weight": -0.55, "alpha": 0.82},
    {"from": "gait_phase_r", "to": "gait_phase_l", "weight": -0.55, "alpha": 0.82},

    # === Foot contact feedback ===
    {"from": "foot_contact_l", "to": "support_bias", "weight": 0.40, "alpha": 0.80},
    {"from": "foot_contact_r", "to": "support_bias", "weight": -0.40, "alpha": 0.80},
    {"from": "foot_contact_l", "to": "posture_stability", "weight": 0.25, "alpha": 0.75},
    {"from": "foot_contact_r", "to": "posture_stability", "weight": 0.25, "alpha": 0.75},

    # === Arms for balance (contralateral swing) ===
    {"from": "intent_arm_counterbalance", "to": "lshoulder", "weight": 0.25, "alpha": 0.60},
    {"from": "intent_arm_counterbalance", "to": "rshoulder", "weight": -0.25, "alpha": 0.60},
    {"from": "intent_arm_counterbalance", "to": "torso_roll", "weight": -0.28, "alpha": 0.65},
    {"from": "support_bias", "to": "intent_arm_counterbalance", "weight": 0.20, "alpha": 0.62},
]


# ─── Layer 2: Spinal Reflexes ────────────────────────────────────────────────
# Fast reactive rules: if condition → adjust intent
# These fire BEFORE the main agent loop, like brainstem reflexes.
# Format: (condition_var, threshold, comparison, target_var, delta)
#   comparison: "lt" (less than), "gt" (greater than)
#   delta: added to current value of target_var

REFLEX_TABLE: list[dict] = [
    # Low com_z → crouch and stabilize (do NOT lean further forward)
    {"sensor": "com_z", "threshold": 0.35, "cmp": "lt",
     "target": "intent_stop_recover", "delta": 0.28},
    {"sensor": "com_z", "threshold": 0.35, "cmp": "lt",
     "target": "intent_torso_forward", "delta": -0.12},

    # Torso tilting too much forward/back → correct
    {"sensor": "torso_pitch", "threshold": 0.65, "cmp": "gt",
     "target": "intent_torso_forward", "delta": -0.22},
    {"sensor": "torso_pitch", "threshold": 0.38, "cmp": "lt",
     "target": "intent_torso_forward", "delta": 0.12},
    {"sensor": "torso_roll", "threshold": 0.65, "cmp": "gt",
     "target": "intent_support_left", "delta": 0.15},
    {"sensor": "torso_roll", "threshold": 0.35, "cmp": "lt",
     "target": "intent_support_right", "delta": 0.15},

    # Low posture → widen stance and activate recovery
    {"sensor": "posture_stability", "threshold": 0.40, "cmp": "lt",
     "target": "intent_stop_recover", "delta": 0.20},
    {"sensor": "posture_stability", "threshold": 0.40, "cmp": "lt",
     "target": "intent_gait_coupling", "delta": -0.08},

    # Lost foot contact → shift weight to stance foot
    {"sensor": "foot_contact_l", "threshold": 0.30, "cmp": "lt",
     "target": "intent_support_right", "delta": 0.15},
    {"sensor": "foot_contact_r", "threshold": 0.30, "cmp": "lt",
     "target": "intent_support_left", "delta": 0.15},
]


# ─── Layer 3: Stand-up Motor Program ─────────────────────────────────────────
# Prone → kneel → upright. Values aligned with system2 recovery_schedule and
# physical_curriculum static_stance (moderate torso, no extreme forward lean).

STAND_PROGRAM: list[dict] = [
    {
        "ticks": 50,
        "phase": "tuck",
        "intents": {
            "intent_stop_recover": 0.72,
            "intent_torso_forward": 0.48,
            "intent_support_left": 0.58,
            "intent_support_right": 0.58,
            "intent_stride": 0.46,
            "intent_gait_coupling": 0.76,
        },
    },
    {
        "ticks": 45,
        "phase": "torso_lift",
        "intents": {
            "intent_stop_recover": 0.68,
            "intent_torso_forward": 0.52,
            "intent_support_left": 0.56,
            "intent_support_right": 0.56,
            "intent_stride": 0.48,
            "intent_gait_coupling": 0.78,
            "intent_arm_counterbalance": 0.52,
        },
    },
    {
        "ticks": 50,
        "phase": "push_up",
        "intents": {
            "intent_stop_recover": 0.64,
            "intent_torso_forward": 0.54,
            "intent_support_left": 0.58,
            "intent_support_right": 0.54,
            "intent_stride": 0.48,
            "intent_gait_coupling": 0.80,
            "intent_arm_counterbalance": 0.54,
        },
    },
    {
        "ticks": 40,
        "phase": "kneel",
        "intents": {
            "intent_stop_recover": 0.62,
            "intent_torso_forward": 0.52,
            "intent_support_left": 0.58,
            "intent_support_right": 0.58,
            "intent_stride": 0.50,
            "intent_gait_coupling": 0.78,
        },
    },
    {
        "ticks": 35,
        "phase": "release_stand",
        "intents": {
            "intent_stop_recover": 0.65,
            "intent_torso_forward": 0.52,
            "intent_support_left": 0.58,
            "intent_support_right": 0.58,
            "intent_stride": 0.50,
            "intent_gait_coupling": 0.78,
            "intent_arm_counterbalance": 0.55,
        },
    },
]


# ─── Layer 4: Anthropomorphic walk CPG (intent keyframes) ───────────────────
# One full gait cycle: heel-strike → mid-stance → push-off → swing (×2 legs).
# Alternating support and mild contralateral arm bias.

WALK_CYCLE_TICKS_DEFAULT = 40

WALK_PROGRAM: list[dict] = [
    {
        "phase": "left_heel_strike",
        "intents": {
            "intent_stride": 0.56,
            "intent_torso_forward": 0.58,
            "intent_support_left": 0.62,
            "intent_support_right": 0.40,
            "intent_gait_coupling": 0.88,
            "intent_stop_recover": 0.54,
            "intent_arm_counterbalance": 0.56,
        },
    },
    {
        "phase": "left_mid_stance",
        "intents": {
            "intent_stride": 0.57,
            "intent_torso_forward": 0.59,
            "intent_support_left": 0.68,
            "intent_support_right": 0.36,
            "intent_gait_coupling": 0.90,
            "intent_stop_recover": 0.52,
            "intent_arm_counterbalance": 0.58,
        },
    },
    {
        "phase": "left_push_off",
        "intents": {
            "intent_stride": 0.58,
            "intent_torso_forward": 0.60,
            "intent_support_left": 0.64,
            "intent_support_right": 0.38,
            "intent_gait_coupling": 0.90,
            "intent_stop_recover": 0.52,
            "intent_arm_counterbalance": 0.62,
        },
    },
    {
        "phase": "right_swing",
        "intents": {
            "intent_stride": 0.58,
            "intent_torso_forward": 0.59,
            "intent_support_left": 0.36,
            "intent_support_right": 0.68,
            "intent_gait_coupling": 0.88,
            "intent_stop_recover": 0.52,
            "intent_arm_counterbalance": 0.64,
        },
    },
    {
        "phase": "right_heel_strike",
        "intents": {
            "intent_stride": 0.56,
            "intent_torso_forward": 0.58,
            "intent_support_left": 0.40,
            "intent_support_right": 0.62,
            "intent_gait_coupling": 0.88,
            "intent_stop_recover": 0.52,
            "intent_arm_counterbalance": 0.56,
        },
    },
    {
        "phase": "right_mid_stance",
        "intents": {
            "intent_stride": 0.57,
            "intent_torso_forward": 0.59,
            "intent_support_left": 0.36,
            "intent_support_right": 0.68,
            "intent_gait_coupling": 0.90,
            "intent_stop_recover": 0.52,
            "intent_arm_counterbalance": 0.58,
        },
    },
    {
        "phase": "right_push_off",
        "intents": {
            "intent_stride": 0.58,
            "intent_torso_forward": 0.60,
            "intent_support_left": 0.38,
            "intent_support_right": 0.64,
            "intent_gait_coupling": 0.90,
            "intent_stop_recover": 0.52,
            "intent_arm_counterbalance": 0.62,
        },
    },
    {
        "phase": "left_swing",
        "intents": {
            "intent_stride": 0.58,
            "intent_torso_forward": 0.59,
            "intent_support_left": 0.68,
            "intent_support_right": 0.36,
            "intent_gait_coupling": 0.88,
            "intent_stop_recover": 0.52,
            "intent_arm_counterbalance": 0.64,
        },
    },
]


def apply_causal_priors(graph) -> int:
    """Inject innate causal edges into the GNN graph. Returns count of edges set."""
    count = 0
    for p in CAUSAL_PRIORS:
        fr, to = p["from"], p["to"]
        if fr in graph._node_ids and to in graph._node_ids:
            graph.set_edge(fr, to, float(p["weight"]), alpha=float(p["alpha"]))
            count += 1
    return count


def apply_reflexes(obs: dict, motor_state: dict) -> dict:
    """
    Apply spinal reflexes: fast reactive adjustments to motor intents.
    Returns updated motor_state dict.
    """
    out = dict(motor_state)
    for r in REFLEX_TABLE:
        val = float(obs.get(r["sensor"], obs.get(f"phys_{r['sensor']}", 0.5)))
        fire = (r["cmp"] == "lt" and val < r["threshold"]) or \
               (r["cmp"] == "gt" and val > r["threshold"])
        if fire:
            tgt = r["target"]
            if tgt in out:
                out[tgt] = float(np.clip(out[tgt] + r["delta"], 0.05, 0.95))
    return out


def get_stand_program() -> list[dict]:
    """Return the innate stand-up motor program."""
    return list(STAND_PROGRAM)


def get_walk_program() -> list[dict]:
    """Return one full anthropomorphic gait cycle (intent keyframes)."""
    return list(WALK_PROGRAM)


def walk_cycle_ticks() -> int:
    try:
        n = int(os.environ.get("RKK_GENOME_WALK_CYCLE_TICKS", str(WALK_CYCLE_TICKS_DEFAULT)))
    except ValueError:
        n = WALK_CYCLE_TICKS_DEFAULT
    return max(8, min(n, 200))


def walk_phase_index(tick: int, cycle_ticks: int | None = None) -> int:
    """Map simulation tick → index into WALK_PROGRAM."""
    cycle = cycle_ticks if cycle_ticks is not None else walk_cycle_ticks()
    n = len(WALK_PROGRAM)
    if n <= 0:
        return 0
    rel = int(tick) % cycle
    return int(rel * n // cycle) % n


def walk_intents_at_tick(tick: int, cycle_ticks: int | None = None) -> dict[str, float]:
    """Target motor intents for the current gait phase."""
    prog = WALK_PROGRAM
    if not prog:
        return {}
    idx = walk_phase_index(tick, cycle_ticks)
    return dict(prog[idx]["intents"])


def genome_walk_enabled() -> bool:
    return os.environ.get("RKK_GENOME_WALK", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def genome_walk_innate_enabled() -> bool:
    """Innate gait when upright — does not wait for curriculum walk goal."""
    return os.environ.get("RKK_GENOME_WALK_INNATE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def genome_walk_gain() -> float:
    try:
        g = float(os.environ.get("RKK_GENOME_WALK_GAIN", "0.14"))
    except ValueError:
        g = 0.14
    return float(np.clip(g, 0.02, 0.40))


def compute_walk_residuals(
    current: dict[str, float],
    tick: int,
    *,
    gain: float | None = None,
    cycle_ticks: int | None = None,
) -> dict[str, float]:
    """
    Soft nudge toward innate walk keyframe (residual deltas, not absolute setpoints).
    """
    targets = walk_intents_at_tick(tick, cycle_ticks)
    if not targets:
        return {}
    g = genome_walk_gain() if gain is None else float(np.clip(gain, 0.02, 0.40))
    residuals: dict[str, float] = {}
    for k, tgt in targets.items():
        cur = float(current.get(k, 0.5))
        delta = (float(tgt) - cur) * g
        if abs(delta) >= 0.008:
            residuals[k] = float(np.clip(delta, -0.18, 0.18))
    return residuals


# Direct leg/torso targets per gait phase (anthropomorphic alternating stance/swing).
WALK_PHASE_JOINTS: dict[str, dict[str, float]] = {
    "left_heel_strike": {
        "lhip": 0.50, "rhip": 0.56, "lknee": 0.52, "rknee": 0.60,
        "lankle": 0.50, "rankle": 0.48, "spine_pitch": 0.60,
    },
    "left_mid_stance": {
        "lhip": 0.48, "rhip": 0.58, "lknee": 0.50, "rknee": 0.62,
        "lankle": 0.51, "rankle": 0.47, "spine_pitch": 0.61,
    },
    "left_push_off": {
        "lhip": 0.46, "rhip": 0.60, "lknee": 0.48, "rknee": 0.64,
        "lankle": 0.52, "rankle": 0.46, "spine_pitch": 0.62,
    },
    "right_swing": {
        "lhip": 0.56, "rhip": 0.50, "lknee": 0.60, "rknee": 0.52,
        "lankle": 0.48, "rankle": 0.50, "spine_pitch": 0.61,
    },
    "right_heel_strike": {
        "lhip": 0.56, "rhip": 0.50, "lknee": 0.60, "rknee": 0.52,
        "lankle": 0.47, "rankle": 0.50, "spine_pitch": 0.60,
    },
    "right_mid_stance": {
        "lhip": 0.58, "rhip": 0.48, "lknee": 0.62, "rknee": 0.50,
        "lankle": 0.47, "rankle": 0.51, "spine_pitch": 0.61,
    },
    "right_push_off": {
        "lhip": 0.60, "rhip": 0.46, "lknee": 0.64, "rknee": 0.48,
        "lankle": 0.46, "rankle": 0.52, "spine_pitch": 0.62,
    },
    "left_swing": {
        "lhip": 0.50, "rhip": 0.56, "lknee": 0.52, "rknee": 0.60,
        "lankle": 0.50, "rankle": 0.48, "spine_pitch": 0.61,
    },
}



def walk_phase_name_at_tick(tick: int, cycle_ticks: int | None = None) -> str:
    prog = WALK_PROGRAM
    if not prog:
        return ""
    idx = walk_phase_index(tick, cycle_ticks)
    return str(prog[idx].get("phase") or f"phase_{idx}")


def genome_walk_physics_enabled() -> bool:
    return os.environ.get("RKK_GENOME_WALK_PHYSICS", "0").strip().lower() not in (
        "0", "false", "no", "off",
    )


def genome_walk_force() -> bool:
    return os.environ.get("RKK_GENOME_WALK_FORCE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def genome_walk_leg_blend() -> float:
    try:
        b = float(os.environ.get("RKK_GENOME_WALK_LEG_BLEND", "0.52"))
    except ValueError:
        b = 0.78
    return float(np.clip(b, 0.20, 0.90))


def genome_walk_cpg_boost() -> float:
    try:
        b = float(os.environ.get("RKK_GENOME_WALK_CPG_BOOST", "2.4"))
    except ValueError:
        b = 2.4
    return float(np.clip(b, 1.0, 4.0))


def walk_leg_joints_at_tick(tick: int, cycle_ticks: int | None = None) -> dict[str, float]:
    name = walk_phase_name_at_tick(tick, cycle_ticks)
    return dict(WALK_PHASE_JOINTS.get(name, {}))


def walk_burst_pairs(tick: int, cycle_ticks: int | None = None) -> list[tuple[str, float]]:
    """Intent-only burst; leg joints applied separately after CPG."""
    return [(k, float(v)) for k, v in walk_intents_at_tick(tick, cycle_ticks).items()]


def genome_walk_eligible(
    obs: dict,
    *,
    goal_walk: bool,
    is_fallen: bool,
    fixed_root: bool,
) -> bool:
    if not genome_walk_enabled() or is_fallen or fixed_root:
        return False
    posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
    cz = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
    if cz < 0.42 or posture < 0.52:
        return False
    foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
    foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
    grounded = min(foot_l, foot_r) >= 0.48
    if not grounded:
        return False
    if genome_walk_innate_enabled() and posture >= 0.55 and cz >= 0.40:
        return True
    if genome_walk_force() or goal_walk:
        return True
    return False
