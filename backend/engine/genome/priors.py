"""
genome/priors.py — Innate biological priors ("DNA") for the humanoid.

Like mammalian neonates who can stand within hours of birth, these priors
encode the minimum viable motor programs and causal knowledge that the
agent starts life with. Everything else is learned online.

Three layers:
  1. CAUSAL_PRIORS  — strong directed edges in the GNN (alpha_trust=0.8+)
  2. REFLEX_TABLE   — fast reactive mappings (observation → intent delta)
  3. STAND_PROGRAM  — a minimal motor sequence to get upright from the ground

These are NOT hardcoded controllers — they are initial conditions that
the online learning system can modify, override, or extend.
"""
from __future__ import annotations

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

    # === Locomotion intent → legs ===
    {"from": "intent_stride", "to": "lhip",  "weight": 0.45, "alpha": 0.75},
    {"from": "intent_stride", "to": "rhip",  "weight": -0.45, "alpha": 0.75},
    {"from": "intent_stride", "to": "com_x", "weight": 0.30, "alpha": 0.70},

    # === Support/balance ===
    {"from": "intent_support_left",  "to": "lhip",  "weight": 0.30, "alpha": 0.75},
    {"from": "intent_support_right", "to": "rhip",  "weight": 0.30, "alpha": 0.75},
    {"from": "intent_stop_recover", "to": "lknee", "weight": -0.35, "alpha": 0.80},
    {"from": "intent_stop_recover", "to": "rknee", "weight": -0.35, "alpha": 0.80},

    # === Torso → posture ===
    {"from": "intent_torso_forward", "to": "spine_pitch", "weight": 0.40, "alpha": 0.75},
    {"from": "intent_torso_forward", "to": "com_x", "weight": 0.25, "alpha": 0.70},
    {"from": "torso_pitch", "to": "posture_stability", "weight": -0.50, "alpha": 0.85},
    {"from": "torso_roll",  "to": "posture_stability", "weight": -0.50, "alpha": 0.85},
    {"from": "com_z", "to": "posture_stability", "weight": 0.60, "alpha": 0.90},

    # === Gait coupling ===
    {"from": "intent_gait_coupling", "to": "gait_phase_l", "weight": 0.35, "alpha": 0.70},
    {"from": "intent_gait_coupling", "to": "gait_phase_r", "weight": 0.35, "alpha": 0.70},
    {"from": "gait_phase_l", "to": "lhip",  "weight": 0.30, "alpha": 0.70},
    {"from": "gait_phase_r", "to": "rhip",  "weight": 0.30, "alpha": 0.70},
    {"from": "gait_phase_l", "to": "lknee", "weight": -0.25, "alpha": 0.70},
    {"from": "gait_phase_r", "to": "rknee", "weight": -0.25, "alpha": 0.70},

    # === Foot contact feedback ===
    {"from": "foot_contact_l", "to": "support_bias", "weight": 0.40, "alpha": 0.80},
    {"from": "foot_contact_r", "to": "support_bias", "weight": -0.40, "alpha": 0.80},
    {"from": "foot_contact_l", "to": "posture_stability", "weight": 0.25, "alpha": 0.75},
    {"from": "foot_contact_r", "to": "posture_stability", "weight": 0.25, "alpha": 0.75},

    # === Arms for balance ===
    {"from": "intent_arm_counterbalance", "to": "lshoulder", "weight": 0.25, "alpha": 0.60},
    {"from": "intent_arm_counterbalance", "to": "rshoulder", "weight": -0.25, "alpha": 0.60},
]


# ─── Layer 2: Spinal Reflexes ────────────────────────────────────────────────
# Fast reactive rules: if condition → adjust intent
# These fire BEFORE the main agent loop, like brainstem reflexes.
# Format: (condition_var, threshold, comparison, target_var, delta)
#   comparison: "lt" (less than), "gt" (greater than)
#   delta: added to current value of target_var

REFLEX_TABLE: list[dict] = [
    # Low com_z → emergency recovery (crouch and stabilize)
    {"sensor": "com_z", "threshold": 0.35, "cmp": "lt",
     "target": "intent_stop_recover", "delta": 0.30},
    {"sensor": "com_z", "threshold": 0.35, "cmp": "lt",
     "target": "intent_torso_forward", "delta": 0.15},

    # Torso tilting too much → correct
    {"sensor": "torso_pitch", "threshold": 0.65, "cmp": "gt",
     "target": "intent_torso_forward", "delta": -0.20},
    {"sensor": "torso_pitch", "threshold": 0.35, "cmp": "lt",
     "target": "intent_torso_forward", "delta": 0.20},
    {"sensor": "torso_roll", "threshold": 0.65, "cmp": "gt",
     "target": "intent_support_left", "delta": 0.15},
    {"sensor": "torso_roll", "threshold": 0.35, "cmp": "lt",
     "target": "intent_support_right", "delta": 0.15},

    # Low posture → widen stance and activate recovery
    {"sensor": "posture_stability", "threshold": 0.40, "cmp": "lt",
     "target": "intent_stop_recover", "delta": 0.20},
    {"sensor": "posture_stability", "threshold": 0.40, "cmp": "lt",
     "target": "intent_gait_coupling", "delta": 0.10},

    # Lost foot contact → shift weight to other foot
    {"sensor": "foot_contact_l", "threshold": 0.30, "cmp": "lt",
     "target": "intent_support_right", "delta": 0.15},
    {"sensor": "foot_contact_r", "threshold": 0.30, "cmp": "lt",
     "target": "intent_support_left", "delta": 0.15},
]


# ─── Layer 3: Stand-up Motor Program ─────────────────────────────────────────
# A minimal hardwired sequence to get from ground to standing.
# Each step is a dict of intent targets held for N ticks.
# The online learner can eventually replace this with a learned program.

STAND_PROGRAM: list[dict] = [
    # Phase 1: Tuck (bring legs under body, lean forward)
    {
        "ticks": 30,
        "intents": {
            "intent_stop_recover": 0.85,
            "intent_torso_forward": 0.70,
            "intent_support_left": 0.60,
            "intent_support_right": 0.60,
            "intent_stride": 0.48,
            "intent_gait_coupling": 0.90,
        },
    },
    # Phase 2: Push up (extend legs, lean torso forward for balance)
    {
        "ticks": 40,
        "intents": {
            "intent_stop_recover": 0.65,
            "intent_torso_forward": 0.62,
            "intent_support_left": 0.55,
            "intent_support_right": 0.55,
            "intent_stride": 0.50,
            "intent_gait_coupling": 0.85,
        },
    },
    # Phase 3: Stabilize (neutral stance, slight forward lean)
    {
        "ticks": 30,
        "intents": {
            "intent_stop_recover": 0.55,
            "intent_torso_forward": 0.55,
            "intent_support_left": 0.50,
            "intent_support_right": 0.50,
            "intent_stride": 0.50,
            "intent_gait_coupling": 0.80,
            "intent_arm_counterbalance": 0.55,
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
    import numpy as np
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
