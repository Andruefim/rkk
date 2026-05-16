from __future__ import annotations

from typing import Any

# Макро: патчи self_* в граф + один приоритетный intent-кандидат + мягкие residual к моторике.
MACRO_TABLE: dict[str, dict[str, Any]] = {
    "IDLE": {
        "graph": {},
        "candidate": None,
        "residuals": {},
    },
    "RECOVER_POSTURE": {
        "graph": {
            "self_goal_active": 0.42,
            "self_goal_target_dist": 0.52,
        },
        "candidate": {
            "variable": "intent_torso_forward",
            "value": 0.68,
            "target": "posture_stability",
            "uncertainty": 0.40,
            "expected_ig": 0.90,
        },
        "residuals": {
            "intent_lean_forward": 0.05,
            "intent_stop_recover": 0.06,
            "intent_arm_counterbalance": 0.04,
        },
    },
    "LOCOMOTE_DELIVERY": {
        "graph": {
            "self_goal_active": 0.88,
            "self_goal_target_dist": 0.36,
        },
        "candidate": {
            "variable": "intent_stride",
            "value": 0.58,
            "target": "target_dist",
            "uncertainty": 0.38,
            "expected_ig": 0.88,
        },
        "residuals": {
            "intent_torso_forward": 0.04,
            "intent_lean_forward": 0.03,
        },
    },
    "EXPLORE": {
        "graph": {"self_goal_active": 0.35, "self_goal_target_dist": 0.55},
        "candidate": {
            "variable": "intent_reach_left",
            "value": 0.62,
            "target": "target_dist",
            "uncertainty": 0.55,
            "expected_ig": 0.55,
        },
        "residuals": {
            "intent_wave": 0.06,
        },
    },
}


def macro_bundle(macro: str) -> dict[str, Any]:
    m = str(macro or "").strip().upper()
    return MACRO_TABLE.get(m, MACRO_TABLE["IDLE"])
