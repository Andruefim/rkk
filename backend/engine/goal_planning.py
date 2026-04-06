"""
Этап E — целевое планирование (imagination rollout).

Переменные self_goal_active / self_goal_target_dist в SELF_VARS; при активной цели
агент ищет действие (и опционально 2-шаговый план), минимизирующее предсказанный
target_dist в world model (propagate_from + rollout_step_free).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def goal_planning_globally_disabled() -> bool:
    return os.environ.get("RKK_GOAL_PLANNING", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    )


def resolve_humanoid_base(env: Any) -> Any | None:
    """Среда humanoid или base_env внутри EnvironmentVisual."""
    if getattr(env, "preset", None) == "humanoid":
        return env
    b = getattr(env, "base_env", None)
    if b is not None and getattr(b, "preset", None) == "humanoid":
        return b
    return None


def motor_allow_set(humanoid_env: Any) -> set[str]:
    """Имена суставов в пространстве humanoid (без префикса phys_)."""
    from engine.environment_humanoid import ARM_VARS, SPINE_VARS, HEAD_VARS, LEG_VARS

    if getattr(humanoid_env, "_fixed_root", False):
        return set(ARM_VARS + SPINE_VARS + HEAD_VARS)
    return set(ARM_VARS + SPINE_VARS + HEAD_VARS + LEG_VARS)


def planning_graph_motor_vars(env: Any, graph_node_ids: list[str]) -> list[str]:
    """
    Идентификаторы узлов графа, по которым разрешено планировать do(),
    согласованные с hybrid (phys_*) и прямым humanoid.
    """
    base = resolve_humanoid_base(env)
    if base is None:
        return []
    allow = motor_allow_set(base)
    out: list[str] = []
    for nid in graph_node_ids:
        if nid.startswith("self_") or nid.startswith("slot_") or nid.startswith("concept_"):
            continue
        if nid == "target_dist":
            continue
        key = nid[5:] if nid.startswith("phys_") else nid
        if key in allow:
            out.append(nid)
    return out


def parse_plan_value_levels() -> list[float]:
    raw = os.environ.get("RKK_PLAN_VALUES", "0.38,0.52,0.62")
    levels: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            levels.append(float(np.clip(float(p), 0.06, 0.94)))
        except ValueError:
            continue
    return levels if levels else [0.38, 0.52, 0.62]


def plan_depth() -> int:
    try:
        d = int(os.environ.get("RKK_PLAN_DEPTH", "1"))
    except ValueError:
        d = 1
    return max(1, min(3, d))


def plan_beam_k() -> int:
    try:
        k = int(os.environ.get("RKK_PLAN_BEAM", "6"))
    except ValueError:
        k = 6
    return max(2, min(16, k))


def plan_max_branch() -> int:
    try:
        m = int(os.environ.get("RKK_PLAN_MAX_BRANCH", "64"))
    except ValueError:
        m = 64
    return max(12, min(200, m))
