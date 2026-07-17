"""Task-conditioned observation keys and embodiment distance thresholds.

Predicate-driven constants — not verb/object heuristics.
"""
from __future__ import annotations

import math
import os
from typing import Any

from engine.task_goal import TaskGoal

TASK_TARGET_DIST = "task_target_dist_m"
TASK_CONTACT = "task_contact"
# Aliases consumed by WM graph nodes and legacy predicate readers.
CONTACT_SIGNAL = "contact_signal"
GRASP_CONTACT = "grasp_contact"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def nav_stop_m() -> float:
    """Locomotion stop distance for reduce_distance predicate."""
    return _env_float("RKK_NAV_STOP_M", 0.55)


def reach_start_m() -> float:
    """Distance at which closed-loop reach manipulation begins."""
    return _env_float("RKK_REACH_START_M", 0.72)


def contact_reach_m() -> float:
    """Max COM-to-target XY distance where contact is physically plausible."""
    return _env_float("RKK_CONTACT_REACH_M", 0.95)


def goal_near_m_legacy() -> float:
    """Legacy curriculum near distance (avoid for task predicates)."""
    return _env_float("RKK_GOAL_NEAR_M", 0.9)


def compute_task_target_dist(
    agent_xy: tuple[float, float] | None,
    target_xy: tuple[float, float] | None,
) -> float | None:
    if agent_xy is None or target_xy is None:
        return None
    try:
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    return float(math.hypot(tx - ax, ty - ay))


def build_task_observations(
    *,
    agent_xy: tuple[float, float] | None,
    target_xy: tuple[float, float] | None,
    contact: float,
) -> dict[str, float]:
    out: dict[str, float] = {
        TASK_CONTACT: float(max(0.0, min(1.0, contact))),
        CONTACT_SIGNAL: float(max(0.0, min(1.0, contact))),
        GRASP_CONTACT: float(max(0.0, min(1.0, contact))),
    }
    dist = compute_task_target_dist(agent_xy, target_xy)
    if dist is not None:
        out[TASK_TARGET_DIST] = float(dist)
    return out


def inject_task_observations(
    obs: dict[str, float],
    task_obs: dict[str, float],
) -> dict[str, float]:
    merged = dict(obs)
    for k, v in task_obs.items():
        merged[k] = float(v)
    return merged


def task_observation_keys_for_goal(goal: TaskGoal | None) -> list[str]:
    if goal is None:
        return []
    keys: list[str] = []
    for p in goal.predicates:
        if p.kind == "state_key" and p.key:
            keys.append(str(p.key))
        elif p.kind == "reduce_distance":
            keys.append(TASK_TARGET_DIST)
        elif p.kind == "contact":
            keys.extend([TASK_CONTACT, CONTACT_SIGNAL, GRASP_CONTACT])
        elif p.kind == "displace":
            keys.append(TASK_TARGET_DIST)
    return list(dict.fromkeys(keys))


def sync_task_obs_to_graph(graph: Any, task_obs: dict[str, float]) -> None:
    nodes = getattr(graph, "nodes", None)
    if nodes is None or not isinstance(nodes, dict):
        return
    for k, v in task_obs.items():
        nodes[k] = float(v)
