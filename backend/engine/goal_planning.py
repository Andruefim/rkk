"""
Этап E — целевое планирование (imagination rollout).

Переменные self_goal_active / self_goal_target_dist в SELF_VARS; при активной цели
агент ищет действие (beam search на RKK_PLAN_DEPTH шагов), минимизируя предсказанный
target_dist в world model (propagate_from + rollout_step_free).

Биологическая аналогия:
  - PFC / hippocampus: multi-step prospective simulation (receding-horizon MPC).
  - Cerebellum: короткий free-rollout после каждого мысленного do().
  - Motor cortex: исполняет первый шаг плана, затем перепланирование на следующем тике.
"""
from __future__ import annotations

import os
from collections.abc import Callable
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
    """Intent-only action space for embodied planning."""
    from engine.environment_humanoid import MOTOR_INTENT_VARS

    return set(MOTOR_INTENT_VARS)


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
        if nid.startswith("intent_"):
            out.append(nid)
            continue
        if nid.startswith("phys_intent_"):
            out.append(nid)
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


def plan_depth_max() -> int:
    try:
        cap = int(os.environ.get("RKK_PLAN_DEPTH_MAX", "12"))
    except ValueError:
        cap = 12
    return max(3, min(24, cap))


def plan_depth() -> int:
    try:
        d = int(os.environ.get("RKK_PLAN_DEPTH", "5"))
    except ValueError:
        d = 5
    return max(1, min(plan_depth_max(), d))


def plan_action_discount() -> float:
    """Вес предыдущих шагов в beam (receding horizon / hippocampal discount)."""
    try:
        g = float(os.environ.get("RKK_PLAN_ACTION_DISCOUNT", "0.38"))
    except ValueError:
        g = 0.38
    return float(np.clip(g, 0.05, 0.95))


def plan_branch_per_beam() -> int:
    """Fan-out действий на каждом уровне beam после первого."""
    try:
        b = int(os.environ.get("RKK_PLAN_BRANCH_PER_BEAM", "24"))
    except ValueError:
        b = 24
    return max(6, min(96, b))


def imagination_steps_default() -> int:
    """Free-rollout шагов WM после каждого мысленного do (cerebellar forward model)."""
    try:
        h = int(os.environ.get("RKK_IMAGINATION_STEPS", "12"))
    except ValueError:
        h = 12
    return max(0, min(48, h))


def imagination_steps_fallen() -> int:
    try:
        h = int(os.environ.get("RKK_IMAGINATION_STEPS_FALLEN", "6"))
    except ValueError:
        h = 6
    return max(0, min(imagination_steps_default(), h))


def imagination_steps_fixed_root() -> int:
    try:
        h = int(os.environ.get("RKK_IMAGINATION_STEPS_FIXED", "4"))
    except ValueError:
        h = 4
    return max(0, min(imagination_steps_default(), h))


def imagination_steps_for_context(
    *,
    fallen: bool = False,
    fallen_override: bool = False,
    fixed_root: bool = False,
) -> int:
    if fixed_root:
        return imagination_steps_fixed_root()
    if fallen_override or fallen:
        return imagination_steps_fallen()
    return imagination_steps_default()


def vl_imagination_steps_fallen() -> int:
    """VL safety check horizon when posture is low (shorter than planner)."""
    try:
        h = int(os.environ.get("RKK_VL_IMAGINATION_FALLEN", "4"))
    except ValueError:
        h = 4
    return max(0, min(imagination_steps_fallen(), h))


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


def plan_max_branch_effective(*, fixed_root: bool) -> int:
    """Меньше ветвлений в curriculum fixed_root — меньше WM-forward на тик."""
    m = plan_max_branch()
    if not fixed_root:
        return m
    try:
        cap = int(os.environ.get("RKK_PLAN_MAX_BRANCH_FIXED", "16"))
    except ValueError:
        cap = 16
    return max(8, min(m, cap))


def subsample_actions(
    actions: list[tuple[str, float]],
    max_n: int,
    *,
    rng: np.random.Generator | None = None,
) -> list[tuple[str, float]]:
    if len(actions) <= max_n:
        return list(actions)
    r = rng if rng is not None else np.random.default_rng()
    idx = r.choice(len(actions), size=max_n, replace=False)
    return [actions[int(i)] for i in idx]


def beam_search_first_action(
    agent: Any,
    *,
    state0: dict[str, float],
    actions: list[tuple[str, float]],
    depth: int,
    beam_k: int,
    rollout_horizon: int,
    score_fn: Callable[[dict[str, float], str, float, dict[str, float]], float],
    accept_fn: Callable[[dict[str, float], str, float], bool] | None = None,
    maximize: bool = True,
) -> tuple[tuple[str, float] | None, float]:
    """
    N-step beam search; returns (first_action, best_aggregate_score).

    score_fn(state0, var, val, state_after) — higher is better (use -cost for minimization).
    """
    if not actions or depth < 1:
        return None, float("-inf") if maximize else float("inf")

    accept = accept_fn or (lambda _s, _v, _x: True)
    discount = plan_action_discount()
    branch_later = plan_branch_per_beam()
    rng = np.random.default_rng()

    # frontier: (state_after, first_action, aggregate_score)
    frontier: list[tuple[dict[str, float], tuple[str, float], float]] = []

    for level in range(depth):
        level_actions = (
            actions if level == 0 else subsample_actions(actions, branch_later, rng=rng)
        )
        row_actions: list[tuple[str, float]] = []
        row_bases: list[dict[str, float]] = []
        row_meta: list[tuple[tuple[str, float] | None, float]] = []

        if level == 0:
            row_actions = list(level_actions)
            meta0 = [(None, 0.0)] * len(row_actions)
            try:
                states_out = agent._batch_rollout_imagination_states(
                    state0, row_actions, horizon=rollout_horizon
                )
            except Exception:
                return None, float("-inf") if maximize else float("inf")
        else:
            for state, first_act, agg in frontier:
                for var, val in level_actions:
                    row_bases.append(dict(state))
                    row_actions.append((var, val))
                    row_meta.append((first_act, agg))
            meta0 = row_meta
            try:
                states_out = agent._batch_rollout_imagination_states(
                    state0, row_actions, row_bases=row_bases, horizon=rollout_horizon
                )
            except Exception:
                break

        candidates: list[tuple[float, dict[str, float], tuple[str, float]]] = []
        for j, (var, val) in enumerate(row_actions):
            if j >= len(states_out):
                break
            s_next = states_out[j]
            if not accept(s_next, var, val):
                continue
            sc = score_fn(state0, var, val, s_next)
            first_act, agg = meta0[j]
            if level == 0:
                fa: tuple[str, float] = (var, val)
                total = sc
            else:
                assert first_act is not None
                fa = first_act
                total = sc + discount * agg
            candidates.append((total, s_next, fa))

        if not candidates:
            break
        candidates.sort(key=lambda t: t[0], reverse=maximize)
        frontier = []
        seen: set[tuple[str, float]] = set()
        for total, s_next, fa in candidates:
            if fa in seen:
                continue
            seen.add(fa)
            frontier.append((s_next, fa, total))
            if len(frontier) >= beam_k:
                break

    if not frontier:
        return None, float("-inf") if maximize else float("inf")
    best = max(frontier, key=lambda t: t[2]) if maximize else min(frontier, key=lambda t: t[2])
    return best[1], float(best[2])
