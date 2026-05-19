"""
S2-gated world-model planner: imagination rollouts under System 2 task + energy cost.

System 2 задаёт макрос / expected_state; WM перебирает intent-действия и выбирает
первый шаг (опционально 2–3 шага beam) по скору задачи, а не по EIG.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.goal_planning import (
    plan_beam_k,
    plan_depth,
    plan_max_branch_effective,
    planning_graph_motor_vars,
)
from engine.symbolic_verifier import symbolic_verifier_enabled, verify_normalized_prediction
from engine.system2.schema import EpisodeSuccessSpec
from engine.system2.success_predicates import (
    obs_value_for_key,
    prediction_error_total,
    resolve_max_prediction_error,
)


def s2_wm_planner_enabled() -> bool:
    return os.environ.get("RKK_S2_WM_PLANNER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def s2_wm_fast_override_enabled() -> bool:
    """При fallen_override — bundle/кэш без полного WM-beam каждый тик."""
    if not s2_wm_planner_enabled():
        return False
    return os.environ.get("RKK_S2_WM_FAST_OVERRIDE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def s2_wm_recover_max_branch() -> int:
    try:
        return max(8, int(os.environ.get("RKK_S2_WM_RECOVER_MAX_BRANCH", "16")))
    except ValueError:
        return 16


def s2_wm_gate_strict() -> bool:
    """При активной задаче S2 — только WM-кандидат (без EIG / CEM / legacy goal_plan)."""
    if not s2_wm_planner_enabled():
        return False
    return os.environ.get("RKK_S2_WM_GATE_STRICT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _env_float(key: str, default: str) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return float(default)


def _parse_value_levels(raw: str, fallback: list[float]) -> list[float]:
    levels: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            levels.append(float(np.clip(float(p), 0.06, 0.94)))
        except ValueError:
            continue
    return levels if levels else fallback


def _obs_f(state: dict[str, float], key: str, default: float = 0.5) -> float:
    v = state.get(key, state.get(f"phys_{key}", default))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class S2WmTask:
    macro: str
    fallen: bool = False
    fallen_override: bool = False
    expected_state: dict[str, float] = field(default_factory=dict)
    max_prediction_error: float | None = None
    skill_id: str | None = None
    goal_target_dist: float = 0.42
    self_goal_active: float = 0.0

    @property
    def active(self) -> bool:
        m = str(self.macro or "").strip().upper()
        if m == "IDLE" and not self.fallen_override:
            return False
        if self.fallen_override or self.fallen:
            return True
        if self.self_goal_active > 0.45 and m != "IDLE":
            return True
        if self.self_goal_active > 0.45 and m in (
            "RECOVER_POSTURE",
            "LOCOMOTE_DELIVERY",
            "EXPLORE",
        ):
            return True
        return False


def task_from_planning_context(ctx: dict[str, Any] | None, graph_nodes: dict[str, float]) -> S2WmTask:
    ctx = ctx or {}
    macro = str(ctx.get("macro") or "IDLE").strip().upper()
    if ctx.get("fallen_override_active"):
        macro = "RECOVER_POSTURE"
    es_raw = ctx.get("expected_state")
    es = dict(es_raw) if isinstance(es_raw, dict) else {}
    mx = ctx.get("max_prediction_error")
    try:
        mx_f = float(mx) if mx is not None else None
    except (TypeError, ValueError):
        mx_f = None
    return S2WmTask(
        macro=macro,
        fallen=bool(ctx.get("fallen", False)),
        fallen_override=bool(ctx.get("fallen_override_active", False)),
        expected_state=es,
        max_prediction_error=mx_f,
        skill_id=str(ctx["skill_id"]) if ctx.get("skill_id") else None,
        goal_target_dist=float(
            graph_nodes.get("self_goal_target_dist", graph_nodes.get("target_dist", 0.42))
        ),
        self_goal_active=float(graph_nodes.get("self_goal_active", 0.0)),
    )


def _action_var_whitelist(task: S2WmTask, motor: list[str]) -> list[str]:
    m = task.macro
    recover_set = {
        "intent_torso_forward",
        "intent_stop_recover",
        "intent_support_left",
        "intent_support_right",
        "intent_lean_forward",
        "intent_arm_counterbalance",
    }
    loco_set = {
        "intent_stride",
        "intent_torso_forward",
        "intent_lean_forward",
        "intent_support_left",
        "intent_support_right",
    }
    explore_set = {
        "intent_reach_left",
        "intent_reach_right",
        "intent_wave",
        "intent_torso_forward",
        "intent_stride",
    }
    if task.fallen_override or (task.fallen and m == "RECOVER_POSTURE"):
        allowed = recover_set
    elif m == "RECOVER_POSTURE":
        allowed = recover_set | {"intent_stride"}
    elif m == "LOCOMOTE_DELIVERY":
        allowed = loco_set
    elif m == "EXPLORE":
        allowed = explore_set
    else:
        allowed = set(motor)
    out: list[str] = []
    for v in motor:
        bare = v[5:] if v.startswith("phys_intent_") else v
        if v in allowed or bare in allowed:
            out.append(v)
    return out


def _value_levels_for_task(task: S2WmTask) -> list[float]:
    if task.macro == "RECOVER_POSTURE" or task.fallen_override:
        return _parse_value_levels(
            os.environ.get("RKK_S2_WM_RECOVER_VALUES", "0.52,0.62,0.72,0.82"),
            [0.52, 0.62, 0.72, 0.82],
        )
    if task.macro == "LOCOMOTE_DELIVERY":
        return _parse_value_levels(
            os.environ.get("RKK_S2_WM_LOCOMOTE_VALUES", "0.42,0.52,0.62,0.72"),
            [0.42, 0.52, 0.62, 0.72],
        )
    return _parse_value_levels(
        os.environ.get("RKK_S2_WM_VALUES", "0.38,0.52,0.62"),
        [0.38, 0.52, 0.62],
    )


def score_wm_trajectory(
    state0: dict[str, float],
    state_end: dict[str, float],
    task: S2WmTask,
    *,
    action_var: str,
    action_val: float,
) -> float:
    """Больше = лучше. Учитывает задачу S2, штраф за stride при recovery, энергию, усилие."""
    w_post = _env_float("RKK_S2_WM_W_POSTURE", "2.2")
    w_com = _env_float("RKK_S2_WM_W_COMZ", "1.6")
    w_td = _env_float("RKK_S2_WM_W_TARGET_DIST", "2.0")
    w_rec = _env_float("RKK_S2_WM_W_RECOVER_INTENT", "0.35")
    w_energy = _env_float("RKK_S2_WM_W_ENERGY", "0.45")
    w_effort = _env_float("RKK_S2_WM_W_EFFORT", "0.25")
    w_stride_pen = _env_float("RKK_S2_WM_STRIDE_PENALTY", "2.5")

    score = 0.0
    ps0, ps1 = _obs_f(state0, "posture_stability"), _obs_f(state_end, "posture_stability")
    cz0, cz1 = _obs_f(state0, "com_z"), _obs_f(state_end, "com_z")
    td0, td1 = _obs_f(state0, "target_dist"), _obs_f(state_end, "target_dist")
    e0, e1 = _obs_f(state0, "intero_energy", 0.5), _obs_f(state_end, "intero_energy", 0.5)

    m = task.macro
    if m in ("RECOVER_POSTURE",) or task.fallen_override:
        score += w_post * (ps1 - ps0)
        score += w_com * (cz1 - cz0)
        if "recover" in action_var or action_var.endswith("stop_recover"):
            score += w_rec * max(0.0, action_val - 0.5)
        if "torso" in action_var:
            score += 0.4 * max(0.0, action_val - 0.5)
        if task.fallen_override or task.fallen:
            if "stride" in action_var and action_val > 0.52:
                score -= w_stride_pen * (action_val - 0.5)
    elif m == "LOCOMOTE_DELIVERY":
        score += w_td * (td0 - td1)
        score += 0.5 * w_post * (ps1 - ps0)
        if "stride" in action_var:
            score += 0.35 * (action_val - 0.45)
    elif m == "EXPLORE":
        score += 0.6 * w_td * (td0 - td1)
        score += 0.3 * (ps1 - ps0)
    else:
        score += 0.5 * (td0 - td1)

    score -= w_energy * max(0.0, e0 - e1)
    score -= w_effort * abs(float(action_val) - 0.5)

    if task.expected_state:
        max_pe = resolve_max_prediction_error(
            task.max_prediction_error,
            n_keys=len(task.expected_state),
            macro=task.macro,
            skill_id=task.skill_id,
        )
        pe = prediction_error_total(state_end, task.expected_state)
        if pe <= max_pe:
            score += _env_float("RKK_S2_WM_PE_BONUS", "0.35")
        else:
            score -= _env_float("RKK_S2_WM_PE_PENALTY", "1.2") * (pe - max_pe)

    return float(score)


def _bundle_fallback_quick(
    planning_context: dict[str, Any] | None,
    task: S2WmTask,
    agent: Any,
) -> dict[str, Any] | None:
    """Лёгкий кандидат для fallen_override без batch WM-rollout."""
    ctx = planning_context or {}
    bundle_c = ctx.get("recovery_schedule_candidate")
    if not isinstance(bundle_c, dict) or not bundle_c.get("variable"):
        bundle_c = ctx.get("bundle_candidate")
    if not isinstance(bundle_c, dict) or not bundle_c.get("variable"):
        return None
    var = str(bundle_c["variable"])
    val = float(bundle_c.get("value", 0.5))
    target = "posture_stability"
    if task.macro in ("LOCOMOTE_DELIVERY", "EXPLORE"):
        target = "target_dist"
    try:
        feat = agent._features_for_intervention_pair(var, target)
    except Exception:
        feat = []
    return {
        "variable": var,
        "target": target,
        "value": float(np.clip(val, 0.06, 0.94)),
        "uncertainty": 0.38,
        "features": feat,
        "expected_ig": 0.62,
        "from_system2": True,
        "from_s2_wm_planner": True,
        "s2_wm_score": 0.0,
        "s2_wm_macro": task.macro,
        "s2_wm_horizon": 0,
        "s2_wm_bundle_fallback": True,
        "s2_wm_fast_override": True,
    }


def plan_s2_wm_candidate(
    agent: Any,
    *,
    planning_context: dict[str, Any] | None,
    enable_l3: bool,
    fixed_root: bool,
) -> dict[str, Any] | None:
    """
    WM imagination под задачей System 2. Возвращает кандидат для agent.step (как goal_plan / S2).
    """
    if not s2_wm_planner_enabled() or not enable_l3:
        return None
    if agent.graph._core is None:
        return None
    try:
        min_steps = int(os.environ.get("RKK_S2_WM_MIN_WM_STEPS", "24"))
    except ValueError:
        min_steps = 24
    state0 = dict(agent.graph.nodes)
    task = task_from_planning_context(planning_context, state0)
    if not task.active:
        return None

    if task.fallen_override and s2_wm_fast_override_enabled():
        fb_fast = _bundle_fallback_quick(planning_context, task, agent)
        if fb_fast is not None:
            return fb_fast

    def _bundle_fallback() -> dict[str, Any] | None:
        ctx = planning_context or {}
        bundle_c = ctx.get("bundle_candidate")
        if not isinstance(bundle_c, dict) or not bundle_c.get("variable"):
            return None
        var = str(bundle_c["variable"])
        val = float(bundle_c.get("value", 0.5))
        target = "posture_stability"
        if task.macro in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            target = "target_dist"
        try:
            feat = agent._features_for_intervention_pair(var, target)
        except Exception:
            feat = []
        return {
            "variable": var,
            "target": target,
            "value": float(np.clip(val, 0.06, 0.94)),
            "uncertainty": 0.38,
            "features": feat,
            "expected_ig": 0.62,
            "from_system2": True,
            "from_s2_wm_planner": True,
            "s2_wm_score": 0.0,
            "s2_wm_macro": task.macro,
            "s2_wm_horizon": 0,
            "s2_wm_bundle_fallback": True,
        }

    if getattr(agent, "_notears_steps", 0) < min_steps:
        if task.fallen_override or task.macro == "RECOVER_POSTURE":
            return _bundle_fallback()
        return None

    motor = _action_var_whitelist(
        task, planning_graph_motor_vars(agent.env, list(agent.graph._node_ids))
    )
    if not motor:
        return None

    levels = _value_levels_for_task(task)
    actions: list[tuple[str, float]] = [(v, x) for v in motor for x in levels]
    max_b = plan_max_branch_effective(fixed_root=fixed_root)
    if task.fallen_override or (task.fallen and task.macro == "RECOVER_POSTURE"):
        max_b = min(max_b, s2_wm_recover_max_branch())
    if len(actions) > max_b:
        idx = np.random.default_rng().choice(len(actions), size=max_b, replace=False)
        actions = [actions[int(i)] for i in idx]

    depth = plan_depth()
    beam_k = plan_beam_k()
    horizon = agent._effective_imagination_horizon(enable_l3=True)

    def _accept(sfin: dict[str, float], var: str, val: float) -> bool:
        if symbolic_verifier_enabled():
            ok, _ = verify_normalized_prediction(dict(sfin), agent.env)
            if not ok:
                return False
        return True

    best_score = float("-inf")
    best_first: tuple[str, float] | None = None

    if depth <= 1:
        try:
            states_fin = agent._batch_rollout_imagination_states(state0, actions)
        except Exception:
            return None
        for i, (var, val) in enumerate(actions):
            if i >= len(states_fin):
                break
            sfin = states_fin[i]
            if not _accept(sfin, var, val):
                continue
            sc = score_wm_trajectory(state0, sfin, task, action_var=var, action_val=val)
            if sc > best_score:
                best_score = sc
                best_first = (var, val)
    else:
        scored: list[tuple[float, str, float, dict[str, float]]] = []
        try:
            states1 = agent._batch_rollout_imagination_states(state0, actions)
        except Exception:
            return None
        for i, (var, val) in enumerate(actions):
            if i >= len(states1):
                break
            s1 = states1[i]
            if not _accept(s1, var, val):
                continue
            sc = score_wm_trajectory(state0, s1, task, action_var=var, action_val=val)
            scored.append((sc, var, val, dict(s1)))
        scored.sort(key=lambda t: -t[0])
        row_bases: list[dict[str, float]] = []
        row_actions: list[tuple[str, float]] = []
        row_meta: list[tuple[str, float, float]] = []
        for sc1, v1, x1, s1 in scored[:beam_k]:
            for v2, x2 in actions:
                row_bases.append(s1)
                row_actions.append((v2, x2))
                row_meta.append((sc1, v1, x1))
        try:
            states2 = agent._batch_rollout_imagination_states(
                state0, row_actions, row_bases=row_bases
            )
        except Exception:
            states2 = []
        for j, sfin in enumerate(states2):
            if j >= len(row_meta):
                break
            sc1, v1, x1 = row_meta[j]
            v2, x2 = row_actions[j]
            if not _accept(sfin, v2, x2):
                continue
            sc2 = score_wm_trajectory(state0, sfin, task, action_var=v2, action_val=x2)
            sc = sc1 * 0.35 + sc2
            if sc > best_score:
                best_score = sc
                best_first = (v1, x1)

    if best_first is None:
        return _bundle_fallback()

    var, val = best_first
    target = "posture_stability"
    if task.macro == "LOCOMOTE_DELIVERY" or task.macro == "EXPLORE":
        target = "target_dist"
    try:
        feat = agent._features_for_intervention_pair(var, target)
    except Exception:
        feat = []

    return {
        "variable": var,
        "target": target,
        "value": float(np.clip(val, 0.06, 0.94)),
        "uncertainty": 0.32,
        "features": feat,
        "expected_ig": float(np.clip(0.55 + best_score * 0.15, 0.2, 0.98)),
        "from_system2": True,
        "from_s2_wm_planner": True,
        "s2_wm_score": round(best_score, 5),
        "s2_wm_macro": task.macro,
        "s2_wm_horizon": int(horizon),
    }
