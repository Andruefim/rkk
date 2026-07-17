"""Stage-gated motor intents for human task execution (predicate-driven, not verb tables)."""
from __future__ import annotations

from engine.goal_interventions import interventions_for_predicate
from engine.task_goal import GoalPredicate, TaskGoal

LOCOMOTION_STAGES = frozenset({"resolve_target", "approach", "approach_target"})
MANIPULATION_STAGES = frozenset({"reach_contact", "reach_target", "push_target"})
VERIFY_STAGES = frozenset({"verify_goal", "verify_target", "verify_posture"})

_UPPER_BODY_INTENTS = frozenset(
    {
        "intent_reach_left",
        "intent_reach_right",
        "intent_grasp",
        "intent_head_yaw",
        "intent_head_pitch",
    }
)
_BALANCE_INTENTS = frozenset(
    {
        "intent_stride",
        "intent_torso_forward",
        "intent_gait_coupling",
        "intent_support_left",
        "intent_support_right",
        "intent_lean_forward",
    }
)


def active_tree_stage_kind(sim: object) -> str:
    tt = getattr(sim, "_task_tree_ctrl", None)
    if tt is None:
        return ""
    active = getattr(tt, "active_node", None)
    return str(getattr(active, "kind", "") or "")


def intent_allowed_for_stage(intent_key: str, stage_kind: str) -> bool:
    """Return whether an intent_* key may be registered during ``stage_kind``."""
    k = str(intent_key or "")
    if not k.startswith("intent_"):
        return True
    kind = str(stage_kind or "")
    if not kind:
        # No active tree stage — block balance (navigation owns); allow upper-body.
        return k not in _BALANCE_INTENTS
    if kind in VERIFY_STAGES:
        return False
    if k in _UPPER_BODY_INTENTS:
        return kind in MANIPULATION_STAGES
    if k in _BALANCE_INTENTS:
        # Locomotion balance fields come from navigation source during approach.
        return kind in MANIPULATION_STAGES or kind in frozenset({"push_target"})
    return kind not in LOCOMOTION_STAGES


def filter_motor_targets_for_stage(
    targets: dict[str, float],
    stage_kind: str,
) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in targets.items()
        if intent_allowed_for_stage(str(k), stage_kind)
    }


def predicate_for_stage(stage_kind: str, goal: TaskGoal | None) -> GoalPredicate | None:
    if goal is None:
        return None
    preds = list(goal.predicates or [])
    if stage_kind in LOCOMOTION_STAGES:
        for p in preds:
            if p.kind == "reduce_distance":
                return p
    if stage_kind in MANIPULATION_STAGES:
        for p in preds:
            if p.kind == "contact":
                return p
        for p in preds:
            if p.kind == "displace":
                return p
    return None


def motor_for_stage(
    goal: TaskGoal | None,
    stage_kind: str,
    *,
    agent_xy: tuple[float, float] | None = None,
    target_xy: tuple[float, float] | None = None,
    agent_forward: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Motor interventions appropriate for the current tree stage only."""
    pred = predicate_for_stage(stage_kind, goal)
    if pred is None:
        return {}
    return interventions_for_predicate(
        pred,
        agent_xy=agent_xy,
        target_xy=target_xy,
        agent_forward=agent_forward,
    )


def neutralize_blocked_graph_intents(graph: object, stage_kind: str) -> None:
    """Reset autonomous graph intents that conflict with the active stage."""
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return
    defaults = {
        "intent_reach_left": 0.5,
        "intent_reach_right": 0.5,
        "intent_grasp": 0.5,
        "intent_head_yaw": 0.5,
        "intent_head_pitch": 0.5,
    }
    for key, val in defaults.items():
        if key in nodes and not intent_allowed_for_stage(key, stage_kind):
            nodes[key] = float(val)


def human_task_executive_active(sim: object) -> bool:
    """True when a human command task is bound and active on ``sim``."""
    arb = getattr(sim, "_motor_arbiter", None)
    if arb is not None and arb.human_task_active():
        return True
    tb = getattr(sim, "_task_binding", None)
    if tb is None:
        return False
    task = tb.active_task
    if task is None:
        task = getattr(tb, "_active", None)
    if task is None:
        return False
    status = str(getattr(task, "status", "") or "active").strip().lower()
    return status in ("active", "running", "")


def human_task_suppresses_autonomous_locomotion(sim: object) -> bool:
    """Block skills / S2 LOCOMOTE while executive navigation owns the body."""
    if not human_task_executive_active(sim):
        return False
    stage = active_tree_stage_kind(sim)
    if stage in LOCOMOTION_STAGES or stage in MANIPULATION_STAGES:
        return True
    return stage in ("resolve_target", "verify_goal", "verify_target", "verify_posture")


def human_task_suppresses_s2_locomote(sim: object, *, fallen: bool = False) -> bool:
    """S2 must not run LOCOMOTE/EXPLORE macros during an active human task."""
    if fallen:
        return False
    return human_task_executive_active(sim)
