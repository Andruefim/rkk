"""
Event-driven hierarchical task tree (wave 1 core).

Flat node store + O(1) tick transitions; nested snapshot on demand only.
Integration with Simulation is handled elsewhere.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from engine.task_goal import GoalPredicate, TaskGoal

TASK_STATUSES = frozenset(
    {"pending", "active", "verifying", "done", "failed", "cancelled"}
)
TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})

DECOMPOSE_MANIPULATE = (
    "resolve_target",
    "approach_target",
    "reach_target",
    "push_target",
    "verify_target",
)
DECOMPOSE_RECOVER = ("recover_posture", "verify_posture")
DECOMPOSE_GENERIC = ("imagine_goal", "execute_goal", "verify_goal")

_MOTOR_APPROACH: dict[str, float] = {
    "intent_stride": 0.62,
    "intent_torso_forward": 0.55,
}
_MOTOR_REACH: dict[str, float] = {
    "intent_reach_right": 0.58,
    "intent_grasp": 0.45,
}
_MOTOR_PUSH: dict[str, float] = {
    "intent_lean_forward": 0.52,
    "intent_stride": 0.48,
    "intent_reach_right": 0.42,
}
_MOTOR_RECOVER: dict[str, float] = {
    "intent_stop_recover": 0.72,
    "intent_torso_forward": 0.48,
}

_STEP_LABELS: dict[str, str] = {
    "resolve_target": "Resolve target",
    "approach": "Approach",
    "reach_contact": "Reach and contact",
    "approach_target": "Approach target",
    "reach_target": "Reach target",
    "push_target": "Push target",
    "verify_target": "Verify target",
    "recover_posture": "Recover posture",
    "verify_posture": "Verify posture",
    "imagine_goal": "Imagine goal",
    "execute_goal": "Execute goal",
    "verify_goal": "Verify goal",
}

_PREDICATE_KIND_ORDER: tuple[str, ...] = (
    "reduce_distance",
    "contact",
    "displace",
    "state_key",
)


def _env_int(key: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def task_replan_max() -> int:
    return _env_int("RKK_TASK_REPLAN_MAX", 2)


def task_tree_max_nodes() -> int:
    return max(4, _env_int("RKK_TASK_TREE_MAX_NODES", 32))


def task_tree_max_depth() -> int:
    return max(1, _env_int("RKK_TASK_TREE_MAX_DEPTH", 8))


def task_deadline_ticks() -> int:
    return max(1, _env_int("RKK_TASK_DEADLINE_TICKS", 2400))


def task_tree_enabled() -> bool:
    """Hierarchical task tree; default on when RKK_TASK_BINDING=1 unless RKK_TASK_TREE=0."""
    explicit = os.environ.get("RKK_TASK_TREE", "").strip().lower()
    if explicit in ("0", "false", "no", "off"):
        return False
    if explicit in ("1", "true", "yes", "on"):
        return True
    from engine.task_binding import task_binding_enabled

    return task_binding_enabled()


def _round_float(v: float, ndigits: int = 4) -> float:
    return round(float(v), ndigits)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _round_float(value) if isinstance(value, float) else value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _status_for_kind(kind: str) -> str:
    if str(kind).startswith("verify"):
        return "verifying"
    return "active"


def _motor_for_kind(kind: str) -> dict[str, float]:
    if kind in ("approach", "approach_target"):
        return dict(_MOTOR_APPROACH)
    if kind in ("reach_contact", "reach_target"):
        return dict(_MOTOR_REACH)
    if kind == "push_target":
        return dict(_MOTOR_PUSH)
    if kind == "recover_posture":
        return dict(_MOTOR_RECOVER)
    return {}


def _sort_predicates(predicates: list[GoalPredicate]) -> list[GoalPredicate]:
    order = {k: i for i, k in enumerate(_PREDICATE_KIND_ORDER)}

    def _key(p: GoalPredicate) -> tuple[float, int]:
        return (-float(p.weight), order.get(str(p.kind), 99))

    return sorted(predicates, key=_key)


def _decompose_from_goal(goal: TaskGoal, *, needs_target: bool) -> tuple[str, ...]:
    """Build stage kinds from observable predicates (not command_kind)."""
    kinds: list[str] = []
    if needs_target:
        kinds.append("resolve_target")

    preds = _sort_predicates(list(goal.predicates or []))
    has_displace_flow = False
    has_generic_flow = False

    for pred in preds:
        pk = str(pred.kind)
        if pk == "reduce_distance":
            kinds.append("approach")
        elif pk == "contact":
            kinds.append("reach_contact")
        elif pk == "displace":
            if "approach" not in kinds:
                kinds.append("approach")
            if not has_displace_flow:
                kinds.extend(["reach_target", "push_target", "verify_target"])
                has_displace_flow = True
        elif pk == "state_key":
            if not has_generic_flow:
                kinds.extend(["imagine_goal", "execute_goal", "verify_goal"])
                has_generic_flow = True

    if not kinds and not needs_target:
        kinds.extend(list(DECOMPOSE_GENERIC))
    return tuple(kinds)


def _expected_state_for_kind(
    kind: str,
    pred: GoalPredicate | None,
    *,
    root_expected: dict[str, float],
) -> dict[str, float]:
    if kind.startswith("verify"):
        return dict(root_expected)
    if kind == "approach" and pred is not None and pred.kind == "reduce_distance":
        return {"stop_distance": float(pred.target_value)}
    if kind == "reach_contact" and pred is not None and pred.kind == "contact":
        return {"contact_threshold": float(pred.target_value)}
    if kind == "state_key" and pred is not None and pred.kind == "state_key" and pred.key:
        return {str(pred.key): float(pred.target_value)}
    return {}


def _predicate_for_kind(kind: str, goal: TaskGoal) -> GoalPredicate | None:
    preds = list(goal.predicates or [])
    if kind == "approach":
        for p in preds:
            if p.kind == "reduce_distance":
                return p
    if kind == "reach_contact":
        for p in preds:
            if p.kind == "contact":
                return p
    if kind in ("reach_target", "push_target", "verify_target"):
        for p in preds:
            if p.kind == "displace":
                return p
    if kind in ("imagine_goal", "execute_goal", "verify_goal"):
        for p in preds:
            if p.kind == "state_key":
                return p
    return None


def _decompose_kinds(command_kind: str) -> tuple[str, ...]:
    ck = str(command_kind or "generic").strip().lower()
    if ck in ("manipulate", "manipulate_object", "object"):
        return DECOMPOSE_MANIPULATE
    if ck in ("recover", "recovery", "getup"):
        return DECOMPOSE_RECOVER
    return DECOMPOSE_GENERIC


@dataclass
class TaskNode:
    id: str
    parent_id: str | None
    label: str
    kind: str
    status: str = "pending"
    expected_state: dict[str, float] = field(default_factory=dict)
    motor_targets: dict[str, float] = field(default_factory=dict)
    target_ref: str | None = None
    tick_started: int | None = None
    tick_deadline: int | None = None
    last_pe: float | None = None
    attempts: int = 0
    failure_reason: str | None = None
    children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "label": self.label,
            "kind": self.kind,
            "status": self.status,
            "expected_state": {k: _round_float(v) for k, v in self.expected_state.items()},
            "motor_targets": {k: _round_float(v) for k, v in self.motor_targets.items()},
            "target_ref": self.target_ref,
            "tick_started": self.tick_started,
            "tick_deadline": self.tick_deadline,
            "last_pe": _round_float(self.last_pe) if self.last_pe is not None else None,
            "attempts": int(self.attempts),
            "failure_reason": self.failure_reason,
            "children": list(self.children),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskNode:
        expected = raw.get("expected_state") or {}
        motor = raw.get("motor_targets") or {}
        return cls(
            id=str(raw["id"]),
            parent_id=raw.get("parent_id"),
            label=str(raw.get("label", "")),
            kind=str(raw.get("kind", "")),
            status=str(raw.get("status", "pending")),
            expected_state={str(k): float(v) for k, v in expected.items()},
            motor_targets={str(k): float(v) for k, v in motor.items()},
            target_ref=raw.get("target_ref"),
            tick_started=raw.get("tick_started"),
            tick_deadline=raw.get("tick_deadline"),
            last_pe=float(raw["last_pe"]) if raw.get("last_pe") is not None else None,
            attempts=int(raw.get("attempts", 0)),
            failure_reason=raw.get("failure_reason"),
            children=[str(c) for c in (raw.get("children") or [])],
        )


@dataclass
class TaskTree:
    session_id: str
    command_text: str
    root_id: str
    active_node_id: str | None
    root_status: str
    created_tick: int
    completed_tick: int | None = None
    cleared: bool = False
    nodes: dict[str, TaskNode] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "command_text": self.command_text,
            "root_id": self.root_id,
            "active_node_id": self.active_node_id,
            "root_status": self.root_status,
            "created_tick": self.created_tick,
            "completed_tick": self.completed_tick,
            "cleared": bool(self.cleared),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskTree:
        nodes_raw = raw.get("nodes") or {}
        nodes = {str(k): TaskNode.from_dict(v) for k, v in nodes_raw.items()}
        return cls(
            session_id=str(raw["session_id"]),
            command_text=str(raw.get("command_text", "")),
            root_id=str(raw["root_id"]),
            active_node_id=raw.get("active_node_id"),
            root_status=str(raw.get("root_status", "pending")),
            created_tick=int(raw.get("created_tick", 0)),
            completed_tick=raw.get("completed_tick"),
            cleared=bool(raw.get("cleared", False)),
            nodes=nodes,
        )


class TaskTreeController:
    """Session-scoped hierarchical task tree with single active leaf."""

    def __init__(self) -> None:
        self._tree: TaskTree | None = None
        self._cleared_pulse: bool = False
        self._node_seq: int = 0

    @property
    def tree(self) -> TaskTree | None:
        return self._tree

    @property
    def is_active(self) -> bool:
        if self._cleared_pulse:
            return False
        t = self._tree
        if t is None:
            return False
        return t.root_status not in TERMINAL_STATUSES

    @property
    def active_node(self) -> TaskNode | None:
        t = self._tree
        if t is None or not t.active_node_id:
            return None
        return t.nodes.get(t.active_node_id)

    def motor_targets(self) -> dict[str, float]:
        n = self.active_node
        if n is None:
            return {}
        return dict(n.motor_targets)

    def _new_id(self) -> str:
        self._node_seq += 1
        return f"n{self._node_seq}"

    def _cancel_tree(self, tick: int, reason: str) -> None:
        t = self._tree
        if t is None:
            return
        for node in t.nodes.values():
            if node.status not in TERMINAL_STATUSES:
                node.status = "cancelled"
                node.failure_reason = reason
        t.root_status = "cancelled"
        t.active_node_id = None
        t.completed_tick = int(tick)

    def bind_command(
        self,
        text: str,
        tick: int,
        *,
        command_kind: str = "generic",
        target_ref: str | None = None,
        expected_state: dict[str, float] | None = None,
    ) -> TaskTree:
        if self.is_active or (self._tree is not None and self._tree.root_status == "active"):
            self._cancel_tree(int(tick), "preempted")
        self._cleared_pulse = False

        session_id = str(uuid.uuid4())
        root_id = self._new_id()
        created = int(tick)
        deadline = created + task_deadline_ticks()
        exp = {str(k): float(v) for k, v in (expected_state or {}).items()}

        root = TaskNode(
            id=root_id,
            parent_id=None,
            label=str(text or "").strip()[:120] or "command",
            kind=str(command_kind).strip().lower(),
            status="active",
            expected_state=dict(exp),
            target_ref=target_ref,
            tick_started=created,
            tick_deadline=deadline,
        )

        nodes: dict[str, TaskNode] = {root_id: root}
        kinds = _decompose_kinds(command_kind)
        max_nodes = task_tree_max_nodes()
        child_ids: list[str] = []

        for kind in kinds:
            if len(nodes) >= max_nodes:
                break
            cid = self._new_id()
            motor = _motor_for_kind(kind)
            verify_exp = dict(exp) if kind.startswith("verify") else {}
            child = TaskNode(
                id=cid,
                parent_id=root_id,
                label=_STEP_LABELS.get(kind, kind.replace("_", " ").title()),
                kind=kind,
                status="pending",
                expected_state=verify_exp,
                motor_targets=motor,
                target_ref=target_ref,
            )
            nodes[cid] = child
            child_ids.append(cid)

        root.children = child_ids
        active_id: str | None = None
        if child_ids:
            active_id = child_ids[0]
            first = nodes[active_id]
            first.status = _status_for_kind(first.kind)
            first.tick_started = created
            first.tick_deadline = deadline
            if first.attempts == 0:
                first.attempts = 1

        tree = TaskTree(
            session_id=session_id,
            command_text=str(text or "").strip(),
            root_id=root_id,
            active_node_id=active_id,
            root_status="active",
            created_tick=created,
            nodes=nodes,
        )
        self._tree = tree
        return tree

    def bind_goal(
        self,
        goal: TaskGoal,
        tick: int,
        *,
        needs_target: bool = False,
        target_ref: str | None = None,
        expected_state: dict[str, float] | None = None,
    ) -> TaskTree:
        """Bind a predicate-based TaskGoal to a hierarchical stage tree."""
        if self.is_active or (self._tree is not None and self._tree.root_status == "active"):
            self._cancel_tree(int(tick), "preempted")
        self._cleared_pulse = False

        session_id = str(uuid.uuid4())
        root_id = self._new_id()
        created = int(tick)
        deadline = created + task_deadline_ticks()
        exp = {str(k): float(v) for k, v in (expected_state or {}).items()}
        text = str(goal.text or "").strip()

        root = TaskNode(
            id=root_id,
            parent_id=None,
            label=text[:120] or "command",
            kind="goal",
            status="active",
            expected_state=dict(exp),
            target_ref=target_ref or goal.target_ref,
            tick_started=created,
            tick_deadline=deadline,
        )

        nodes: dict[str, TaskNode] = {root_id: root}
        kinds = _decompose_from_goal(goal, needs_target=needs_target)
        max_nodes = task_tree_max_nodes()
        child_ids: list[str] = []

        for kind in kinds:
            if len(nodes) >= max_nodes:
                break
            cid = self._new_id()
            pred = _predicate_for_kind(kind, goal)
            motor = _motor_for_kind(kind)
            verify_exp = _expected_state_for_kind(kind, pred, root_expected=exp)
            child = TaskNode(
                id=cid,
                parent_id=root_id,
                label=_STEP_LABELS.get(kind, kind.replace("_", " ").title()),
                kind=kind,
                status="pending",
                expected_state=verify_exp,
                motor_targets=motor,
                target_ref=target_ref or goal.target_ref,
            )
            nodes[cid] = child
            child_ids.append(cid)

        root.children = child_ids
        active_id: str | None = None
        if child_ids:
            active_id = child_ids[0]
            first = nodes[active_id]
            first.status = _status_for_kind(first.kind)
            first.tick_started = created
            first.tick_deadline = deadline
            if first.attempts == 0:
                first.attempts = 1

        tree = TaskTree(
            session_id=session_id,
            command_text=text,
            root_id=root_id,
            active_node_id=active_id,
            root_status="active",
            created_tick=created,
            nodes=nodes,
        )
        self._tree = tree
        return tree

    def complete_active(
        self,
        tick: int,
        diagnostics: dict[str, Any] | None = None,
    ) -> TaskTree | None:
        t = self._tree
        if t is None or not t.active_node_id:
            return None
        active = t.nodes.get(t.active_node_id)
        if active is None or active.status not in ("active", "verifying"):
            return None

        active.status = "done"
        active.failure_reason = None
        if diagnostics:
            pe = diagnostics.get("pe_total", diagnostics.get("last_pe"))
            if pe is not None:
                try:
                    active.last_pe = float(pe)
                except (TypeError, ValueError):
                    pass

        parent_id = active.parent_id
        if parent_id is None:
            t.root_status = "done"
            t.active_node_id = None
            t.completed_tick = int(tick)
            t.nodes[t.root_id].status = "done"
            return t

        parent = t.nodes[parent_id]
        siblings = parent.children
        try:
            idx = siblings.index(active.id)
        except ValueError:
            idx = -1

        next_id: str | None = None
        if idx >= 0 and idx + 1 < len(siblings):
            next_id = siblings[idx + 1]

        if next_id is not None:
            nxt = t.nodes[next_id]
            nxt.status = _status_for_kind(nxt.kind)
            nxt.tick_started = int(tick)
            nxt.tick_deadline = t.nodes[t.root_id].tick_deadline
            if nxt.attempts == 0:
                nxt.attempts = 1
            t.active_node_id = next_id
            return t

        t.root_status = "done"
        t.active_node_id = None
        t.completed_tick = int(tick)
        t.nodes[t.root_id].status = "done"
        return t

    def fail_active(
        self,
        tick: int,
        reason: str,
        *,
        retryable: bool = False,
    ) -> TaskTree | None:
        t = self._tree
        if t is None or not t.active_node_id:
            return None
        active = t.nodes.get(t.active_node_id)
        if active is None or active.status not in ("active", "verifying"):
            return None

        active.failure_reason = str(reason)
        max_retries = task_replan_max()

        if retryable and active.attempts <= max_retries:
            active.attempts += 1
            active.status = _status_for_kind(active.kind)
            active.tick_started = int(tick)
            active.tick_deadline = int(tick) + task_deadline_ticks()
            return t

        active.status = "failed"
        t.root_status = "failed"
        t.active_node_id = None
        t.completed_tick = int(tick)
        t.nodes[t.root_id].status = "failed"
        return t

    def cancel(self, tick: int, reason: str) -> TaskTree | None:
        if self._tree is None:
            return None
        self._cancel_tree(int(tick), str(reason))
        return self._tree

    def clear(self, tick: int) -> TaskTree | None:
        prev = self._tree
        if prev is not None:
            if prev.root_status not in TERMINAL_STATUSES:
                self._cancel_tree(int(tick), "cleared")
            prev.cleared = True
        self._cleared_pulse = True
        return prev

    def acknowledge_clear(self) -> None:
        """Consume one-shot cleared pulse; subsequent snapshots are inactive."""
        self._cleared_pulse = False
        self._tree = None

    def consume_clear(self) -> None:
        """Alias for acknowledge_clear."""
        self.acknowledge_clear()

    def _progress(self, t: TaskTree) -> float:
        root = t.nodes.get(t.root_id)
        if root is None or not root.children:
            if t.root_status == "done":
                return 1.0
            if t.root_status in TERMINAL_STATUSES:
                return 0.0
            return 0.0
        steps = [t.nodes[cid] for cid in root.children if cid in t.nodes]
        if not steps:
            return 0.0
        done = sum(1 for s in steps if s.status == "done")
        return _round_float(done / len(steps))

    def _nested_nodes(self, node_id: str, t: TaskTree) -> dict[str, Any]:
        node = t.nodes[node_id]
        out: dict[str, Any] = {
            "id": node.id,
            "label": node.label,
            "kind": node.kind,
            "status": node.status,
            "target_ref": node.target_ref,
            "tick_started": node.tick_started,
            "tick_deadline": node.tick_deadline,
            "last_pe": _round_float(node.last_pe) if node.last_pe is not None else None,
            "attempts": int(node.attempts),
            "failure_reason": node.failure_reason,
            "motor_targets": {k: _round_float(v) for k, v in node.motor_targets.items()},
            "expected_state": {k: _round_float(v) for k, v in node.expected_state.items()},
            "children": [
                self._nested_nodes(cid, t) for cid in node.children if cid in t.nodes
            ],
        }
        return _json_safe(out)

    def snapshot(self, tick: int) -> dict[str, Any]:
        if self._cleared_pulse and self._tree is not None:
            t = self._tree
            return _json_safe(
                {
                    "active": False,
                    "session_id": t.session_id,
                    "command_text": t.command_text,
                    "root_status": t.root_status,
                    "current_node_id": None,
                    "progress": self._progress(t),
                    "nodes": [self._nested_nodes(t.root_id, t)],
                    "cleared": True,
                    "tick": int(tick),
                }
            )

        t = self._tree
        if t is None:
            return _json_safe(
                {
                    "active": False,
                    "session_id": None,
                    "command_text": "",
                    "root_status": None,
                    "current_node_id": None,
                    "progress": 0.0,
                    "nodes": None,
                    "cleared": False,
                    "tick": int(tick),
                }
            )

        return _json_safe(
            {
                "active": self.is_active,
                "session_id": t.session_id,
                "command_text": t.command_text,
                "root_status": t.root_status,
                "current_node_id": t.active_node_id,
                "progress": self._progress(t),
                "nodes": [self._nested_nodes(t.root_id, t)],
                "cleared": False,
                "tick": int(tick),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree": self._tree.to_dict() if self._tree else None,
            "cleared_pulse": bool(self._cleared_pulse),
            "node_seq": int(self._node_seq),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskTreeController:
        ctrl = cls()
        tree_raw = raw.get("tree")
        if tree_raw:
            ctrl._tree = TaskTree.from_dict(tree_raw)
        ctrl._cleared_pulse = bool(raw.get("cleared_pulse", False))
        ctrl._node_seq = int(raw.get("node_seq", 0))
        return ctrl
