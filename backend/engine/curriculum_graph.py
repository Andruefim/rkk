"""
Track G Phase 5: CurriculumGraph DAG — autonomous curriculum replacing physical_curriculum.

Nodes are goals with prerequisites; supports cross-world goal transfer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def curriculum_graph_enabled() -> bool:
    return os.environ.get("RKK_CURRICULUM_GRAPH_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def curriculum_human_seed() -> bool:
    return os.environ.get("RKK_CURRICULUM_GRAPH_HUMAN_SEED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def goal_transfer_enabled() -> bool:
    return os.environ.get("RKK_GOAL_TRANSFER_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ei(key: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class CurriculumNode:
    node_id: str
    var_id: str
    prerequisites: list[str] = field(default_factory=list)
    world_id: str = "humanoid"
    source: str = "seeded"  # seeded | generated | transferred
    intent_targets: dict[str, float] = field(default_factory=dict)
    min_ticks: int = 200
    status: str = "pending"  # pending | active | completed | failed
    success_rate: float = 0.0
    tick_started: int = 0
    tick_completed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "var_id": self.var_id,
            "prerequisites": list(self.prerequisites),
            "world_id": self.world_id,
            "source": self.source,
            "intent_targets": dict(self.intent_targets),
            "min_ticks": self.min_ticks,
            "status": self.status,
            "success_rate": round(self.success_rate, 4),
            "tick_started": self.tick_started,
            "tick_completed": self.tick_completed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurriculumNode:
        return cls(
            node_id=str(d.get("node_id", "")),
            var_id=str(d.get("var_id", "")),
            prerequisites=[str(x) for x in d.get("prerequisites") or []],
            world_id=str(d.get("world_id", "humanoid")),
            source=str(d.get("source", "seeded")),
            intent_targets={str(k): float(v) for k, v in (d.get("intent_targets") or {}).items()},
            min_ticks=int(d.get("min_ticks", 200)),
            status=str(d.get("status", "pending")),
            success_rate=float(d.get("success_rate", 0.0)),
            tick_started=int(d.get("tick_started", 0)),
            tick_completed=(
                int(d["tick_completed"]) if d.get("tick_completed") is not None else None
            ),
        )


class CurriculumGraph:
    """DAG of curriculum nodes; drives intent_targets when enabled."""

    def __init__(self) -> None:
        self._nodes: dict[str, CurriculumNode] = {}
        self._active_id: str | None = None
        self._generated_count: int = 0
        self._frozen_human_curriculum: bool = False

    def seed_from_physical_curriculum(self, physical_curriculum: Any | None) -> int:
        """Import ALL_SKILLS as seeded DAG nodes."""
        if not curriculum_human_seed() or physical_curriculum is None:
            return 0
        try:
            from engine.physical_curriculum import ALL_SKILLS
        except ImportError:
            return 0
        added = 0
        for skill in ALL_SKILLS:
            stage = skill.stage
            var_id = "posture_stability"
            targets = dict(stage.intent_targets)
            if targets:
                var_id = next(iter(targets.keys()), var_id)
            nid = f"seed_{skill.skill_id}"
            if nid in self._nodes:
                continue
            self._nodes[nid] = CurriculumNode(
                node_id=nid,
                var_id=var_id,
                prerequisites=[f"seed_{p}" for p in skill.prerequisites if p],
                world_id="humanoid",
                source="seeded",
                intent_targets=targets,
                min_ticks=int(stage.min_ticks),
            )
            added += 1
        return added

    def add_generated_node(
        self,
        candidate: Any,
        *,
        tick: int,
        prerequisites: list[str] | None = None,
    ) -> CurriculumNode | None:
        max_gen = _ei("RKK_CURRICULUM_MAX_GENERATED", 20)
        if self._generated_count >= max_gen:
            return None
        var_id = getattr(candidate, "var_id", str(candidate))
        nid = f"gen_{var_id}_{tick}"
        if nid in self._nodes:
            return None
        prereq = list(prerequisites or [])
        if self._active_id:
            prereq = prereq or [self._active_id]
        targets = {var_id: float(getattr(candidate, "target_val", 0.62))}
        node = CurriculumNode(
            node_id=nid,
            var_id=str(var_id),
            prerequisites=prereq,
            world_id=str(getattr(candidate, "world_id", "humanoid")),
            source="generated",
            intent_targets=targets,
            min_ticks=200,
            status="pending",
        )
        self._nodes[nid] = node
        self._generated_count += 1
        return node

    def _unlocked(self, node: CurriculumNode) -> bool:
        return all(
            self._nodes[p].status == "completed"
            for p in node.prerequisites
            if p in self._nodes
        )

    def get_next_pending(self, world_id: str | None = None) -> CurriculumNode | None:
        pending = [
            n
            for n in self._nodes.values()
            if n.status == "pending"
            and self._unlocked(n)
            and (world_id is None or n.world_id == world_id)
        ]
        if not pending:
            return None
        pending.sort(key=lambda n: (n.source != "seeded", n.node_id))
        return pending[0]

    def activate_next(self, tick: int, world_id: str = "humanoid") -> CurriculumNode | None:
        if self._active_id:
            cur = self._nodes.get(self._active_id)
            if cur and cur.status == "active":
                return cur
        nxt = self.get_next_pending(world_id)
        if nxt is None:
            return None
        nxt.status = "active"
        nxt.tick_started = tick
        self._active_id = nxt.node_id
        return nxt

    def mark_completed(
        self,
        node_id: str,
        *,
        success_rate: float,
        tick: int,
    ) -> list[str]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        node.status = "completed"
        node.success_rate = float(success_rate)
        node.tick_completed = tick
        if self._active_id == node_id:
            self._active_id = None
        return [
            n.node_id
            for n in self._nodes.values()
            if n.status == "pending" and self._unlocked(n)
        ]

    def get_active_intent_targets(self) -> dict[str, float]:
        if not self._active_id:
            return {}
        node = self._nodes.get(self._active_id)
        if node is None or node.status != "active":
            return {}
        return dict(node.intent_targets)

    def transfer_goals_to_world(
        self,
        from_world: str,
        to_world: str,
        role_map: dict[str, str] | None = None,
    ) -> list[CurriculumNode]:
        """Cross-world goal transfer for completed nodes above SR threshold."""
        if not goal_transfer_enabled():
            return []
        min_sr = _ef("RKK_GOAL_TRANSFER_MIN_SUCCESS", 0.40)
        transferred: list[CurriculumNode] = []
        for node in list(self._nodes.values()):
            if node.world_id != from_world or node.status != "completed":
                continue
            if node.success_rate < min_sr:
                continue
            mapped_var = node.var_id
            if role_map and mapped_var in role_map:
                mapped_var = role_map[mapped_var]
            nid = f"xfer_{to_world}_{node.node_id}"
            if nid in self._nodes:
                continue
            targets = {mapped_var: float(node.intent_targets.get(node.var_id, 0.62))}
            xfer = CurriculumNode(
                node_id=nid,
                var_id=mapped_var,
                prerequisites=[],
                world_id=to_world,
                source="transferred",
                intent_targets=targets,
                min_ticks=node.min_ticks,
                status="pending",
            )
            self._nodes[nid] = xfer
            transferred.append(xfer)
        return transferred

    def freeze_human_curriculum(self) -> None:
        self._frozen_human_curriculum = True

    def human_curriculum_frozen(self) -> bool:
        return self._frozen_human_curriculum and curriculum_graph_enabled()

    def snapshot(self) -> dict[str, Any]:
        completed_gen = [
            n.to_dict()
            for n in self._nodes.values()
            if n.status == "completed" and n.source == "generated"
        ]
        return {
            "enabled": curriculum_graph_enabled(),
            "frozen_human_curriculum": self._frozen_human_curriculum,
            "active_node": self._active_id,
            "active_intent_targets": self.get_active_intent_targets(),
            "node_count": len(self._nodes),
            "generated_count": self._generated_count,
            "completed_generated": completed_gen[-5:],
            "pending_count": sum(1 for n in self._nodes.values() if n.status == "pending"),
            "completed_count": sum(1 for n in self._nodes.values() if n.status == "completed"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "active_id": self._active_id,
            "generated_count": self._generated_count,
            "frozen_human_curriculum": self._frozen_human_curriculum,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if not data:
            return
        self._nodes = {
            k: CurriculumNode.from_dict(v)
            for k, v in (data.get("nodes") or {}).items()
        }
        self._active_id = data.get("active_id")
        self._generated_count = int(data.get("generated_count", 0))
        self._frozen_human_curriculum = bool(data.get("frozen_human_curriculum", False))
