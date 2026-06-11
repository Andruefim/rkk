"""
Intention Cortex — long-horizon goal memory, stack, and executive projection.

Biological mapping:
  - dlPFC / mPFC: maintained intention stack + narrative (what & why)
  - Hippocampus: curriculum DAG + episodic bias into future subgoals
  - Basal ganglia gate: macro_hint → System 2 (mode selection)
  - Premotor (L3): intent_targets / self_goal_* → agent WM beam (motor planning)

Runs *before* System2 each tick so macros and graph self-state reflect current intention.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.curriculum_graph import CurriculumGraph, curriculum_graph_enabled
from engine.goal_generator import GoalCandidate, GoalGenerator, goal_gen_enabled
from engine.meta_causal import WMetaEnsemble, build_meta_observation, meta_causal_enabled


def intention_cortex_enabled() -> bool:
    return os.environ.get("RKK_INTENTION_CORTEX", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _ei(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def phase5_component_enabled(flag_fn: Any) -> bool:
    """Master switch OR per-component env flag."""
    if not intention_cortex_enabled():
        return bool(flag_fn())
    return bool(flag_fn())


@dataclass
class SubGoal:
    """One step on the long-horizon intention stack."""

    subgoal_id: str
    var_id: str
    target_val: float
    intent_targets: dict[str, float] = field(default_factory=dict)
    world_id: str = "humanoid"
    tick_start: int = 0
    tick_deadline: int = 0
    min_ticks: int = 120
    source: str = "curriculum"
    priority: float = 0.5
    status: str = "pending"  # pending | active | completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "var_id": self.var_id,
            "target_val": round(self.target_val, 4),
            "intent_targets": {k: round(float(v), 4) for k, v in self.intent_targets.items()},
            "world_id": self.world_id,
            "tick_start": self.tick_start,
            "tick_deadline": self.tick_deadline,
            "min_ticks": self.min_ticks,
            "source": self.source,
            "priority": round(self.priority, 4),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubGoal:
        return cls(
            subgoal_id=str(d.get("subgoal_id", "")),
            var_id=str(d.get("var_id", "")),
            target_val=float(d.get("target_val", 0.62)),
            intent_targets={str(k): float(v) for k, v in (d.get("intent_targets") or {}).items()},
            world_id=str(d.get("world_id", "humanoid")),
            tick_start=int(d.get("tick_start", 0)),
            tick_deadline=int(d.get("tick_deadline", 0)),
            min_ticks=int(d.get("min_ticks", 120)),
            source=str(d.get("source", "curriculum")),
            priority=float(d.get("priority", 0.5)),
            status=str(d.get("status", "pending")),
        )

    @classmethod
    def from_goal_candidate(cls, g: GoalCandidate) -> SubGoal:
        targets = {}
        if g.var_id.startswith("intent_"):
            targets[g.var_id] = float(g.target_val)
        return cls(
            subgoal_id=f"gen_{g.var_id}_{g.tick_proposed}",
            var_id=g.var_id,
            target_val=float(g.target_val),
            intent_targets=targets,
            world_id=g.world_id,
            tick_start=g.tick_proposed,
            source="generated",
            priority=float(0.45 + 0.35 * g.meta_success_pred),
        )


@dataclass
class IntentionContext:
    """Executive snapshot consumed by System2 / WM planner / snapshot."""

    macro_hint: str = "IDLE"
    graph_patch: dict[str, float] = field(default_factory=dict)
    intent_residuals: dict[str, float] = field(default_factory=dict)
    horizon_ticks: int = 120
    primary: SubGoal | None = None
    stack_depth: int = 0
    narrative: str = ""
    expected_state: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_hint": self.macro_hint,
            "graph_patch": {k: round(float(v), 4) for k, v in self.graph_patch.items()},
            "intent_residuals": {
                k: round(float(v), 4) for k, v in self.intent_residuals.items()
            },
            "horizon_ticks": self.horizon_ticks,
            "primary": self.primary.to_dict() if self.primary else None,
            "stack_depth": self.stack_depth,
            "narrative": self.narrative,
            "expected_state": {
                k: round(float(v), 4) for k, v in self.expected_state.items()
            },
        }


class IntentionCortex:
    def __init__(self) -> None:
        self._goal_generator = GoalGenerator()
        self._curriculum_graph = CurriculumGraph()
        self._stack: list[SubGoal] = []
        self._completed_stack: list[SubGoal] = []
        self._last_replan_tick: int = -10**9
        self._last_context: IntentionContext = IntentionContext()
        self._narrative_lines: list[str] = []
        self._seeded_curriculum = False

    @property
    def goal_generator(self) -> GoalGenerator:
        return self._goal_generator

    @property
    def curriculum_graph(self) -> CurriculumGraph:
        return self._curriculum_graph

    @property
    def context(self) -> IntentionContext:
        return self._last_context

    def ensure_curriculum_seed(self, physical_curriculum: Any | None) -> int:
        if self._seeded_curriculum or not curriculum_graph_enabled():
            return 0
        n = self._curriculum_graph.seed_from_physical_curriculum(physical_curriculum)
        if n > 0:
            self._curriculum_graph.freeze_human_curriculum()
        self._seeded_curriculum = True
        return n

    def tick_pre_control(
        self,
        sim: Any,
        *,
        tick: int,
        obs: dict[str, float],
        fallen: bool,
    ) -> IntentionContext:
        """Update stack, project intention into graph — call before System2."""
        if not intention_cortex_enabled():
            return self._last_context

        self.ensure_curriculum_seed(getattr(sim, "_physical_curriculum", None))
        agent = sim.agent
        world_id = str(getattr(sim, "current_world", "humanoid"))

        self._goal_generator.on_tick(tick)
        self._maybe_propose_goal(agent, tick, world_id)

        if curriculum_graph_enabled():
            self._curriculum_graph.activate_next(tick, world_id=world_id)

        self._maybe_rebuild_stack(sim, tick, world_id)
        self._advance_stack_if_done(agent, obs, tick)

        primary = self._stack[0] if self._stack else None
        ctx = self._build_context(primary, obs, fallen, tick)
        self._project_to_graph(agent, ctx, primary)
        self._project_hierarchical(sim, primary, tick)
        self._last_context = ctx
        return ctx

    def tick_post_step(self, sim: Any, snap: dict[str, Any]) -> None:
        """Meta observe, curriculum completion, episodic narrative — after agent step."""
        if not intention_cortex_enabled():
            return

        tick = int(getattr(sim, "tick", 0))
        self._observe_meta(sim, snap, tick)
        self._maybe_complete_curriculum(sim, snap, tick)
        self._maybe_world_transfer(sim, tick)
        self._update_narrative(snap, tick)

    def _maybe_propose_goal(self, agent: Any, tick: int, world_id: str) -> None:
        if not goal_gen_enabled():
            return
        every = _ei("RKK_GOAL_PROPOSE_EVERY", 200)
        if tick % every != 0:
            return
        role_map: dict[str, str] = {}
        try:
            role_map = agent.graph.role_type_map()
        except Exception:
            pass
        w_meta = getattr(agent, "_w_meta", None)
        cand = self._goal_generator.propose(
            agent.graph, w_meta, role_map=role_map, tick=tick, world_id=world_id
        )
        if cand is not None and curriculum_graph_enabled():
            self._curriculum_graph.add_generated_node(cand, tick=tick)
            sg = SubGoal.from_goal_candidate(cand)
            sg.tick_deadline = tick + _ei("RKK_INTENTION_SUBGOAL_TICKS", 400)
            self._insert_subgoal(sg)

    def _maybe_rebuild_stack(self, sim: Any, tick: int, world_id: str) -> None:
        every = _ei("RKK_INTENTION_REPLAN_EVERY", 120)
        if tick - self._last_replan_tick < every and self._stack:
            return
        self._last_replan_tick = tick
        max_depth = _ei("RKK_INTENTION_STACK_DEPTH", 12)
        new_stack: list[SubGoal] = []

        if curriculum_graph_enabled():
            cg = self._curriculum_graph
            active = cg.get_active_intent_targets()
            if cg._active_id:
                node = cg._nodes.get(cg._active_id)
                if node and node.status == "active":
                    sg = SubGoal(
                        subgoal_id=node.node_id,
                        var_id=node.var_id,
                        target_val=float(
                            node.intent_targets.get(node.var_id, 0.62)
                        ),
                        intent_targets=dict(node.intent_targets),
                        world_id=node.world_id,
                        tick_start=node.tick_started,
                        min_ticks=node.min_ticks,
                        source="curriculum_active",
                        priority=0.9,
                    )
                    sg.tick_deadline = tick + max(node.min_ticks, 200)
                    new_stack.append(sg)

            pending_cursor = tick
            for nxt in cg.pending_chain(world_id, max_depth - len(new_stack)):
                tv = float(nxt.intent_targets.get(nxt.var_id, 0.62))
                sg = SubGoal(
                    subgoal_id=nxt.node_id,
                    var_id=nxt.var_id,
                    target_val=tv,
                    intent_targets=dict(nxt.intent_targets),
                    world_id=nxt.world_id,
                    min_ticks=nxt.min_ticks,
                    source="curriculum_pending",
                    priority=0.55,
                )
                pending_cursor += max(nxt.min_ticks, 120)
                sg.tick_deadline = pending_cursor
                new_stack.append(sg)

        for g in list(self._goal_generator._active):
            sg = SubGoal.from_goal_candidate(g)
            sg.tick_deadline = g.tick_proposed + _ei("RKK_INTENTION_SUBGOAL_TICKS", 400)
            if not any(s.subgoal_id == sg.subgoal_id for s in new_stack):
                new_stack.append(sg)

        new_stack.sort(key=lambda s: (-s.priority, s.tick_deadline))
        if self._stack and self._stack[0].status == "active":
            head = self._stack[0]
            if not any(s.subgoal_id == head.subgoal_id for s in new_stack):
                new_stack.insert(0, head)
        self._stack = new_stack[:max_depth]
        if self._stack:
            self._stack[0].status = "active"

    def _insert_subgoal(self, sg: SubGoal) -> None:
        if any(s.subgoal_id == sg.subgoal_id for s in self._stack):
            return
        self._stack.append(sg)
        self._stack.sort(key=lambda s: (-s.priority, s.tick_deadline))
        cap = _ei("RKK_INTENTION_STACK_DEPTH", 12)
        self._stack = self._stack[:cap]

    def _subgoal_reached(
        self, sg: SubGoal, agent: Any, obs: dict[str, float]
    ) -> bool:
        nodes = agent.graph.nodes
        var = sg.var_id
        if var in nodes:
            v = float(nodes[var])
            if abs(v - sg.target_val) < 0.12:
                return True
        if var in obs:
            if abs(float(obs[var]) - sg.target_val) < 0.12:
                return True
        if var == "target_dist" and "target_dist" in obs:
            return float(obs["target_dist"]) <= sg.target_val + 0.04
        if var == "posture_stability":
            ps = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
            return ps >= sg.target_val - 0.06
        return False

    def _advance_stack_if_done(
        self, agent: Any, obs: dict[str, float], tick: int
    ) -> None:
        if not self._stack:
            return
        head = self._stack[0]
        min_age = max(20, head.min_ticks // 4)
        age_ok = tick - head.tick_start >= min_age if head.tick_start > 0 else tick > min_age
        bench_after = _ei("RKK_GOAL_BENCH_COMPLETE_AFTER", 80)
        time_ok = (
            head.source == "generated"
            and head.tick_start > 0
            and tick - head.tick_start >= bench_after
        )
        curriculum_time_ok = (
            head.source.startswith("curriculum")
            and head.tick_start > 0
            and tick - head.tick_start >= max(head.min_ticks, bench_after)
        )

        if not (
            self._subgoal_reached(head, agent, obs) or time_ok or curriculum_time_ok
        ):
            if tick > head.tick_deadline:
                head.status = "expired"
                self._completed_stack.append(head)
                self._stack.pop(0)
            return

        head.status = "completed"
        self._completed_stack.append(head)
        self._stack.pop(0)

        if head.source.startswith("curriculum") and curriculum_graph_enabled():
            self._curriculum_graph.mark_completed(
                head.subgoal_id,
                success_rate=0.72,
                tick=tick,
            )

        if head.source == "generated" and goal_gen_enabled():
            self._goal_generator.complete_goal(
                head.var_id, success_rate=0.72, tick=tick
            )

        if self._stack:
            self._stack[0].status = "active"
            self._stack[0].tick_start = tick

    def _build_context(
        self,
        primary: SubGoal | None,
        obs: dict[str, float],
        fallen: bool,
        tick: int,
    ) -> IntentionContext:
        if primary is None:
            return IntentionContext(
                macro_hint="IDLE" if not fallen else "RECOVER_POSTURE",
                horizon_ticks=_ei("RKK_INTENTION_MACRO_TICKS", 180),
            )

        macro = self._macro_for_subgoal(primary, obs, fallen)
        patch, residuals, expected = self._patches_for_subgoal(primary, obs)
        horizon = max(
            _ei("RKK_INTENTION_MACRO_TICKS", 180),
            primary.tick_deadline - tick if primary.tick_deadline > tick else 60,
        )
        narrative = self._format_narrative(primary, macro)
        return IntentionContext(
            macro_hint=macro,
            graph_patch=patch,
            intent_residuals=residuals,
            horizon_ticks=horizon,
            primary=primary,
            stack_depth=len(self._stack),
            narrative=narrative,
            expected_state=expected,
        )

    def _macro_for_subgoal(
        self, sg: SubGoal, obs: dict[str, float], fallen: bool
    ) -> str:
        if fallen:
            return "RECOVER_POSTURE"
        ps = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        cz = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        if cz < 0.46 or ps < 0.38:
            return "RECOVER_POSTURE"
        intents = sg.intent_targets or {}
        keys = set(intents) | {sg.var_id}
        if sg.var_id == "target_dist" or "target_dist" in keys:
            return "LOCOMOTE_DELIVERY"
        if any("stride" in k or "torso_forward" in k or "lean" in k for k in keys):
            return "LOCOMOTE_DELIVERY"
        if any("reach" in k or "wave" in k or "arm" in k for k in keys):
            return "EXPLORE"
        if sg.var_id in ("posture_stability", "com_z", "self_posture_target"):
            return "IDLE" if ps > 0.55 else "RECOVER_POSTURE"
        return "LOCOMOTE_DELIVERY"

    def _patches_for_subgoal(
        self, sg: SubGoal, obs: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        patch: dict[str, float] = {
            "self_goal_active": 0.88,
            "self_attention": float(np.clip(0.55 + 0.35 * sg.priority, 0.4, 0.95)),
        }
        residuals: dict[str, float] = {}
        expected: dict[str, float] = {}

        if sg.var_id == "target_dist" or "target_dist" in sg.intent_targets:
            td = float(sg.target_val if sg.var_id == "target_dist" else sg.intent_targets.get("target_dist", 0.38))
            patch["self_goal_target_dist"] = float(np.clip(td, 0.08, 0.92))
            expected["target_dist"] = patch["self_goal_target_dist"]

        if sg.var_id == "posture_stability":
            patch["self_posture_target"] = float(np.clip(sg.target_val, 0.2, 0.95))
            expected["posture_stability"] = patch["self_posture_target"]

        if sg.var_id == "com_z" or "com_z" in sg.var_id:
            patch["self_com_z_target"] = float(np.clip(sg.target_val, 0.25, 0.85))
            expected["com_z"] = patch["self_com_z_target"]

        for k, v in sg.intent_targets.items():
            fv = float(np.clip(v, 0.06, 0.94))
            if k.startswith("intent_"):
                residuals[k] = residuals.get(k, 0.0) + (fv - 0.5) * 0.35
                if k.replace("intent_", "") in ("stride", "torso_forward"):
                    expected[k.replace("intent_", "")] = fv
            elif k in ("posture_stability", "target_dist", "com_z"):
                expected[k] = fv

        if sg.var_id.startswith("intent_"):
            residuals[sg.var_id] = residuals.get(sg.var_id, 0.0) + (sg.target_val - 0.5) * 0.4

        return patch, residuals, expected

    def _project_to_graph(
        self, agent: Any, ctx: IntentionContext, primary: SubGoal | None
    ) -> None:
        nodes = agent.graph.nodes
        for k, v in ctx.graph_patch.items():
            if k in nodes:
                nodes[k] = float(np.clip(v, 0.02, 0.98))
        base = getattr(agent.env, "base_env", None) or agent.env
        fn = getattr(base, "apply_self_state_patch", None)
        if callable(fn):
            patch = {k: float(nodes[k]) for k in ctx.graph_patch if k in nodes}
            if patch:
                try:
                    fn(patch)
                except Exception:
                    pass

    def _project_hierarchical(self, sim: Any, primary: SubGoal | None, tick: int) -> None:
        if primary is None:
            return
        hg = getattr(sim, "_hierarchical_graph", None)
        if hg is None:
            return
        horizon = _ei("RKK_INTENTION_L3_HORIZON", 48)
        tv = primary.target_val
        if primary.var_id == "target_dist":
            hg.set_l3_goal("target_dist", tv, horizon=horizon)
        elif primary.var_id in hg.L2.nodes:
            hg.set_l3_goal(primary.var_id, tv, horizon=horizon)

    def _observe_meta(self, sim: Any, snap: dict[str, Any], tick: int) -> None:
        w_meta = getattr(sim.agent, "_w_meta", None)
        if w_meta is None or not meta_causal_enabled():
            return
        success = snap.get("behavioral_score")
        if success is None:
            success = 1.0 - float(snap.get("prediction_error", 0.5))
        cur_step = int(snap.get("curriculum_step", 0))
        obs = build_meta_observation(
            sim.agent,
            tick=tick,
            curriculum_step=cur_step,
            success_rate=float(success) if success is not None else None,
        )
        w_meta.observe(obs, tick=tick)

    def _maybe_complete_curriculum(self, sim: Any, snap: dict[str, Any], tick: int) -> None:
        if not curriculum_graph_enabled():
            return
        active_id = self._curriculum_graph._active_id
        if not active_id:
            return
        node = self._curriculum_graph._nodes.get(active_id)
        if node is None or node.status != "active":
            return
        if tick - node.tick_started < node.min_ticks:
            return
        sr = snap.get("behavioral_score")
        if sr is None:
            sr = 1.0 - float(snap.get("prediction_error", 0.45))
        if float(sr) >= _ef("RKK_CURRICULUM_COMPLETE_SR", 0.55):
            self._curriculum_graph.mark_completed(active_id, success_rate=float(sr), tick=tick)

    def _maybe_world_transfer(self, sim: Any, tick: int) -> None:
        if not goal_gen_enabled():
            return
        every = _ei("RKK_GOAL_WORLD_SWITCH_EVERY", 600)
        if tick % every != 0:
            return
        from engine.core.world import is_humanoid_topology

        sw = getattr(sim, "switcher", None)
        if sw is None or not is_humanoid_topology(sim.current_world):
            return
        from_world = str(sim.current_world)
        target = "humanoid_variant" if from_world == "humanoid" else "humanoid"
        role_map: dict[str, str] = {}
        try:
            role_map = sim.agent.graph.role_type_map()
        except Exception:
            pass
        if curriculum_graph_enabled():
            self._curriculum_graph.transfer_goals_to_world(
                from_world, target, role_map=role_map
            )
        sw.switch(target)
        sim.current_world = target

    def _format_narrative(self, sg: SubGoal, macro: str) -> str:
        intents = ", ".join(f"{k}→{v:.2f}" for k, v in list(sg.intent_targets.items())[:4])
        return (
            f"[{macro}] {sg.source}:{sg.var_id}→{sg.target_val:.2f}"
            f" stack={len(self._stack)} {intents}"
        ).strip()

    def _update_narrative(self, snap: dict[str, Any], tick: int) -> None:
        if self._last_context.narrative:
            self._narrative_lines.append(f"t{tick}: {self._last_context.narrative}")
        cap = _ei("RKK_INTENTION_NARRATIVE_CAP", 24)
        if len(self._narrative_lines) > cap:
            self._narrative_lines = self._narrative_lines[-cap:]
        snap["intention_cortex"] = self.snapshot(tick)

    def snapshot(self, tick: int = 0) -> dict[str, Any]:
        return {
            "enabled": intention_cortex_enabled(),
            "tick": tick,
            "context": self._last_context.to_dict(),
            "stack": [s.to_dict() for s in self._stack[:8]],
            "completed_recent": [s.to_dict() for s in self._completed_stack[-6:]],
            "narrative": list(self._narrative_lines)[-6:],
            "goal_generator": self._goal_generator.snapshot(),
            "curriculum_graph": self._curriculum_graph.snapshot(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack": [s.to_dict() for s in self._stack],
            "completed_stack": [s.to_dict() for s in self._completed_stack[-48:]],
            "last_replan_tick": self._last_replan_tick,
            "narrative_lines": list(self._narrative_lines),
            "seeded_curriculum": self._seeded_curriculum,
            "goal_generator": self._goal_generator.to_dict(),
            "curriculum_graph": self._curriculum_graph.to_dict(),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if not data:
            return
        self._stack = [SubGoal.from_dict(x) for x in data.get("stack") or []]
        self._completed_stack = [
            SubGoal.from_dict(x) for x in data.get("completed_stack") or []
        ]
        self._last_replan_tick = int(data.get("last_replan_tick", -10**9))
        self._narrative_lines = list(data.get("narrative_lines") or [])
        self._seeded_curriculum = bool(data.get("seeded_curriculum", False))
        self._goal_generator.load_dict(data.get("goal_generator") or {})
        self._curriculum_graph.load_dict(data.get("curriculum_graph") or {})
