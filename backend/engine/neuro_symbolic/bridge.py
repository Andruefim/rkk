"""
Layer 3: Neuro-Symbolic Bridge — differentiable translator between latent vectors and symbols.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.neuro_symbolic.knowledge_graph import KnowledgeGraph, bootstrap_humanoid_ontology
from engine.neuro_symbolic.planner import (
    HUMANOID_ACTIONS,
    SymbolicAction,
    discover_actions_from_graph,
    macro_to_goal,
    plan_to_goal,
)
from engine.neuro_symbolic.predicates import (
    PATH_BLOCKED_FORWARD_MAX,
    PATH_BLOCKED_TURN_MIN,
    PathBlockedHysteresis,
    ProbabilisticState,
    ground_humanoid_state,
    path_forward_blocked,
    path_turn_recommended,
)


def neuro_symbolic_enabled() -> bool:
    return os.environ.get("RKK_NEURO_SYMBOLIC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class SymbolicPlanContext:
    macro: str = "IDLE"
    facts: dict[str, float] = field(default_factory=dict)
    plan_steps: list[str] = field(default_factory=list)
    motor_priors: dict[str, float] = field(default_factory=dict)
    graph_patch: dict[str, float] = field(default_factory=dict)
    precision_weights: dict[str, float] = field(default_factory=dict)
    attention_focus: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    narrative: str = ""
    safety_veto: bool = False
    safety_reasons: list[str] = field(default_factory=list)
    plan_invalidated: bool = False
    invalidation_reasons: list[str] = field(default_factory=list)


class NeuroSymbolicBridge:
    """
    Vector ↔ Symbol bidirectional bridge.
    Ascent: graph/obs → ProbabilisticState (fuzzy predicates).
    Descent: symbolic plan → motor/graph priors for Active Inference.
    """

    def __init__(self) -> None:
        self._kg = bootstrap_humanoid_ontology()
        self._last_state = ProbabilisticState()
        self._last_plan = SymbolicPlanContext()
        self._last_skeleton_rules: list[str] = []
        self._active_plan_steps: list[str] = []
        self._active_plan_macro: str = "IDLE"
        self._path_blocked_hyst = PathBlockedHysteresis()

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        return self._kg

    def perceive(
        self,
        obs: dict[str, float],
        graph_nodes: dict[str, float] | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> ProbabilisticState:
        ctx = dict(context or {})
        ctx["_path_blocked_hysteresis"] = self._path_blocked_hyst
        self._last_state = ground_humanoid_state(obs, graph_nodes, context=ctx)
        blocked = self._last_state.best("PathBlocked")
        self._kg.set_runtime_fact("PathBlocked", blocked)
        return self._last_state

    @staticmethod
    def _context_from_sim(sim: Any | None) -> dict[str, Any]:
        if sim is None:
            return {}
        ctx: dict[str, Any] = {}
        ctx["hai_pe_fwd_ema"] = float(getattr(sim, "_hai_pe_fwd_ema", 0.0) or 0.0)
        diag = getattr(sim, "_hai_last_diag", None) or {}
        if isinstance(diag, dict) and "actual_delta" in diag:
            ctx["actual_delta"] = float(diag["actual_delta"])
        beh = getattr(sim, "_behavioral_snap", None)
        if isinstance(beh, dict):
            if "com_x_vel_ema" in beh:
                ctx["com_x_vel_ema"] = float(beh["com_x_vel_ema"])
        pack = getattr(sim, "_skill_exec", None) or {}
        skill = pack.get("skill")
        if skill is not None:
            ctx["active_skill"] = str(getattr(skill, "name", skill))
        ic = getattr(sim, "_intention_state", None)
        if ic is not None:
            ctx["macro_hint"] = str(getattr(ic, "macro_hint", "") or "")
        ns_ctx = getattr(sim, "_ns_last_ctx", None) or {}
        steps = ns_ctx.get("plan_steps") or []
        if steps:
            ctx["ns_plan_head"] = str(steps[0])
        bridge = getattr(sim, "_ns_bridge", None)
        if bridge is not None and getattr(bridge, "_last_plan", None) is not None:
            lp = bridge._last_plan.plan_steps
            if lp:
                ctx["ns_plan_head"] = str(lp[0])
        ctx["visual_depth"] = NeuroSymbolicBridge._forward_visual_depth(sim)
        try:
            agent = sim.agent
            nodes = agent.graph.nodes
            ctx["contact_stress"] = float(
                nodes.get("intero_stress", nodes.get("phys_intero_stress", 0.0))
            )
            ms = getattr(getattr(agent, "env", None), "_motor_state", None) or {}
            if isinstance(ms, dict):
                drive = max(
                    float(ms.get("motor_drive_l", 0.5)),
                    float(ms.get("motor_drive_r", 0.5)),
                )
                ctx["contact_stress"] = max(float(ctx.get("contact_stress", 0.0)), drive)
        except Exception:
            pass
        return ctx

    @staticmethod
    def _forward_visual_depth(sim: Any) -> float:
        """Normalized forward clearance in [0,1]; lower = obstacle closer ahead."""
        vis = getattr(sim, "_visual_env", None)
        if vis is not None:
            slots = getattr(vis, "_last_slots", None)
            if slots is not None:
                arr = np.asarray(slots.detach().cpu().numpy()).reshape(-1)
                if arr.size:
                    peak = float(np.max(np.abs(arr - 0.5)) * 2.0)
                    return float(np.clip(1.0 - peak, 0.0, 1.0))
        vs = getattr(sim, "_last_vision_state", None) or {}
        slot_vals = vs.get("slot_values") or []
        if slot_vals:
            peak = float(max(abs(float(v) - 0.5) * 2.0 for v in slot_vals))
            return float(np.clip(1.0 - peak, 0.0, 1.0))
        return 1.0

    def _check_plan_invalidation(
        self,
        macro: str,
        path: list[SymbolicAction],
        st: ProbabilisticState,
    ) -> tuple[bool, list[str]]:
        forward_actions = {"StepForward", "ApproachTarget", "ApproachObject"}
        blocked = st.best("PathBlocked")
        reasons: list[str] = []
        if path_forward_blocked(blocked):
            prev = self._active_plan_steps
            if (
                str(macro).upper() in ("LOCOMOTE_DELIVERY", "EXPLORE")
                and prev
                and any(s in forward_actions for s in prev)
                and "Turn" not in prev
            ):
                reasons.append("PathBlocked:forward_plan_invalidated")
            new_names = [a.name for a in path]
            if new_names and new_names[0] in forward_actions:
                reasons.append("PathBlocked:StepForward_precondition_failed")
        return bool(reasons), reasons

    def plan_for_macro(
        self,
        macro: str,
        state: ProbabilisticState | None = None,
        *,
        graph_nodes: dict[str, float] | None = None,
    ) -> SymbolicPlanContext:
        st = state or self._last_state
        goal = macro_to_goal(macro)
        actions = discover_actions_from_graph(
            graph_nodes or {},
            st,
            kg=self._kg,
        )
        path = plan_to_goal(st, goal, actions=actions)
        ctx = SymbolicPlanContext(macro=str(macro).upper(), facts=st.to_dict())
        invalidated, reasons = self._check_plan_invalidation(macro, path, st)
        ctx.plan_invalidated = invalidated
        ctx.invalidation_reasons = reasons

        forward_actions = {"StepForward", "ApproachTarget", "ApproachObject"}
        blocked_lvl = st.best("PathBlocked")
        if path_turn_recommended(blocked_lvl) and path:
            if path[0].name in forward_actions:
                turn = next((a for a in HUMANOID_ACTIONS if a.name == "Turn"), None)
                if turn is not None:
                    path = [turn] + path
                    ctx.plan_invalidated = True
                    if "PathBlocked:replan_with_Turn" not in ctx.invalidation_reasons:
                        ctx.invalidation_reasons.append("PathBlocked:replan_with_Turn")
        elif path_forward_blocked(blocked_lvl) and not path:
            turn = next((a for a in HUMANOID_ACTIONS if a.name == "Turn"), None)
            if turn is not None:
                path = [turn]
                ctx.plan_invalidated = True
                ctx.invalidation_reasons.append("PathBlocked:replan_with_Turn")

        if path:
            ctx.plan_steps = [a.name for a in path]
            motor: dict[str, float] = {}
            for act in path:
                for k, v in act.motor_priors.items():
                    motor[k] = float(v)
            ctx.motor_priors = motor
        else:
            act = self._fallback_action(macro, st)
            if act is not None:
                ctx.plan_steps = [act.name]
                ctx.motor_priors = dict(act.motor_priors)
        ctx.graph_patch = self._motor_to_graph_patch(ctx.motor_priors)
        ctx.precision_weights = self._motor_to_precision_weights(
            ctx.motor_priors, ctx.plan_steps
        )
        ctx.attention_focus = self._plan_attention_focus(ctx.plan_steps, ctx.motor_priors)
        ctx.narrative = self._format_narrative(ctx)
        self._last_plan = ctx
        self._active_plan_steps = list(ctx.plan_steps)
        self._active_plan_macro = ctx.macro
        return ctx

    def _fallback_action(self, macro: str, st: ProbabilisticState) -> SymbolicAction | None:
        m = str(macro).upper()
        blocked_lvl = st.best("PathBlocked")
        if path_turn_recommended(blocked_lvl):
            for act in HUMANOID_ACTIONS:
                if act.name == "Turn":
                    return act
        for act in HUMANOID_ACTIONS:
            if m == "RECOVER_POSTURE" and act.name == "RecoverPosture":
                return act
            if m == "LOCOMOTE_DELIVERY" and act.name in ("StepForward", "ApproachTarget", "ApproachObject"):
                if path_forward_blocked(blocked_lvl):
                    continue
                if st.best("GoalActive") > 0.55 and act.name == "ApproachTarget":
                    return act
                if act.name == "StepForward":
                    return act
            if m == "IDLE" and act.name == "HoldStance":
                return act
        if path_forward_blocked(blocked_lvl):
            for act in HUMANOID_ACTIONS:
                if act.name == "Turn":
                    return act
        return None

    def _motor_to_graph_patch(self, motor: dict[str, float]) -> dict[str, float]:
        patch: dict[str, float] = {}
        blend = _ef("RKK_NS_PRIOR_BLEND", 0.35)
        for k, target in motor.items():
            if k.startswith("self_"):
                patch[k] = float(np.clip(target, 0.05, 0.95))
            elif k.startswith("intent_"):
                patch[k] = float(np.clip(0.5 + blend * (target - 0.5), 0.05, 0.95))
        return patch

    def _motor_to_precision_weights(
        self,
        motor: dict[str, float],
        plan_steps: list[str],
    ) -> dict[str, float]:
        """Downward symbolic → precision: plan-relevant nodes get higher PE weight."""
        weights: dict[str, float] = {}
        scale = _ef("RKK_NS_PRECISION_SCALE", 2.4)
        for k, target in motor.items():
            weights[k] = float(1.0 + scale * abs(float(target) - 0.5))
        step_targets = {
            "RecoverPosture": ("posture_stability", "com_z", "intent_stop_recover"),
            "StepForward": ("target_dist", "intent_stride", "intent_torso_forward"),
            "Turn": ("intent_look_at", "intent_gait_coupling", "intent_stride"),
            "ApproachTarget": ("self_goal_active", "target_dist", "intent_stride"),
            "HoldStance": ("posture_stability", "foot_contact_l", "foot_contact_r"),
        }
        for step in plan_steps:
            for nid in step_targets.get(step, ()):
                weights[nid] = max(weights.get(nid, 1.0), 1.0 + scale * 0.55)
        return weights

    def _plan_attention_focus(
        self,
        plan_steps: list[str],
        motor: dict[str, float],
    ) -> list[str]:
        focus: list[str] = []
        for step in plan_steps[:3]:
            focus.append(step)
        for k in sorted(motor.keys(), key=lambda x: -abs(motor[x] - 0.5))[:6]:
            if k not in focus:
                focus.append(k)
        return focus[:12]

    def priors_for_active_inference(
        self,
        macro: str,
        obs: dict[str, float],
        graph_nodes: dict[str, float] | None = None,
        *,
        sim: Any | None = None,
    ) -> SymbolicPlanContext:
        """Full slow-loop: perceive → plan → motor priors."""
        ctx = self._context_from_sim(sim)
        ctx["macro_hint"] = str(macro).upper()
        st = self.perceive(obs, graph_nodes, context=ctx)
        return self.plan_for_macro(macro, st, graph_nodes=graph_nodes)

    def inject_skeleton_rules(self, rules: list[str]) -> None:
        self._last_skeleton_rules = list(rules[:32])
        for rule in rules[:16]:
            parts = rule.replace(" ", "").split("->")
            if len(parts) == 2:
                self._kg.add(parts[0], "causes", parts[1], confidence=0.7)

    def apply_priors_to_graph(
        self,
        graph_nodes: dict[str, float],
        ctx: SymbolicPlanContext,
    ) -> None:
        blend = _ef("RKK_NS_GRAPH_BLEND", 0.38)
        for k, target in ctx.graph_patch.items():
            if k not in graph_nodes:
                continue
            cur = float(graph_nodes[k])
            graph_nodes[k] = float(
                np.clip(cur + blend * (target - cur), 0.05, 0.95)
            )

    def apply_symbolic_precision_to_graph(
        self,
        graph: Any,
        ctx: SymbolicPlanContext,
    ) -> None:
        """Downward channel: symbolic plan → GNN precision / attention reweighting."""
        if not ctx.precision_weights:
            return
        fn = getattr(graph, "apply_symbolic_precision", None)
        if callable(fn):
            fn(ctx.precision_weights)
        gate_fn = getattr(graph, "apply_attention_focus", None)
        if callable(gate_fn) and ctx.attention_focus:
            gate_fn(ctx.attention_focus, ctx.precision_weights)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": neuro_symbolic_enabled(),
            "facts": self._last_state.to_dict(),
            "plan": {
                "macro": self._last_plan.macro,
                "steps": self._last_plan.plan_steps,
                "motor_priors": {
                    k: round(v, 4) for k, v in self._last_plan.motor_priors.items()
                },
                "narrative": self._last_plan.narrative,
                "safety_veto": self._last_plan.safety_veto,
                "plan_invalidated": self._last_plan.plan_invalidated,
                "invalidation_reasons": list(self._last_plan.invalidation_reasons),
            },
            "skeleton_rules": self._last_skeleton_rules[:8],
            "knowledge_graph": self._kg.snapshot(),
        }

    @staticmethod
    def _format_narrative(ctx: SymbolicPlanContext) -> str:
        steps = "→".join(ctx.plan_steps) if ctx.plan_steps else "—"
        mp = ",".join(f"{k.split('_')[-1]}={v:.2f}" for k, v in list(ctx.motor_priors.items())[:4])
        inv = ""
        if ctx.plan_invalidated:
            inv = f" INVALID[{','.join(ctx.invalidation_reasons[:2])}]"
        return f"NS[{ctx.macro}] {steps} priors({mp}){inv}"
