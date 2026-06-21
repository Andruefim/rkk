"""Simulation mixin: Neuro-Symbolic slow loop (Layer 3 + 4)."""
from __future__ import annotations

import os
from typing import Any

from engine.neuro_symbolic.bridge import NeuroSymbolicBridge, neuro_symbolic_enabled
from engine.neuro_symbolic.engine import SymbolicCognitiveEngine, symbolic_engine_enabled


def _ns_slow_every() -> int:
    try:
        return max(1, int(os.environ.get("RKK_NS_SLOW_EVERY", "10")))
    except ValueError:
        return 10


class SimulationNeuroSymbolicMixin:
    def _ensure_neuro_symbolic(self) -> None:
        if getattr(self, "_ns_ready", False):
            return
        self._ns_bridge = NeuroSymbolicBridge()
        self._ns_engine = SymbolicCognitiveEngine()
        self._ns_last_ctx: dict[str, Any] = {}
        self._ns_ready = True

    def _ns_snapshot_meta(self) -> dict[str, Any]:
        if not neuro_symbolic_enabled():
            return {"enabled": False}
        self._ensure_neuro_symbolic()
        out: dict[str, Any] = {
            "enabled": True,
            "symbolic_engine_enabled": symbolic_engine_enabled(),
            "slow_every": _ns_slow_every(),
        }
        if self._ns_bridge is not None:
            out["bridge"] = self._ns_bridge.snapshot()
        if self._ns_engine is not None:
            out["engine"] = self._ns_engine.snapshot()
        if self._ns_last_ctx:
            out["last_context"] = self._ns_last_ctx
        return out

    def _tick_neuro_symbolic_slow(self, *, fallen: bool) -> None:
        if not neuro_symbolic_enabled():
            return
        tick = int(getattr(self, "tick", 0))
        if tick % _ns_slow_every() != 0:
            return
        self._ensure_neuro_symbolic()
        agent = self.agent
        obs = dict(self._graph_vec_cached())

        macro = "IDLE"
        ctx_int = getattr(self, "_intention_state", None)
        if ctx_int is not None:
            macro = str(getattr(ctx_int, "macro_hint", "IDLE") or "IDLE")
        elif isinstance(getattr(self, "_system2_last", None), dict):
            macro = str(self._system2_last.get("macro") or "IDLE")

        if fallen:
            macro = "RECOVER_POSTURE"

        # Refresh skeleton for symbolic grounding loop
        try:
            from engine.genome.meta_invariants import extract_skeleton_from_graph
            from engine.symbolic_grounding import symbolic_grounding_enabled

            sk = extract_skeleton_from_graph(agent.graph)
            self._last_skeleton = sk
            if symbolic_grounding_enabled():
                sg = getattr(self, "_symbolic_grounding", None)
                if sg is not None:
                    rules = sg.skeleton_to_rules(sk)
                    self._ns_bridge.inject_skeleton_rules(rules)
        except Exception:
            pass

        revision = None
        plan_ctx = self._ns_bridge.priors_for_active_inference(
            macro,
            obs,
            dict(agent.graph.nodes),
            sim=self,
        )
        revision = self._ns_engine.suggest_goal_revision(
            self._ns_bridge._last_state, macro
        )
        if revision and not fallen:
            macro = str(revision.get("macro", macro))
            plan_ctx = self._ns_bridge.priors_for_active_inference(
                macro,
                obs,
                dict(agent.graph.nodes),
                sim=self,
            )
        prox_veto = self._ns_engine.check_human_proximity(obs)
        fuzzy_veto = self._ns_engine.check_fuzzy_safety(self._ns_bridge._last_state)
        plan_ctx.safety_veto = not prox_veto.allowed or not fuzzy_veto.allowed
        plan_ctx.safety_reasons = list(prox_veto.violations) + list(fuzzy_veto.violations)

        # Symbolic hypotheses + KG online learning
        hyps = self._ns_engine.generate_hypotheses(self._ns_bridge._last_state)
        plan_ctx.hypotheses = [
            {
                "predicate": h.predicate,
                "confidence": round(h.confidence, 4),
                "macro": h.suggested_macro,
                "action": h.suggested_action,
            }
            for h in hyps
        ]
        kg = self._ns_bridge.knowledge_graph
        st_facts = self._ns_bridge._last_state.to_dict()
        for pred, conf in st_facts.items():
            expected = kg.get_runtime_fact(pred, conf)
            if kg.learn_from_surprise(pred, conf, expected, tick=tick):
                plan_ctx.invalidation_reasons.append(f"kg_surprise:{pred}")
        if plan_ctx.plan_invalidated and plan_ctx.invalidation_reasons:
            for reason in plan_ctx.invalidation_reasons[:2]:
                if ":" in reason:
                    kg.learn_from_plan_failure(macro, reason.split(":")[-1], tick=tick)
        kg.forget_stale(tick)

        if not plan_ctx.safety_veto and plan_ctx.motor_priors:
            self._ns_bridge.apply_priors_to_graph(agent.graph.nodes, plan_ctx)
            self._ns_bridge.apply_symbolic_precision_to_graph(agent.graph, plan_ctx)
            base = self._unwrap_base_env(agent.env)
            fn = getattr(base, "apply_motor_intent_residuals", None)
            if callable(fn):
                gain = float(os.environ.get("RKK_NS_MOTOR_GAIN", "0.32"))
                residuals = {
                    k: (v - float(getattr(base, "_motor_state", {}).get(k, 0.5))) * gain
                    for k, v in plan_ctx.motor_priors.items()
                    if k.startswith("intent_")
                }
                residuals = {k: v for k, v in residuals.items() if abs(v) >= 0.004}
                if residuals:
                    try:
                        fn(residuals)
                    except Exception:
                        pass

        self._ns_last_ctx = {
            "tick": tick,
            "macro": macro,
            "plan_steps": plan_ctx.plan_steps,
            "motor_priors": plan_ctx.motor_priors,
            "narrative": plan_ctx.narrative,
            "safety_veto": plan_ctx.safety_veto,
            "plan_invalidated": plan_ctx.plan_invalidated,
            "path_blocked": float(
                self._ns_bridge._last_state.best("PathBlocked")
            ),
        }

        ic = getattr(self, "_intention_state", None)
        if ic is not None and plan_ctx.narrative:
            base_n = str(getattr(ic, "narrative", "") or "")
            if plan_ctx.narrative not in base_n:
                ic.narrative = f"{base_n} | {plan_ctx.narrative}".strip(" |")
            if plan_ctx.graph_patch:
                gp = dict(getattr(ic, "graph_patch", None) or {})
                gp.update(plan_ctx.graph_patch)
                ic.graph_patch = gp
            for k, v in plan_ctx.motor_priors.items():
                if k.startswith("intent_"):
                    ir = dict(getattr(ic, "intent_residuals", None) or {})
                    ir[k] = ir.get(k, 0.0) + (float(v) - 0.5) * 0.25
                    ic.intent_residuals = ir

    def get_symbolic_engine(self) -> SymbolicCognitiveEngine | None:
        if not neuro_symbolic_enabled():
            return None
        self._ensure_neuro_symbolic()
        return self._ns_engine
