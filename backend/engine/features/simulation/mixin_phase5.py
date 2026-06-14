"""Simulation mixin: Phase 5 — Intention Cortex (meta, goals, curriculum DAG)."""
from __future__ import annotations

from typing import Any

from engine.eval_mode import meta_pe_rolling_window
from engine.intention_cortex import IntentionCortex, intention_cortex_enabled
from engine.meta_causal import WMetaEnsemble, meta_causal_enabled
from engine.meta_circuit_breaker import meta_cb_enabled
from engine.goal_generator import goal_gen_enabled
from engine.curriculum_graph import curriculum_graph_enabled


class SimulationPhase5Mixin:
    def _ensure_intention_cortex(self) -> IntentionCortex:
        ic = getattr(self, "_intention_cortex", None)
        if ic is not None:
            return ic
        ic = IntentionCortex()
        self._intention_cortex = ic
        self._goal_generator = ic.goal_generator
        self._curriculum_graph = ic.curriculum_graph
        if meta_causal_enabled() and getattr(self, "_w_meta", None) is None:
            self._w_meta = WMetaEnsemble(self.device)
            self.agent._w_meta = self._w_meta
        if meta_cb_enabled() and getattr(self, "_meta_cb", None) is None:
            from engine.meta_circuit_breaker import MetaCircuitBreaker

            self._meta_cb = MetaCircuitBreaker()
        ic.ensure_curriculum_seed(getattr(self, "_physical_curriculum", None))
        try:
            from engine.deliberation_worker import DeliberationService, deliberation_enabled

            if deliberation_enabled() and getattr(self, "_deliberation", None) is None:
                self._deliberation = DeliberationService(self)
                self._deliberation.ensure_started()
        except Exception:
            pass
        self._phase5_ready = True
        return ic

    def _ensure_phase5(self) -> None:
        if intention_cortex_enabled():
            self._ensure_intention_cortex()
            return
        if getattr(self, "_phase5_ready", False):
            return
        self._ensure_intention_cortex()

    def _tick_intention_pre_system2(self, *, fallen: bool) -> None:
        """Project long-horizon intention before System2 macro selection."""
        if not intention_cortex_enabled():
            return
        ic = self._ensure_intention_cortex()
        obs = dict(self._graph_vec_cached())
        ctx = ic.tick_pre_control(self, tick=int(self.tick), obs=obs, fallen=fallen)
        self._intention_state = ctx

    def _phase5_snapshot_meta(self) -> dict[str, Any]:
        self._ensure_phase5()
        ic = getattr(self, "_intention_cortex", None)
        out: dict[str, Any] = {
            "intention_cortex_enabled": intention_cortex_enabled(),
            "meta_causal_enabled": meta_causal_enabled(),
            "goal_gen_enabled": goal_gen_enabled(),
            "curriculum_graph_enabled": curriculum_graph_enabled(),
        }
        if ic is not None:
            out.update(ic.snapshot(int(getattr(self, "tick", 0))))
        cb = getattr(self, "_meta_cb", None)
        if cb is not None and meta_cb_enabled():
            out["meta_circuit_breaker"] = cb.snapshot(int(getattr(self, "tick", 0)))
            out["wmeta_active"] = cb.wmeta_active
        if getattr(self, "_w_meta", None) is not None:
            out["w_meta"] = self._w_meta.snapshot()
            out["meta_prediction_error"] = self._w_meta.meta_prediction_error_rolling(
                meta_pe_rolling_window()
            )
        ctx = getattr(self, "_intention_state", None)
        if ctx is not None:
            out["intention_context"] = ctx.to_dict()
        delib = getattr(self, "_deliberation", None)
        if delib is not None:
            latest = delib.latest()
            if latest is not None:
                out["deliberation"] = latest.to_dict()
        return out

    def _tick_phase5(self, snap: dict[str, Any]) -> None:
        if not (
            intention_cortex_enabled()
            or meta_causal_enabled()
            or goal_gen_enabled()
            or curriculum_graph_enabled()
        ):
            return
        ic = self._ensure_intention_cortex()
        ic.tick_post_step(self, snap)
        if self._w_meta is not None and meta_cb_enabled():
            cb = getattr(self, "_meta_cb", None)
            tick = int(self.tick)
            if cb is not None:
                meta_pe = self._w_meta.meta_prediction_error_rolling(
                    meta_pe_rolling_window()
                )
                meta_age = tick - int(getattr(self._w_meta, "_last_update_tick", tick))
                prev = cb.state
                cb.observe(meta_pe, meta_age, tick)
                if cb.state == cb.HALF_OPEN and prev == cb.OPEN:
                    cb.reset_w_meta_if_needed(self._w_meta)
                snap["wmeta_active"] = cb.wmeta_active
                snap["meta_circuit_breaker"] = cb.snapshot(tick)
                snap["meta_recovery_ticks"] = cb.recovery_ticks(tick)
            snap["meta_prediction_error"] = self._w_meta.meta_prediction_error_rolling(
                meta_pe_rolling_window()
            )
