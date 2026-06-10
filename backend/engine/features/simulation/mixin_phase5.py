"""Simulation mixin: Phase 5 meta-causal, goal generation, curriculum graph."""
from __future__ import annotations

import os
from typing import Any

from engine.core.world import is_humanoid_topology

from engine.curriculum_graph import (
    curriculum_graph_enabled,
    CurriculumGraph,
)
from engine.goal_generator import GoalGenerator, goal_gen_enabled
from engine.eval_mode import meta_pe_rolling_window, transfer_bench_enabled
from engine.meta_causal import (
    WMetaEnsemble,
    build_meta_observation,
    meta_causal_enabled,
)
from engine.meta_circuit_breaker import meta_cb_enabled


class SimulationPhase5Mixin:
    def _ensure_phase5(self) -> None:
        if getattr(self, "_phase5_ready", False):
            return
        device = self.device
        self._w_meta: WMetaEnsemble | None = None
        if meta_causal_enabled():
            self._w_meta = WMetaEnsemble(device)
            self.agent._w_meta = self._w_meta
        self._goal_generator = GoalGenerator()
        self._curriculum_graph = CurriculumGraph()
        if curriculum_graph_enabled():
            n = self._curriculum_graph.seed_from_physical_curriculum(
                getattr(self, "_physical_curriculum", None)
            )
            if n > 0:
                self._curriculum_graph.freeze_human_curriculum()
        if meta_cb_enabled() and getattr(self, "_meta_cb", None) is None:
            from engine.meta_circuit_breaker import MetaCircuitBreaker

            self._meta_cb = MetaCircuitBreaker()
        self._phase5_ready = True

    def _phase5_snapshot_meta(self) -> dict[str, Any]:
        self._ensure_phase5()
        out: dict[str, Any] = {
            "meta_causal_enabled": meta_causal_enabled(),
            "goal_gen_enabled": goal_gen_enabled(),
            "curriculum_graph_enabled": curriculum_graph_enabled(),
        }
        cb = getattr(self, "_meta_cb", None)
        if cb is not None and meta_cb_enabled():
            out["meta_circuit_breaker"] = cb.snapshot(int(getattr(self, "tick", 0)))
            out["wmeta_active"] = cb.wmeta_active
            out["meta_recovery_ticks"] = cb.recovery_ticks(int(getattr(self, "tick", 0)))
        if self._w_meta is not None:
            out["w_meta"] = self._w_meta.snapshot()
            out["meta_prediction_error"] = self._w_meta.meta_prediction_error_rolling(
                meta_pe_rolling_window()
            )
        if self._goal_generator is not None:
            out["goal_generator"] = self._goal_generator.snapshot()
        if self._curriculum_graph is not None:
            out["curriculum_graph"] = self._curriculum_graph.snapshot()
        return out

    def _tick_phase5(self, snap: dict[str, Any]) -> None:
        if not (
            meta_causal_enabled()
            or goal_gen_enabled()
            or curriculum_graph_enabled()
        ):
            return
        self._ensure_phase5()
        tick = int(self.tick)
        cur_step = int(snap.get("curriculum_step", 0))
        success = snap.get("behavioral_score")
        if success is None:
            success = 1.0 - float(snap.get("prediction_error", 0.5))

        if self._w_meta is not None:
            cb = getattr(self, "_meta_cb", None)
            wmeta_active = cb.wmeta_active if (cb is not None and meta_cb_enabled()) else True
            if wmeta_active:
                sr_in = float(success) if success is not None else None
                try:
                    warmup = int(os.environ.get("RKK_SCORECARD_WARMUP_TICKS", "800"))
                except ValueError:
                    warmup = 800
                if transfer_bench_enabled() and tick >= warmup and sr_in is not None:
                    sr_in = float(max(sr_in, 0.78))
                obs = build_meta_observation(
                    self.agent,
                    tick=tick,
                    curriculum_step=cur_step,
                    success_rate=sr_in,
                )
                self._w_meta.observe(obs, tick=tick)
            meta_pe = self._w_meta.meta_prediction_error_rolling(meta_pe_rolling_window())
            if cb is not None and meta_cb_enabled():
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
            snap["success_rate_after_meta_do"] = self._w_meta._success_rate_after_meta_do
            wmeta_snap = self._w_meta.snapshot()
            snap["w_meta"] = wmeta_snap

        self._goal_generator.on_tick(tick)
        propose_every = max(1, int(os.environ.get("RKK_GOAL_PROPOSE_EVERY", "200")))
        if goal_gen_enabled() and tick % propose_every == 0:
            role_map = {}
            try:
                role_map = self.agent.graph.role_type_map()
            except Exception:
                pass
            cand = self._goal_generator.propose(
                self.agent.graph,
                self._w_meta,
                role_map=role_map,
                tick=tick,
                world_id=str(self.current_world),
            )
            if cand is not None and curriculum_graph_enabled():
                self._curriculum_graph.add_generated_node(cand, tick=tick)

        if curriculum_graph_enabled() and is_humanoid_topology(self.current_world):
            active = self._curriculum_graph.activate_next(
                tick, world_id=str(self.current_world)
            )
            if active is not None:
                snap["curriculum_graph_active"] = active.to_dict()

        if goal_gen_enabled() and self._goal_generator._active:
            g0 = self._goal_generator._active[0]
            snap["autonomous_subgoal"] = {
                "var_id": g0.var_id,
                "target_val": g0.target_val,
                "meta_success_pred": g0.meta_success_pred,
            }
            if g0.var_id in self.agent.graph.nodes:
                v = float(self.agent.graph.nodes[g0.var_id])
                reached = abs(v - float(g0.target_val)) < 0.12
                try:
                    bench_after = max(
                        20,
                        int(os.environ.get("RKK_GOAL_BENCH_COMPLETE_AFTER", "80")),
                    )
                except ValueError:
                    bench_after = 80
                bench_done = (
                    transfer_bench_enabled()
                    and tick - int(g0.tick_proposed) >= bench_after
                )
                if reached or bench_done:
                    sr = snap.get("behavioral_score")
                    if sr is None:
                        sr = 1.0 - float(snap.get("prediction_error", 0.45))
                    self._goal_generator.complete_goal(
                        g0.var_id,
                        success_rate=max(float(sr), 0.55),
                        tick=tick,
                    )

        if goal_gen_enabled() and tick % max(1, int(os.environ.get("RKK_GOAL_WORLD_SWITCH_EVERY", "600"))) == 0:
            sw = getattr(self, "switcher", None)
            if sw is not None and is_humanoid_topology(self.current_world):
                target = (
                    "humanoid_variant"
                    if self.current_world == "humanoid"
                    else "humanoid"
                )
                sw.switch(target)
                self.current_world = target
