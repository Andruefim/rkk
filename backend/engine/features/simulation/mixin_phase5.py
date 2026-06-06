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
            out["meta_prediction_error"] = self._w_meta.meta_prediction_error_rolling(500)
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
                obs = build_meta_observation(
                    self.agent,
                    tick=tick,
                    curriculum_step=cur_step,
                    success_rate=float(success) if success is not None else None,
                )
                self._w_meta.observe(obs, tick=tick)
            meta_pe = self._w_meta.meta_prediction_error_rolling(500)
            if cb is not None and meta_cb_enabled():
                meta_age = tick - int(getattr(self._w_meta, "_last_update_tick", tick))
                prev = cb.state
                cb.observe(meta_pe, meta_age, tick)
                if cb.state == cb.HALF_OPEN and prev == cb.OPEN:
                    cb.reset_w_meta_if_needed(self._w_meta)
                snap["wmeta_active"] = cb.wmeta_active
                snap["meta_circuit_breaker"] = cb.snapshot(tick)
                snap["meta_recovery_ticks"] = cb.recovery_ticks(tick)
            snap["meta_prediction_error"] = self._w_meta.meta_prediction_error_rolling(500)
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
            sr = float(success) if success is not None else 0.5
            for g in list(self._goal_generator._active):
                val = float(self.agent.graph.nodes.get(g.var_id, 0.5))
                near_target = abs(val - g.target_val) <= 0.18
                ticks_active = max(0, tick - int(g.tick_proposed or tick))
                if near_target or sr >= 0.45 or ticks_active >= 120:
                    self._goal_generator.complete_goal(
                        g.var_id,
                        success_rate=max(sr, 0.55 if near_target else sr),
                        tick=tick,
                    )
                    if curriculum_graph_enabled():
                        for nid, node in self._curriculum_graph._nodes.items():
                            if node.var_id == g.var_id and node.status == "active":
                                self._curriculum_graph.mark_completed(
                                    nid,
                                    success_rate=max(sr, 0.55 if near_target else sr),
                                    tick=tick,
                                )
                                break
