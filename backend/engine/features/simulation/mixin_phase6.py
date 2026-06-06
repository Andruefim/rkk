"""Simulation mixin: Phase 6 — non-physical stubs, symbolic grounding, EWC, health monitor, meta CB."""
from __future__ import annotations

from typing import Any

from engine.causal_health_monitor import (
    CausalHealthMonitor,
    health_check_every,
    health_monitor_enabled,
)
from engine.elastic_role_protector import ElasticRoleProtector, ewc_enabled
from engine.environment_grid_nav import EnvironmentGridNav
from engine.environment_symbolic import EnvironmentSymbolic
from engine.meta_circuit_breaker import MetaCircuitBreaker, meta_cb_enabled
from engine.symbolic_grounding import SymbolicGrounding, symbolic_grounding_enabled


class SimulationPhase6Mixin:
    def _ensure_phase6(self) -> None:
        if getattr(self, "_phase6_ready", False):
            return
        self._symbolic_grounding = SymbolicGrounding()
        self._ewc_protector = ElasticRoleProtector()
        self._health_monitor = CausalHealthMonitor()
        self._meta_cb = MetaCircuitBreaker()
        self._snap_window: list[dict[str, Any]] = []
        self._phase6_ready = True
        self.agent._ewc_protector = self._ewc_protector
        self.agent.graph._ewc_protector = self._ewc_protector
        if meta_cb_enabled() and getattr(self, "_meta_cb", None) is None:
            self._meta_cb = MetaCircuitBreaker()

    def _phase6_snapshot_meta(self) -> dict[str, Any]:
        self._ensure_phase6()
        out: dict[str, Any] = {
            "symbolic_grounding_enabled": symbolic_grounding_enabled(),
            "ewc_enabled": ewc_enabled(),
            "health_monitor_enabled": health_monitor_enabled(),
            "meta_cb_enabled": meta_cb_enabled(),
        }
        if self._symbolic_grounding is not None:
            out["symbolic_grounding"] = self._symbolic_grounding.snapshot()
        if self._ewc_protector is not None:
            out["ewc"] = self._ewc_protector.snapshot()
            out["continual_forgetting_ratio"] = self._ewc_protector._continual_forgetting_ratio
            out["ewc_stable_edge_count"] = self._ewc_protector._stable_edge_count
            out["ewc_recompute_count"] = self._ewc_protector._ewc_recompute_count
        if self._health_monitor is not None:
            out["health_monitor"] = self._health_monitor.snapshot()
        if self._meta_cb is not None:
            out["meta_circuit_breaker"] = self._meta_cb.snapshot(int(self.tick))
            out["meta_recovery_ticks"] = self._meta_cb.recovery_ticks(int(self.tick))
            out["wmeta_active"] = self._meta_cb.wmeta_active
        env = self.agent.env
        if isinstance(env, (EnvironmentGridNav, EnvironmentSymbolic)):
            out["world_metrics"] = env.autonomy_metrics()
        return out

    def _tick_phase6(self, snap: dict[str, Any]) -> None:
        if not (
            symbolic_grounding_enabled()
            or ewc_enabled()
            or health_monitor_enabled()
            or meta_cb_enabled()
        ):
            return
        self._ensure_phase6()
        tick = int(self.tick)
        graph = self.agent.graph

        if ewc_enabled():
            self._ewc_protector.maybe_update(graph)
            snap.update(self._ewc_protector.metrics())

        sr = float(snap.get("behavioral_score", snap.get("success_rate", 0.0)) or 0.0)
        if not hasattr(self, "_world_success_last"):
            self._world_success_last: dict[str, float] = {}
        if not hasattr(self, "_world_success_baseline"):
            self._world_success_baseline: dict[str, float] = {}
        wid = str(self.current_world)
        self._world_success_last[wid] = sr
        if wid not in self._world_success_baseline or self._world_success_baseline[wid] <= 0:
            self._world_success_baseline[wid] = max(sr, self._world_success_baseline.get(wid, 0.0))

        env = self.agent.env
        if isinstance(env, (EnvironmentGridNav, EnvironmentSymbolic)):
            wm = env.autonomy_metrics()
            snap.setdefault("worlds", {})
            if isinstance(snap["worlds"], dict):
                snap["worlds"][str(self.current_world)] = wm
            for k, v in wm.items():
                snap[k] = v

        if symbolic_grounding_enabled() and getattr(self, "_last_skeleton", None) is not None:
            sk = self._last_skeleton
            rules = self._symbolic_grounding.skeleton_to_rules(sk)
            snap["symbolic_rules"] = rules[:16]
            snap["symbolic_rules_n"] = len(rules)

        if health_monitor_enabled() and tick % health_check_every() == 0:
            self._snap_window.append(dict(snap))
            if len(self._snap_window) > 64:
                self._snap_window.pop(0)
            report = self._health_monitor.diagnose(self._snap_window)
            repair = self._health_monitor.suggest_repair(report)
            snap["health_degraded"] = report.degraded
            snap["health_symptoms"] = list(report.symptoms)
            snap["health_repair_action"] = repair.action
            if report.degraded:
                self._health_monitor.apply_repair(repair, self)

    def on_world_switch_phase6(self, new_world: str) -> None:
        self._ensure_phase6()
        if ewc_enabled():
            prot = self._ewc_protector
            old_world = str(getattr(self, "current_world", "humanoid"))
            baseline = float(getattr(self, "_world_success_baseline", {}).get(old_world, 0.0))
            current = float(getattr(self, "_world_success_last", {}).get(old_world, 0.0))
            if baseline > 0:
                prot.update_forgetting_ratio(baseline, current)
            prot.on_world_switch(self.agent.graph, new_world)
