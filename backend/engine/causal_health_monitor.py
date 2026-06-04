"""
Track I2: CausalHealthMonitor — degradation diagnosis + self-repair suggestions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env_flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def health_monitor_enabled() -> bool:
    return _env_flag("RKK_HEALTH_MONITOR_ENABLED", False)


def health_check_every() -> int:
    try:
        return max(1, int(os.environ.get("RKK_HEALTH_CHECK_EVERY", "100")))
    except ValueError:
        return 100


def health_discovery_min() -> float:
    try:
        return float(os.environ.get("RKK_HEALTH_DISCOVERY_MIN", "0.40"))
    except ValueError:
        return 0.40


def health_ensemble_min_ent() -> float:
    try:
        return float(os.environ.get("RKK_HEALTH_ENSEMBLE_MIN_ENT", "0.20"))
    except ValueError:
        return 0.20


def health_meta_pe_max() -> float:
    try:
        return float(os.environ.get("RKK_HEALTH_META_PE_MAX", "0.20"))
    except ValueError:
        return 0.20


def health_repair_dry_run() -> bool:
    return _env_flag("RKK_HEALTH_REPAIR_DRY_RUN", True)


@dataclass
class HealthReport:
    degraded: bool = False
    symptoms: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RepairAction:
    action: str
    reason: str
    dry_run: bool = True
    params: dict[str, Any] = field(default_factory=dict)


class CausalHealthMonitor:
    def __init__(self) -> None:
        self._baseline_cross_sr: float | None = None
        self._detection_count = 0
        self._check_count = 0
        self._last_report: HealthReport | None = None
        self._last_repair: RepairAction | None = None

    def _snap_metric(self, snap: dict[str, Any], key: str, default: float = 0.0) -> float:
        if key in snap:
            return float(snap[key])
        phase5 = snap.get("phase5") or {}
        if isinstance(phase5, dict) and key in phase5:
            return float(phase5[key])
        ge = snap.get("graph_ensemble") or {}
        if isinstance(ge, dict) and key in ge:
            return float(ge[key])
        return default

    def diagnose(self, snapshot_window: list[dict[str, Any]]) -> HealthReport:
        symptoms: list[str] = []
        scores: dict[str, float] = {}
        if not snapshot_window:
            return HealthReport(degraded=False, symptoms=symptoms, scores=scores)

        recent = snapshot_window[-min(32, len(snapshot_window)) :]
        disc_vals = [
            self._snap_metric(s, "discovery_new_frac")
            for s in recent
        ]
        disc = float(sum(disc_vals) / max(1, len(disc_vals)))
        scores["discovery_new_frac"] = disc
        if disc < health_discovery_min():
            symptoms.append("low_discovery")

        ent_vals = []
        for s in recent:
            ge = s.get("graph_ensemble") or {}
            if isinstance(ge, dict) and "entropy" in ge:
                ent_vals.append(float(ge["entropy"]))
        ent = float(sum(ent_vals) / max(1, len(ent_vals))) if ent_vals else 1.0
        scores["ensemble_entropy"] = ent
        if ent_vals and ent < health_ensemble_min_ent():
            symptoms.append("low_ensemble_entropy")

        pe_vals = [
            self._snap_metric(s, "meta_prediction_error", 0.0)
            for s in recent
        ]
        meta_pe = float(sum(pe_vals) / max(1, len(pe_vals)))
        scores["meta_prediction_error"] = meta_pe
        if meta_pe > health_meta_pe_max():
            symptoms.append("high_meta_pe")

        cross_sr = self._snap_metric(recent[-1], "cross_env_success_rate_200", 0.5)
        scores["cross_env_success_rate_200"] = cross_sr
        if self._baseline_cross_sr is None:
            self._baseline_cross_sr = cross_sr
        elif self._baseline_cross_sr > 0.05:
            drop = (self._baseline_cross_sr - cross_sr) / self._baseline_cross_sr
            scores["cross_env_drop"] = drop
            if drop > 0.20:
                symptoms.append("cross_env_sr_drop")

        degraded = len(symptoms) > 0
        report = HealthReport(degraded=degraded, symptoms=symptoms, scores=scores)
        self._last_report = report
        self._check_count += 1
        if degraded:
            self._detection_count += 1
        return report

    def suggest_repair(self, report: HealthReport) -> RepairAction:
        dry = health_repair_dry_run()
        if "high_meta_pe" in report.symptoms:
            action = RepairAction(
                action="wmeta_rollback",
                reason="meta_prediction_error above threshold",
                dry_run=dry,
                params={"reset_w_meta": True},
            )
        elif "low_discovery" in report.symptoms:
            action = RepairAction(
                action="latent_reinject",
                reason="discovery_new_frac collapsed",
                dry_run=dry,
                params={"reinject_latent": True},
            )
        elif "low_ensemble_entropy" in report.symptoms:
            action = RepairAction(
                action="alpha_trust_decay",
                reason="ensemble entropy collapsed",
                dry_run=dry,
                params={"decay": 0.05},
            )
        elif "cross_env_sr_drop" in report.symptoms:
            action = RepairAction(
                action="ewc_reset",
                reason="cross-world success rate dropped",
                dry_run=dry,
                params={"reanchor": True},
            )
        else:
            action = RepairAction(
                action="none",
                reason="healthy",
                dry_run=dry,
            )
        self._last_repair = action
        return action

    def apply_repair(self, action: RepairAction, sim: Any) -> bool:
        if action.dry_run or action.action == "none":
            return False
        agent = getattr(sim, "agent", None)
        graph = getattr(agent, "graph", None) if agent else None
        if graph is None:
            return False
        if action.action == "ewc_reset":
            prot = getattr(sim, "_ewc_protector", None)
            if prot is not None:
                prot.anchor_weights(graph)
                prot.compute_fisher(graph)
            return True
        if action.action == "alpha_trust_decay":
            decay = float(action.params.get("decay", 0.05))
            core = graph._core
            if core is not None and hasattr(core, "alpha_trust"):
                with __import__("torch").no_grad():
                    core.alpha_trust.mul_(1.0 - decay)
            return True
        if action.action == "latent_reinject":
            fn = getattr(sim, "_reinject_latent_confounder", None)
            if callable(fn):
                fn()
                return True
        if action.action == "wmeta_rollback":
            wmeta = getattr(sim, "_w_meta", None)
            if wmeta is not None:
                wmeta.load_dict({})
            cb = getattr(sim, "_meta_cb", None)
            if cb is not None:
                cb.force_open(int(getattr(sim, "tick", 0)))
            return True
        return False

    def detection_rate(self) -> float:
        if self._check_count <= 0:
            return 0.0
        return float(self._detection_count) / float(self._check_count)

    def snapshot(self) -> dict[str, Any]:
        rep = self._last_report
        repair = self._last_repair
        return {
            "enabled": health_monitor_enabled(),
            "check_every": health_check_every(),
            "dry_run": health_repair_dry_run(),
            "detection_rate": round(self.detection_rate(), 4),
            "check_count": self._check_count,
            "detection_count": self._detection_count,
            "last_degraded": bool(rep.degraded) if rep else False,
            "last_symptoms": list(rep.symptoms) if rep else [],
            "last_repair_action": repair.action if repair else None,
        }
