"""
Track D Phase 3: per-world autonomy contracts + abstract A1/A4 probe mapping.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class WorldAutonomyContract:
    world_id: str
    recovery_macros: tuple[str, ...]
    script_override_sources: tuple[str, ...]
    emergency_override_snapshot_key: str
    success_field: str
    warmup_ticks: int = 800
    metrics_applicable: bool = True
    a1_probe_key: str = ""
    a4_probe_key: str = ""


def _humanoid_a1(snap: dict[str, Any]) -> float:
    s2 = snap.get("system2") or {}
    return float(
        s2.get(
            "script_override_frac_post_warmup",
            snap.get("s2_override_frac", 0.0),
        )
    )


def _humanoid_a4(snap: dict[str, Any]) -> float:
    s2 = snap.get("system2") or {}
    warmup = _env_int("RKK_SCORECARD_WARMUP_TICKS", "800")
    tick = int(snap.get("tick", 0))
    if tick < warmup:
        return 0.0
    return float(
        s2.get(
            "emergency_override_frac_post_warmup",
            snap.get("fallen_override_frac_post_warmup", 0.0),
        )
    )


def _stub_frac(snap: dict[str, Any], key: str) -> float:
    worlds = snap.get("worlds") or {}
    w = worlds.get(snap.get("current_world", ""), {}) if isinstance(worlds, dict) else {}
    if isinstance(w, dict) and key in w:
        return float(w.get(key, 0.0))
    return float(snap.get(key, 0.0))


CONTRACTS: dict[str, WorldAutonomyContract] = {
    "humanoid": WorldAutonomyContract(
        world_id="humanoid",
        recovery_macros=("RECOVER_POSTURE",),
        script_override_sources=("s2_scripted", "fallen_override", "recovery_schedule"),
        emergency_override_snapshot_key="fallen_override_frac_post_warmup",
        success_field="posture_stability",
        a1_probe_key="s2_override_frac",
        a4_probe_key="fallen_override_frac_post_800",
    ),
    "humanoid_variant": WorldAutonomyContract(
        world_id="humanoid_variant",
        recovery_macros=("RECOVER_POSTURE",),
        script_override_sources=("s2_scripted", "fallen_override"),
        emergency_override_snapshot_key="fallen_override_frac_post_warmup",
        success_field="posture_stability",
        metrics_applicable=True,
        a1_probe_key="s2_override_frac",
        a4_probe_key="fallen_override_frac_post_800",
    ),
    "cartpole": WorldAutonomyContract(
        world_id="cartpole",
        recovery_macros=("BALANCE_RECOVER",),
        script_override_sources=("replan_script", "balance_recovery"),
        emergency_override_snapshot_key="balance_emergency_override",
        success_field="upright",
        metrics_applicable=False,
        a1_probe_key="replan_script_override_frac",
        a4_probe_key="balance_emergency_override",
    ),
    "grid_nav": WorldAutonomyContract(
        world_id="grid_nav",
        recovery_macros=("UNSTUCK_RECOVER",),
        script_override_sources=("pathfinder", "stuck_recovery"),
        emergency_override_snapshot_key="stuck_override_active",
        success_field="goal_reached",
        metrics_applicable=False,
        a1_probe_key="pathfinder_override_frac",
        a4_probe_key="stuck_override_active",
    ),
    "symbolic_control": WorldAutonomyContract(
        world_id="symbolic_control",
        recovery_macros=("CONSTRAINT_REPAIR",),
        script_override_sources=("rule_engine", "constraint_repair"),
        emergency_override_snapshot_key="constraint_violation_override",
        success_field="constraints_satisfied",
        metrics_applicable=False,
        a1_probe_key="rule_engine_bailout_frac",
        a4_probe_key="constraint_violation_override",
    ),
}

_PROBE_EXTRACTORS: dict[str, Callable[[dict[str, Any]], float]] = {
    "humanoid": _humanoid_a1,
    "humanoid_variant": _humanoid_a1,
}


def get_contract(world_id: str) -> WorldAutonomyContract | None:
    return CONTRACTS.get(str(world_id))


def registered_world_ids() -> list[str]:
    return sorted(CONTRACTS.keys())


def extract_a1_a4(
    world_id: str,
    snap: dict[str, Any] | None,
) -> tuple[float, float, bool]:
    """
    Return (a1_metric, a4_metric, metrics_applicable).
    Humanoid maps Phase 2 frozen probes (#1 script override, #4 emergency).
    """
    contract = get_contract(world_id)
    if contract is None:
        return 0.0, 0.0, False
    s = snap or {}
    if world_id in ("humanoid", "humanoid_variant"):
        return _humanoid_a1(s), _humanoid_a4(s), contract.metrics_applicable
    a1 = _stub_frac(s, contract.a1_probe_key)
    a4 = _stub_frac(s, contract.a4_probe_key)
    return a1, a4, contract.metrics_applicable


def build_world_metrics(
    world_id: str,
    snap: dict[str, Any] | None,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    contract = get_contract(world_id)
    if contract is None:
        return {"world_id": world_id, "metrics_applicable": False}
    a1, a4, applicable = extract_a1_a4(world_id, snap)
    th = thresholds or {}
    a1_max = float(th.get("a1_max", 0.20))
    a4_max = float(th.get("a4_max", 0.15))
    return {
        "world_id": world_id,
        "warmup_ticks": contract.warmup_ticks,
        "recovery_macros": list(contract.recovery_macros),
        "script_override_frac_post_warmup": round(a1, 4),
        "emergency_override_frac_post_warmup": round(a4, 4),
        "a1_probe": contract.a1_probe_key,
        "a4_probe": contract.a4_probe_key,
        "a1_pass": a1 < a1_max if applicable else None,
        "a4_pass": a4 < a4_max if applicable else None,
        "metrics_applicable": applicable,
        "success_field": contract.success_field,
    }
