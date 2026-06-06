"""
Track D: autonomy scorecard JSON hooks (A1/A4 thresholds + worlds{}).
Phase 3: WorldAutonomyContract per-world probe mapping.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from engine.scorecard.world_autonomy_contract import (
    build_world_metrics,
    registered_world_ids,
)


def _env_float(key: str, default: str) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return float(default)


def default_thresholds() -> dict[str, float]:
    return {
        "a1_max": _env_float("RKK_SCORECARD_A1_MAX", "0.20"),
        "a4_max": _env_float("RKK_SCORECARD_A4_MAX", "0.15"),
        "discovery_min": _env_float("RKK_SCORECARD_DISCOVERY_MIN", "0.60"),
        "meta_pe_max": _env_float("RKK_SCORECARD_META_PE_MAX", "0.15"),
        "continual_forgetting_min": _env_float(
            "RKK_SCORECARD_CONTINUAL_FORGETTING_MIN", "0.50"
        ),
        "meta_recovery_max_ticks": _env_float(
            "RKK_SCORECARD_META_RECOVERY_MAX_TICKS", "1000"
        ),
    }


def _world_autonomy_stub(world_id: str, snap: dict[str, Any] | None) -> dict[str, Any]:
    """Per-world metrics via WorldAutonomyContract (humanoid #1/#4 frozen mapping)."""
    th = default_thresholds()
    return build_world_metrics(world_id, snap, thresholds=th)


def _metric_from_snap(snap: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in snap and snap[key] is not None:
            try:
                return float(snap[key])
            except (TypeError, ValueError):
                pass
        phase5 = snap.get("phase5") or {}
        if isinstance(phase5, dict) and key in phase5:
            try:
                return float(phase5[key])
            except (TypeError, ValueError):
                pass
        ewc = snap.get("ewc") or {}
        if isinstance(ewc, dict) and key in ewc:
            try:
                return float(ewc[key])
            except (TypeError, ValueError):
                pass
    return default


def build_scorecard(
    sim_snap: dict[str, Any] | None = None,
    *,
    worlds: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sim_snap = sim_snap or {}
    extra = extra or {}
    world_list = worlds or ["humanoid"]
    th = default_thresholds()
    worlds_out = {
        wid: _world_autonomy_stub(wid, sim_snap)
        for wid in world_list
    }
    h = worlds_out.get("humanoid", {})
    discovery_frac = float(sim_snap.get("discovery_new_frac", 0.0))
    pass_core = (
        bool(h.get("a1_pass"))
        and bool(h.get("a4_pass"))
        and discovery_frac >= th["discovery_min"]
    )
    meta_pe = _metric_from_snap(sim_snap, "meta_prediction_error", default=1.0)
    goal_metrics = (sim_snap.get("phase5") or {}).get("goal_generator") or sim_snap.get(
        "goal_generator"
    ) or {}
    if not isinstance(goal_metrics, dict):
        goal_metrics = {}
    meta_pe_pass = meta_pe < th["meta_pe_max"]
    goals_crossworld_pass = bool(
        goal_metrics.get("autonomous_goals_crossworld_pass", False)
    )
    pass_extended = pass_core and meta_pe_pass and goals_crossworld_pass

    nonphys_worlds = [w for w in ("grid_nav", "symbolic_control") if w in worlds_out]
    autonomy_integrity_nonphys = bool(nonphys_worlds) and all(
        bool(worlds_out[w].get("a1_pass"))
        and bool(worlds_out[w].get("a4_pass"))
        for w in nonphys_worlds
    )

    continual_forgetting = _metric_from_snap(
        sim_snap, "continual_forgetting_ratio", default=0.0
    )
    meta_recovery = sim_snap.get("meta_recovery_ticks")
    if meta_recovery is None:
        mcb = sim_snap.get("meta_circuit_breaker") or {}
        if isinstance(mcb, dict):
            meta_recovery = mcb.get("meta_recovery_ticks")
    cross_env_sr = extra.get("cross_env_success_rate_200")
    if cross_env_sr is None:
        te = extra.get("transfer_eval") or {}
        if isinstance(te, dict):
            cross_env_sr = te.get("cross_env_success_rate_200")
    skeleton_nonphys = extra.get("skeleton_nonphys_success_500")
    if skeleton_nonphys is None:
        te = extra.get("transfer_eval") or {}
        if isinstance(te, dict):
            skeleton_nonphys = te.get("skeleton_nonphys_success_500")

    pass_full = (
        pass_extended
        and autonomy_integrity_nonphys
        and continual_forgetting >= th["continual_forgetting_min"]
        and meta_recovery is not None
        and float(meta_recovery) <= th["meta_recovery_max_ticks"]
    )

    card: dict[str, Any] = {
        "pass_agi_full": pass_full,
        "pass_agi_extended": pass_extended,
        "pass_core_embodied": pass_core,
        "pass_core": pass_core,
        "autonomy_integrity_nonphys": autonomy_integrity_nonphys,
        "worlds": worlds_out,
        "discovery_new_frac": discovery_frac,
        "cross_env_success_rate_200": cross_env_sr,
        "meta_prediction_error": round(meta_pe, 4),
        "meta_prediction_error_pass": meta_pe_pass,
        "autonomous_goals_crossworld_pass": goals_crossworld_pass,
        "continual_forgetting_ratio": continual_forgetting,
        "ewc_stable_edge_count": _metric_from_snap(
            sim_snap, "ewc_stable_edge_count", default=0.0
        ),
        "meta_recovery_ticks": meta_recovery,
        "skeleton_nonphys_success_500": skeleton_nonphys,
        "thresholds": th,
    }
    if extra:
        safe_extra = {k: v for k, v in extra.items() if k != "worlds"}
        card.update(safe_extra)
    return card


def write_scorecard(
    card: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    p = Path(path or os.environ.get("RKK_SCORECARD_PATH", "logs/autonomy_scorecard.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
