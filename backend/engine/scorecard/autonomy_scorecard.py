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


def build_scorecard(
    sim_snap: dict[str, Any] | None = None,
    *,
    worlds: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sim_snap = sim_snap or {}
    world_list = worlds or ["humanoid"]
    th = default_thresholds()
    worlds_out = {
        wid: _world_autonomy_stub(wid, sim_snap)
        for wid in world_list
    }
    h = worlds_out.get("humanoid", {})
    pass_core = bool(h.get("a1_pass")) and bool(h.get("a4_pass"))
    discovery_frac = float(sim_snap.get("discovery_new_frac", 0.0))
    meta_pe = float(
        sim_snap.get(
            "meta_prediction_error",
            (sim_snap.get("phase5") or {}).get("meta_prediction_error", 1.0),
        )
    )
    phase5 = sim_snap.get("phase5") or {}
    goal_metrics = phase5.get("goal_generator") or sim_snap.get("goal_generator") or {}
    if meta_pe >= 1.0 and phase5.get("meta_prediction_error") is not None:
        meta_pe = float(phase5["meta_prediction_error"])
    meta_pe_pass = meta_pe < th["meta_pe_max"]
    goals_crossworld_pass = bool(
        goal_metrics.get("autonomous_goals_crossworld_pass", False)
    )
    pass_extended = (
        pass_core
        and discovery_frac >= th["discovery_min"]
        and meta_pe_pass
        and goals_crossworld_pass
    )
    phase6 = sim_snap.get("phase6") or {}
    continual = float(
        phase6.get(
            "continual_forgetting_ratio",
            sim_snap.get("continual_forgetting_ratio", 0.0),
        )
    )
    meta_recovery = phase6.get("meta_recovery_ticks", sim_snap.get("meta_recovery_ticks"))
    non_phys_ok = all(
        bool(worlds_out.get(w, {}).get("a1_pass"))
        and bool(worlds_out.get(w, {}).get("a4_pass"))
        for w in world_list
        if w != "humanoid"
        and worlds_out.get(w, {}).get("metrics_applicable")
    )
    pass_full = (
        pass_extended
        and non_phys_ok
        and continual >= th["continual_forgetting_min"]
        and (
            meta_recovery is None
            or float(meta_recovery) <= th["meta_recovery_max_ticks"]
        )
    )
    card: dict[str, Any] = {
        "pass_agi_full": pass_full,
        "pass_agi_extended": pass_extended,
        "pass_core_embodied": pass_core,
        "pass_core": pass_core,
        "autonomy_integrity_nonphys": non_phys_ok,
        "worlds": worlds_out,
        "discovery_new_frac": discovery_frac,
        "meta_prediction_error": round(meta_pe, 4),
        "meta_prediction_error_pass": meta_pe_pass,
        "autonomous_goals_crossworld_pass": goals_crossworld_pass,
        "continual_forgetting_ratio": round(continual, 4),
        "meta_recovery_ticks": meta_recovery,
        "ewc_stable_edge_count": phase6.get(
            "ewc_stable_edge_count", sim_snap.get("ewc_stable_edge_count")
        ),
        "thresholds": th,
    }
    if extra:
        card.update(extra)
    return card


def write_scorecard(
    card: dict[str, Any],
    path: str | Path | None = None,
) -> Path:
    p = Path(path or os.environ.get("RKK_SCORECARD_PATH", "logs/autonomy_scorecard.json"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
