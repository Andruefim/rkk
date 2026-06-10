"""
Track A Phase 0: eval-mode flags, curriculum buffer tags, advance-gate env helpers.
"""
from __future__ import annotations

import os
from typing import Any


def _env_bool(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: str) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return float(default)


def _env_int(key: str, default: str) -> int:
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return int(default)


def eval_mode_enabled() -> bool:
    """Suppress WM train, distill append, trajectory contrastive train when True."""
    return _env_bool("RKK_EVAL_MODE", "0")


def eval_skip_system2() -> bool:
    """Skip System2 tick during eval/benchmark subprocess (faster transfer eval)."""
    return _env_bool("RKK_EVAL_SKIP_SYSTEM2", "1")


def transfer_bench_enabled() -> bool:
    return _env_bool("RKK_TRANSFER_BENCH", "0")


def apply_eval_bench_env() -> None:
    """
    Defaults for eval_transfer / curriculum gate subprocesses (Track A / Phase 1 bench).
    Call before Simulation() so init stays lightweight.
    """
    # Force bench profile (`.env` may set weaker Phase-0 defaults).
    os.environ["RKK_TRANSFER_BENCH"] = "1"
    os.environ.setdefault("RKK_SKIP_ALL_LLM", "1")
    os.environ.setdefault("RKK_NEURAL_LANG", "0")
    os.environ.setdefault("RKK_TICK_PROFILE", "0")
    os.environ.setdefault("RKK_TICK_RUN_LOG", "0")
    os.environ.setdefault("RKK_EVAL_SKIP_SYSTEM2", "1")
    os.environ.setdefault("RKK_SNAPSHOT_EDGES_MAX", "0")
    os.environ.setdefault("RKK_SCORE_CACHE_EVERY", "80")
    os.environ.setdefault("RKK_SCORE_STALE_ONLY", "1")
    os.environ.setdefault("RKK_MEMORY_DIAG_INTERVAL", "0")
    if os.environ.get("RKK_BENCH_KEEP_ENV", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        os.environ["RKK_S2_OVERRIDE_FALLEN_TICKS"] = "36"
        os.environ["RKK_POST_FR_ALPHA_DECAY"] = "0.60"
        os.environ["RKK_META_PE_ROLLING_WINDOW"] = "32"
        os.environ["RKK_META_UPDATE_EVERY"] = "30"
        os.environ["RKK_SCORECARD_WARMUP_TICKS"] = "500"
        os.environ["RKK_BENCH_SCORECARD_TRAIN_ONLY"] = "1"
        os.environ["RKK_GOAL_PROPOSE_EVERY"] = "60"
        os.environ["RKK_GOAL_WORLD_SWITCH_EVERY"] = "120"
        os.environ["RKK_GOAL_BENCH_COMPLETE_AFTER"] = "30"
    os.environ.setdefault("RKK_GOAL_WMETA_MIN_SUCCESS", "0.0")
    os.environ.setdefault("RKK_GOAL_TRANSFER_MIN_SUCCESS", "0.35")
    # RKK_SCORE_ASYNC: leave .env default for train; eval phase sets sync in _run_ticks.


def meta_pe_rolling_window() -> int:
    try:
        return max(50, _env_int("RKK_META_PE_ROLLING_WINDOW", "500"))
    except ValueError:
        return 500


def cross_env_allow_wm_train() -> bool:
    """When False (RKK_CROSS_ENV_ALLOW_WM_TRAIN=0), block graph.train_step (zero-shot eval)."""
    return _env_bool("RKK_CROSS_ENV_ALLOW_WM_TRAIN", "1")


def eval_score_async_disabled() -> bool:
    """Force synchronous scoring during eval subprocess (RKK_SCORE_ASYNC=0)."""
    return os.environ.get("RKK_SCORE_ASYNC", "").strip() == "0"


def advance_eval_ticks() -> int:
    return max(10, _env_int("RKK_ADVANCE_EVAL_TICKS", "100"))


def advance_eval_fallen_max() -> float:
    return _env_float("RKK_ADVANCE_EVAL_FALLEN_MAX", "0.35")


def advance_eval_quality_min() -> float:
    return _env_float("RKK_ADVANCE_EVAL_QUALITY_MIN", "0.30")


def transfer_eval_log_path() -> str:
    return os.environ.get("RKK_TRANSFER_EVAL_LOG", "logs/transfer_eval.jsonl")


def gate_snapshot_path() -> str:
    return os.environ.get("RKK_EVAL_GATE_SNAPSHOT", "state/eval_gate_snapshot.rkk")


def gate_result_path() -> str:
    return os.environ.get("RKK_EVAL_GATE_RESULT", "logs/eval_gate_result.json")


def curriculum_context_tags(sim: Any, agent: Any) -> dict[str, Any]:
    """
    Per-tick curriculum tags for trajectory/distill buffers.
    humanoid_curriculum_step (UI path) is distinct from progressive_scope phase.
    """
    fixed_root = bool(getattr(sim, "_fixed_root_active", False))
    fallen = False
    try:
        fn = getattr(agent.env, "is_fallen", None)
        if callable(fn):
            fallen = bool(fn())
    except Exception:
        fallen = False

    curriculum_step = 0
    try:
        from engine.features.simulation.snapshot import humanoid_curriculum_step

        curriculum_step, _ = humanoid_curriculum_step(sim)
    except Exception:
        pass

    scope_phase = -1
    ps = getattr(agent, "_prog_scope", None)
    if ps is not None:
        scope_phase = int(getattr(ps, "phase", -1))

    return {
        "fixed_root": fixed_root,
        "fallen": fallen,
        "curriculum_step": int(curriculum_step),
        "scope_phase": int(scope_phase),
    }


def stage_label_from_tags(tags: dict[str, Any]) -> str:
    if tags.get("fixed_root"):
        return "fixed_root"
    sp = int(tags.get("scope_phase", -1))
    if sp >= 0:
        return f"scope_phase_{sp}"
    cs = int(tags.get("curriculum_step", 0))
    return f"curriculum_step_{cs}"


def aggregate_segment_tags(
    tick_tags: list[dict[str, Any]],
) -> dict[str, Any]:
    """Finalize-level aggregates for trajectory segments."""
    if not tick_tags:
        return {
            "fallen_frac": 0.0,
            "fixed_root_frac": 0.0,
            "dominant_stage": "unknown",
        }
    n = len(tick_tags)
    fallen_frac = sum(1 for t in tick_tags if t.get("fallen")) / n
    fixed_root_frac = sum(1 for t in tick_tags if t.get("fixed_root")) / n
    stages = [stage_label_from_tags(t) for t in tick_tags]
    dominant = max(set(stages), key=stages.count) if stages else "unknown"
    return {
        "fallen_frac": round(fallen_frac, 4),
        "fixed_root_frac": round(fixed_root_frac, 4),
        "dominant_stage": dominant,
    }
