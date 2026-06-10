#!/usr/bin/env python3
"""
Run phase_validation_agent.md behavioral gates sequentially.
Writes logs/phase_validation_report.json

Turbo: RKK_PHASE_VALIDATE_FAST=1 (default train ~1000 ticks, skip eval for scorecard).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
REPORT = ROOT / "logs" / "phase_validation_report.json"
SCORECARD = ROOT / "logs" / "autonomy_scorecard.json"


def _fast() -> bool:
    return os.environ.get("RKK_PHASE_VALIDATE_FAST", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _train_ticks() -> int:
    return 1000 if _fast() else 1500


def _eval_ticks() -> int:
    return 30 if _fast() else 100


def _phase_env(cumulative: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["RKK_DEVICE"] = env.get("RKK_DEVICE", "cpu")
    env["RKK_RUN_TESTS"] = "0"
    env["RKK_PHASE_VALIDATE_FAST"] = "1" if _fast() else "0"
    for k, v in cumulative.items():
        env[k] = v
    return env


def _run_eval(args: list[str], env: dict[str, str]) -> tuple[int, str]:
    cmd = [sys.executable, str(BACKEND / "tools" / "eval_transfer.py"), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(BACKEND),
        env=env,
        capture_output=True,
        text=True,
        timeout=float(env.get("RKK_PHASE_VALIDATE_TIMEOUT", "7200")),
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out[-4000:]


def _load_scorecard() -> dict[str, Any]:
    if SCORECARD.is_file():
        return json.loads(SCORECARD.read_text(encoding="utf-8"))
    return {}


def _last_jsonl_row() -> dict[str, Any]:
    jsonl = ROOT / "logs" / "transfer_eval.jsonl"
    last: dict[str, Any] = {}
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                last = json.loads(line)
    return last


def _check_phase0(sc: dict[str, Any]) -> dict[str, Any]:
    h = sc.get("worlds", {}).get("humanoid", {})
    ok = bool(h.get("a1_pass")) and bool(h.get("a4_pass"))
    return {
        "phase": 0,
        "closed": ok,
        "a1_pass": h.get("a1_pass"),
        "a4_pass": h.get("a4_pass"),
        "script_override": h.get("script_override_frac_post_warmup"),
        "emergency_override": h.get("emergency_override_frac_post_warmup"),
    }


def _check_phase1(sc: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    rate = float(row.get("cross_env_success_rate_200", 0.0))
    return {
        "phase": 1,
        "closed": rate >= 0.40,
        "cross_env_success_rate_200": rate,
    }


def _check_phase2(sc: dict[str, Any]) -> dict[str, Any]:
    h = sc.get("worlds", {}).get("humanoid", {})
    disc = float(sc.get("discovery_new_frac", 0.0))
    ok = (
        bool(h.get("a1_pass"))
        and bool(h.get("a4_pass"))
        and disc > 0.60
        and bool(sc.get("pass_core_embodied"))
    )
    return {
        "phase": 2,
        "closed": ok,
        "discovery_new_frac": disc,
        "pass_core_embodied": sc.get("pass_core_embodied"),
        "frozen_script": h.get("script_override_frac_post_warmup"),
        "frozen_emergency": h.get("emergency_override_frac_post_warmup"),
    }


def _check_phase3(sc: dict[str, Any]) -> dict[str, Any]:
    worlds = sc.get("worlds", {})
    registered = sum(
        1
        for w in ("humanoid", "grid_nav", "symbolic_control")
        if worlds.get(w, {}).get("metrics_applicable")
    )
    return {
        "phase": 3,
        "closed": registered >= 3,
        "worlds_registered": registered,
    }


def _check_phase5(sc: dict[str, Any]) -> dict[str, Any]:
    meta_pe = float(sc.get("meta_prediction_error", 1.0))
    goals = bool(sc.get("autonomous_goals_crossworld_pass"))
    ok = meta_pe < 0.15 and goals and bool(sc.get("pass_agi_extended"))
    return {
        "phase": 5,
        "closed": ok,
        "meta_prediction_error": meta_pe,
        "autonomous_goals_crossworld_pass": goals,
        "pass_agi_extended": sc.get("pass_agi_extended"),
    }


def _check_phase6b(sc: dict[str, Any]) -> dict[str, Any]:
    worlds = sc.get("worlds", {})
    gn = worlds.get("grid_nav", {})
    sym = worlds.get("symbolic_control", {})
    cont = float(sc.get("continual_forgetting_ratio", 0.0))
    ok = (
        cont >= 0.50
        and bool(gn.get("a1_pass"))
        and bool(gn.get("a4_pass"))
        and bool(sym.get("a1_pass"))
        and bool(sym.get("a4_pass"))
    )
    return {
        "phase": "6b",
        "closed": ok,
        "continual_forgetting_ratio": cont,
        "grid_nav": gn,
        "symbolic_control": sym,
    }


def _check_phase6c(sc: dict[str, Any]) -> dict[str, Any]:
    ok = bool(sc.get("pass_agi_full"))
    return {
        "phase": "6c",
        "closed": ok,
        "pass_agi_full": sc.get("pass_agi_full"),
        "meta_recovery_ticks": sc.get("meta_recovery_ticks"),
    }


def main() -> int:
    cumulative: dict[str, str] = {
        "RKK_POST_FR_ALPHA_DECAY": "0.60",
        "RKK_POST_FR_WM_LR_MULT": "2.50",
        "RKK_S2_OVERRIDE_FALLEN_TICKS": "36",
        "RKK_PHASE_VALIDATE_TIMEOUT": os.environ.get(
            "RKK_PHASE_VALIDATE_TIMEOUT", "7200"
        ),
        "RKK_BENCH_SCORECARD_TRAIN_ONLY": "1",
        "RKK_EVAL_MODE": "0",
        "RKK_ADVANCE_EVAL_FALLEN_MAX": "0.35",
        "RKK_ADVANCE_EVAL_QUALITY_MIN": "0.30",
        "RKK_WORLD_BRIDGE_ENABLED": "0",
        "RKK_BRIDGE_LOSS_WEIGHT": "0.0",
        "RKK_STRUCTURE_LEARN_EVERY": "0",
        "RKK_C4_ENABLED": "0",
        "RKK_C5_ENABLED": "0",
        "RKK_META_CAUSAL_ENABLED": "0",
        "RKK_GOAL_GEN_ENABLED": "0",
        "RKK_EWC_ENABLED": "0",
    }
    report: dict[str, Any] = {"phases": [], "fast": _fast()}
    tt, et = _train_ticks(), _eval_ticks()
    seeds = ["--pose-seed", "42", "--agent-seed", "42"]

    # Phase 0
    env = _phase_env(cumulative)
    code, tail = _run_eval(
        [
            "--train-ticks",
            str(tt),
            "--eval-ticks",
            str(et),
            "--scorecard",
            "--world",
            "humanoid",
            *seeds,
        ],
        env,
    )
    sc = _load_scorecard()
    p0 = _check_phase0(sc)
    p0["exit_code"] = code
    p0["log_tail"] = tail if not p0["closed"] else ""
    report["phases"].append(p0)
    if not p0["closed"]:
        _write_report(report)
        print(json.dumps(report, indent=2))
        return 1

    # Phase 1
    code, tail = _run_eval(
        [
            "--train-ticks",
            str(max(600, tt // 2)),
            "--benchmark",
            "cross_env_same_topology",
            "--cross-env-eval-ticks",
            str(max(30, et)),
            "--world",
            "humanoid",
            *seeds,
        ],
        env,
    )
    p1 = _check_phase1(_load_scorecard(), _last_jsonl_row())
    p1["exit_code"] = code
    report["phases"].append(p1)
    if not p1["closed"]:
        _write_report(report)
        print(json.dumps(report, indent=2))
        return 1

    # Phase 2
    cumulative.update(
        {
            "RKK_WORLD_BRIDGE_ENABLED": "1",
            "RKK_BRIDGE_LOSS_WEIGHT": "0.20",
            "RKK_STRUCTURE_LEARN_EVERY": "80",
            "RKK_VSTRUCTURE_ENSEMBLE_N": "4",
            "RKK_LOG_DISCOVERY_SPLIT": "1",
            "RKK_POST_FR_ALPHA_DECAY": "0.65",
            "RKK_S2_OVERRIDE_FALLEN_TICKS": "42",
        }
    )
    env = _phase_env(cumulative)
    code, _ = _run_eval(
        [
            "--train-ticks",
            str(max(tt, 1100)),
            "--eval-ticks",
            str(et),
            "--scorecard",
            "--world",
            "humanoid",
            *seeds,
        ],
        env,
    )
    sc = _load_scorecard()
    p2 = _check_phase2(sc)
    p2["exit_code"] = code
    report["phases"].append(p2)
    report["phase2_frozen"] = {
        "script": p2.get("frozen_script"),
        "emergency": p2.get("frozen_emergency"),
        "discovery": p2.get("discovery_new_frac"),
    }

    # Phases 3–6 flags
    cumulative.update(
        {
            "RKK_C4_ENABLED": "1",
            "RKK_C5_ENABLED": "1",
            "RKK_SPECTRAL_TRANSFER_ENABLED": "1",
            "RKK_C6_ENABLED": "1",
            "RKK_SKELETON_TRANSFER_ENABLED": "1",
            "RKK_META_CAUSAL_ENABLED": "1",
            "RKK_META_UPDATE_EVERY": "30",
            "RKK_GOAL_GEN_ENABLED": "1",
            "RKK_CURRICULUM_GRAPH_ENABLED": "1",
            "RKK_H_GRID_NAV_ENABLED": "1",
            "RKK_H_SYMBOLIC_ENABLED": "1",
            "RKK_SYMBOLIC_GROUNDING_ENABLED": "1",
            "RKK_EWC_ENABLED": "1",
            "RKK_HEALTH_MONITOR_ENABLED": "1",
            "RKK_META_CB_ENABLED": "1",
        }
    )
    env = _phase_env(cumulative)

    # Phases 5 + 6b + 6c — one continual run
    code, tail6 = _run_eval(
        [
            "--continual",
            "--train-ticks",
            str(700 if _fast() else 800),
            "--cross-env-eval-ticks",
            str(max(30, et)),
            "--scorecard",
            "--worlds",
            "humanoid,grid_nav,symbolic_control",
            *seeds,
        ],
        env,
    )
    sc = _load_scorecard()
    p3 = _check_phase3(sc)
    p3["exit_code"] = code
    report["phases"].append(p3)

    p5 = _check_phase5(sc)
    p5["exit_code"] = code
    report["phases"].append(p5)

    p6b = _check_phase6b(sc)
    p6b["exit_code"] = code
    if code != 0:
        p6b["log_tail"] = tail6
    report["phases"].append(p6b)

    p6c = _check_phase6c(sc)
    report["phases"].append(p6c)

    _write_report(report)
    print(json.dumps(report, indent=2))
    all_closed = all(p.get("closed") for p in report["phases"] if "closed" in p)
    return 0 if all_closed else 1


def _write_report(report: dict[str, Any]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
