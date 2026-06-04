"""
Track A Phase 0: dual-criterion curriculum advance via subprocess eval_transfer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from engine.eval_mode import (
    advance_eval_fallen_max,
    advance_eval_quality_min,
    advance_eval_ticks,
    gate_result_path,
    gate_snapshot_path,
)
from engine.persistence import save_simulation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _eval_script() -> Path:
    return _repo_root() / "backend" / "tools" / "eval_transfer.py"


def write_gate_snapshot(sim: Any, path: str | Path | None = None) -> dict[str, Any]:
    """Persist graph/temporal state for subprocess eval (--load-snapshot)."""
    p = Path(path or gate_snapshot_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    out = save_simulation(sim, p)
    tags = {}
    try:
        from engine.eval_mode import curriculum_context_tags

        tags = curriculum_context_tags(sim, sim.agent)
    except Exception:
        pass
    meta = {
        "tick": int(sim.tick),
        "current_world": str(getattr(sim, "current_world", "")),
        "tags": tags,
    }
    meta_path = p.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**out, "meta_path": str(meta_path), "tags": tags}


def evaluate_gate_metrics(
    *,
    fallen_frac: float,
    success_rate: float,
    quality: float | None = None,
) -> dict[str, Any]:
    fallen_max = advance_eval_fallen_max()
    quality_min = advance_eval_quality_min()
    q = float(quality if quality is not None else success_rate)
    passed = (
        float(fallen_frac) <= fallen_max
        and float(success_rate) >= quality_min
        and q >= quality_min
    )
    return {
        "passed": passed,
        "fallen_frac": round(float(fallen_frac), 4),
        "success_rate": round(float(success_rate), 4),
        "quality": round(q, 4),
        "thresholds": {
            "fallen_max": fallen_max,
            "quality_min": quality_min,
        },
    }


def write_gate_result(result: dict[str, Any], path: str | Path | None = None) -> Path:
    p = Path(path or gate_result_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_gate_result(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path or gate_result_path())
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _eval_subprocess_timeout_s() -> float:
    try:
        return max(5.0, float(os.environ.get("RKK_CURRICULUM_EVAL_GATE_TIMEOUT", "600")))
    except ValueError:
        return 600.0


def _skip_blocking_gate_in_live_ui() -> bool:
    """Decoupled agent loop: never block the sim thread on 100-tick eval_transfer."""
    if os.environ.get("RKK_CURRICULUM_EVAL_GATE_LIVE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    try:
        from engine.core.constants import agent_loop_hz_from_env

        return agent_loop_hz_from_env() > 0.0
    except Exception:
        return False


def run_eval_subprocess(
    *,
    snapshot_path: str | Path | None = None,
    eval_ticks: int | None = None,
    extra_env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Spawn eval_transfer with RKK_EVAL_MODE=1; return parsed gate result."""
    if timeout_s is None:
        timeout_s = _eval_subprocess_timeout_s()
    snap = Path(snapshot_path or gate_snapshot_path())
    ticks = eval_ticks if eval_ticks is not None else advance_eval_ticks()
    script = _eval_script()
    if not script.is_file():
        return {"passed": False, "error": f"missing script: {script}"}

    env = os.environ.copy()
    env["RKK_EVAL_MODE"] = "1"
    env["RKK_SCORE_ASYNC"] = "0"
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable,
        str(script),
        "--load-snapshot",
        str(snap),
        "--eval-ticks",
        str(int(ticks)),
        "--eval-only",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root() / "backend"),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "eval subprocess timeout", "cmd": cmd}
    except OSError as e:
        return {"passed": False, "error": str(e), "cmd": cmd}

    result = load_gate_result() or {}
    result["subprocess_exit_code"] = int(proc.returncode)
    if proc.returncode != 0 and not result.get("passed"):
        result.setdefault("error", (proc.stderr or proc.stdout or "")[-2000:])
    write_gate_result(result)
    return result


def dual_gate_allows(
    sim: Any,
    *,
    reason: str,
    run_subprocess: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Train-side gate: write snapshot, subprocess eval, check fallen/quality thresholds.
    reason: 'scope_advance' | 'fr_release' | other (logged).
    """
    snap_info = write_gate_snapshot(sim)
    diag: dict[str, Any] = {
        "reason": reason,
        "snapshot": snap_info,
        "passed": False,
    }
    if not run_subprocess:
        diag["passed"] = True
        diag["skipped_subprocess"] = True
        write_gate_result(diag)
        return True, diag

    if _skip_blocking_gate_in_live_ui():
        diag["passed"] = True
        diag["skipped_subprocess"] = True
        diag["skipped_live_ui"] = True
        write_gate_result(diag)
        return True, diag

    eval_out = run_eval_subprocess()
    diag["eval"] = eval_out
    passed = bool(eval_out.get("passed"))
    if not passed and "fallen_frac" in eval_out:
        check = evaluate_gate_metrics(
            fallen_frac=float(eval_out.get("fallen_frac", 1.0)),
            success_rate=float(eval_out.get("success_rate", 0.0)),
            quality=eval_out.get("quality"),
        )
        passed = bool(check.get("passed"))
        diag["metrics"] = check
        diag["passed"] = passed
    else:
        diag["passed"] = passed
    write_gate_result(diag)
    return passed, diag


def maybe_gate_scope_advance(sim: Any, agent: Any) -> bool:
    """Call before ProgressiveScope phase advance; returns False to block."""
    if os.environ.get("RKK_CURRICULUM_EVAL_GATE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return True
    ok, _ = dual_gate_allows(sim, reason="scope_advance")
    return ok


def maybe_gate_fr_release(sim: Any) -> bool:
    """Call before _fr_curriculum_finalize_release; returns False to block."""
    if os.environ.get("RKK_CURRICULUM_EVAL_GATE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return True
    ok, _ = dual_gate_allows(sim, reason="fr_release")
    return ok
