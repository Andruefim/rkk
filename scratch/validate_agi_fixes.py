#!/usr/bin/env python3
"""Acceptance checks for AGI Fixes v2 (3000+ tick jsonl log)."""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_ticks(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "tick":
                rows.append(obj)
    return rows


def _last_n(rows: list[dict], n: int) -> list[dict]:
    return rows[-n:] if len(rows) >= n else rows


def validate(rows: list[dict]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not rows:
        results["run"] = {"pass": False, "detail": "no tick rows"}
        return results

    tail = _last_n(rows, 1000)
    fallen = sum(1 for r in tail if (r.get("body") or {}).get("fallen")) / max(1, len(tail))
    results["fall_rate_last_1000"] = {
        "pass": fallen < 0.08,
        "value": round(fallen, 4),
        "threshold": "< 0.08",
    }

    step3 = [r for r in rows if (r.get("curriculum") or {}).get("step", 0) >= 3]
    com_vels = [
        float((r.get("behavioral") or {}).get("com_x_vel_ema", 0.0))
        for r in step3[-500:]
    ]
    com_med = sorted(com_vels)[len(com_vels) // 2] if com_vels else 0.0
    results["com_x_vel_ema_step3"] = {
        "pass": abs(com_med) > 0.00015,
        "value": round(com_med, 5),
        "threshold": "> 0.00015 (forward m/tick window rate)",
    }

    max_w = 0.0
    for r in tail:
        w = ((r.get("wm") or {}).get("ensemble") or {}).get("weights") or []
        if w:
            max_w = max(max_w, max(float(x) for x in w))
    results["ensemble_weights_max"] = {
        "pass": max_w > 0.35,
        "value": round(max_w, 4),
        "threshold": "> 0.35",
    }

    learned = 0
    for r in rows:
        s2 = r.get("system2") or {}
        if s2.get("override_recovered"):
            learned += 1
        ex = (r.get("events") or [])
        if any("learned_recovery" in str(e) for e in ex):
            learned += 1
    results["learned_recovery_episodes"] = {
        "pass": learned >= 1,
        "value": learned,
        "threshold": ">= 1",
    }

    spike_events = 0
    hist: deque[int] = deque(maxlen=100)
    for r in rows:
        d = int((r.get("hud") or {}).get("edge_delta", 0))
        hist.append(d)
        fallen_now = bool((r.get("body") or {}).get("fallen"))
        if len(hist) == 100 and sum(hist) > 500 and not fallen_now:
            spike_events += 1
    results["edge_spike_without_fall"] = {
        "pass": spike_events == 0,
        "value": spike_events,
        "threshold": "0 events >500/100 ticks without fall",
    }

    after_1500 = [r for r in rows if int(r.get("tick", 0)) >= 1500]
    scripted = 0
    total_rec = 0
    for r in after_1500:
        s2 = r.get("system2") or {}
        if not s2.get("fallen_override_active"):
            continue
        total_rec += 1
        src = str(s2.get("recovery_schedule_source", ""))
        if src in ("scripted", "fallback"):
            scripted += 1
    share = scripted / max(1, total_rec)
    results["scripted_fallback_share"] = {
        "pass": share < 0.5 or total_rec == 0,
        "value": round(share, 3),
        "threshold": "< 0.5 after tick 1500",
    }

    return results


def main() -> int:
    log_path = Path(os.environ.get("RKK_TICK_RUN_LOG_PATH", "logs/rkk_run.jsonl"))
    if not log_path.is_absolute():
        log_path = _REPO / log_path
    rows = _load_ticks(log_path)
    results = validate(rows)
    all_pass = all(v.get("pass") for v in results.values())
    print(json.dumps({"ticks": len(rows), "checks": results, "pass": all_pass}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
