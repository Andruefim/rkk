#!/usr/bin/env python3
"""
Wave A: analyze System2 distill JSONL — success rates, blend readiness hints.

Usage (from repo root):
  python backend/tools/analyze_s2_distill.py
  python backend/tools/analyze_s2_distill.py --path backend/logs/system2_distill.jsonl --window 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from engine.system2.distill_log import (  # noqa: E402
    analyze_distill_file,
    blend_ready_student_conf,
    distill_log_path,
    distill_min_success_rate,
    distill_recover_min_success_rate,
)


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze system2_distill.jsonl")
    ap.add_argument(
        "--path",
        type=Path,
        default=None,
        help="JSONL path (default: RKK_SYSTEM2_DISTILL_LOG or logs/system2_distill.jsonl)",
    )
    ap.add_argument("--window", type=int, default=None, help="Last N rows")
    ap.add_argument("--json", action="store_true", help="Print raw report JSON")
    args = ap.parse_args()

    path = args.path or distill_log_path()
    if not path.is_absolute():
        candidates = [_BACKEND / path, _REPO / path, path]
        path = next((c for c in candidates if c.is_file()), _BACKEND / path)

    report = analyze_distill_file(path, window=args.window)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"Distill report: {report['path']}")
    print(f"  rows (window): {report['rows']}")
    print(f"  overall success: {_pct(report.get('overall_success_rate'))}")
    print(f"  llm share: {_pct(report.get('llm_share'))}")
    print(f"  student share: {_pct(report.get('student_share'))}")

    print("\nBy macro:")
    for k, v in (report.get("by_macro") or {}).items():
        print(f"  {k}: {_pct(v)}")

    print("\nBy source:")
    for k, v in (report.get("by_source") or {}).items():
        print(f"  {k}: {_pct(v)}")

    rec = report.get("recover") or {}
    print("\nRECOVER / fallen_override:")
    print(f"  count: {rec.get('count', 0)}")
    print(f"  success rate: {_pct(rec.get('success_rate'))}")
    print(f"  median d_com_z: {rec.get('median_d_com_z')}")
    print(f"  median d_posture: {rec.get('median_d_posture')}")
    print(f"  PE pass rate: {_pct(rec.get('pe_pass_rate'))}")

    health = report.get("health") or {}
    print("\nBlend readiness (rolling tracker):")
    print(f"  distill_blend_ready: {health.get('distill_blend_ready')}")
    print(f"  recover conf median: {health.get('distill_recover_conf_median')}")
    print(f"  quality_warn: {health.get('distill_quality_warn')}")

    print("\nThresholds (.env):")
    print(f"  RKK_S2_DISTILL_MIN_SUCCESS_RATE={distill_min_success_rate()}")
    print(f"  RKK_S2_DISTILL_RECOVER_MIN_SUCCESS_RATE={distill_recover_min_success_rate()}")
    print(f"  RKK_S2_DISTILL_BLEND_READY_STUDENT_CONF={blend_ready_student_conf()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
