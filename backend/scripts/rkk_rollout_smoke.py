#!/usr/bin/env python3
"""
Cross-cutting rollout checklist (Strong AI plan): print phase env flags and optional
baseline placeholders. Run before/after a milestone to compare numbers manually or in CI.

  cd backend && set PYTHONPATH=. && python scripts/rkk_rollout_smoke.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    flags = [
        "RKK_PEARL_API",
        "RKK_PRECISION_GROUPS",
        "RKK_PRECISION_CALIB_WINDOW",
        "RKK_VALUE_VETO",
        "RKK_PEARL_CONTEXT",
        "RKK_SZ_SPLIT",
        "RKK_SZ_FREEZE_S_STEPS",
        "RKK_EFFERENCE_COPY",
        "RKK_TEMPORAL_PRECISION_ROUTING",
        "RKK_HIERARCHICAL_AI",
        "RKK_SLEEP_REM_SURPRISE_SORT",
    ]
    out = sys.stdout
    out.write("# RKK phase flags (unset = default in code)\n")
    for f in flags:
        v = os.environ.get(f)
        out.write(f"{f}={v!r}\n")
    out.write(
        "\n# Baseline slots (fill after a 500+ tick dry run): "
        "mean_pe, mean_posture_stability, mean_tick_s, falls_per_N, …\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
