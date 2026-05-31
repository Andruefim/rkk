#!/usr/bin/env python3
"""Print ranked tick profile from a running backend (GET /api/tick_profile)."""
from __future__ import annotations

import json
import sys
import urllib.request


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/api/tick_profile"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except Exception as exc:
        print(f"Failed to fetch {url}: {exc}", file=sys.stderr)
        return 1
    if not data.get("enabled"):
        print("Tick profiler disabled (set RKK_TICK_PROFILE=1 and restart backend).")
        return 2
    print(
        f"tick={data.get('tick')}  wall={data.get('last_wall_ms')}ms  "
        f"~{data.get('effective_hz')}Hz  window={data.get('window_ticks')}/{data.get('window_max')}"
    )
    print("\nLast tick spans (ms):")
    for name, ms in (data.get("last_tick_spans_ms") or {}).items():
        print(f"  {name}: {ms}")
    print("\nRanked (EMA, by avg_ms):")
    for i, row in enumerate(data.get("ranked") or [], 1):
        print(
            f"  {i:2d}. {row['name']:<28} ema={row['avg_ms']:7.1f}ms  "
            f"pct={row['pct']:5.1f}%  max={row['max_ms']:7.1f}ms  n={row['count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
