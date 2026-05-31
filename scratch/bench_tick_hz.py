#!/usr/bin/env python3
"""Measure agent tick Hz (sync path) + tick profiler ranked spans."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT / "backend")

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("RKK_SKIP_ALL_LLM", "1")
os.environ["RKK_AGENT_LOOP_HZ"] = "0"
os.environ["RKK_TICK_RUN_LOG"] = "0"
os.environ["RKK_TICK_PROFILE"] = "1"
os.environ["RKK_TICK_PROFILE_REPORT_EVERY"] = "100000"

from engine.simulation import Simulation  # noqa: E402
from engine.tick_profiler import get_tick_profiler  # noqa: E402


def main() -> None:
    device = os.environ.get("RKK_DEVICE", "cpu")
    n = int(os.environ.get("BENCH_TICKS", "80"))
    warm = int(os.environ.get("BENCH_WARMUP", "15"))
    sim = Simulation(device_str=device, start_world="humanoid")
    get_tick_profiler()._last_report_tick = 10**9

    for _ in range(warm):
        with sim._sim_step_lock:
            sim._run_single_agent_timestep_inner()

    times: list[float] = []
    tail: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        with sim._sim_step_lock:
            sim._run_single_agent_timestep_inner()
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        if sim.tick >= 650:
            tail.append(ms)

    times.sort()
    tail.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    mx = max(times)
    hz = 1000.0 / p50 if p50 > 0 else 0.0
    print(f"device={device} ticks={n} warmup={warm} final_tick={sim.tick}")
    print(f"all:   median={p50:.1f}ms p95={p95:.1f}ms max={mx:.1f}ms  ~{hz:.2f}Hz")
    if tail:
        tp50 = tail[len(tail) // 2]
        tp95 = tail[int(len(tail) * 0.95)]
        print(
            f"tick>=650: n={len(tail)} median={tp50:.1f}ms p95={tp95:.1f}ms "
            f"~{1000.0 / tp50:.2f}Hz"
        )
    print("\nRanked profiler (top 12):")
    for i, row in enumerate(get_tick_profiler().ranked(top_n=12), 1):
        print(
            f"  {i:2d}. {row['name']:<28} ema={row['avg_ms']:7.1f}ms "
            f"pct={row['pct']:5.1f}% max={row['max_ms']:7.1f}ms"
        )


if __name__ == "__main__":
    main()
