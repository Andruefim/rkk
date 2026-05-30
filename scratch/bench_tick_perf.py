"""Quick tick timing: ticks 1..60 after warm-up."""
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

os.environ["RKK_DEVICE"] = "cpu"
os.environ["RKK_SKIP_ALL_LLM"] = "1"
os.environ["RKK_AGENT_LOOP_HZ"] = "0"
os.environ["RKK_TICK_RUN_LOG"] = "0"

from engine.simulation import Simulation  # noqa: E402


def main() -> None:
    sim = Simulation(device_str=os.environ.get("RKK_DEVICE", "cpu"), start_world="humanoid")
    sim._bg.ensure_rkk_agent_loop()
    # warm
    for _ in range(3):
        sim.tick_step()
    times: list[float] = []
    for t in range(1, 61):
        t0 = time.perf_counter()
        sim.tick_step()
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        if t in (1, 5, 10, 25, 26, 40, 50, 60):
            print(f"tick {t:3d}: {ms:7.1f} ms")
    p50 = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]
    mx = max(times)
    print(f"median={p50:.1f}ms p95={p95:.1f}ms max={mx:.1f}ms")


if __name__ == "__main__":
    main()
