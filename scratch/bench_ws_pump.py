"""Simulate WS pump: read agent cache at 15Hz (no tick_step/sanitize)."""
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
os.environ["RKK_AGENT_LOOP_HZ"] = "15"
os.environ["RKK_TICK_RUN_LOG"] = "0"

from engine.json_util import sanitize_for_json  # noqa: E402
from engine.simulation import Simulation  # noqa: E402


def main() -> None:
    sim = Simulation(device_str="cpu", start_world="humanoid")
    sim._bg.ensure_rkk_agent_loop()
    time.sleep(2.0)
    hz = 15.0
    period = 1.0 / hz
    n = 90
    t_read: list[float] = []
    t_bad: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        with sim._sim_step_lock:
            payload = sim._agent_step_response
        t_read.append((time.perf_counter() - t0) * 1000.0)
        if payload and i % 3 == 0:
            t1 = time.perf_counter()
            sanitize_for_json(payload)
            t_bad.append((time.perf_counter() - t1) * 1000.0)
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    print(f"cache read: median={sorted(t_read)[n//2]:.2f}ms max={max(t_read):.2f}ms")
    if t_bad:
        print(f"re-sanitize(30x): median={sorted(t_bad)[len(t_bad)//2]:.1f}ms max={max(t_bad):.1f}ms")


if __name__ == "__main__":
    main()
