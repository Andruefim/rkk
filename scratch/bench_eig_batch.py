"""Compare sequential W-swap vs batched forward_dynamics_batched_W."""
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

os.environ["RKK_SKIP_ALL_LLM"] = "1"
os.environ["RKK_AGENT_LOOP_HZ"] = "0"
device = os.environ.get("RKK_DEVICE", "cuda")
os.environ["RKK_DEVICE"] = device

import torch

from engine.hypothesis_testing import eig_for_action, intent_vars_in_graph
from engine.simulation import Simulation


def main() -> None:
    sim = Simulation(device_str=device, start_world="humanoid")
    g = sim.agent.graph
    ens = g._ensemble
    for _ in range(3):
        sim.tick_step()
    obs = dict(sim.agent.env.observe())
    intents = intent_vars_in_graph(g)[:12]
    candidates = [(v, 0.5) for v in intents]
    K, N = len(candidates), ens.n if ens else 1
    print(f"device={device} K={K} N={N} core.d={g._core.d}")

    os.environ["RKK_EIG_BATCH_W"] = "0"
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        eig_for_action(g, obs, candidates, return_best=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seq_ms = (time.perf_counter() - t0) / 5 * 1000

    os.environ["RKK_EIG_BATCH_W"] = "1"
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        eig_for_action(g, obs, candidates, return_best=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    bat_ms = (time.perf_counter() - t0) / 5 * 1000

    print(f"sequential (RKK_EIG_BATCH_W=0): {seq_ms:.1f} ms")
    print(f"batched    (RKK_EIG_BATCH_W=1): {bat_ms:.1f} ms")
    print(f"speedup: {seq_ms / max(bat_ms, 1e-6):.2f}x")


if __name__ == "__main__":
    main()
