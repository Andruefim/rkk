"""Time eig_for_action with K candidates × N ensemble."""
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

from engine.hypothesis_testing import eig_for_action, intent_vars_in_graph
from engine.simulation import Simulation


def main() -> None:
    sim = Simulation(device_str="cpu", start_world="humanoid")
    g = sim.agent.graph
    print(f"[GRAPH] d={g._d} nodes={len(g._node_ids)} edges={g.edge_count}")
    ens = getattr(g, "_ensemble", None)
    print(f"[ENSEMBLE] n={ens.n if ens else 0}")

    for _ in range(5):
        sim.tick_step()

    obs = dict(sim.agent.env.observe())
    intents = intent_vars_in_graph(g)[:12]
    candidates = [(v, 0.5) for v in intents]
    k = len(candidates)
    n = ens.n if ens else 1

    t0 = time.perf_counter()
    for _ in range(3):
        eig_for_action(g, obs, candidates)
    ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"eig_for_action K={k} N={n}: ~{ms:.1f}ms/call (~{k*n} core forwards)")

    t0 = time.perf_counter()
    eig_for_action(g, obs, candidates, return_best=True)
    print(f"return_best: {(time.perf_counter()-t0)*1000:.1f}ms")


if __name__ == "__main__":
    main()
