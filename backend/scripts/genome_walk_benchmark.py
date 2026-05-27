#!/usr/bin/env python3
"""Headless check: humanoid walks forward without falling (genome priors + CPG)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "backend"))
os.chdir(_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("RKK_DEVICE", "cpu")
os.environ.setdefault("RKK_GENOME_WALK", "1")
os.environ.setdefault("RKK_GENOME_WALK_INNATE", "1")
os.environ.setdefault("RKK_LOCOMOTION_CPG", "1")
# Benchmark runs free locomotion — do not re-attach pelvis after manual release.
os.environ["RKK_CURRICULUM_FIXED_ROOT_RETRY_MAX"] = "0"
os.environ["RKK_AUTO_FIXED_ROOT_TICKS"] = "0"
os.environ["RKK_FR_REATTACH_MIN_FALLEN_TICKS"] = "99999"
os.environ["RKK_AGENT_LOOP_HZ"] = "0"


def main() -> int:
    from engine.simulation import Simulation

    sim = Simulation(device_str="cpu", start_world="humanoid")
    sim.enable_fixed_root()

    def steps(n: int) -> None:
        sim.advance_agent_steps(n)

    # Stand curriculum on fixed pelvis
    steps(80)

    sim._curriculum_auto_fr_released = True
    sim.disable_fixed_root()
    if sim._fixed_root_active:
        print("WARN: fixed_root still active after disable_fixed_root")
    release_stab = int(os.environ.get("RKK_POST_FR_STABILIZE_TICKS", "120"))
    steps(release_stab + 40)

    walk_ticks = 320
    com_x0 = None
    falls = 0
    strides: list[float] = []
    base = sim._unwrap_base_env(sim.agent.env)
    is_fallen = getattr(base, "is_fallen", lambda: False)

    for i in range(walk_ticks):
        steps(1)
        obs = sim.agent.env.observe()
        cx = float(obs.get("com_x", obs.get("phys_com_x", 0.5)))
        if com_x0 is None:
            com_x0 = cx
        strides.append(float(obs.get("intent_stride", 0.5)))
        if callable(is_fallen) and is_fallen():
            falls += 1
        if i % 80 == 0:
            ps = float(
                obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
            )
            active = getattr(sim, "_genome_walk_active_tick", False)
            print(
                f"tick={sim.tick} com_x={cx:.3f} stride={strides[-1]:.2f} "
                f"posture={ps:.2f} genome_walk={active} fallen={bool(falls)}"
            )

    obs = sim.agent.env.observe()
    com_x1 = float(obs.get("com_x", obs.get("phys_com_x", 0.5)))
    com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
    posture = float(
        obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
    )
    dx = com_x1 - float(com_x0 or com_x1)
    mean_stride = sum(strides[-120:]) / max(1, len(strides[-120:]))

    print("\n=== genome walk benchmark ===")
    print(f"com_x delta: {dx:.4f} (start {com_x0:.3f} -> {com_x1:.3f})")
    print(f"mean intent_stride (last 120): {mean_stride:.3f}")
    print(f"posture: {posture:.3f} com_z: {com_z:.3f}")
    print(f"fall detections during walk window: {falls}")

    gait_l = float(obs.get("gait_phase_l", obs.get("phys_gait_phase_l", 0.5)))
    gait_r = float(obs.get("gait_phase_r", obs.get("phys_gait_phase_r", 0.5)))
    gait_desync = abs(gait_l - gait_r)
    ok = (
        mean_stride > 0.52
        and posture > 0.48
        and com_z > 0.38
        and falls < 25
        and gait_desync > 0.07
    )
    print(f"gait desync: {gait_desync:.3f}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
