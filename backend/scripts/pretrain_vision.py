#!/usr/bin/env python3
"""
Претрейн зрительной коры на кадрах реальной симуляции.

Онлайн кора учится медленно: полный рендер идёт раз в RKK_VISION_ENCODE_EVERY
интервенций. Скрипт собирает кадры из работающей симуляции и прогоняет по ним
много шагов self-supervised реконструкции, после чего сохраняет веса в чекпоинт —
дальше гуманоид стартует с уже сегментирующим зрением.

    cd backend && RKK_DEVICE=cpu python3 scripts/pretrain_vision.py --ticks 200 --steps 600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def _peakiness(cortex, frames) -> float:
    peaks = []
    with torch.no_grad():
        for f in frames:
            cortex.encode(f)
            peaks.append(cortex.mask_peakiness())
    return float(np.mean(peaks)) if peaks else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200, help="тиков симуляции для сбора кадров")
    ap.add_argument("--steps", type=int, default=600, help="шагов обучения реконструкции")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--no-save", action="store_true", help="не писать чекпоинт")
    args = ap.parse_args()

    from engine.simulation import Simulation

    sim = Simulation()
    vis = getattr(sim, "_visual_env", None)
    if vis is None or getattr(vis, "cortex", None) is None:
        print("[pretrain] визуальная среда выключена — включите RKK_AUTO_VISUAL=1")
        return 1
    cortex = vis.cortex

    frames: list[np.ndarray] = []
    for i in range(args.ticks):
        sim.tick_step()
        frame = vis._get_raw_frame()
        if frame is not None and i % 2 == 0:
            frames.append(frame)
            cortex.remember_frame(frame)
    if len(frames) < 4:
        print(f"[pretrain] собрано мало кадров: {len(frames)}")
        return 1
    holdout = frames[-4:]
    print(f"[pretrain] кадров собрано: {len(frames)}")

    peak0 = _peakiness(cortex, holdout)
    loss0 = float(np.mean(list(cortex.recon_losses)[-20:])) if cortex.recon_losses else 0.0
    print(f"[pretrain] before: recon_loss={loss0:.5f} mask_peakiness={peak0:.3f}")

    report_every = max(1, args.steps // 6)
    for i in range(args.steps):
        cortex.train_reconstruction(batch_size=args.batch, steps=1)
        if (i + 1) % report_every == 0:
            loss = float(np.mean(list(cortex.recon_losses)[-30:]))
            print(
                f"[pretrain] step {i + 1:5d}: recon_loss={loss:.5f} "
                f"mask_peakiness={_peakiness(cortex, holdout):.3f}",
                flush=True,
            )

    peak1 = _peakiness(cortex, holdout)
    loss1 = float(np.mean(list(cortex.recon_losses)[-30:]))
    print(f"[pretrain] after:  recon_loss={loss1:.5f} mask_peakiness={peak1:.3f}")

    if not args.no_save:
        out = sim.memory_save()
        print(f"[pretrain] чекпоинт: ok={out.get('ok')} path={out.get('path')}")
    sim.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
