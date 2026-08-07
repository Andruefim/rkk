#!/usr/bin/env python3
"""
Демонстрация того, что зрительная кора действительно учится: на синтетической
сцене с объектами меряем recon loss и «пиковость» слот-масок до и после обучения.

    cd backend && python3 scripts/check_vision_learning.py --steps 400

Пиковость (peak/mean маски) — тот самый порог, по которому `vision_resolve`
решает, годится ли слот как цель команды (RKK_SLOT_MASK_PEAKINESS_MIN, 1.8).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def _scene(rng: np.random.Generator, h: int = 216, w: int = 288) -> np.ndarray:
    """Кадр: серый фон + 3 цветных объекта в случайных местах."""
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    img[int(h * 0.72) :, :] = 130  # «пол»
    colors = [(220, 60, 50), (60, 200, 90), (70, 110, 240)]
    for color in colors:
        bh, bw = rng.integers(28, 52), rng.integers(28, 52)
        y = int(rng.integers(10, h - bh - 10))
        x = int(rng.integers(10, w - bw - 10))
        img[y : y + bh, x : x + bw] = color
    return img


def _measure(cortex, frames: list[np.ndarray]) -> tuple[float, float, float]:
    import torch.nn.functional as F

    def _peak(m: torch.Tensor) -> float:
        f = m.reshape(m.shape[0], -1).float()
        f = f / f.sum(-1, keepdim=True).clamp_min(1e-8)
        return float((f.max(-1).values / f.mean(-1)).mean())

    peaks: list[float] = []
    alphas: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for f in frames:
            cortex.encode(f)
            peaks.append(cortex.mask_peakiness())
            x = cortex.preprocess(f)
            slots, _ = cortex.attention(cortex.encoder(x))
            recon, alpha = cortex.decoder(slots)
            alphas.append(_peak(alpha.squeeze(0)))
            losses.append(float(F.mse_loss(recon, cortex._recon_target(x))))
    return float(np.mean(losses)), float(np.mean(peaks)), float(np.mean(alphas))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    from engine.causal_vision import make_visual_cortex

    cortex = make_visual_cortex(torch.device("cpu"), n_slots=6)
    frames = [_scene(rng) for _ in range(args.frames)]
    holdout = [_scene(rng) for _ in range(6)]

    loss0, peak0, alpha0 = _measure(cortex, holdout)
    print(f"before: recon_loss={loss0:.5f} attn_peakiness={peak0:.3f} alpha_peakiness={alpha0:.1f}")

    for f in frames:
        cortex.remember_frame(f)
    for i in range(args.steps):
        cortex.train_reconstruction(batch_size=4, steps=1)
        if (i + 1) % max(1, args.steps // 4) == 0:
            l, p, a = _measure(cortex, holdout)
            print(f"  step {i + 1:4d}: recon_loss={l:.5f} attn_peakiness={p:.3f} alpha_peakiness={a:.1f}")

    loss1, peak1, alpha1 = _measure(cortex, holdout)
    print(f"after:  recon_loss={loss1:.5f} attn_peakiness={peak1:.3f} alpha_peakiness={alpha1:.1f}")
    print(f"\nrecon loss:     {loss0:.5f} -> {loss1:.5f}")
    print(f"attn peakiness: {peak0:.3f} -> {peak1:.3f} (порог vision_resolve = 1.8)")
    print(f"alpha peakiness:{alpha0:.1f} -> {alpha1:.1f}")
    ok = loss1 < loss0 and peak1 > max(1.8, peak0 * 1.5)
    print("RESULT:", "OK" if ok else "NO LEARNING")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
