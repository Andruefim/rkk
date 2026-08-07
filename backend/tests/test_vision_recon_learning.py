"""Зрительная кора должна реально обучаться: recon-лосс падает, маски становятся пиковыми."""
from __future__ import annotations

import numpy as np
import torch

from engine.causal_vision import make_visual_cortex


def _scene(rng: np.random.Generator, h: int = 128, w: int = 160) -> np.ndarray:
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    img[int(h * 0.72) :, :] = 130
    for color in ((220, 60, 50), (60, 200, 90), (70, 110, 240)):
        bh, bw = int(rng.integers(24, 40)), int(rng.integers(24, 40))
        y = int(rng.integers(6, h - bh - 6))
        x = int(rng.integers(6, w - bw - 6))
        img[y : y + bh, x : x + bw] = color
    return img


def _mask_peak(cortex, frames) -> float:
    peaks = []
    for f in frames:
        cortex.encode(f)
        peaks.append(cortex.mask_peakiness())
    return float(np.mean(peaks))


def test_reconstruction_training_sharpens_slot_masks():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cortex = make_visual_cortex(torch.device("cpu"), n_slots=4)
    frames = [_scene(rng) for _ in range(8)]
    holdout = [_scene(rng) for _ in range(2)]

    peak_before = _mask_peak(cortex, holdout)
    for f in frames:
        cortex.remember_frame(f)

    losses = []
    for _ in range(220):
        loss = cortex.train_reconstruction(batch_size=2, steps=1)
        if loss is not None:
            losses.append(loss)

    assert len(losses) > 200
    assert np.mean(losses[-20:]) < 0.75 * np.mean(losses[:20])

    peak_after = _mask_peak(cortex, holdout)
    assert peak_after > peak_before * 1.25
    assert cortex.n_recon_train == len(losses)


def test_snapshot_exposes_vision_learning_metrics():
    cortex = make_visual_cortex(torch.device("cpu"), n_slots=4)
    rng = np.random.default_rng(1)
    for _ in range(4):
        cortex.remember_frame(_scene(rng))
    cortex.train_reconstruction(batch_size=2, steps=1)
    cortex.encode(_scene(rng))

    snap = cortex.snapshot()
    assert snap["n_recon_train"] >= 1
    assert snap["mean_recon_loss"] > 0.0
    assert snap["mask_peakiness"] > 0.0


def test_decoder_weights_are_part_of_checkpoint():
    from engine.checkpoint_modules import pack_learnable_modules

    sim = type("S", (), {})()
    sim.agent = type("A", (), {})()
    sim._visual_env = type("V", (), {})()
    sim._visual_env.cortex = make_visual_cortex(torch.device("cpu"), n_slots=4)

    packed = pack_learnable_modules(sim)
    keys = packed["sections"]["vision_cortex"]["modules"]["."].keys()
    assert any(k.startswith("decoder.") for k in keys)
