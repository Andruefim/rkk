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


def _alpha_peak(cortex, frames) -> float:
    """Пиковость альфа-масок декодера — прямой признак того, что слоты делят сцену."""
    peaks = []
    with torch.no_grad():
        for f in frames:
            slots, _ = cortex.attention(cortex.encoder(cortex.preprocess(f)))
            _, alpha = cortex.decoder(slots)
            a = alpha.squeeze(0).reshape(alpha.shape[1], -1).float()
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
            peaks.append(float((a.max(-1).values / a.mean(-1)).mean()))
    return float(np.mean(peaks))


def test_reconstruction_training_sharpens_slot_masks():
    torch.manual_seed(0)
    np.random.seed(0)  # train_reconstruction сэмплирует батч глобальным numpy RNG
    rng = np.random.default_rng(0)
    cortex = make_visual_cortex(torch.device("cpu"), n_slots=4)
    frames = [_scene(rng) for _ in range(8)]
    holdout = [_scene(rng) for _ in range(2)]

    peak_before = _mask_peak(cortex, holdout)
    alpha_before = _alpha_peak(cortex, holdout)
    for f in frames:
        cortex.remember_frame(f)

    losses = []
    for _ in range(220):
        loss = cortex.train_reconstruction(batch_size=2, steps=1)
        if loss is not None:
            losses.append(loss)

    assert len(losses) > 200
    assert np.mean(losses[-20:]) < 0.75 * np.mean(losses[:20])

    # Слоты перестают быть взаимозаменяемыми и делят кадр между собой.
    assert _alpha_peak(cortex, holdout) > max(5.0, alpha_before * 3.0)
    # Attention-маски (их читает vision_resolve) тоже сходятся к пикам, но медленнее:
    # порог 1.8 достигается за сотни шагов, см. scripts/pretrain_vision.py.
    assert _mask_peak(cortex, holdout) > peak_before
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


def test_vision_config_scales_from_env(monkeypatch):
    from engine.causal_vision import vision_config_from_env

    monkeypatch.setenv("RKK_VISION_FRAME_H", "130")
    monkeypatch.setenv("RKK_VISION_FRAME_W", "128")
    monkeypatch.setenv("RKK_VISION_CNN_CHANNELS", "32,64,64")
    monkeypatch.setenv("RKK_VISION_ITERS", "3")
    monkeypatch.setenv("RKK_VISION_SLOT_DIM", "96")
    monkeypatch.setenv("RKK_VISION_FORCE_GPU_PROFILE", "1")

    cfg = vision_config_from_env(n_slots=5)
    # Разрешение округляется вниз до кратного 4: энкодер отдаёт сетку H/4 × W/4.
    assert (cfg.frame_h, cfg.frame_w) == (128, 128)
    assert cfg.cnn_channels == [32, 64, 64]
    assert cfg.n_iters == 3 and cfg.slot_dim == 96 and cfg.n_slots == 5

    # CPU: shrink frame / iters (checkpoint-safe). Keep explicit CNN channels —
    # those tensors do not transfer across 16,32,32 vs 32,64,64.
    monkeypatch.setenv("RKK_VISION_FORCE_GPU_PROFILE", "0")
    monkeypatch.setenv("RKK_DEVICE", "cpu")
    cfg_cpu = vision_config_from_env(n_slots=5, device=torch.device("cpu"))
    assert (cfg_cpu.frame_h, cfg_cpu.frame_w) == (64, 64)
    assert cfg_cpu.cnn_channels == [32, 64, 64]
    assert cfg_cpu.n_iters == 2

    monkeypatch.delenv("RKK_VISION_CNN_CHANNELS", raising=False)
    monkeypatch.delenv("RKK_VISION_FRAME_H", raising=False)
    monkeypatch.delenv("RKK_VISION_FRAME_W", raising=False)
    monkeypatch.delenv("RKK_VISION_ITERS", raising=False)
    cfg_cpu_default = vision_config_from_env(n_slots=5, device=torch.device("cpu"))
    assert cfg_cpu_default.cnn_channels == [16, 32, 32]
    assert (cfg_cpu_default.frame_h, cfg_cpu_default.frame_w) == (64, 64)


def test_cpu_cam_and_recon_follow_env_at_call_time(monkeypatch):
    from engine.causal_vision import recon_batch, recon_steps
    from engine.environment_visual import _vision_cam_h, _vision_cam_w

    monkeypatch.setenv("RKK_DEVICE", "cpu")
    monkeypatch.setenv("RKK_VISION_FORCE_GPU_PROFILE", "0")
    monkeypatch.setenv("RKK_VISION_CAM_W", "384")
    monkeypatch.setenv("RKK_VISION_CAM_H", "288")
    monkeypatch.setenv("RKK_VISION_RECON_BATCH", "8")
    monkeypatch.setenv("RKK_VISION_RECON_STEPS", "8")
    assert _vision_cam_w() == 288
    assert _vision_cam_h() == 216
    assert recon_batch(torch.device("cpu")) == 2
    assert recon_steps(torch.device("cpu")) == 2

    monkeypatch.setenv("RKK_VISION_FORCE_GPU_PROFILE", "1")
    assert _vision_cam_w() == 384
    assert _vision_cam_h() == 288
    assert recon_batch(torch.device("cpu")) == 8
    assert recon_steps(torch.device("cpu")) == 8


def test_higher_resolution_cortex_trains(monkeypatch):
    from engine.causal_vision import make_visual_cortex as make

    monkeypatch.setenv("RKK_VISION_FRAME_H", "128")
    monkeypatch.setenv("RKK_VISION_FRAME_W", "128")
    monkeypatch.setenv("RKK_VISION_FORCE_GPU_PROFILE", "1")
    cortex = make(torch.device("cpu"), n_slots=3)
    assert cortex.cfg.frame_h == 128

    rng = np.random.default_rng(2)
    for _ in range(4):
        cortex.remember_frame(_scene(rng))
    loss = cortex.train_reconstruction(batch_size=2, steps=1)
    assert loss is not None and loss > 0.0

    vals, vecs, attn = cortex.encode(_scene(rng))
    assert attn.shape == (3, 32, 32)


def test_checkpoint_transfers_across_resolutions(monkeypatch):
    from engine.causal_vision import make_visual_cortex as make
    from engine.checkpoint_modules import _load_compatible  # noqa: PLC2701

    monkeypatch.setenv("RKK_VISION_FORCE_GPU_PROFILE", "1")
    monkeypatch.setenv("RKK_VISION_FRAME_H", "64")
    monkeypatch.setenv("RKK_VISION_FRAME_W", "64")
    small = make(torch.device("cpu"), n_slots=3)
    with torch.no_grad():
        for p in small.parameters():
            p.mul_(0.5).add_(0.01)

    monkeypatch.setenv("RKK_VISION_FRAME_H", "128")
    monkeypatch.setenv("RKK_VISION_FRAME_W", "128")
    monkeypatch.setenv("RKK_VISION_FORCE_GPU_PROFILE", "1")
    big = make(torch.device("cpu"), n_slots=3)
    loaded = _load_compatible(big, {k: v.clone() for k, v in small.state_dict().items()})

    assert loaded > 0
    assert torch.allclose(
        big.encoder.backbone[0].weight, small.encoder.backbone[0].weight
    )
    # Позиционные сетки зависят от разрешения и остаются пересозданными под новый размер.
    assert big.encoder.pos_grid.shape[-1] == 32
    big.encode(_scene(np.random.default_rng(5)))


def test_decoder_weights_are_part_of_checkpoint():
    from engine.checkpoint_modules import pack_learnable_modules

    sim = type("S", (), {})()
    sim.agent = type("A", (), {})()
    sim._visual_env = type("V", (), {})()
    sim._visual_env.cortex = make_visual_cortex(torch.device("cpu"), n_slots=4)

    packed = pack_learnable_modules(sim)
    keys = packed["sections"]["vision_cortex"]["modules"]["."].keys()
    assert any(k.startswith("decoder.") for k in keys)
