"""Vision-loss fixes: self-slot skip, adaptive encode, soft hold, ring dump."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from engine.environment_visual import vision_encode_should_run
from engine.goal_navigation import navigation_intents_from_bearing_range
from engine.object_working_memory import (
    LatentSceneMemory,
    ObjectWorkingMemory,
    scene_sigma_max,
)
from engine.vision_loss_trace import VisionLossTrace, snapshot_from_scene
from engine.vision_resolve import (
    _mask_bbox,
    collect_vision_slots,
    is_self_vision_slot,
    resolve_visual_target,
)
from engine.vision_target import VisualTarget, bearing_from_u
from engine.visual_grounding import compute_slot_joint_overlap


def _hash_embed(text: str, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(str(text).lower())) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= float(np.linalg.norm(v) + 1e-9)
    return v


class _FakeVisual:
    def __init__(self, lexicon: dict, n: int = 2) -> None:
        self._slot_lexicon = lexicon
        self._last_attn = None
        self._last_slots = np.linspace(0.4, 0.8, n)
        self._last_slot_vecs = None


def test_is_self_vision_slot_matches_ego_grounding_and_flag() -> None:
    assert is_self_vision_slot({"label": "[EGO] left knee"})
    assert is_self_vision_slot({"source": "grounding", "label": "left knee"})
    assert is_self_vision_slot({"self_slot": True, "label": "hand"})
    assert not is_self_vision_slot({"label": "chair", "source": "vlm"})


def test_collect_vision_slots_skips_grounding_and_ego() -> None:
    env = _FakeVisual(
        {
            "slot_0": {
                "label": "left knee",
                "source": "grounding",
                "self_slot": True,
                "confidence": 0.9,
            },
            "slot_1": {"label": "chair", "source": "vlm", "confidence": 0.8},
        },
        n=2,
    )
    slots = collect_vision_slots(env)
    ids = {s["slot_id"] for s in slots}
    assert "slot_0" not in ids
    assert "slot_1" in ids


def test_collect_vision_slots_skips_existing_ego_without_grounding_flag() -> None:
    env = _FakeVisual(
        {
            "slot_0": {"label": "[EGO] left arm", "confidence": 0.7},
            "slot_1": {"label": "box", "confidence": 0.6},
        },
        n=2,
    )
    slots = collect_vision_slots(env)
    ids = {s["slot_id"] for s in slots}
    assert "slot_0" not in ids
    assert "slot_1" in ids


def test_resolve_skips_self_slots_passed_directly() -> None:
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.4,
            "v": 0.4,
            "label": "[EGO] left knee",
            "activation": 0.9,
            "source": "grounding",
            "self_slot": True,
            "uv_valid": True,
            "mask_peakiness": 3.0,
            "match_score": 0.9,
        }
    ]
    target, diag = resolve_visual_target(
        "approach the chair",
        slots=slots,
        embed_fn=_hash_embed,
        require_range=False,
    )
    assert target is None
    assert diag.get("reason") == "no_vision_slots"


def test_mask_bbox_and_bearing_uv_convention() -> None:
    assert bearing_from_u(0.5) == pytest.approx(0.0, abs=1e-6)
    assert bearing_from_u(1.0) > 0.0
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:5, 4:7] = 1.0
    bbox = _mask_bbox(mask, thresh=0.25)
    assert bbox is not None
    u_min, v_min, u_max, v_max = bbox
    assert u_min < u_max
    assert v_min < v_max
    assert 0.4 < (u_min + u_max) / 2 < 0.9


def test_slot_joint_overlap_uses_same_uv_as_bearing_from_u() -> None:
    import torch

    masks = torch.zeros(1, 8, 8)
    masks[0, 3:5, 3:5] = 1.0  # center ~ (0.5, 0.5)
    overlap = compute_slot_joint_overlap(
        masks, {"left_knee": (0.5, 0.5)}, dist_threshold=0.22
    )
    assert overlap[0]["left_knee"] > 0.8
    assert bearing_from_u(0.5) == pytest.approx(0.0, abs=1e-6)


def test_adaptive_encode_rare_when_still_fires_on_yaw_jump() -> None:
    assert vision_encode_should_run(
        2,
        pending_dtheta=0.0,
        last_encode_stride=1,
        encode_every=48,
        turn_rad=0.08,
        min_every=4,
    ) is False
    assert vision_encode_should_run(
        49,
        pending_dtheta=0.0,
        last_encode_stride=1,
        encode_every=48,
        min_every=4,
    ) is True
    assert vision_encode_should_run(
        5,
        pending_dtheta=0.10,
        last_encode_stride=1,
        encode_every=48,
        turn_rad=0.08,
        min_every=4,
    ) is True
    assert vision_encode_should_run(
        3,
        pending_dtheta=0.20,
        last_encode_stride=1,
        encode_every=48,
        turn_rad=0.08,
        min_every=4,
    ) is False


def test_hard_lock_odom_does_not_refresh_last_vision_tick() -> None:
    owm = ObjectWorkingMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="object",
        confidence=0.8,
        bearing=0.0,
        range_m=2.0,
        range_conf=0.9,
    )
    owm.bind_from_visual(vt, tick=10, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    assert owm.last_vision_tick == 10
    owm.observe_vision(
        None,
        tick=11,
        agent_xy=(0.5, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert owm.last_vision_tick == 10
    assert owm.holding is True
    assert owm.x_fwd == pytest.approx(1.5, abs=0.05)
    assert owm.is_usable(11)
    assert owm.bearing_sigma > 0.04


def test_sigma_grows_under_lock_without_live_uv_then_shrinks() -> None:
    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="object",
        confidence=0.9,
        bearing=0.0,
        range_m=2.0,
    )
    scene.bind_visual_target(
        vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0)
    )
    ent = scene.active()
    assert ent is not None
    sigma0 = float(ent.bearing_sigma)
    xy = (0.0, 0.0)
    fwd = (1.0, 0.0)
    last_usable_tick = 1
    for t in range(2, 90):
        yaw = 0.12 * (t - 1)
        fwd = (math.cos(yaw), math.sin(yaw))
        scene.update(
            tick=t,
            percepts=[],
            agent_xy=xy,
            agent_forward=fwd,
        )
        ent = scene.active()
        assert ent is not None
        if ent.is_usable(t):
            last_usable_tick = t
    ent = scene.active()
    assert ent is not None
    assert float(ent.bearing_sigma) > sigma0
    assert float(ent.bearing_sigma) >= scene_sigma_max() or not ent.is_usable(89)
    assert last_usable_tick < 89
    scene.release_hard_lock()
    grown = float(ent.bearing_sigma)
    scene.update(
        tick=90,
        percepts=[
            {
                "slot_id": "slot_0",
                "bearing": float(ent.bearing),
                "range_m": float(ent.range_m),
                "confidence": 0.9,
                "activation": 0.9,
                "u": 0.5,
                "v": 0.5,
            }
        ],
        agent_xy=xy,
        agent_forward=fwd,
    )
    ent2 = scene.active()
    assert ent2 is not None
    assert float(ent2.bearing_sigma) < grown
    assert ent2.last_live_uv_tick == 90


def test_nav_widens_deadzone_with_sigma_does_not_pause() -> None:
    sharp = navigation_intents_from_bearing_range(
        0.15, 2.0, 0.55, posture_stability=0.9, bearing_sigma=0.0
    )
    soft = navigation_intents_from_bearing_range(
        0.15, 2.0, 0.55, posture_stability=0.9, bearing_sigma=0.4
    )
    assert sharp and soft
    assert abs(float(soft["intent_gait_coupling"]) - 0.5) <= abs(
        float(sharp["intent_gait_coupling"]) - 0.5
    ) + 1e-9


def test_loss_ring_dumps_on_usable_true_to_false(tmp_path: Path) -> None:
    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="object",
        confidence=0.9,
        bearing=0.0,
        range_m=2.0,
    )
    scene.bind_visual_target(
        vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0)
    )
    trace = VisionLossTrace(maxlen=16)
    prev = True
    dumped = None
    xy = (0.0, 0.0)
    for t in range(2, 80):
        yaw = 0.15 * (t - 1)
        fwd = (math.cos(yaw), math.sin(yaw))
        scene.update(tick=t, percepts=[], agent_xy=xy, agent_forward=fwd)
        snap = snapshot_from_scene(tick=t, scene=scene, slots=[], dtheta=0.15)
        trace.push(snap)
        now = bool(scene.active() and scene.active().is_usable(t))
        if prev and not now:
            dumped = trace.dump_to_dir(tmp_path / f"loss_tick_{t}")
            break
        prev = now
    assert dumped is not None
    assert dumped.exists()
    text = dumped.read_text(encoding="utf-8")
    assert "bearing_sigma" in text
    assert "owm" in text
