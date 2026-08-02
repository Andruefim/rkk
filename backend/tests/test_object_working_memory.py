"""Object working memory + egocentric navigation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine.goal_navigation import navigation_intents_from_ego_xy
from engine.object_working_memory import (
    LatentSceneMemory,
    ObjectWorkingMemory,
    SceneEntity,
    bearing_range_from_ego,
    ego_from_bearing_range,
)
from engine.vision_target import VisualTarget


def test_ego_bearing_roundtrip() -> None:
    xf, yr = ego_from_bearing_range(0.0, 2.0)
    assert xf == pytest.approx(2.0, abs=1e-5)
    assert yr == pytest.approx(0.0, abs=1e-5)
    b, r = bearing_range_from_ego(xf, yr)
    assert b == pytest.approx(0.0, abs=1e-5)
    assert r == pytest.approx(2.0, abs=1e-5)


def test_owm_ema_and_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_OWM_EMA_ALPHA", "0.5")
    monkeypatch.setenv("RKK_OWM_HOLD_TICKS", "20")
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
    assert owm.range_m == pytest.approx(2.0)
    assert owm.is_usable(10)
    assert owm.scene.hard_lock_active is True

    # Move forward 0.5 m — ego x should shrink (hard-lock odometry)
    owm.observe_vision(
        None,
        tick=11,
        agent_xy=(0.5, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert owm.holding is True
    assert owm.x_fwd == pytest.approx(1.5, abs=0.05)
    assert owm.is_usable(11)

    # Vision EMA fuse requires releasing approach hard-lock
    owm.scene.release_hard_lock()
    vt2 = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="object",
        confidence=0.85,
        bearing=0.0,
        range_m=1.0,
        range_conf=0.9,
    )
    owm.observe_vision(
        vt2,
        tick=12,
        agent_xy=(0.5, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert owm.holding is False
    assert 1.0 < owm.range_m < 1.6  # EMA between prev and 1.0


def test_owm_yaw_odometry() -> None:
    owm = ObjectWorkingMemory()
    vt = VisualTarget(
        slot_id="slot_1",
        u=0.5,
        v=0.5,
        label="object",
        confidence=0.9,
        bearing=0.0,
        range_m=3.0,
        range_conf=0.9,
    )
    owm.bind_from_visual(vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    # Turn 90° left (CCW): forward becomes +Y
    owm.observe_vision(
        None,
        tick=2,
        agent_xy=(0.0, 0.0),
        agent_forward=(0.0, 1.0),
    )
    # Target was ahead; after left turn it lies to the agent's left (−y_right)
    assert owm.y_right < -1.0
    assert abs(owm.x_fwd) < 1.0


def test_nav_from_ego_xy() -> None:
    intents = navigation_intents_from_ego_xy(2.0, 0.0, stop_distance=0.55, posture_stability=0.9)
    assert "intent_stride" in intents
    assert intents.get("vision_range_m") == pytest.approx(2.0, abs=0.05)
    # Already close
    assert navigation_intents_from_ego_xy(0.3, 0.0, stop_distance=0.55) == {}


def test_owm_graph_payload() -> None:
    owm = ObjectWorkingMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.6,
        v=0.5,
        label="object",
        confidence=0.7,
        bearing=0.2,
        range_m=1.5,
        range_conf=0.8,
    )
    owm.bind_from_visual(vt, tick=5, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    p = owm.graph_payload()
    assert "task_target_x" in p
    assert "task_target_y" in p
    assert p["task_target_dist_m"] == pytest.approx(owm.range_m)
    assert p["self_goal_active"] == 1.0
    assert p["scene_n_entities"] == 1.0
    assert p["scene_n_active"] == 1.0
    ov = owm.scene.overlay_payload(tick=5)
    assert ov["active"] is not None
    assert ov["active"]["range_m"] == pytest.approx(1.5, abs=0.05)
    assert 0.0 <= ov["active"]["u"] <= 1.0
    assert ov["n_entities"] >= 1


def test_latent_scene_multi_entity_and_active() -> None:
    scene = LatentSceneMemory()
    scene.update(
        tick=1,
        percepts=[
            {
                "slot_id": "slot_a",
                "bearing": 0.0,
                "range_m": 2.0,
                "label": "chair",
                "confidence": 0.8,
                "activation": 0.8,
            },
            {
                "slot_id": "slot_b",
                "bearing": 0.4,
                "range_m": 3.0,
                "label": "box",
                "confidence": 0.7,
                "activation": 0.7,
            },
        ],
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert len(scene.entities) == 2
    assert scene.active() is None  # no focus until bind/focus

    scene.focus("slot_b", exclusive=True)
    act = scene.active()
    assert act is not None
    assert act.slot_id == "slot_b"
    assert act.range_m == pytest.approx(3.0, abs=0.05)

    # Odometry moves both; active stays slot_b
    # slot_b started at bearing 0.4 / 3m → x≈2.43; after +0.4m forward → x≈2.03
    scene.update(
        tick=2,
        percepts=[],
        agent_xy=(0.4, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert len(scene.entities) == 2
    assert scene.active_ids == ["slot_b"]
    assert scene.active().x_fwd == pytest.approx(2.027, abs=0.05)
    assert scene.entities["slot_a"].x_fwd == pytest.approx(1.6, abs=0.1)

    p = scene.graph_payload(tick=2)
    assert p["scene_n_entities"] == 2.0
    assert p["scene_n_active"] == 1.0
    assert p["task_target_dist_m"] == pytest.approx(scene.active().range_m)
    assert "scene_e0_x" in p
    assert "scene_e1_conf" in p


def test_bind_sets_exclusive_active() -> None:
    scene = LatentSceneMemory()
    scene.update(
        tick=1,
        percepts=[
            {"slot_id": "slot_a", "bearing": 0.0, "range_m": 2.0, "confidence": 0.8},
            {"slot_id": "slot_b", "bearing": 0.2, "range_m": 2.5, "confidence": 0.7},
        ],
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    vt = VisualTarget(
        slot_id="slot_a",
        u=0.5,
        v=0.5,
        label="chair",
        confidence=0.9,
        bearing=0.0,
        range_m=1.8,
        range_conf=0.9,
    )
    scene.bind_visual_target(vt, tick=2, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    assert scene.active_ids == ["slot_a"]
    assert scene.active().label == "chair"
    # Facade mirrors active
    owm = ObjectWorkingMemory(scene)
    assert owm.slot_id == "slot_a"
    assert owm.is_usable(2)


def test_apply_odometry_includes_lateral_shift() -> None:
    from engine.object_working_memory import _apply_odometry_to_ego

    xf, yr = 2.0, 0.4
    prev_xy = (0.0, 0.0)
    prev_fwd = (1.0, 0.0)
    agent_xy = (0.0, 0.1)
    agent_fwd = (1.0, 0.0)
    xf2, yr2 = _apply_odometry_to_ego(
        xf,
        yr,
        prev_xy=prev_xy,
        prev_fwd=prev_fwd,
        agent_xy=agent_xy,
        agent_forward=agent_fwd,
    )
    assert yr2 < yr


def test_refresh_active_syncs_ego_with_bearing() -> None:
    from engine.vision_depth import ArrayDepthCamera, DepthFrame
    from engine.vision_target import bearing_from_u

    scene = LatentSceneMemory()
    ent = scene.entities.setdefault(
        "slot_0",
        SceneEntity(entity_id="slot_0"),
    )
    scene.active_ids = ["slot_0"]
    ent.x_fwd, ent.y_right = 4.5, 0.6
    ent.bearing = 0.14
    ent.range_m = 4.8
    ent.u, ent.v = 0.5, 0.55
    ent.confidence = 0.8
    ent.last_vision_tick = 1

    h, w = 40, 48
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[8:22, 10:20] = 1.5
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    ok = scene.refresh_active_from_live_camera(cam, tick=2, blend=1.0)
    assert ok
    act = scene.active()
    assert act is not None
    xf, yr = ego_from_bearing_range(act.bearing, act.range_m)
    assert abs(act.x_fwd - xf) < 1e-4
    assert abs(act.y_right - yr) < 1e-4
    assert abs(act.bearing - bearing_from_u(act.u)) < 0.05


def test_refresh_active_from_live_camera_updates_uv() -> None:
    from engine.vision_depth import ArrayDepthCamera, DepthFrame

    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.55,
        label="cylinder",
        confidence=0.8,
        bearing=0.0,
        range_m=4.5,
        range_conf=0.8,
    )
    scene.bind_visual_target(vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    # Unlocked soft-track may rewrite bearing from live UV.
    scene.release_hard_lock()
    h, w = 40, 48
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[8:22, 22:30] = 1.5
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    ok = scene.refresh_active_from_live_camera(cam, tick=2, blend=1.0)
    assert ok
    act = scene.active()
    assert act is not None
    assert 0.42 < act.u < 0.68
    assert act.range_m < 3.0


def test_hard_lock_refresh_does_not_yank_bearing() -> None:
    """Far live peak under hard_lock must get near-zero Kalman gain (no yank)."""

    class _FarCam:
        def live_at_bearing(self, bearing, **_kwargs):
            # Far-left UV → live bearing ≈ -0.9, well outside slack.
            return 0.05, 0.55, 1.2, 0.9

    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.55,
        label="cylinder",
        confidence=0.8,
        bearing=0.1,
        range_m=2.3,
        range_conf=0.8,
    )
    scene.bind_visual_target(vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    assert scene.hard_lock_active is True
    locked_bearing = float(scene.active().bearing)

    ok = scene.refresh_active_from_live_camera(_FarCam(), tick=2, blend=0.78)
    assert ok
    act = scene.active()
    assert act is not None
    # Continuous fusion may apply a tiny residual nudge — never a yank.
    assert abs(float(act.bearing) - locked_bearing) < 0.03
    assert float(act.diagnostics.get("kalman_gain") or 1.0) < 0.08
    assert abs(float(act.diagnostics.get("bearing_nudge") or 1.0)) < 0.03
    assert act.diagnostics.get("source") == "hard_lock_bayesian_kalman"


def test_hard_lock_near_live_applies_bounded_nudge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Near live hit under hard_lock gets higher Kalman gain than a far outlier."""
    monkeypatch.setenv("RKK_HARD_LOCK_BEARING_SLACK", "0.25")
    from engine.object_working_memory import _bayesian_vision_kalman_gain

    k_near, _ = _bayesian_vision_kalman_gain(0.05, 0.9, outlier_scale=0.25)
    k_far, _ = _bayesian_vision_kalman_gain(0.90, 0.9, outlier_scale=0.25)
    assert k_near > k_far
    assert k_far < 0.08
    assert k_near > 0.20

    from engine.vision_depth import ArrayDepthCamera, DepthFrame

    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.55,
        label="cylinder",
        confidence=0.8,
        bearing=0.0,
        range_m=2.5,
        range_conf=0.8,
    )
    scene.bind_visual_target(vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    before = float(scene.active().bearing)

    h, w = 40, 48
    depth = np.full((h, w), 4.0, dtype=np.float32)
    # Peak slightly right of center → small positive bearing.
    depth[8:22, 26:34] = 1.4
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    ok = scene.refresh_active_from_live_camera(cam, tick=2, blend=1.0)
    assert ok
    act = scene.active()
    assert act is not None
    assert abs(float(act.bearing) - before) < 0.20
    assert act.diagnostics.get("live_bearing") is not None
    assert act.diagnostics.get("kalman_gain") is not None


def test_odom_discontinuity_skips_warp_and_reseeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_SCENE_ODOM_MAX_STEP_M", "0.75")
    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="cylinder",
        confidence=0.9,
        bearing=0.0,
        range_m=2.0,
        range_conf=0.9,
    )
    scene.bind_visual_target(
        vt, tick=1, agent_xy=(3.5, -1.5), agent_forward=(1.0, 0.0)
    )
    assert scene.hard_lock_active is True
    assert scene.active().range_m == pytest.approx(2.0, abs=0.05)

    # Teleport back to spawn (~1.1 m) — must NOT inflate range via odometry
    scene.update(
        tick=2,
        percepts=[
            {
                "slot_id": "slot_0",
                "bearing": 0.0,
                "range_m": 2.4,
                "confidence": 0.85,
                "activation": 0.85,
            }
        ],
        agent_xy=(2.4, -1.5),
        agent_forward=(1.0, 0.0),
    )
    assert scene.last_odom_discontinuity is True
    assert scene.last_odom_jump_m == pytest.approx(1.1, abs=0.05)
    # Reseeded from vision, not 2.0+1.1
    assert scene.active().range_m == pytest.approx(2.4, abs=0.05)
    assert scene.active().diagnostics.get("source") == "hard_lock_reseed"

    # Continuous step still warps normally
    scene.update(
        tick=3,
        percepts=[],
        agent_xy=(2.55, -1.5),
        agent_forward=(1.0, 0.0),
    )
    assert scene.last_odom_discontinuity is False
    assert scene.active().range_m == pytest.approx(2.25, abs=0.08)


def test_hud_filters_learning_opportunity() -> None:
    from engine.vision_resolve import hud_safe_label

    assert hud_safe_label("LEARNING_OPPORTUNITY") == "target"
    assert hud_safe_label("cylinder") == "cylinder"
