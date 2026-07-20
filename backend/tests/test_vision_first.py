"""Vision-first AGI: VisualTarget, depth, embedding resolve, ego servoing contracts."""
from __future__ import annotations

import numpy as np
import pytest

from engine.goal_navigation import navigation_intents_from_bearing_range
from engine.manipulation_control import manipulation_intents_from_bearing_range
from engine.vision_depth import (
    ArrayDepthCamera,
    DepthFrame,
    UvDepthTrack,
    adaptive_fov_u_half,
    attach_range_to_target,
    attention_guided_range,
    buffer_to_metric_depth,
    depth_at_uv,
    live_uv_fov_base,
    live_uv_range_at_bearing,
    salient_objectness_peak,
    salient_objectness_peak_near_bearing,
    track_search_fov_u_half,
)
from engine.vision_resolve import resolve_visual_target
from engine.vision_target import (
    VisualTarget,
    bearing_from_u,
    task_resolve_mode,
    vision_resolve_enabled,
)


def _hash_embed(text: str, dim: int = 32) -> np.ndarray:
    """Deterministic unit embedding for tests (same string → same vector)."""
    rng = np.random.default_rng(abs(hash(str(text).lower())) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= float(np.linalg.norm(v) + 1e-9)
    return v


def test_task_resolve_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_TASK_RESOLVE", "vision")
    assert task_resolve_mode() == "vision"
    assert vision_resolve_enabled() is True
    monkeypatch.setenv("RKK_TASK_RESOLVE", "oracle")
    assert task_resolve_mode() == "oracle"
    assert vision_resolve_enabled() is False


def test_visual_target_ready_requires_range() -> None:
    vt = VisualTarget(
        slot_id="slot_1",
        u=0.5,
        v=0.55,
        label="chair",
        confidence=0.8,
        bearing=0.0,
        range_m=None,
    )
    assert vt.is_ready(require_range=True) is False
    assert vt.is_ready(require_range=False) is True
    vt2 = vt.with_range(1.2, range_conf=0.9)
    assert vt2.is_ready(require_range=True) is True
    assert "body_id" not in vt2.to_dict()
    assert vt2.ref.startswith("vision:")


def test_bearing_from_u() -> None:
    assert bearing_from_u(0.5) == pytest.approx(0.0)
    assert bearing_from_u(1.0) > 0.0
    assert bearing_from_u(0.0) < 0.0


def test_buffer_to_metric_and_depth_at_uv() -> None:
    buf = np.full((40, 60), 0.3, dtype=np.float32)
    depth_m = buffer_to_metric_depth(buf, near_m=0.1, far_m=15.0)
    assert depth_m.shape == (40, 60)
    assert 0.1 < float(np.median(depth_m)) < 15.0

    frame = DepthFrame(depth_m=depth_m, near_m=0.1, far_m=15.0)
    r, var, conf = depth_at_uv(frame, 0.5, 0.5, window=2)
    assert r is not None and r > 0.1
    assert conf is not None and conf > 0.0

    cam = ArrayDepthCamera(frame)
    r2, _, c2 = cam.range_at_uv(0.5, 0.5)
    assert r2 == pytest.approx(r, rel=1e-3)


def test_resolve_by_label_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.3")
    depth = np.full((48, 64), 1.5, dtype=np.float32)
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.48,
            "v": 0.55,
            "label": "chair",
            "activation": 0.9,
            "vector": None,
        },
        {
            "slot_id": "slot_1",
            "u": 0.9,
            "v": 0.2,
            "label": "lamp",
            "activation": 0.9,
            "vector": None,
        },
    ]
    vt, diag = resolve_visual_target(
        "подойди к стулу chair",
        slots=slots,
        depth_camera=cam,
        embed_fn=_hash_embed,
        require_range=True,
    )
    # Hash embed: exact label substring won't match unless command==label.
    # Use identical strings for principled cosine match.
    vt, diag = resolve_visual_target(
        "chair",
        slots=slots,
        depth_camera=cam,
        embed_fn=_hash_embed,
        require_range=True,
    )
    assert vt is not None, diag
    assert diag["reason"] == "ok"
    assert vt.slot_id == "slot_0"
    assert vt.range_m is not None
    assert "body_id" not in vt.to_dict()


def test_resolve_fails_unlabeled_without_ontology_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrelated command + unlabeled slots → no link."""
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.9")
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.5,
            "v": 0.5,
            "label": "",
            "activation": 0.9,
            "vector": np.ones(8, dtype=np.float32),
        }
    ]
    # Hash embed: ontology descriptions won't match random command string
    vt, diag = resolve_visual_target(
        "xyzzy_no_such_referent_qqq",
        slots=slots,
        depth_camera=None,
        embed_fn=_hash_embed,
        require_range=False,
    )
    assert vt is None
    assert diag["reason"] in ("no_language_vision_link", "low_vision_confidence")


def test_resolve_via_visual_referent_ontology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Command ↔ ontology descriptions (embeddings) × slot activation."""
    from engine.grounded_language import FallbackEmbeddingClient
    from engine.visual_referent_ontology import clear_visual_referent_cache

    clear_visual_referent_cache()
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.20")
    emb = FallbackEmbeddingClient(embed_dim=64)
    depth = np.full((32, 32), 1.2, dtype=np.float32)
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.5,
            "v": 0.55,
            "label": "",
            "activation": 0.85,
            "vector": None,
        },
        {
            "slot_id": "slot_1",
            "u": 0.2,
            "v": 0.2,
            "label": "",
            "activation": 0.15,
            "vector": None,
        },
    ]
    vt, diag = resolve_visual_target(
        "подойди к объекту перед тобой и дотронься до него",
        slots=slots,
        depth_camera=cam,
        embed_fn=emb.embed,
        require_range=True,
    )
    assert vt is not None, diag
    assert diag["reason"] == "ok"
    assert vt.slot_id == "slot_0"
    assert vt.range_m is not None


def test_resolve_via_concept_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.3")
    depth = np.full((32, 32), 1.2, dtype=np.float32)
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.5,
            "v": 0.5,
            "label": "",
            "activation": 0.8,
            "vector": np.ones(8, dtype=np.float32),
        }
    ]

    def project(_vec: np.ndarray) -> list[tuple[str, float]]:
        return [("chair", 0.9)]

    vt, diag = resolve_visual_target(
        "chair",
        slots=slots,
        depth_camera=cam,
        embed_fn=_hash_embed,
        concept_project_fn=project,
        require_range=True,
    )
    assert vt is not None, diag
    assert diag["reason"] == "ok"
    assert vt.label == "chair"


def test_resolve_fails_without_slots() -> None:
    vt, diag = resolve_visual_target(
        "touch object", slots=[], embed_fn=_hash_embed, require_range=True
    )
    assert vt is None
    assert diag["reason"] == "no_vision_slots"


def test_resolve_fails_without_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.3")
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.5,
            "v": 0.5,
            "label": "object",
            "activation": 0.9,
            "vector": None,
        }
    ]
    vt, diag = resolve_visual_target(
        "object",
        slots=slots,
        depth_camera=None,
        embed_fn=_hash_embed,
        require_range=True,
    )
    assert vt is None
    assert diag["reason"] == "missing_or_invalid_range"


def test_objectness_prefers_protrusion_over_floor() -> None:
    """Flat ground strip must not win over a closer protruding object blob."""
    from engine.vision_depth import attention_guided_range

    h, w = 60, 80
    # Floor-like: depth increases smoothly with v (look-down)
    yy = np.linspace(1.5, 4.0, h, dtype=np.float32)[:, None]
    depth = np.repeat(yy, w, axis=1)
    # Protruding object (tree) left-center, closer than floor at that row
    depth[18:38, 25:45] = 2.0
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)
    u, v, r, _var, conf = attention_guided_range(frame, None, prefer_objects=True)
    assert r is not None and conf is not None
    assert 1.6 < r < 2.6
    assert 0.25 < u < 0.65
    # Should not lock to extreme bottom floor
    assert v < 0.85


def test_attention_guided_range_prefers_near_surface() -> None:
    """Center UV may hit far plane; attention×1/Z must lock onto nearer blob."""
    from engine.vision_depth import attention_guided_range

    h, w = 48, 64
    depth = np.full((h, w), 12.0, dtype=np.float32)  # far / sky
    # Near object blob left-of-center
    depth[20:36, 10:28] = 2.2
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)

    # Point sample at center → invalid (far plane rejected)
    r_pt, _, _ = depth_at_uv(frame, 0.5, 0.5, window=2)
    assert r_pt is None

    # Diffuse attention (uniform) still pulls to nearer surface via 1/Z
    u, v, r, _var, conf = attention_guided_range(frame, None)
    assert r is not None
    assert conf is not None and conf > 0.05
    assert 1.5 < r < 3.0
    assert u < 0.45  # pulled toward left blob

    cam = ArrayDepthCamera(frame)
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="object",
        confidence=0.8,
        bearing=0.0,
    )
    vt2 = attach_range_to_target(vt, cam, attn_mask=None)
    assert vt2.range_m is not None
    assert 1.5 < float(vt2.range_m) < 3.0
    assert float(vt2.u) < 0.45


def test_hard_lock_odometry_ignores_vision_yank() -> None:
    from engine.object_working_memory import LatentSceneMemory
    from engine.vision_target import VisualTarget

    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.45,
        label="object",
        confidence=0.9,
        bearing=0.0,
        range_m=2.5,
        range_conf=0.9,
    )
    scene.bind_visual_target(vt, tick=1, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0))
    assert scene.hard_lock_active is True
    assert scene.active().range_m == pytest.approx(2.5, abs=0.05)

    # Walk forward — odometry shrinks range
    scene.update(
        tick=2,
        percepts=[
            {
                "slot_id": "slot_0",
                "bearing": 0.0,
                "range_m": 4.0,  # would yank — must be ignored under hard-lock
                "u": 0.5,
                "v": 0.9,
                "confidence": 0.9,
            }
        ],
        agent_xy=(0.5, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert scene.active().diagnostics.get("source") == "hard_lock_odom"
    assert scene.active().range_m == pytest.approx(2.0, abs=0.1)
    scene.release_hard_lock()
    assert scene.hard_lock_active is False


def test_active_track_gates_floor_relock() -> None:
    """Active entity rejects vision that jumps onto a different surface."""
    from engine.object_working_memory import LatentSceneMemory

    scene = LatentSceneMemory()
    scene.update(
        tick=1,
        percepts=[
            {
                "slot_id": "slot_0",
                "bearing": 0.0,
                "range_m": 2.5,
                "u": 0.5,
                "v": 0.45,
                "confidence": 0.8,
                "activation": 0.8,
                "label": "object",
            }
        ],
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    scene.focus("slot_0")
    assert scene.active().range_m == pytest.approx(2.5, abs=0.05)

    # Walk forward — odometry shrinks ego x
    scene.update(
        tick=2,
        percepts=[],
        agent_xy=(0.4, 0.0),
        agent_forward=(1.0, 0.0),
    )
    held = float(scene.active().range_m)
    assert held < 2.5

    # Bad vision: "floor" at 3.2m ahead — must not yank track
    scene.update(
        tick=3,
        percepts=[
            {
                "slot_id": "slot_0",
                "bearing": 0.0,
                "range_m": 3.2,
                "u": 0.5,
                "v": 0.85,
                "confidence": 0.3,
                "activation": 0.3,
            }
        ],
        agent_xy=(0.4, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert scene.active().diagnostics.get("source") == "gate_reject"
    assert abs(float(scene.active().range_m) - held) < 0.15


def test_resolve_rejects_far_plane_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.3")
    # Entire frame at far-plane depth → no valid control range
    depth = np.full((32, 32), 11.0, dtype=np.float32)
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.5,
            "v": 0.5,
            "label": "object",
            "activation": 0.9,
            "vector": None,
        }
    ]
    vt, diag = resolve_visual_target(
        "object",
        slots=slots,
        depth_camera=cam,
        embed_fn=_hash_embed,
        require_range=True,
    )
    assert vt is None
    assert diag["reason"] == "missing_or_invalid_range"


def test_navigation_from_bearing_range() -> None:
    intents = navigation_intents_from_bearing_range(
        0.4, 2.0, stop_distance=0.55, posture_stability=0.9
    )
    assert "intent_stride" in intents
    assert intents.get("vision_range_m") == pytest.approx(2.0)
    assert navigation_intents_from_bearing_range(0.0, 0.4, stop_distance=0.55) == {}


def test_manipulation_from_bearing_range() -> None:
    intents = manipulation_intents_from_bearing_range(0.3, 0.5)
    assert "intent_reach_right" in intents or "intent_reach_left" in intents
    assert manipulation_intents_from_bearing_range(0.0, 2.0) == {}


def test_skill_none_not_visual_label() -> None:
    from engine.vision_resolve import _is_visual_concept, hud_safe_label

    assert _is_visual_concept("SKILL_NONE") is False
    assert _is_visual_concept("SKILL_HOME") is False
    assert hud_safe_label("SKILL_NONE") == "target"
    assert hud_safe_label("cylinder") == "cylinder"


def test_diffuse_uv_penalized_vs_peaked_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ontology×activation must not pick center-floor SKILL noise."""
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.05")
    monkeypatch.setenv("RKK_SLOT_MASK_PEAKINESS_MIN", "1.8")
    from engine.vision_resolve import score_slots_for_command

    def embed(text: str) -> np.ndarray:
        # Same vector for everything → ontology path dominates via activation
        return np.ones(8, dtype=np.float32)

    slots = [
        {
            "slot_id": "slot_noise",
            "u": 0.5,
            "v": 0.5,
            "label": "SKILL_NONE",
            "activation": 0.99,
            "vector": None,
            "uv_valid": False,
            "mask_peakiness": 1.0,
        },
        {
            "slot_id": "slot_obj",
            "u": 0.35,
            "v": 0.42,
            "label": "cylinder",
            "activation": 0.55,
            "vector": None,
            "uv_valid": True,
            "mask_peakiness": 4.0,
        },
    ]
    scored, _ = score_slots_for_command(slots, "цилиндр", embed_fn=embed)
    by_id = {s["slot_id"]: s for s in scored}
    assert by_id["slot_obj"]["match_score"] > by_id["slot_noise"]["match_score"]


def test_objectness_peak_when_mask_invalid() -> None:
    """Diffuse mask → ignore it; UV snaps to depth protrusion (cylinder proxy)."""
    h, w = 40, 48
    depth = np.full((h, w), 4.0, dtype=np.float32)
    # Protruding blob left of center (closer)
    depth[8:22, 6:16] = 1.4
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    flat = np.ones((8, 8), dtype=np.float32)
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.5,
            "v": 0.5,
            "label": "cylinder",
            "activation": 0.9,
            "vector": None,
            "uv_valid": False,
            "mask_peakiness": 1.0,
            "attn_mask": flat,
        }
    ]
    vt, diag = resolve_visual_target(
        "cylinder",
        slots=slots,
        depth_camera=cam,
        embed_fn=_hash_embed,
        require_range=True,
    )
    assert vt is not None, diag
    assert vt.u < 0.45  # pulled toward left protrusion, not dead center
    assert (vt.diagnostics or {}).get("geometry") == "objectness_peak"


def test_salient_peak_prefers_protrusion_over_floor_centroid() -> None:
    """Full-image centroid stays near center; salient peak snaps to closer blob."""
    h, w = 48, 64
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[6:18, 44:58] = 1.3
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)
    u_cent, v_cent, _, _, _ = attention_guided_range(frame, None)
    u_peak, v_peak, r, _, _, pstr = salient_objectness_peak(frame)
    assert u_cent is not None and u_peak is not None and r is not None
    assert u_peak > 0.55
    assert u_peak > u_cent + 0.05
    assert pstr > 0.1


def test_lateral_tilt_not_visual_concept() -> None:
    from engine.vision_resolve import _is_visual_concept

    assert not _is_visual_concept("LATERAL_TILT_RIGHT")
    assert not _is_visual_concept("TILT_LEFT")


def test_live_uv_at_bearing_tracks_protrusion() -> None:
    h, w = 48, 64
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[8:20, 50:60] = 1.6
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)
    cam = ArrayDepthCamera(frame)
    u, v, r, conf = cam.live_at_bearing(0.35, range_hint=2.0)
    assert u is not None and r is not None
    assert u > 0.55
    assert r < 2.5


def test_salient_peak_near_bearing_respects_window() -> None:
    h, w = 48, 64
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[8:20, 50:60] = 1.4  # right protrusion in bearing window
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)
    u_peak, v_peak, r, _, _, _ = salient_objectness_peak_near_bearing(frame, 0.35)
    assert u_peak is not None and r is not None
    assert u_peak > 0.52
    assert r < 2.0


def test_live_uv_rejects_floor_when_hint_closer() -> None:
    h, w = 48, 64
    depth = np.full((h, w), 3.2, dtype=np.float32)
    depth[10:22, 30:38] = 1.1
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)
    u, v, r, conf = live_uv_range_at_bearing(
        ArrayDepthCamera(frame), 0.0, range_hint=1.2
    )
    assert r is not None and r < 1.8


def test_adaptive_fov_follows_track_not_range() -> None:
    """FOV must not shrink from range_hint alone (no object-size / 1/r heuristic)."""
    base = live_uv_fov_base()
    assert adaptive_fov_u_half(1.0) == pytest.approx(base)
    assert adaptive_fov_u_half(4.0) == pytest.approx(base)
    stable = UvDepthTrack.from_list([[0.50, 0.40], [0.51, 0.40], [0.505, 0.41]])
    jumpy = UvDepthTrack.from_list([[0.30, 0.35], [0.55, 0.45], [0.80, 0.50]])
    fov_stable = track_search_fov_u_half(stable)
    fov_jumpy = track_search_fov_u_half(jumpy)
    assert fov_stable <= base
    assert fov_jumpy >= fov_stable
    assert track_search_fov_u_half(None) == pytest.approx(base)


def test_live_uv_relaxes_range_gate_inside_track() -> None:
    """Tight range_hint with only far depth near track must not return empty."""
    h, w = 48, 64
    depth = np.full((h, w), 4.0, dtype=np.float32)
    # Blob near prior track UV, but deeper than a tight oracle-like hint.
    depth[14:24, 28:38] = 2.4
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)
    cam = ArrayDepthCamera(frame)
    track = UvDepthTrack.from_list([[0.50, 0.40], [0.51, 0.40]])
    u, v, r, _ = live_uv_range_at_bearing(
        cam, 0.0, range_hint=1.5, uv_track=track
    )
    assert u is not None and r is not None
    assert abs(u - 0.5) < 0.15
    assert r < 3.0


def test_live_uv_continuity_prefers_prev_track() -> None:
    h, w = 48, 64
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[10:22, 12:20] = 1.4  # left blob
    depth[10:22, 44:52] = 1.35  # right blob, slightly closer
    frame = DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0)
    cam = ArrayDepthCamera(frame)
    track = UvDepthTrack.from_list([[0.28, 0.35]])
    u, v, r, _ = live_uv_range_at_bearing(
        cam, 0.0, range_hint=1.5, uv_track=track
    )
    assert u is not None and r is not None
    assert u < 0.42


def test_ontology_objectness_fallback_when_scores_crushed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: uv penalty must not yield 'Не удалось' when depth can lock."""
    from engine.grounded_language import FallbackEmbeddingClient
    from engine.visual_referent_ontology import clear_visual_referent_cache

    clear_visual_referent_cache()
    monkeypatch.setenv("RKK_VISION_RESOLVE_MIN_CONF", "0.35")
    emb = FallbackEmbeddingClient(embed_dim=64)
    h, w = 40, 48
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[10:24, 8:20] = 1.5  # cylinder-like protrusion
    cam = ArrayDepthCamera(DepthFrame(depth_m=depth, near_m=0.1, far_m=15.0))
    slots = [
        {
            "slot_id": "slot_0",
            "u": 0.5,
            "v": 0.5,
            "label": "",
            "activation": 0.9,
            "vector": None,
            "uv_valid": False,
            "mask_peakiness": 1.0,
            "attn_mask": np.ones((8, 8), dtype=np.float32),
        }
    ]
    vt, diag = resolve_visual_target(
        "подойди к цилиндрическому объекту перед тобой",
        slots=slots,
        depth_camera=cam,
        embed_fn=emb.embed,
        require_range=True,
    )
    assert vt is not None, diag
    assert diag.get("geometry_fallback") == "objectness_peak" or (
        vt.diagnostics or {}
    ).get("geometry") == "objectness_peak"
    assert vt.range_m is not None


def test_llm_decompose_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_TASK_TREE_LLM", "0")
    from engine.task_goal import GoalPredicate, TaskGoal
    from engine.task_tree import _decompose_from_goal
    from engine.task_tree_llm import _parse_stages_json

    assert _parse_stages_json(
        '["resolve_target", "approach", "reach_contact", "verify_goal"]'
    ) == ["resolve_target", "approach", "reach_contact", "verify_goal"]
    goal = TaskGoal(
        text="touch chair",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.55),
            GoalPredicate(kind="contact", target_value=1.0),
        ],
    )
    kinds = _decompose_from_goal(goal, needs_target=True)
    assert "resolve_target" in kinds
    assert "approach" in kinds
    assert "reach_contact" in kinds
