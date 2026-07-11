"""Tests for manipulation object resolver."""
from __future__ import annotations

import math

from engine.grounded_language import FallbackEmbeddingClient
from engine.object_resolver import (
    ResolvedObject,
    collect_scene_candidates,
    extract_content_tokens,
    infer_semantic_from_query,
    resolve_manipulation_target,
)


def _movable_chair(x: float = 1.0, y: float = 1.2) -> dict:
    return {
        "registry": [{
            "ref": "manip_chair_front",
            "body_id": 42,
            "semantic": "chair",
            "movable": True,
            "mass": 5.5,
            "x": x,
            "y": y,
            "z": 0.4,
            "source": "manip_chair",
        }],
        "static_geometry": [],
    }


def test_extract_content_tokens_strips_verbs_and_stopwords() -> None:
    tokens = extract_content_tokens("передвинь стул перед тобой")
    assert "стул" in tokens
    assert "передвинь" not in tokens
    assert "перед" not in tokens
    assert "тобой" not in tokens


def test_infer_semantic_from_query_noun_or_generic() -> None:
    assert infer_semantic_from_query("передвинь стул перед тобой") == "стул"
    assert infer_semantic_from_query("move the chair") == "chair"
    assert infer_semantic_from_query("move the object") == "object"
    assert infer_semantic_from_query("передвинь это") == "object"
    assert infer_semantic_from_query("иди вперёд") is None


def test_resolver_chooses_movable_chair_by_semantic() -> None:
    extras = _movable_chair()
    extras["static_geometry"] = [
        {"ref": "cafe_seat_0", "tx": 0.5, "ty": 0.4, "tz": 0.5, "semantic": "seat", "static": True},
    ]
    fb = FallbackEmbeddingClient(embed_dim=64)
    resolved, diag = resolve_manipulation_target(
        "передвинь стул перед тобой",
        extras,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
        embed_fn=fb.embed,
    )
    assert resolved is not None
    assert isinstance(resolved, ResolvedObject)
    assert resolved.ref == "manip_chair_front"
    assert resolved.movable is True
    assert resolved.semantic == "chair"
    assert diag["reason"] == "resolved"


def test_resolver_finds_box_by_semantic() -> None:
    extras = {
        "registry": [
            {
                "ref": "box_a",
                "semantic": "box",
                "label": "коробка",
                "movable": True,
                "mass": 2.0,
                "x": 1.0,
                "y": 0.0,
                "z": 0.3,
            }
        ],
    }
    resolved, diag = resolve_manipulation_target(
        "передвинь коробку",
        extras,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert resolved is not None, diag
    assert resolved.ref == "box_a"
    assert resolved.semantic == "box"


def test_resolver_no_semantic_match_reports_scene() -> None:
    extras = _movable_chair()
    resolved, diag = resolve_manipulation_target(
        "передвинь пианино",
        extras,
        agent_xy=(0.0, 0.0),
    )
    assert resolved is None
    assert diag["reason"] == "no_semantic_match"
    assert "chair" in diag.get("scene_semantics", [])


def test_resolver_rejects_static_only_targets() -> None:
    extras = {
        "registry": [],
        "static_geometry": [
            {"ref": "cafe_chair_0", "tx": 0.8, "ty": 0.2, "tz": 0.5, "semantic": "chair"},
            {"ref": "cafe_chair_1", "tx": 1.0, "ty": 0.3, "tz": 0.5, "semantic": "chair"},
        ],
    }
    resolved, diag = resolve_manipulation_target(
        "move the chair",
        extras,
        agent_xy=(0.0, 0.0),
    )
    assert resolved is None
    assert diag["reason"] == "target_not_movable"


def test_resolver_prefers_forward_cone_without_noun() -> None:
    extras = {
        "registry": [
            {
                "ref": "chair_behind",
                "semantic": "chair",
                "movable": True,
                "mass": 5.0,
                "x": -1.2,
                "y": 0.0,
                "z": 0.4,
            },
            {
                "ref": "chair_front",
                "semantic": "chair",
                "movable": True,
                "mass": 5.0,
                "x": 1.1,
                "y": 0.0,
                "z": 0.4,
            },
        ],
    }
    resolved, _ = resolve_manipulation_target(
        "передвинь это",
        extras,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert resolved is not None
    assert resolved.ref == "chair_front"


def test_resolver_prefers_forward_cone_with_noun() -> None:
    extras = {
        "registry": [
            {
                "ref": "chair_behind",
                "semantic": "chair",
                "movable": True,
                "mass": 5.0,
                "x": -1.2,
                "y": 0.0,
                "z": 0.4,
            },
            {
                "ref": "chair_front",
                "semantic": "chair",
                "movable": True,
                "mass": 5.0,
                "x": 1.1,
                "y": 0.0,
                "z": 0.4,
            },
        ],
    }
    resolved, _ = resolve_manipulation_target(
        "chair",
        extras,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    assert resolved is not None
    assert resolved.ref == "chair_front"


def test_collect_scene_candidates_includes_props() -> None:
    extras = {
        "props": [{"x": 2.0, "y": 3.0, "z": 0.1, "mass": 0.3, "semantic": "box", "movable": True}],
        "registry": [],
    }
    cands = collect_scene_candidates(extras)
    assert len(cands) == 1
    assert cands[0]["source"] == "props"
    assert math.isclose(cands[0]["x"], 2.0)
