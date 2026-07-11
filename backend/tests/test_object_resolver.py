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


def test_collect_scene_candidates_includes_generic_ball_extra() -> None:
    extras = {
        "ball": {"x": 1.5, "y": 0.2, "z": 0.25, "body_id": 77, "movable": True},
    }
    cands = collect_scene_candidates(extras)
    assert len(cands) == 1
    assert cands[0]["ref"] == "ball"
    assert cands[0]["semantic"] == "ball"
    assert cands[0]["movable"] is True


def test_resolver_ball_lexical_and_ru_embed() -> None:
    extras = {
        "ball": {"x": 1.5, "y": 0.0, "z": 0.2, "body_id": 100, "movable": True},
    }
    resolved_en, diag_en = resolve_manipulation_target(
        "touch ball",
        extras,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
        require_movable=False,
    )
    assert resolved_en is not None, diag_en
    assert resolved_en.ref == "ball"

    fb = FallbackEmbeddingClient(embed_dim=64)

    def _ru_ball_embed(text: str):
        vec = fb.embed(text)
        if vec is None:
            return None
        if "шар" in text.lower():
            return fb.embed("ball")
        return vec

    resolved_ru, diag_ru = resolve_manipulation_target(
        "дотронься до шара",
        extras,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
        embed_fn=_ru_ball_embed,
        require_movable=False,
    )
    assert resolved_ru is not None, diag_ru
    assert resolved_ru.ref == "ball"


def test_resolver_embed_equal_to_lexical() -> None:
    """Embed match is combined with lexical (max), not secondary fallback only."""
    extras = {
        "registry": [
            {
                "ref": "crate_a",
                "semantic": "crate",
                "label": "crate",
                "movable": True,
                "mass": 3.0,
                "x": 1.0,
                "y": 0.0,
                "z": 0.3,
            }
        ],
    }
    fb = FallbackEmbeddingClient(embed_dim=64)

    def _crate_embed(text: str):
        vec = fb.embed(text)
        if vec is None:
            return None
        if "ящик" in text.lower():
            return fb.embed("crate")
        return vec

    resolved, diag = resolve_manipulation_target(
        "передвинь ящик",
        extras,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
        embed_fn=_crate_embed,
    )
    assert resolved is not None, diag
    assert resolved.ref == "crate_a"
    assert diag["reason"] == "resolved"


def test_collect_scene_candidates_dedup_by_ref() -> None:
    extras = {
        "registry": [
            {
                "ref": "dup_obj",
                "semantic": "chair",
                "movable": False,
                "x": 1.0,
                "y": 0.0,
                "z": 0.4,
            }
        ],
        "chair": {
            "ref": "dup_obj",
            "semantic": "chair",
            "movable": True,
            "x": 1.0,
            "y": 0.0,
            "z": 0.4,
            "body_id": 9,
        },
    }
    cands = collect_scene_candidates(extras)
    refs = [c["ref"] for c in cands if c["ref"] == "dup_obj"]
    assert len(refs) == 1
    movable = [c for c in cands if c["ref"] == "dup_obj"][0]
    assert movable["movable"] is True
