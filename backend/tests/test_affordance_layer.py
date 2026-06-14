"""Tests for affordance predicates and scene graph (Sprint 7)."""
from __future__ import annotations

from engine.neuro_symbolic.predicates import (
    can_interact_confidence,
    ground_humanoid_state,
    has_target_confidence,
    in_reach_confidence,
)
from engine.scene_graph import SceneGraphObserver


def test_in_reach_predicate() -> None:
    near = {"scene_target_dist": 0.1, "posture_stability": 0.8}
    far = {"scene_target_dist": 0.95, "posture_stability": 0.8}
    assert in_reach_confidence(near, "target") > 0.5
    assert in_reach_confidence(far, "target") < 0.3


def test_ground_humanoid_affordance_facts() -> None:
    obs = {
        "posture_stability": 0.82,
        "com_z": 0.55,
        "intent_stride": 0.64,
        "foot_contact_l": 0.7,
        "foot_contact_r": 0.7,
        "scene_has_target": 1.0,
        "scene_target_dist": 0.15,
        "intent_grasp": 0.5,
        "target_dist": 0.4,
    }
    st = ground_humanoid_state(obs)
    assert st.best("HasTarget") > 0.5
    assert st.best("InReach") > 0.4
    assert can_interact_confidence(obs) > 0.3


def test_scene_graph_observer_snapshot() -> None:
    obs = SceneGraphObserver()
    snap = obs.snapshot()
    assert "objects_tracked" in snap
    assert "affordances_available" in snap
