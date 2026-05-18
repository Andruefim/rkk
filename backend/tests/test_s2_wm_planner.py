"""S2-gated WM planner scoring and gating."""
from __future__ import annotations

from engine.system2.wm_planner import (
    S2WmTask,
    _bundle_fallback_quick,
    score_wm_trajectory,
    s2_wm_fast_override_enabled,
    s2_wm_gate_strict,
    s2_wm_planner_enabled,
    task_from_planning_context,
)


def test_recover_penalizes_stride_when_fallen():
    s0 = {
        "posture_stability": 0.3,
        "com_z": 0.35,
        "intero_energy": 0.6,
        "target_dist": 0.7,
    }
    s1 = dict(s0)
    task = S2WmTask(macro="RECOVER_POSTURE", fallen=True, fallen_override=True)
    torso = score_wm_trajectory(
        s0, s1, task, action_var="intent_torso_forward", action_val=0.72
    )
    stride = score_wm_trajectory(
        s0, s1, task, action_var="intent_stride", action_val=0.62
    )
    assert torso > stride


def test_task_active_on_fallen_override():
    ctx = {"macro": "IDLE", "fallen_override_active": True, "fallen": True}
    t = task_from_planning_context(ctx, {"self_goal_active": 0.2})
    assert t.active
    assert t.macro == "RECOVER_POSTURE"


def test_planner_enabled_by_default():
    assert s2_wm_planner_enabled()
    assert s2_wm_gate_strict()
    assert s2_wm_fast_override_enabled()


def test_bundle_fallback_quick_from_context():
    class _Agent:
        def _features_for_intervention_pair(self, a, b):
            return [0.1, 0.2]

    ctx = {
        "bundle_candidate": {
            "variable": "intent_torso_forward",
            "value": 0.68,
        }
    }
    task = S2WmTask(macro="RECOVER_POSTURE", fallen_override=True)
    cand = _bundle_fallback_quick(ctx, task, _Agent())
    assert cand is not None
    assert cand["variable"] == "intent_torso_forward"
    assert cand.get("s2_wm_fast_override") is True


def test_recover_improves_posture_score():
    s0 = {"posture_stability": 0.2, "com_z": 0.3, "intero_energy": 0.7}
    s1 = {"posture_stability": 0.35, "com_z": 0.38, "intero_energy": 0.68}
    task = S2WmTask(macro="RECOVER_POSTURE", fallen_override=True)
    sc = score_wm_trajectory(
        s0, s1, task, action_var="intent_torso_forward", action_val=0.72
    )
    assert sc > 0
