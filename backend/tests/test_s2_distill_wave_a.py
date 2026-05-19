"""Wave A: distill JSONL schema and analyze helpers."""
from __future__ import annotations

import json
from pathlib import Path

from engine.system2.success_predicates import override_recovered_posture_ok
from engine.system2.success_predicates import override_recovered_posture_ok
from engine.system2.distill_log import (
    DistillHealthTracker,
    analyze_distill_file,
    compress_recovery_steps,
    proposal_distill_extra,
)
from engine.system2.schema import System2Proposal


def test_compress_recovery_steps():
    steps = [
        {"ticks": 12, "intent_deltas": {"intent_torso_forward": 0.08, "bad": 1.0}},
        {"ticks": 200, "intent_deltas": {"intent_stop_recover": 0.1}},
    ]
    out = compress_recovery_steps(steps)
    assert len(out) == 2
    assert out[0]["ticks"] == 12
    assert "bad" not in out[0]["intent_deltas"]
    assert out[1]["ticks"] == 80


def test_proposal_distill_extra_llm():
    p = System2Proposal(
        macro="RECOVER_POSTURE",
        intent_deltas={"intent_torso_forward": 0.06},
        expected_state={"com_z": 0.55},
        skill_id="test_skill",
    )
    ex = proposal_distill_extra(p, source="llm", llm_macro="RECOVER_POSTURE")
    assert ex["plan_source"] == "llm"
    assert ex["llm_macro"] == "RECOVER_POSTURE"
    assert "intent_torso_forward" in ex["intent_deltas"]


def test_distill_health_blend_ready():
    t = DistillHealthTracker()
    for _ in range(30):
        t.record(success=True, macro="RECOVER_POSTURE", student_conf=0.5)
    snap = t.snapshot()
    assert snap["distill_recover_success_rate"] == 1.0
    assert snap["distill_blend_ready"] is True


def test_analyze_distill_file(tmp_path: Path):
    p = tmp_path / "distill.jsonl"
    rows = [
        {
            "macro": "RECOVER_POSTURE",
            "source": "fallen_override:recovered",
            "success": True,
            "delta": {"d_com_z": 0.02, "d_posture": 0.05},
            "student_conf": 0.44,
            "recovery_llm": True,
            "recovery_steps": [{"ticks": 8, "intent_deltas": {}}],
        },
        {
            "macro": "EXPLORE",
            "source": "student_learned",
            "success": False,
            "delta": {"d_com_z": 0.0, "d_posture": -0.01},
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rep = analyze_distill_file(p, window=10)
    assert rep["rows"] == 2
    assert rep["by_macro"]["RECOVER_POSTURE"] == 1.0
    assert rep["recover"]["count"] == 1


def test_override_recovered_posture_gate():
    ok, _ = override_recovered_posture_ok(
        {
            "posture_stability": 0.55,
            "com_z": 0.48,
            "foot_contact_l": 0.4,
            "foot_contact_r": 0.35,
        }
    )
    assert ok
    ok2, diag = override_recovered_posture_ok(
        {"posture_stability": 0.2, "com_z": 0.5, "foot_contact_l": 0.5, "foot_contact_r": 0.5}
    )
    assert not ok2
    assert diag.get("override_exit_block") == "posture_low"
