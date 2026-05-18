"""Unified System2+VLM JSON: macro payload must ignore extra slot_labels."""
from __future__ import annotations

from engine.system2.schema import proposal_from_dict
from engine.system2.validate import validate_proposal


def test_proposal_from_dict_ignores_slot_labels_key() -> None:
    raw = {
        "macro": "RECOVER_POSTURE",
        "rationale": "test",
        "slot_labels": {
            "slot_0": {
                "label": "floor",
                "likely_phys": ["com_z"],
                "confidence": 0.4,
            }
        },
    }
    p = validate_proposal(proposal_from_dict(raw))
    assert p is not None
    assert p.normalized_macro() == "RECOVER_POSTURE"
