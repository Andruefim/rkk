"""Schedule-only WM candidate must include System1 features."""
from __future__ import annotations

from unittest.mock import MagicMock

from engine.agent import RKKAgent


def test_enrich_schedule_candidate_adds_features():
    agent = MagicMock(spec=RKKAgent)
    agent._features_for_intervention_pair = MagicMock(return_value=[0.1, 0.2, 0.3])
    enriched = RKKAgent._enrich_s2_wm_candidate(
        agent,
        {
            "variable": "intent_stop_recover",
            "value": 0.62,
            "target": "posture_stability",
        },
        macro="RECOVER_POSTURE",
    )
    assert enriched["features"] == [0.1, 0.2, 0.3]
    assert enriched.get("from_s2_wm_planner") is True
    agent._features_for_intervention_pair.assert_called_once_with(
        "intent_stop_recover", "posture_stability"
    )
