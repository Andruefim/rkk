"""phys_intent_* → intent_* mapping in EnvironmentHumanoid."""
from __future__ import annotations

import unittest

from engine.features.humanoid.environment import (
    EnvironmentHumanoid,
    canonical_motor_intent_variable,
)


class PhysIntentMappingTests(unittest.TestCase):
    def test_canonical_known_suffix(self) -> None:
        self.assertEqual(
            canonical_motor_intent_variable("phys_intent_lean_forward"),
            "intent_lean_forward",
        )
        self.assertEqual(
            canonical_motor_intent_variable("intent_lean_forward"),
            "intent_lean_forward",
        )

    def test_canonical_unknown_returns_original(self) -> None:
        self.assertEqual(
            canonical_motor_intent_variable("phys_intent_not_a_real_intent"),
            "phys_intent_not_a_real_intent",
        )

    def test_intervene_phys_intent_updates_motor_state(self) -> None:
        env = EnvironmentHumanoid(fixed_root=True)
        env.intervene("phys_intent_lean_forward", 0.71, count_intervention=False)
        self.assertAlmostEqual(float(env._motor_state["intent_lean_forward"]), 0.71, places=5)


if __name__ == "__main__":
    unittest.main()
