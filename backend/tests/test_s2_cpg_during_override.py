"""CPG during S2 fallen override."""
from __future__ import annotations

from engine.features.simulation.mixin_locomotion import SimulationLocomotionMixin


def test_s2_cpg_during_override_enabled_default():
    assert SimulationLocomotionMixin._s2_cpg_during_override_enabled()
