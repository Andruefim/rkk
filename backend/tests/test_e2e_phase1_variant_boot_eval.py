"""E2e smoke after second-world: variant boot + role map (skips heavy PyBullet by default)."""
from __future__ import annotations

import os

import pytest
import torch

from engine.causal_graph import CausalGraph
from engine.environment_humanoid_variant import EnvironmentHumanoidVariant
from engine.features.humanoid.constants import VAR_NAMES
from engine.role_types import build_role_map, validate_role_map


@pytest.mark.skipif(
    os.environ.get("RKK_RUN_PYBULLET_E2E", "0").strip() not in ("1", "true", "yes"),
    reason="set RKK_RUN_PYBULLET_E2E=1 for full PyBullet e2e",
)
def test_variant_boot_eval_ticks_no_crash():
    env = EnvironmentHumanoidVariant(device=torch.device("cpu"))
    g = CausalGraph(torch.device("cpu"))
    g.set_env_preset("humanoid_variant")
    obs = env.observe()
    for vid in env.variable_ids:
        g.set_node(vid, float(obs.get(vid, 0.5)))
    role_map = build_role_map(env.variable_ids, env_preset="humanoid_variant")
    validate_role_map(env.variable_ids, role_map)
    for _ in range(5):
        obs = env.observe()
        for vid in env.variable_ids:
            g.set_node(vid, float(obs.get(vid, 0.5)))


def test_variant_role_map_covers_all_vars():
    env = EnvironmentHumanoidVariant(device=torch.device("cpu"))
    role_map = build_role_map(env.variable_ids, env_preset="humanoid_variant")
    assert len(role_map) == len(VAR_NAMES)
    validate_role_map(env.variable_ids, role_map)
