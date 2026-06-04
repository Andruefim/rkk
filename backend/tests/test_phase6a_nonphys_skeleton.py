"""Phase 6a gate smoke: non-physical stubs, skeleton transfer, symbolic grounding."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from engine.core.world import WORLDS, _make_env
from engine.environment_grid_nav import EnvironmentGridNav
from engine.environment_symbolic import EnvironmentSymbolic
from engine.genome.meta_invariants import (
    CausalSkeleton,
    extract_causal_skeleton,
    transfer_skeleton_nonphys,
)
from engine.genome.spectral import GRID_NAV_VARIABLE_IDS, SYMBOLIC_CONTROL_VARIABLE_IDS
from engine.scorecard.autonomy_scorecard import build_scorecard
from engine.scorecard.world_autonomy_contract import get_contract
from engine.symbolic_grounding import SymbolicGrounding


def _tiny_skeleton() -> CausalSkeleton:
    adj = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )
    return CausalSkeleton(
        adjacency=adj,
        scale_structure="hierarchical",
        node_ids=["a", "b", "c"],
    )


def test_worlds_registered_and_stubs_load():
    assert "grid_nav" in WORLDS
    assert "symbolic_control" in WORLDS
    gn = EnvironmentGridNav()
    sc = EnvironmentSymbolic()
    assert gn.preset == "grid_nav"
    assert sc.preset == "symbolic_control"
    _make_env("grid_nav", torch.device("cpu"))
    _make_env("symbolic_control", torch.device("cpu"))


def test_transfer_skeleton_nonphys_grid_nav_500_steps():
    sk = _tiny_skeleton()
    W0 = np.zeros((len(GRID_NAV_VARIABLE_IDS), len(GRID_NAV_VARIABLE_IDS)), dtype=np.float32)
    W_grid = transfer_skeleton_nonphys(sk, W0, "grid_nav", {})
    assert W_grid.shape[0] == len(GRID_NAV_VARIABLE_IDS)
    assert float(W_grid.abs().sum()) > 0.0

    env = EnvironmentGridNav()
    rng = np.random.default_rng(42)
    for _ in range(500):
        env.step_random(rng)
    assert env._ticks == 500


def test_symbolic_grounding_rules_and_prior():
    sk = _tiny_skeleton()
    adj = sk.adjacency.copy()
    adj[0, 1] = 0.15
    adj[1, 2] = 0.18
    sk2 = CausalSkeleton(
        adjacency=adj,
        scale_structure="hierarchical",
        node_ids=["a", "b", "c"],
    )
    g = SymbolicGrounding()
    rules = g.skeleton_to_rules(sk2)
    assert len(rules) >= 1
    assert any(g.rule_cmi(r) > 0.12 for r in rules)

    W_init = torch.zeros(5, 5)
    prior = g.rules_to_skeleton_prior(rules, W_init, node_ids=["a", "b", "c", "d", "e"])
    assert torch.isfinite(prior).all()
    assert float(prior.abs().sum()) > 0.0


def test_contract_a1_a4_probes_registered():
    for wid in ("grid_nav", "symbolic_control"):
        c = get_contract(wid)
        assert c is not None
        assert c.a1_probe_key
        assert c.a4_probe_key
    card = build_scorecard(
        {
            "pathfinder_override_frac": 0.05,
            "stuck_override_active": 0.0,
            "rule_engine_bailout_frac": 0.04,
            "constraint_violation_override": 0.0,
            "current_world": "grid_nav",
        },
        worlds=["grid_nav", "symbolic_control"],
    )
    assert "a1_probe" in card["worlds"]["grid_nav"]
    assert "a4_probe" in card["worlds"]["symbolic_control"]


def test_extract_skeleton_symbolic_vars():
    ids = list(SYMBOLIC_CONTROL_VARIABLE_IDS)
    W = np.zeros((len(ids), len(ids)))
    obs = [{nid: 0.5 for nid in ids} for _ in range(12)]
    sk = extract_causal_skeleton(W, obs, node_ids=ids)
    assert sk.adjacency.shape[0] == len(ids)
