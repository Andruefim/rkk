"""Phase 4 smoke: spectral transfer, role discovery, causal skeleton."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from engine.causal_graph import CausalGraph
from engine.features.humanoid.constants import VAR_NAMES
from engine.genome.learned_roles import LearnedRoleEntry, learned_roles, reset_learned_roles
from engine.genome.meta_invariants import (
    CausalSkeleton,
    extract_causal_skeleton,
    skeleton_similarity,
    transfer_skeleton_nonphys,
    transfer_skeleton_to_env,
)
from engine.genome.role_discovery import discover_roles_in_new_env
from engine.genome.spectral import (
    CARTPOLE_VARIABLE_IDS,
    GRID_NAV_VARIABLE_IDS,
    procrustes_align,
    spectral_fingerprint,
    spectral_similarity,
    transfer_W_spectral,
)
from engine.role_types import build_role_map


def _tiny_humanoid_graph(d: int = 12) -> CausalGraph:
    g = CausalGraph(torch.device("cpu"))
    g.set_env_preset("humanoid")
    for i, nid in enumerate(VAR_NAMES[:d]):
        g.set_node(nid, 0.4 + 0.01 * i)
    g._rebuild_core()
    for t in range(24):
        obs = {nid: float(g.nodes.get(nid, 0.5)) + 0.01 * (t % 5) for nid in g._node_ids}
        g.record_observation(obs)
    return g


def test_spectral_fingerprint_and_procrustes():
    rng = np.random.default_rng(1)
    W = rng.normal(scale=0.1, size=(10, 10))
    W = np.triu(W, 1)
    F = spectral_fingerprint(torch.from_numpy(W), k=4)
    assert F.shape == (10, 4)
    R = procrustes_align(F, F)
    assert R.shape[0] >= 4
    assert spectral_similarity(F, F) >= 0.9


def test_transfer_W_spectral_humanoid_to_cartpole():
    g = _tiny_humanoid_graph()
    W, ids = np.zeros((len(g._node_ids), len(g._node_ids))), list(g._node_ids)
    edges = g.edges.values() if hasattr(g.edges, "values") else g.edges
    for e in edges:
        i, j = ids.index(e.from_), ids.index(e.to)
        W[i, j] = e.weight
    W_tgt, meta = transfer_W_spectral(
        W, ids, list(CARTPOLE_VARIABLE_IDS), env_ref="humanoid", env_target="cartpole"
    )
    assert W_tgt.shape[0] == len(CARTPOLE_VARIABLE_IDS)
    assert "similarity" in meta
    assert W_tgt.sum() >= 0.0


def test_discover_roles_in_new_env(monkeypatch):
    monkeypatch.setenv("RKK_C6_ENABLED", "1")
    reset_learned_roles()
    learned_roles["learned_balance"] = LearnedRoleEntry(
        latent_id="latent_X_test",
        role_type="learned_balance",
        signature=[0.2, 0.5, 0.8, 0.1, 0.3, 0.6, 0.4, 0.2],
        worlds=["humanoid"],
    )
    rng = np.random.default_rng(2)
    n = len(CARTPOLE_VARIABLE_IDS)
    W = rng.normal(scale=0.12, size=(n, n))
    W = np.triu(W, 1)
    g = CausalGraph(torch.device("cpu"))
    for nid in CARTPOLE_VARIABLE_IDS:
        g.set_node(nid, 0.5)
    g._rebuild_core()
    out = discover_roles_in_new_env(g, W=W, node_ids=list(CARTPOLE_VARIABLE_IDS), force=True)
    assert isinstance(out, dict)


def test_extract_and_transfer_skeleton():
    g = _tiny_humanoid_graph()
    ids = list(g._node_ids)
    role_map = build_role_map(ids)
    obs = list(g._obs_buffer)
    W = np.zeros((len(ids), len(ids)))
    sk = extract_causal_skeleton(W, obs, role_map, node_ids=ids)
    assert isinstance(sk, CausalSkeleton)
    assert sk.adjacency.shape[0] == len(ids)
    assert sk.scale_structure in ("hierarchical", "feedback")

    sk2 = extract_causal_skeleton(W, obs, role_map, node_ids=ids[:6])
    assert skeleton_similarity(sk, sk2) >= 0.0

    W0 = np.zeros((len(CARTPOLE_VARIABLE_IDS), len(CARTPOLE_VARIABLE_IDS)), dtype=np.float32)
    W_cp = transfer_skeleton_to_env(sk, W0, "cartpole", force=True)
    assert float(W_cp.abs().sum()) > 0.0

    W_grid = transfer_skeleton_nonphys(sk, W0, "grid_nav", {})
    assert W_grid.shape[0] == len(GRID_NAV_VARIABLE_IDS)

    W_gn = transfer_skeleton_to_env(sk, W0, "grid_control", force=True)
    assert W_gn.shape[0] == len(GRID_NAV_VARIABLE_IDS)
    assert W_cp.shape[0] == len(CARTPOLE_VARIABLE_IDS)
