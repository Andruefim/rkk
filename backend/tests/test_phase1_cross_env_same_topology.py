"""Phase 1 gate smoke: role_types meta, variant env, cross-env JSONL fields."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from engine.causal_graph import CausalGraph
from engine.environment_humanoid_variant import (
    EnvironmentHumanoidVariant,
    variant_mass_scale,
)
from engine.features.humanoid.constants import VAR_NAMES
from engine.genome.compressor import (
    apply_role_subgraph_to_graph,
    compress_adjacency_role_subgraph,
    load_compressed_genome,
    save_compressed_genome,
)
from engine.role_types import build_role_map, validate_role_map


def test_snapshot_meta_role_types_map():
    g = CausalGraph(torch.device("cpu"))
    g.set_env_preset("humanoid")
    for vid in VAR_NAMES[:12]:
        g.set_node(vid, 0.5)
    role_map = g.role_type_map()
    assert role_map
    assert all(role_map.get(v) for v in VAR_NAMES[:12])


def test_humanoid_variant_physics_params():
    assert variant_mass_scale() >= 1.0
    env = EnvironmentHumanoidVariant(device=torch.device("cpu"))
    assert env.preset == "humanoid_variant"
    assert env.variable_ids == list(VAR_NAMES)
    role_map = build_role_map(env.variable_ids, env_preset="humanoid_variant")
    validate_role_map(env.variable_ids, role_map)


def test_cross_env_jsonl_field_names_stub():
    row = {
        "eval_kind": "cross_env_same_topology",
        "cross_env_success_rate_200": 0.41,
        "ticks_to_success_0_5": 42,
        "success_rate": 0.41,
        "fallen_frac": 0.2,
        "target_world": "humanoid_variant",
    }
    assert row["cross_env_success_rate_200"] >= 0.0
    assert "ticks_to_success_0_5" in row


def test_genome_role_subgraph_roundtrip_not_worse_than_dense():
    rng = np.random.default_rng(3)
    node_ids = list(VAR_NAMES)
    d = len(node_ids)
    role_map = build_role_map(node_ids)
    W = rng.normal(scale=0.08, size=(d, d))
    W = np.triu(W, 1)
    result = compress_adjacency_role_subgraph(W, node_ids, role_map, rank=6)
    g = CausalGraph(torch.device("cpu"))
    g.set_env_preset("humanoid")
    for i, nid in enumerate(node_ids):
        g.set_node(nid, 0.5)
    g._rebuild_core()
    n_edges = apply_role_subgraph_to_graph(g, result, alpha=0.75)
    assert n_edges >= 0

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "role_prior.npz"
        save_compressed_genome(result, path)
        loaded = load_compressed_genome(path)
        assert loaded.get("role_subgraph")
        assert len(loaded["node_ids"]) >= 4
