"""
Pearl-style semantics (Phase A): explicit exogenous node set U for honest L3 counterfactuals.

L1 observe_predict: one-step passive rollout f(X, 0) — prediction without intervention.
L2 intervene_predict: P(Y | do(X=x)) via world-model forward_dynamics (same as propagate_from).
L3 counterfactual_predict: after an atomic do() prediction, coordinates in U keep factual values
from the observation dict (exogenous context fixed); endogenous predictions follow the structural
map. This is **not** full Pearl abduction (inferring U from data); U must be labeled via config.

Config:
  RKK_PEARL_API=1          — gate Pearl helpers on CausalGraph (always available; flag for telemetry)
  RKK_EXOGENOUS_NODE_IDS — comma-separated node ids treated as U (merged with defaults below)

Schema version for physics_context / episodic (Phase D/I): bump when changing PyBullet mappings.
"""
from __future__ import annotations

import os


PHYSICS_CONTEXT_SCHEMA_VERSION = "1"


def default_exogenous_node_ids() -> frozenset[str]:
    """
    Conservative defaults: environment / vestibular / extrinsic cues agents do not set via policy.
    Override and extend via RKK_EXOGENOUS_NODE_IDS.
    """
    base = frozenset(
        {
            "vestibular_gx",
            "vestibular_gy",
            "vestibular_gz",
            "floor_friction",
            "cube0_x",
            "cube0_y",
            "cube0_z",
            "cube1_x",
            "cube1_y",
            "cube1_z",
            "cube2_x",
            "cube2_y",
            "cube2_z",
            "ball_x",
            "ball_y",
            "ball_z",
            "target_dist",
            "stack_height",
        }
    )
    raw = os.environ.get("RKK_EXOGENOUS_NODE_IDS", "")
    extra: set[str] = set()
    if raw.strip():
        for part in raw.split(","):
            p = part.strip()
            if p:
                extra.add(p)
    return frozenset(base | extra)


def pearl_api_enabled() -> bool:
    return os.environ.get("RKK_PEARL_API", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def exogenous_ids_for_graph(node_ids: list[str]) -> frozenset[str]:
    """Intersection of configured U with existing graph coordinates."""
    u = default_exogenous_node_ids()
    return frozenset(n for n in node_ids if n in u)
