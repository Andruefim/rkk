"""
Track C5: promote surviving latent confounders to universal learned role_type entries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.latent_confounder import LatentRecord, signature_similarity


def c5_enabled() -> bool:
    return os.environ.get("RKK_C5_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def promote_min_worlds() -> int:
    try:
        return max(2, int(os.environ.get("RKK_PROMOTE_MIN_WORLDS", "2")))
    except ValueError:
        return 2


def promote_signature_match() -> float:
    try:
        return float(os.environ.get("RKK_PROMOTE_SIGNATURE_MATCH", "0.60"))
    except ValueError:
        return 0.60


@dataclass
class LearnedRoleEntry:
    latent_id: str
    role_type: str
    signature: list[float]
    worlds: list[str] = field(default_factory=list)
    source_role_cluster: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "role_type": self.role_type,
            "signature": [round(x, 5) for x in self.signature],
            "worlds": list(self.worlds),
            "source_role_cluster": self.source_role_cluster,
        }


# In-memory registry (persist via genome compressor / checkpoint in later phases).
learned_roles: dict[str, LearnedRoleEntry] = {}


def _next_learned_role_name(cluster: str) -> str:
    base = f"learned_{cluster or 'latent'}"
    if base not in learned_roles:
        return base
    i = 2
    while f"{base}_{i}" in learned_roles:
        i += 1
    return f"{base}_{i}"


def promote_to_universal_concept(
    record: LatentRecord,
    graph: Any,
    *,
    world_id: str,
    force: bool = False,
) -> LearnedRoleEntry | None:
    """
    After TTL pass in ``world_id``, register latent for cross-world promotion.
    When the same signature appears in >= RKK_PROMOTE_MIN_WORLDS worlds, emit
    a learned role_type usable by role map / eval (Track B3-style).
    """
    if not c5_enabled() and not force:
        return None
    if record.ttl_passed is not True:
        return None
    wid = str(world_id or "humanoid")
    if wid not in record.worlds_survived:
        record.worlds_survived.append(wid)
    sig = record.signature_vector(graph)
    match_thresh = promote_signature_match()

    for entry in learned_roles.values():
        es = np.asarray(entry.signature, dtype=np.float64)
        if signature_similarity(sig, es) >= match_thresh:
            if wid not in entry.worlds:
                entry.worlds.append(wid)
            if len(entry.worlds) >= promote_min_worlds():
                _apply_learned_role_to_graph(graph, entry)
                return entry
            return entry

    role_name = _next_learned_role_name(record.role_cluster)
    entry = LearnedRoleEntry(
        latent_id=record.node_id,
        role_type=role_name,
        signature=sig.tolist(),
        worlds=[wid],
        source_role_cluster=record.role_cluster,
    )
    learned_roles[role_name] = entry
    if len(entry.worlds) >= promote_min_worlds():
        _apply_learned_role_to_graph(graph, entry)
    return entry


def _apply_learned_role_to_graph(graph: Any, entry: LearnedRoleEntry) -> None:
    """Tag promoted latent node with learned role_type."""
    nid = entry.latent_id
    if nid not in getattr(graph, "_node_meta", {}):
        return
    meta = graph._node_meta[nid]
    meta.role_type = entry.role_type
    if hasattr(graph, "nodes") and nid in graph.nodes:
        graph.nodes[nid] = float(graph.nodes.get(nid, 0.5))


def try_promote_all_passed(
    manager_records: list[LatentRecord],
    graph: Any,
    *,
    world_id: str,
) -> list[LearnedRoleEntry]:
    promoted: list[LearnedRoleEntry] = []
    for rec in manager_records:
        ent = promote_to_universal_concept(rec, graph, world_id=world_id)
        if ent is not None and len(ent.worlds) >= promote_min_worlds():
            promoted.append(ent)
    return promoted


def learned_roles_snapshot() -> dict[str, Any]:
    return {
        "enabled": c5_enabled(),
        "promote_min_worlds": promote_min_worlds(),
        "entries": [e.as_dict() for e in learned_roles.values()],
    }


def reset_learned_roles() -> None:
    learned_roles.clear()
