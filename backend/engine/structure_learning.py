"""
structure_learning.py — v-structure detection and PC orientation rules (Phase 2).

Collider test: A — C — B with A independent of B given C but dependent given C,S → orient A→C←B.
Meek rules applied after conditional dependence discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DirectedEdge:
    from_idx: int
    to_idx: int
    confidence: float = 1.0


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation ρ(x,y|z) for 1D arrays."""
    if len(x) < 5:
        return 0.0
    xz = np.column_stack([x, z])
    yz = np.column_stack([y, z])
    rx = x - xz @ np.linalg.lstsq(xz, x, rcond=None)[0]
    ry = y - yz @ np.linalg.lstsq(yz, y, rcond=None)[0]
    denom = np.std(rx) * np.std(ry)
    if denom < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def detect_v_structures(
    data: np.ndarray,
    *,
    margin: float = 0.15,
    cond_set_size: int = 1,
) -> list[tuple[int, int, int]]:
    """
    Find collider candidates (A, C, B): A—C—B where A⊥B marginally but A⊥̸B|C.

    data: (T, d) observations.
    Returns list of (A, C, B) index triples (orient A→C←B).
    """
    T, d = data.shape
    if T < 10 or d < 3:
        return []

    colliders: list[tuple[int, int, int]] = []
    for c in range(d):
        for a in range(d):
            if a == c:
                continue
            for b in range(a + 1, d):
                if b == c:
                    continue
                rho_ab = abs(float(np.corrcoef(data[:, a], data[:, b])[0, 1]))
                rho_ac = abs(float(np.corrcoef(data[:, a], data[:, c])[0, 1]))
                rho_bc = abs(float(np.corrcoef(data[:, b], data[:, c])[0, 1]))
                rho_ab_c = abs(_partial_corr(data[:, a], data[:, b], data[:, c]))
                # Collider: A,B marginally independent, both correlate with C
                if rho_ab < 0.35 and rho_ac > 0.25 and rho_bc > 0.25:
                    colliders.append((a, c, b))
                elif rho_ab_c > margin:
                    colliders.append((a, c, b))
    return colliders


def orient_colliders(
    colliders: Iterable[tuple[int, int, int]],
    d: int,
) -> list[DirectedEdge]:
    """Orient A→C←B for each v-structure."""
    edges: list[DirectedEdge] = []
    for a, c, b in colliders:
        edges.append(DirectedEdge(a, c, 0.9))
        edges.append(DirectedEdge(b, c, 0.9))
    return edges


def meek_rule_orient(
    adj: np.ndarray,
    directed: list[DirectedEdge],
) -> list[DirectedEdge]:
    """
    Simple Meek rules: if i→k and i—j—k with no edge i—j, orient j→k.
    adj: undirected skeleton (symmetric, binary).
    """
    out = list(directed)
    known = {(e.from_idx, e.to_idx) for e in out}
    d = adj.shape[0]
    changed = True
    while changed:
        changed = False
        for i in range(d):
            for k in range(d):
                if (i, k) not in known:
                    continue
                for j in range(d):
                    if j in (i, k):
                        continue
                    if adj[i, j] > 0 and adj[j, k] > 0:
                        if (i, j) not in known and (j, i) not in known:
                            if (j, k) not in known:
                                out.append(DirectedEdge(j, k, 0.7))
                                known.add((j, k))
                                changed = True
    return out


def apply_molecular_constraints(
    edges: list[DirectedEdge],
    node_kinds: dict[int, str],
) -> list[DirectedEdge]:
    """
    Molecular tags: sensor has no parents; motor cannot parent another motor.
    """
    filtered: list[DirectedEdge] = []
    for e in edges:
        child_kind = node_kinds.get(e.to_idx, "latent")
        parent_kind = node_kinds.get(e.from_idx, "latent")
        if child_kind == "sensor":
            continue
        if parent_kind == "motor" and child_kind == "motor":
            continue
        filtered.append(e)
    return filtered


def structure_learn_step(
    data: np.ndarray,
    node_kinds: dict[int, str] | None = None,
) -> list[DirectedEdge]:
    """Full pipeline: v-structures → orient → Meek → molecular filter."""
    colliders = detect_v_structures(data)
    edges = orient_colliders(colliders, data.shape[1])
    adj = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            if abs(np.corrcoef(data[:, i], data[:, j])[0, 1]) > 0.2:
                adj[i, j] = adj[j, i] = 1.0
    edges = meek_rule_orient(adj, edges)
    if node_kinds:
        edges = apply_molecular_constraints(edges, node_kinds)
    return edges
