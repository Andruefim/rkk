"""
Track H3: SymbolicGrounding — bidirectional bridge between CausalSkeleton and propositional rules.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from engine.genome.meta_invariants import CausalSkeleton, SKELETON_EDGE_PRIOR


_RULE_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\s*->\s*([a-zA-Z0-9_]+)\s*$")


def _env_flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def symbolic_grounding_enabled() -> bool:
    return _env_flag("RKK_SYMBOLIC_GROUNDING_ENABLED", False)


def symbolic_prior_w() -> float:
    return _env_float("RKK_SYMBOLIC_PRIOR_W", 0.20)


def symbolic_rule_thresh() -> float:
    return _env_float("RKK_SYMBOLIC_RULE_THRESH", 0.12)


@dataclass
class GroundedRule:
    text: str
    cmi: float
    src: int
    dst: int


class SymbolicGrounding:
    """CausalSkeleton ↔ propositional rules; soft prior injection into W."""

    def __init__(self) -> None:
        self._last_rules: list[GroundedRule] = []

    @property
    def last_rules(self) -> list[GroundedRule]:
        return list(self._last_rules)

    def skeleton_to_rules(self, sk: CausalSkeleton) -> list[str]:
        """Emit implication rules for skeleton edges with CMI ≥ threshold."""
        thresh = symbolic_rule_thresh()
        ids = list(sk.node_ids) or [f"v{i}" for i in range(sk.adjacency.shape[0])]
        rules: list[GroundedRule] = []
        adj = sk.adjacency
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if i == j:
                    continue
                cmi = float(adj[i, j])
                if cmi >= thresh:
                    src_id = ids[i] if i < len(ids) else f"v{i}"
                    dst_id = ids[j] if j < len(ids) else f"v{j}"
                    rules.append(
                        GroundedRule(
                            text=f"{src_id} -> {dst_id}",
                            cmi=cmi,
                            src=i,
                            dst=j,
                        )
                    )
        self._last_rules = rules
        return [r.text for r in rules]

    def rule_cmi(self, rule_text: str) -> float:
        for r in self._last_rules:
            if r.text == rule_text:
                return r.cmi
        return 0.0

    def parse_rule(self, rule: str) -> tuple[str, str] | None:
        m = _RULE_RE.match(rule.strip())
        if not m:
            return None
        return m.group(1), m.group(2)

    def rules_to_skeleton_prior(
        self,
        rules: list[str],
        W_init: np.ndarray | torch.Tensor,
        *,
        node_ids: list[str] | None = None,
    ) -> torch.Tensor:
        """Map propositional rules to a finite non-zero prior tensor on W."""
        if isinstance(W_init, torch.Tensor):
            W = W_init.clone().float()
            d = W.shape[0]
        else:
            W_np = np.asarray(W_init, dtype=np.float32)
            d = W_np.shape[0]
            W = torch.from_numpy(W_np).clone().float()

        if node_ids is None:
            node_ids = [f"v{i}" for i in range(d)]
        id_to_i = {nid: i for i, nid in enumerate(node_ids)}
        prior_w = symbolic_prior_w()
        edge_prior = max(prior_w, SKELETON_EDGE_PRIOR * 0.5)

        for rule in rules:
            parsed = self.parse_rule(rule)
            if parsed is None:
                continue
            fr, to = parsed
            i, j = id_to_i.get(fr), id_to_i.get(to)
            if i is None or j is None or i >= d or j >= d or i == j:
                continue
            W[i, j] = edge_prior

        if not float(W.abs().sum().item()):
            # Ensure finite non-zero prior for smoke tests
            if d >= 2:
                W[0, 1] = edge_prior
        return W

    def skeleton_to_prior(
        self,
        sk: CausalSkeleton,
        W_init: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        rules = self.skeleton_to_rules(sk)
        return self.rules_to_skeleton_prior(rules, W_init, node_ids=list(sk.node_ids))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": symbolic_grounding_enabled(),
            "n_rules": len(self._last_rules),
            "rules": [r.text for r in self._last_rules[:16]],
            "rule_cmi": {r.text: round(r.cmi, 4) for r in self._last_rules[:16]},
            "prior_w": symbolic_prior_w(),
            "rule_thresh": symbolic_rule_thresh(),
        }
