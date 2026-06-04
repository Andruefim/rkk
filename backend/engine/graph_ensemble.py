"""
graph_ensemble.py — Bayesian ensemble of causal graph hypotheses (AGI Phase 2).

Holds N adjacency matrices W_k with log-posterior weights π_k.
Executive CausalGNNCore uses MAP/mixture W for fast rollouts; ensemble
tracks structural uncertainty for EIG and Bayes updates.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensemble_size() -> int:
    try:
        return max(1, min(16, int(os.environ.get("RKK_GRAPH_ENSEMBLE_N", "4"))))
    except ValueError:
        return 4


def ensemble_enabled() -> bool:
    return ensemble_size() > 1


class WeightedGraphEnsemble(nn.Module):
    """N weighted hypotheses over d×d adjacency matrices."""

    def __init__(
        self,
        d: int,
        device: torch.device,
        n: int | None = None,
        seed_W: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d = d
        self.device = device
        self.n = n if n is not None else ensemble_size()

        if seed_W is not None:
            base = seed_W.detach().clone().to(device)
            W_stack = base.unsqueeze(0).expand(self.n, -1, -1).clone()
            noise = torch.randn_like(W_stack) * 0.08
            W_stack = W_stack + noise
        else:
            W_stack = torch.randn(self.n, d, d, device=device) * 0.02

        self.register_buffer("W_stack", W_stack)
        self.log_weights = nn.Parameter(torch.zeros(self.n, device=device))

        mask = 1.0 - torch.eye(d, device=device)
        self.register_buffer("mask", mask)

    def posterior(self) -> torch.Tensor:
        return F.softmax(self.log_weights, dim=0)

    def posterior_mean(self) -> torch.Tensor:
        w = self.posterior()
        return torch.einsum("n,nij->ij", w, self.W_stack * self.mask)

    def map_graph(self) -> torch.Tensor:
        idx = int(self.posterior().argmax().item())
        return self.W_stack[idx] * self.mask

    def sample_graph(self) -> torch.Tensor:
        idx = torch.multinomial(self.posterior(), 1).item()
        return self.W_stack[int(idx)] * self.mask

    def entropy(self) -> float:
        p = self.posterior().detach()
        h = -(p * (p + 1e-12).log()).sum()
        return float(h.item())

    @torch.no_grad()
    def update_posterior(self, log_likelihood: torch.Tensor) -> None:
        """Bayes update: log π_k += log_likelihood_k."""
        ll = log_likelihood.detach().flatten()
        if ll.numel() != self.n:
            return
        self.log_weights.add_(ll)

    def sync_from_executive(self, W: torch.Tensor, idx: int = 0) -> None:
        """Copy executive W into hypothesis idx."""
        with torch.no_grad():
            i = max(0, min(self.n - 1, idx))
            d = min(self.d, W.shape[0])
            self.W_stack[i, :d, :d].copy_(W.detach()[:d, :d])

    @torch.no_grad()
    def sync_latent_edges(
        self,
        node_ids: list[str],
        edge_pairs: list[tuple[str, str, float]],
        *,
        latent_id: str | None = None,
    ) -> int:
        """
        C4: copy latent confounder edges identically into all W_k hypotheses.
        """
        if not edge_pairs and latent_id is None:
            return 0
        updated = 0
        for k in range(self.n):
            W = self.W_stack[k]
            for fr, to, w in edge_pairs:
                if fr not in node_ids or to not in node_ids:
                    continue
                i, j = node_ids.index(fr), node_ids.index(to)
                W[i, j] = float(w)
                updated += 1
            if latent_id and latent_id in node_ids:
                li = node_ids.index(latent_id)
                W[li, li] = max(float(W[li, li].item()), 0.08)
        return updated

    @torch.no_grad()
    def apply_vstructure_orientations(
        self,
        idx_a: int,
        idx_c: int,
        idx_b: int,
        *,
        n_orientations: int = 4,
    ) -> int:
        """
        C3: assign distinct collider orientations across ensemble hypotheses.
        Returns number of hypotheses updated.
        """
        n = min(max(1, n_orientations), self.n)
        patterns: list[list[tuple[int, int, float]]] = [
            [(idx_a, idx_c, 0.22), (idx_b, idx_c, 0.22)],
            [(idx_c, idx_a, 0.18), (idx_c, idx_b, 0.18)],
            [(idx_a, idx_c, 0.20), (idx_c, idx_b, 0.16)],
            [(idx_c, idx_a, 0.16), (idx_b, idx_c, 0.20)],
        ]
        for k in range(n):
            W = self.W_stack[k]
            for i, j, w in patterns[k % len(patterns)]:
                if i == j:
                    continue
                old = float(W[i, j].item())
                W[i, j] = 0.65 * old + 0.35 * float(w)
        return n

    def snapshot(self) -> dict:
        p = self.posterior().detach().cpu().tolist()
        out = {
            "N": self.n,
            "d": self.d,
            "entropy": round(self.entropy(), 5),
            "weights": [round(x, 5) for x in p],
            "map_idx": int(self.posterior().argmax().item()),
        }
        if os.environ.get("RKK_LOG_DISCOVERY_SPLIT", "1").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            out["vstructure_orientations"] = min(
                4, max(0, int(os.environ.get("RKK_VSTRUCTURE_ENSEMBLE_N", "4") or 4))
            )
        return out
