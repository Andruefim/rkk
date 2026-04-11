"""
concept_store.py — SemanticConceptStore: семантический словарь внутренних концептов (inner voice).

512 dense embeddings (d=64) — «словарь ситуаций» агента.
Инициализируется вручную из биомеханических понятий,
затем дообучается через LLM distillation.

Концепты разделены по доменам:
  BALANCE   — состояние баланса тела
  GAIT      — качество походки
  STABILITY — устойчивость позы
  FALL      — риск и процесс падения
  RECOVERY  — восстановление после падения
  INTENT    — намерения и цели движения
  SKILL     — активные навыки
  ANOMALY   — аномальные состояния суставов

Поиск: cosine similarity → top-K активных концептов → узлы GNN
Обучение: contrastive loss (CLIP-style) vs LLM text embeddings
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


CONCEPT_DIM = 64    # embedding dimension
VOCAB_SIZE  = 512   # number of concepts


# ── Concept definitions ────────────────────────────────────────────────────────
# (name, domain, description, key_vars_hint)
CONCEPT_DEFS: list[tuple[str, str, str]] = [
    # BALANCE
    ("STABLE_BALANCE",        "BALANCE", "body is well balanced, CoM centered over feet"),
    ("FORWARD_LEAN",          "BALANCE", "torso leaning forward, CoM ahead of feet"),
    ("BACKWARD_LEAN",         "BALANCE", "torso leaning backward, fall risk"),
    ("LATERAL_TILT_LEFT",     "BALANCE", "body tilting left, uneven weight"),
    ("LATERAL_TILT_RIGHT",    "BALANCE", "body tilting right, uneven weight"),
    ("LOSING_BALANCE",        "BALANCE", "balance deteriorating rapidly"),
    ("REGAINING_BALANCE",     "BALANCE", "actively correcting balance"),

    # GAIT
    ("GOOD_STRIDE",           "GAIT", "stride length and symmetry are appropriate"),
    ("OVERSTRIDE",            "GAIT", "stride too wide for current posture, fall risk"),
    ("UNDERSTRIDE",           "GAIT", "stride too small, shuffling gait"),
    ("ASYMMETRIC_GAIT",       "GAIT", "left-right gait asymmetry detected"),
    ("SYMMETRIC_GAIT",        "GAIT", "left-right gait is symmetric"),
    ("GAIT_TRANSITION",       "GAIT", "transitioning between gait phases"),
    ("STANCE_PHASE_LEFT",     "GAIT", "left foot in stance (supporting weight)"),
    ("STANCE_PHASE_RIGHT",    "GAIT", "right foot in stance (supporting weight)"),
    ("SWING_PHASE_LEFT",      "GAIT", "left foot in swing (airborne)"),
    ("SWING_PHASE_RIGHT",     "GAIT", "right foot in swing (airborne)"),
    ("DOUBLE_SUPPORT",        "GAIT", "both feet on ground simultaneously"),
    ("SINGLE_SUPPORT_LEFT",   "GAIT", "only left foot on ground"),
    ("SINGLE_SUPPORT_RIGHT",  "GAIT", "only right foot on ground"),

    # STABILITY
    ("HIGH_STABILITY",        "STABILITY", "posture_stability > 0.75"),
    ("MEDIUM_STABILITY",      "STABILITY", "posture_stability 0.5-0.75"),
    ("LOW_STABILITY",         "STABILITY", "posture_stability < 0.5"),
    ("CRITICAL_STABILITY",    "STABILITY", "posture_stability < 0.3, near fall"),
    ("STATIC_STANCE",         "STABILITY", "standing still, minimal movement"),
    ("COM_HIGH",              "STABILITY", "center of mass at good height"),
    ("COM_LOW",               "STABILITY", "center of mass low, crouched or fallen"),

    # FALL
    ("NO_FALL_RISK",          "FALL", "all metrics nominal, no fall imminent"),
    ("LOW_FALL_RISK",         "FALL", "minor instability, monitoring"),
    ("HIGH_FALL_RISK",        "FALL", "multiple risk factors, fall likely"),
    ("FALLING_NOW",           "FALL", "falling in progress, com_z dropping"),
    ("FALLEN",                "FALL", "body on ground, com_z very low"),
    ("FALL_BACKWARD",         "FALL", "falling backward, torso_pitch negative"),
    ("FALL_FORWARD",          "FALL", "falling forward, torso_pitch strongly positive"),
    ("FALL_LATERAL",          "FALL", "falling sideways"),

    # RECOVERY
    ("NOT_IN_RECOVERY",       "RECOVERY", "normal operation, no recovery needed"),
    ("INITIATING_RECOVERY",   "RECOVERY", "starting to get up from fallen state"),
    ("RECOVERY_IN_PROGRESS",  "RECOVERY", "actively recovering, partial posture regained"),
    ("RECOVERY_COMPLETE",     "RECOVERY", "successfully stood up"),
    ("RECOVERY_FAILED",       "RECOVERY", "recovery attempt unsuccessful"),
    ("SLOW_RECOVERY",         "RECOVERY", "recovery taking longer than expected"),

    # WEIGHT TRANSFER
    ("WEIGHT_LEFT",           "WEIGHT", "weight shifted to left leg"),
    ("WEIGHT_RIGHT",          "WEIGHT", "weight shifted to right leg"),
    ("WEIGHT_CENTERED",       "WEIGHT", "weight evenly distributed"),
    ("WEIGHT_TRANSFER_ACTIVE","WEIGHT", "actively shifting weight between legs"),

    # INTENT / GOAL
    ("INTENT_STAND",          "INTENT", "current goal: stand still"),
    ("INTENT_WALK",           "INTENT", "current goal: walk forward"),
    ("INTENT_RECOVER",        "INTENT", "current goal: recover from fall"),
    ("INTENT_SLOW_DOWN",      "INTENT", "reducing stride to improve stability"),
    ("INTENT_SPEED_UP",       "INTENT", "increasing stride for faster movement"),
    ("INTENT_EXPLORE",        "INTENT", "exploring new movement patterns"),

    # SKILL EXECUTION
    ("SKILL_STAND_ACTIVE",    "SKILL", "stand skill is currently running"),
    ("SKILL_WALK_ACTIVE",     "SKILL", "walk skill is currently running"),
    ("SKILL_NONE",            "SKILL", "no skill currently active"),
    ("SKILL_TRANSITION",      "SKILL", "switching between skills"),

    # CURIOSITY / LEARNING
    ("HIGH_CURIOSITY",        "LEARNING", "prediction error high, unexplored state"),
    ("LOW_CURIOSITY",         "LEARNING", "familiar state, low prediction error"),
    ("LEARNING_OPPORTUNITY",  "LEARNING", "novel situation worth exploring"),
    ("EXPLOITING_KNOWN",      "LEARNING", "using learned behavior"),

    # EMPOWERMENT
    ("HIGH_EMPOWERMENT",      "EMPOWER", "many future states reachable from here"),
    ("LOW_EMPOWERMENT",       "EMPOWER", "few future options, constrained state"),

    # ANOMALY
    ("JOINT_NOMINAL",         "ANOMALY", "all joints operating normally"),
    ("JOINT_STRESS",          "ANOMALY", "one or more joints under stress"),
    ("JOINT_CRITICAL",        "ANOMALY", "joint anomaly critical, survival veto risk"),
    ("ARM_ACTIVE",            "ANOMALY", "arms actively counterbalancing"),
    ("ARM_PASSIVE",           "ANOMALY", "arms not contributing to balance"),

    # TEMPORAL
    ("IMPROVING",             "TEMPORAL", "performance metrics trending upward"),
    ("DEGRADING",             "TEMPORAL", "performance metrics trending downward"),
    ("PLATEAU",               "TEMPORAL", "performance stable but not improving"),
    ("BREAKTHROUGH",          "TEMPORAL", "sudden improvement in key metric"),
]

# Pad to VOCAB_SIZE with generic concepts
_N_DEFINED = len(CONCEPT_DEFS)
for _i in range(VOCAB_SIZE - _N_DEFINED):
    CONCEPT_DEFS.append((f"LATENT_{_i:04d}", "LATENT", f"latent concept slot {_i}"))

assert len(CONCEPT_DEFS) == VOCAB_SIZE


# ── Concept embeddings ────────────────────────────────────────────────────────
class SemanticConceptStore(nn.Module):
    """
    Learnable concept embedding store.

    Usage:
      store = SemanticConceptStore(device)
      # Get top-3 active concepts for current thought embedding
      top_k = store.query(thought_embedding, k=3)
      # → [("FALLING_BACKWARD", 0.87), ("HIGH_FALL_RISK", 0.81), ("LOW_STABILITY", 0.78)]

    Training:
      store.contrastive_step(thought_emb, llm_text_emb)
      → aligns concept embeddings with LLM text embeddings
    """

    def __init__(self, device: torch.device, concept_dim: int = CONCEPT_DIM):
        super().__init__()
        self.device = device
        self.concept_dim = concept_dim
        self.vocab_size = VOCAB_SIZE

        # Learnable embeddings
        self.embeddings = nn.Embedding(VOCAB_SIZE, concept_dim)

        # Initialize with domain-structured noise (concepts in same domain start close)
        self._init_domain_structured()

        # LLM text projection: maps LLM embedding (d_llm) → concept_dim
        # d_llm is unknown until first LLM call; lazy init
        self._text_proj: nn.Linear | None = None

        # Activation threshold
        self.activation_threshold = 0.55

        self.to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Index: name → idx
        self.name_to_idx = {name: i for i, (name, _, _) in enumerate(CONCEPT_DEFS)}
        self.idx_to_name = {i: name for i, (name, _, _) in enumerate(CONCEPT_DEFS)}
        self.idx_to_domain = {i: domain for i, (_, domain, _) in enumerate(CONCEPT_DEFS)}
        self.idx_to_desc = {i: desc for i, (_, _, desc) in enumerate(CONCEPT_DEFS)}

    def _init_domain_structured(self) -> None:
        """Initialize embeddings with domain-cluster structure."""
        with torch.no_grad():
            domains = list({d for _, d, _ in CONCEPT_DEFS})
            domain_centers = {
                d: torch.randn(self.concept_dim) * 0.5
                for d in domains
            }
            data = torch.zeros(VOCAB_SIZE, self.concept_dim)
            for i, (_, domain, _) in enumerate(CONCEPT_DEFS):
                center = domain_centers[domain]
                data[i] = center + torch.randn(self.concept_dim) * 0.15
            # Normalize
            data = F.normalize(data, dim=-1)
            self.embeddings.weight.copy_(data)

    @torch.no_grad()
    def query(
        self,
        thought_emb: torch.Tensor,
        k: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Find top-k concepts matching thought embedding.
        thought_emb: (concept_dim,) or (1, concept_dim)
        Returns: [(name, similarity), ...]
        """
        t = threshold or self.activation_threshold
        emb = F.normalize(thought_emb.view(1, -1), dim=-1)
        all_emb = F.normalize(self.embeddings.weight, dim=-1)  # (V, D)
        sims = (emb @ all_emb.T).squeeze(0)  # (V,)

        top_vals, top_idx = sims.topk(k)
        results = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if val >= t:
                results.append((self.idx_to_name[idx], float(val)))
        return results

    def get_embedding(self, name: str) -> torch.Tensor | None:
        idx = self.name_to_idx.get(name)
        if idx is None:
            return None
        return self.embeddings.weight[idx].detach()

    def ensure_text_proj(self, llm_dim: int) -> None:
        if self._text_proj is None or self._text_proj.in_features != llm_dim:
            self._text_proj = nn.Linear(llm_dim, self.concept_dim, bias=False).to(self.device)
            nn.init.orthogonal_(self._text_proj.weight)
            # Add to optimizer
            self.optim.add_param_group({"params": self._text_proj.parameters()})

    def contrastive_step(
        self,
        thought_emb: torch.Tensor,      # (B, concept_dim)
        target_concept_names: list[str], # B concept names (from LLM annotation)
        temperature: float = 0.07,
    ) -> float:
        """
        CLIP-style contrastive loss:
          thought_emb[i] should be close to concept_embeddings[target_i]
          and far from all others.
        """
        if not target_concept_names:
            return 0.0

        # Get target indices
        target_indices = []
        for name in target_concept_names:
            idx = self.name_to_idx.get(name)
            if idx is not None:
                target_indices.append(idx)

        if not target_indices:
            return 0.0

        B = len(target_indices)
        thought = F.normalize(thought_emb[:B].to(self.device), dim=-1)  # (B, D)
        target_idx_t = torch.tensor(target_indices, device=self.device)
        concept_embs = F.normalize(self.embeddings(target_idx_t), dim=-1)  # (B, D)

        # All concept embeddings for contrastive negatives
        all_embs = F.normalize(self.embeddings.weight, dim=-1)  # (V, D)

        # Similarity: thought vs all concepts
        logits = (thought @ all_embs.T) / temperature  # (B, V)

        # Labels: correct concept index
        loss = F.cross_entropy(logits, target_idx_t)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        return float(loss.item())

    def snapshot(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "concept_dim": self.concept_dim,
            "n_domains": len({d for _, d, _ in CONCEPT_DEFS}),
            "defined_concepts": _N_DEFINED,
            "latent_slots": VOCAB_SIZE - _N_DEFINED,
        }
