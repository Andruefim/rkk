"""
trajectory_contrastive.py — Trajectory-Level Contrastive Learning.

Fundamental insight: single-step transitions (s_t, a_t → s_{t+1}) don't
contain enough information to learn temporally extended behaviors.

This module:
  1. Collects trajectory segments (rolling windows of N ticks)
  2. Labels each segment with continuous outcome metrics
  3. Provides a contrastive loss that trains the GNN encoder to
     distinguish successful vs failed behavioral patterns
  4. Feeds trajectory-level quality back as learning signal

The GNN learns two things:
  OLD — forward_dynamics: s_t → s_{t+1} (single-step)
  NEW — trajectory_quality: encode(segment) → outcome (multi-step)

This gives temporal credit assignment: the GNN learns not just
"what happens next tick" but "what patterns lead to good outcomes."

Config:
  RKK_TRAJECTORY_ENABLED=1           — master switch
  RKK_TRAJECTORY_SEGMENT_LEN=50     — ticks per segment
  RKK_TRAJECTORY_OVERLAP=10          — overlap between consecutive segments
  RKK_TRAJECTORY_BUFFER_SIZE=200     — max completed segments in memory
  RKK_TRAJECTORY_LOSS_WEIGHT=0.25    — weight of trajectory loss in GNN training
  RKK_TRAJECTORY_TRAIN_MIN_SEGS=6    — minimum segments before training starts
  RKK_TRAJECTORY_CONTRAST_MARGIN=0.3 — margin for contrastive pairs
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config helpers ────────────────────────────────────────────────────────────
def trajectory_enabled() -> bool:
    return os.environ.get("RKK_TRAJECTORY_ENABLED", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _ei(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class TrajectorySegment:
    """One completed trajectory segment with outcome labels."""
    observations: list[list[float]]   # (T, d) — vectorized obs per tick
    actions: list[tuple[str, float] | None]  # (T,) — (var, val) or None
    outcome: dict[str, float]         # labeled metrics
    tick_start: int = 0
    tick_end: int = 0


# ── Outcome computation ──────────────────────────────────────────────────────
def compute_segment_outcome(
    raw_obs: list[dict[str, float]],
    fallen_flags: list[bool],
) -> dict[str, float]:
    """
    Compute general outcome metrics for a trajectory segment.

    These metrics are intentionally GENERAL — not walking-specific.
    They capture whether the agent maintained homeostasis, explored,
    and avoided degenerate states.
    """
    n = len(raw_obs)
    if n == 0:
        return {"quality": 0.0, "upright_frac": 0.0,
                "stability_mean": 0.0, "state_diversity": 0.0}

    # 1. Upright fraction: survived without catastrophic failure
    upright = sum(1 for f in fallen_flags if not f) / n

    # 2. Stability: mean posture quality over segment
    stabilities = []
    for obs in raw_obs:
        ps = obs.get("posture_stability",
                      obs.get("phys_posture_stability", 0.5))
        stabilities.append(float(ps))
    stability = float(np.mean(stabilities))

    # 3. State diversity: how much did the state change?
    #    High diversity = agent is exploring, not stuck
    if n > 1:
        keys = list(raw_obs[0].keys())
        changes = []
        for i in range(1, min(n, 10)):
            for k in keys[:20]:  # sample keys to keep O(1)
                v0 = float(raw_obs[0].get(k, 0.5))
                vi = float(raw_obs[i].get(k, 0.5))
                changes.append(abs(vi - v0))
        diversity = float(np.mean(changes)) if changes else 0.0
    else:
        diversity = 0.0

    # 4. Composite quality (general AGI metric)
    quality = (
        upright * 0.4
        + stability * 0.35
        + min(diversity * 2.0, 0.25) * 1.0  # cap diversity contribution
    )

    return {
        "quality": float(np.clip(quality, 0.0, 1.0)),
        "upright_frac": upright,
        "stability_mean": stability,
        "state_diversity": diversity,
    }


# ── Trajectory Collector ─────────────────────────────────────────────────────
class TrajectoryCollector:
    """
    Collects trajectory segments from the agent's tick-by-tick experience.

    Usage in agent.step():
        self._traj_collector.tick(obs_dict, (var, val), is_fallen, node_ids, tick)
    """

    def __init__(self):
        self.segment_len = _ei("RKK_TRAJECTORY_SEGMENT_LEN", 50)
        self.overlap = _ei("RKK_TRAJECTORY_OVERLAP", 10)
        buf_size = _ei("RKK_TRAJECTORY_BUFFER_SIZE", 200)

        self._current_obs: list[dict[str, float]] = []
        self._current_actions: list[tuple[str, float] | None] = []
        self._current_fallen: list[bool] = []
        self._tick_start: int = 0

        self.buffer: deque[TrajectorySegment] = deque(maxlen=buf_size)
        self._node_ids: list[str] = []  # set by agent on first call

    def tick(
        self,
        obs: dict[str, float],
        action: tuple[str, float] | None,
        is_fallen: bool,
        node_ids: list[str],
        engine_tick: int,
    ) -> TrajectorySegment | None:
        """
        Feed one tick of experience. Returns completed segment if ready.
        """
        if not trajectory_enabled():
            return None

        self._node_ids = node_ids
        if not self._current_obs:
            self._tick_start = engine_tick

        self._current_obs.append(dict(obs))
        self._current_actions.append(action)
        self._current_fallen.append(bool(is_fallen))

        if len(self._current_obs) >= self.segment_len:
            return self._finalize(engine_tick)
        return None

    def _finalize(self, engine_tick: int) -> TrajectorySegment:
        """Finalize current segment, label it, add to buffer."""
        obs_list = list(self._current_obs)
        act_list = list(self._current_actions)
        fallen_list = list(self._current_fallen)

        # Vectorize observations for GNN training
        nids = self._node_ids
        obs_vecs = []
        for obs in obs_list:
            vec = [float(obs.get(nid, 0.5)) for nid in nids]
            obs_vecs.append(vec)

        outcome = compute_segment_outcome(obs_list, fallen_list)

        seg = TrajectorySegment(
            observations=obs_vecs,
            actions=act_list,
            outcome=outcome,
            tick_start=self._tick_start,
            tick_end=engine_tick,
        )
        self.buffer.append(seg)

        # Keep overlap for continuity
        keep = max(1, self.overlap)
        self._current_obs = self._current_obs[-keep:]
        self._current_actions = self._current_actions[-keep:]
        self._current_fallen = self._current_fallen[-keep:]
        self._tick_start = engine_tick - keep

        return seg

    def has_enough_segments(self) -> bool:
        min_segs = _ei("RKK_TRAJECTORY_TRAIN_MIN_SEGS", 6)
        return len(self.buffer) >= min_segs

    def sample_contrastive_pairs(
        self, n_pairs: int = 4,
    ) -> list[tuple[TrajectorySegment, TrajectorySegment]]:
        """
        Sample (good, bad) segment pairs for contrastive learning.
        Good = higher quality, Bad = lower quality.
        """
        if len(self.buffer) < 4:
            return []

        segs = list(self.buffer)
        segs.sort(key=lambda s: s.outcome["quality"])

        # Split into bottom half (bad) and top half (good)
        mid = len(segs) // 2
        bad_pool = segs[:mid]
        good_pool = segs[mid:]

        if not bad_pool or not good_pool:
            return []

        rng = np.random.default_rng()
        pairs = []
        for _ in range(min(n_pairs, min(len(good_pool), len(bad_pool)))):
            gi = rng.integers(0, len(good_pool))
            bi = rng.integers(0, len(bad_pool))
            pairs.append((good_pool[gi], bad_pool[bi]))
        return pairs

    def recent_quality(self, window: int = 5) -> float:
        """Mean quality of recent segments."""
        if not self.buffer:
            return 0.0
        recent = list(self.buffer)[-window:]
        return float(np.mean([s.outcome["quality"] for s in recent]))

    def snapshot(self) -> dict[str, Any]:
        quals = [s.outcome["quality"] for s in self.buffer]
        return {
            "enabled": trajectory_enabled(),
            "segment_len": self.segment_len,
            "buffer_size": len(self.buffer),
            "current_len": len(self._current_obs),
            "quality_mean": round(float(np.mean(quals)), 4) if quals else 0.0,
            "quality_std": round(float(np.std(quals)), 4) if quals else 0.0,
            "recent_quality": round(self.recent_quality(), 4),
        }


# ── Trajectory Head (added to GNN) ───────────────────────────────────────────
class TrajectoryHead(nn.Module):
    """
    Small MLP that predicts trajectory outcome from pooled GNN embeddings.

    Input: pooled segment embedding (d * hidden) from GNN's node_enc
    Output: scalar quality prediction
    """
    def __init__(self, embed_dim: int, device: torch.device):
        super().__init__()
        h = min(64, embed_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        ).to(device)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, embed_dim) → (N, 1) quality prediction."""
        return self.net(z)


# ── Trajectory Loss ───────────────────────────────────────────────────────────
def trajectory_contrastive_loss(
    core: nn.Module,
    segments: list[TrajectorySegment],
    node_ids: list[str],
    d: int,
    max_d: int,
    traj_head: TrajectoryHead,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute trajectory-level loss for GNN training.

    Two components:
      1. Regression: predict quality from segment embedding
      2. Contrastive: push good/bad segment embeddings apart

    The loss flows through the GNN's node_enc, teaching it to learn
    temporally meaningful representations — not just single-step dynamics.
    """
    if not segments or core is None or len(segments) < 2:
        return torch.tensor(0.0, device=device)

    margin = _ef("RKK_TRAJECTORY_CONTRAST_MARGIN", 0.3)
    max_segs = min(8, len(segments))
    rng = np.random.default_rng()
    indices = rng.choice(len(segments), size=max_segs, replace=False)
    batch_segs = [segments[i] for i in indices]

    embeddings = []
    qualities = []

    for seg in batch_segs:
        T = len(seg.observations)
        if T == 0:
            continue

        # Vectorize and coerce to d dimensions
        obs_vecs = []
        for row in seg.observations:
            coerced = row[:d] if len(row) >= d else row + [0.5] * (d - len(row))
            obs_vecs.append(coerced)

        X = torch.tensor(obs_vecs, dtype=torch.float32, device=device)

        # Pad to max_d for GNN core
        if X.shape[-1] < max_d:
            X = F.pad(X, (0, max_d - X.shape[-1]))

        # Encode through GNN's node_enc (shared weights!)
        H = core.node_enc(X.unsqueeze(-1))  # (T, max_d, hidden)

        # Pool: mean across time, then across nodes (up to d)
        H_time = H[:, :d, :].mean(dim=0)   # (d, hidden)
        H_flat = H_time.flatten()           # (d * hidden)
        embeddings.append(H_flat)
        qualities.append(seg.outcome["quality"])

    if len(embeddings) < 2:
        return torch.tensor(0.0, device=device)

    Z = torch.stack(embeddings)   # (N, d*hidden)
    Q = torch.tensor(qualities, dtype=torch.float32, device=device)

    # 1. Regression loss: predict quality from embedding
    Q_pred = traj_head(Z).squeeze(-1)  # (N,)
    l_regression = F.mse_loss(Q_pred, Q)

    # 2. Contrastive loss: separate good from bad
    l_contrast = torch.tensor(0.0, device=device)
    n_pairs = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            q_diff = abs(qualities[i] - qualities[j])
            if q_diff < 0.1:
                continue  # too similar, skip
            # Cosine distance
            cos_sim = F.cosine_similarity(
                Z[i].unsqueeze(0), Z[j].unsqueeze(0),
            ).squeeze()
            # Good and bad segments should be dissimilar
            target_sim = -1.0 if q_diff > 0.3 else 0.0
            l_contrast = l_contrast + F.mse_loss(
                cos_sim, torch.tensor(target_sim, device=device),
            )
            n_pairs += 1

    if n_pairs > 0:
        l_contrast = l_contrast / n_pairs

    return l_regression + 0.5 * l_contrast
