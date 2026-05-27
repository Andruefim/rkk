"""
RSSM-lite isolated from executive WM path.

Prefer CausalGNNCore via causal_graph.get_world_model_core().
Re-exports from temporal_world_model for backward compatibility.
"""
from __future__ import annotations

from engine.temporal_world_model import (
    RSSMImagination,
    RSSMLiteCore,
    RSSMTrainer,
    rssm_enabled,
)

__all__ = [
    "RSSMLiteCore",
    "RSSMImagination",
    "RSSMTrainer",
    "rssm_enabled",
]
