"""Genome: innate biological priors for the humanoid AGI."""
from engine.genome.priors import (
    CAUSAL_PRIORS,
    REFLEX_TABLE,
    STAND_PROGRAM,
    WALK_PROGRAM,
    apply_causal_priors,
    apply_reflexes,
    compute_walk_residuals,
    genome_walk_enabled,
    get_stand_program,
    get_walk_program,
    walk_intents_at_tick,
)

__all__ = [
    "CAUSAL_PRIORS",
    "REFLEX_TABLE",
    "STAND_PROGRAM",
    "WALK_PROGRAM",
    "apply_causal_priors",
    "apply_reflexes",
    "compute_walk_residuals",
    "genome_walk_enabled",
    "get_stand_program",
    "get_walk_program",
    "walk_intents_at_tick",
]
