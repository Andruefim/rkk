"""Simulation mixin: skill snapshot (embodied LLM reward удалён)."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationPoseEmbodiedMixin:
    def _skill_snapshot(self) -> dict | None:
        if not self._skill_library_enabled():
            return None
        lib = self._skill_library
        out: dict = {"enabled": True, "active": None}
        if lib is not None:
            out.update(lib.snapshot())
        else:
            out["n_skills"] = 0
            out["skills"] = []
            out["history_len"] = 0
        if self._skill_exec is not None:
            sk = self._skill_exec["skill"]
            out["active"] = {
                "name": sk.name,
                "step": self._skill_exec["index"],
                "total": len(sk.action_sequence),
            }
        return out
