"""Simulation mixin: InnerVoiceNet (τ2, no Ollama)."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationTeacherMixin:
    def _tick_inner_voice(self, tick: int) -> None:
        """InnerVoiceNet τ2 — gated by timescale LEVEL_COGNIT."""
        if not _INNER_VOICE_AVAILABLE or self._inner_voice is None:
            return
        if not is_humanoid_topology(self.current_world):
            return
        if self._timescale is None or not self._timescale.should_run(LEVEL_COGNIT, tick):
            return

        graph = getattr(self.agent, "graph", None)
        if graph is None:
            return

        inf0 = self._inner_voice.total_inferences
        result = self._inner_voice.tick(tick, graph, self.agent.env)
        if self._inner_voice.total_inferences <= inf0:
            return

        self._timescale.mark_ran(LEVEL_COGNIT, tick)

        active = result.get("active_concepts", []) if result else []
        if active:
            top_concept, top_val = active[0]
            fall_concepts = {
                "FALLING_NOW",
                "HIGH_FALL_RISK",
                "FALLEN",
                "JOINT_CRITICAL",
            }
            if top_concept in fall_concepts and top_val > 0.75:
                pass
