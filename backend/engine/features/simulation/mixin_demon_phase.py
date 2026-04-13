"""Simulation mixin: demon, фаза, события."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationDemonPhaseMixin:
    def _step_demon(self, snap: dict):
        try:
            action = self.demon.step([snap], 1 - snap.get("peak_discovery_rate", 0))
        except RuntimeError as e:
            print(f"[Singleton] Demon step skipped: {e}")
            return
        if action is None:
            return
        corrupted = self.agent.demon_disrupt()
        self._add_event(
            f"⚠ Demon [{action.get('mode','?')}] → Nova: {corrupted}",
            "#ff2244", "demon"
        )

    def _update_phase(self, snap: dict) -> float:
        dr = snap.get("discovery_rate", 0)
        self._dr_window.append(dr)
        smoothed = float(np.mean(self._dr_window))

        potential = 1
        for i, t in enumerate(PHASE_THRESHOLDS):
            if smoothed >= t:
                potential = i + 1
        potential = min(potential, 5)

        if potential > self.max_phase:
            if potential == self._candidate_phase:
                self._phase_hold_counter += 1
            else:
                self._candidate_phase    = potential
                self._phase_hold_counter = 1
            if self._phase_hold_counter >= PHASE_HOLD_TICKS:
                self.max_phase = potential
                self.phase     = potential
                self._phase_hold_counter = 0
                self._add_event(f"⬆ Phase {potential}: {PHASE_NAMES[potential]}", "#ffcc00", "phase")
        else:
            self._candidate_phase    = self.max_phase
            self._phase_hold_counter = 0
        self.phase = self.max_phase
        return smoothed

    def _log_step(self, result: dict, fallen: bool):
        if result.get("blocked"):
            self._add_event(
                f"🛡 [BLOCKED] Nova: {result.get('reason','?')}",
                "#884400", "value"
            )
        elif result.get("hierarchy") == "skill":
            sk = result.get("skill", "?")
            var = result.get("variable", "?")
            val = result.get("value", 0)
            done = result.get("skill_done", False)
            try:
                val_sf = float(val)
            except (TypeError, ValueError):
                val_sf = 0.0
            self._add_event(
                f"🦿 Skill [{sk}] do({var}={val_sf:.2f})"
                f"{' ✓' if done else ' …'}",
                "#66ccff",
                "skill",
            )
        elif result.get("updated_edges"):
            cg  = result.get("compression_delta", 0)
            var = result.get("variable", "?")
            val = result.get("value", 0)
            color = WORLDS.get(self.current_world, {}).get("color", "#cc44ff")
            if self._visual_mode:
                color = "#44ffcc"
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                val_f = 0.0
            try:
                cg_f = float(cg)
            except (TypeError, ValueError):
                cg_f = 0.0
            self._add_event(
                f"Nova: do({var}={val_f:.2f}) CG{'+' if cg_f >= 0 else ''}{cg_f:.3f}",
                color, "discovery"
            )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})
