"""Simulation mixin: episodic fall memory."""
from __future__ import annotations

from engine.core.world import is_humanoid_topology

from engine.features.simulation.mixin_imports import *


class SimulationEpisodicRssmMixin:
    def _record_last_action(self, result: dict) -> None:
        """Level 2-D: Track last action for episodic memory."""
        if not result.get("blocked") and not result.get("skipped"):
            var = result.get("variable")
            val = result.get("value")
            if var is not None and val is not None:
                self._last_action_for_memory = (str(var), float(val))

    def _update_episodic_memory(
        self, tick: int, obs: dict, fallen: bool, posture: float
    ) -> None:
        """Level 2-D: Update episodic memory with current state."""
        if not _EPISODIC_MEMORY_AVAILABLE or self._episodic_memory is None:
            return
        if not episode_memory_enabled():
            return
        if not is_humanoid_topology(self.current_world) or self._fixed_root_active:
            return

        physics_ctx: dict[str, float] = {}
        try:
            fn = getattr(self.agent.env, "get_dynamics_params", None)
            if callable(fn):
                physics_ctx = dict(fn())
        except Exception:
            physics_ctx = {}

        self._episodic_memory.tick_update(
            tick=tick,
            obs=obs,
            last_action=self._last_action_for_memory,
            fallen=fallen,
            posture=posture,
            physics_context=physics_ctx,
        )

        if fallen and (tick - self._last_fall_memory_tick) > 5:
            env = self.agent.env
            intents = {}
            try:
                obs_now = dict(env.observe())
                intents = {
                    k: float(
                        obs_now.get(k, obs_now.get(f"phys_{k}", 0.5))
                    )
                    for k in [
                        "intent_stride",
                        "intent_torso_forward",
                        "intent_support_left",
                        "intent_support_right",
                        "intent_stop_recover",
                        "intent_gait_coupling",
                    ]
                }
            except Exception:
                pass
            obs_for_fall = obs
            pending = getattr(self, "_pending_fall_obs_for_memory", None)
            if isinstance(pending, dict) and pending:
                obs_for_fall = pending
            ep = self._episodic_memory.on_fall(
                tick, obs_for_fall, intents, physics_context=physics_ctx
            )
            if ep is not None:
                self._last_fall_memory_tick = tick
                self._pending_fall_obs_for_memory = None
                seeds = self._episodic_memory.get_seeds_from_patterns(
                    set(self.agent.graph.nodes.keys())
                )
                if seeds:
                    self.agent.inject_text_priors(seeds)
