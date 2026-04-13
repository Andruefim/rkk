"""Simulation mixin: эпизоды, curriculum, RSSM."""
from __future__ import annotations

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
        if self.current_world != "humanoid" or self._fixed_root_active:
            return

        self._episodic_memory.tick_update(
            tick=tick,
            obs=obs,
            last_action=self._last_action_for_memory,
            fallen=fallen,
            posture=posture,
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
            ep = self._episodic_memory.on_fall(tick, obs, intents)
            if ep is not None:
                self._last_fall_memory_tick = tick
                seeds = self._episodic_memory.get_seeds_from_patterns(
                    set(self.agent.graph.nodes.keys())
                )
                if seeds:
                    self.agent.inject_text_priors(seeds)

    def _tick_curriculum(self, tick: int, obs: dict, fallen: bool) -> None:
        """Level 2-E: Update curriculum scheduler."""
        if not _CURRICULUM_AVAILABLE or self._curriculum is None:
            return
        if not curriculum_enabled():
            return
        if self.current_world != "humanoid" or self._fixed_root_active:
            return

        fall_rate = (
            float(np.mean(self._mc_fallen_count_window))
            if hasattr(self, "_mc_fallen_count_window")
            and self._mc_fallen_count_window
            else 0.0
        )

        stage, advanced = self._curriculum.tick(tick, obs, fallen, fall_rate)

        if advanced:
            injected = self._curriculum.inject_stage_seeds(self.agent)
            self._add_event(
                f"📚 Curriculum → '{stage.name}': {stage.description[:60]} "
                f"(+{injected} seeds)",
                "#aaffaa",
                "phase",
            )

        if tick % self._curriculum_apply_every == 0:
            self._curriculum.apply_stage_intents(self.agent.env)

        if (
            tick % 300 == 0
            and self._curriculum._current_idx >= len(self._curriculum._stages) - 2
        ):
            fall_summary = ""
            if self._episodic_memory is not None:
                fall_summary = self._episodic_memory.get_llm_context_block(
                    max_falls=3
                )
            skill_stats = self._skill_snapshot() or {}
            pose_metrics: dict = {}
            try:
                cur_obs = dict(self.agent.env.observe())
                pose_metrics = self._curriculum.compute_metrics(cur_obs)
            except Exception:
                pass
            valid_intent = [k for k in self.agent.graph.nodes if k.startswith("intent_")]
            valid_graph = list(self.agent.graph.nodes.keys())

            def _run_curriculum_llm() -> None:
                import asyncio

                asyncio.run(
                    self._curriculum.maybe_generate_next_stage_llm(
                        tick=tick,
                        skill_stats=skill_stats,
                        fall_summary=fall_summary,
                        pose_metrics=pose_metrics,
                        valid_intent_vars=valid_intent,
                        valid_graph_vars=valid_graph,
                        llm_url=get_ollama_generate_url(),
                        llm_model=get_ollama_model(),
                    )
                )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            try:
                if loop is not None:
                    try:
                        loop.create_task(
                            self._curriculum.maybe_generate_next_stage_llm(
                                tick=tick,
                                skill_stats=skill_stats,
                                fall_summary=fall_summary,
                                pose_metrics=pose_metrics,
                                valid_intent_vars=valid_intent,
                                valid_graph_vars=valid_graph,
                                llm_url=get_ollama_generate_url(),
                                llm_model=get_ollama_model(),
                            )
                        )
                    except RuntimeError:
                        self._llm_loop_executor.submit(_run_curriculum_llm)
                else:
                    self._llm_loop_executor.submit(_run_curriculum_llm)
            except Exception as e:
                print(f"[Simulation] curriculum LLM error: {e}")

    def _maybe_upgrade_rssm(self, tick: int) -> None:
        """Level 2-F: Upgrade GNN to RSSM after sufficient GNN training."""
        if not _RSSM_AVAILABLE or self._rssm_upgraded:
            return
        if not rssm_enabled():
            return
        if self.current_world != "humanoid":
            return
        try:
            min_tick = max(500, int(os.environ.get("RKK_WM_RSSM_UPGRADE_TICK", "500")))
        except ValueError:
            min_tick = 500
        if tick < min_tick:
            return
        upgraded, trainer = maybe_upgrade_graph_to_rssm(self.agent.graph, self.device)
        if upgraded and trainer is None:
            self._rssm_upgraded = True
            return
        if upgraded and trainer is not None:
            self._rssm_trainer = trainer
            self._rssm_imagination = RSSMImagination(
                self.agent.graph._core, self.device
            )
            self._rssm_upgraded = True
            self._rssm_upgrade_tick = tick
            try:
                h = int(os.environ.get("RKK_WM_RSSM_IMAGINATION", "12"))
            except ValueError:
                h = 12
            self.agent._imagination_horizon = h
            self._add_event(
                f"🔮 RSSM-lite activated at tick={tick} "
                f"(horizon={self.agent._imagination_horizon})",
                "#88aaff",
                "phase",
            )

    def _rssm_train_step(
        self,
        obs_before: dict[str, float],
        action_var: str,
        action_val: float,
        obs_after: dict[str, float],
    ) -> None:
        """Level 2-F: Push transition to RSSM trainer."""
        if self._rssm_trainer is None or not self._rssm_upgraded:
            return
        try:
            node_ids = list(self.agent.graph._node_ids)
            X_t = [float(obs_before.get(n, 0.0)) for n in node_ids]
            a_t = [float(action_val) if n == action_var else 0.0 for n in node_ids]
            X_tp1 = [float(obs_after.get(n, 0.0)) for n in node_ids]
            self._rssm_trainer.push(X_t, a_t, X_tp1)
            if self.tick % 8 == 0:
                self._rssm_trainer.maybe_train()
        except Exception:
            pass
