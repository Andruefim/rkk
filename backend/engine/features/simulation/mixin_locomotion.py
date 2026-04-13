"""Simulation mixin: CPG, motor cortex, reward coord."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationLocomotionMixin:
    @staticmethod
    def _unwrap_base_env(env):
        e = env
        while hasattr(e, "base_env"):
            e = e.base_env
        return e

    def _locomotion_cpg_enabled(self) -> bool:
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _cpg_decoupled_enabled(self) -> bool:
        return self._locomotion_cpg_enabled() and _cpg_loop_hz_from_env() > 0.0

    def _stop_cpg_background_loop(self) -> None:
        self._bg.stop_cpg_loop()

    def _ensure_cpg_background_loop(self) -> None:
        self._bg.ensure_cpg_background_loop()

    def _publish_cpg_node_snapshot(self) -> None:
        self._bg.publish_cpg_node_snapshot()

    def _maybe_apply_cpg_locomotion(self, fallen: bool) -> None:
        """Phase A+D: CPG + Motor Cortex blended locomotion."""
        dt = 0.05
        if self._cpg_decoupled_enabled():
            return
        if not self._locomotion_cpg_enabled():
            return
        if self.current_world != "humanoid" or self._fixed_root_active:
            return
        base = self._unwrap_base_env(self.agent.env)
        fn = getattr(base, "apply_cpg_leg_targets", None)
        if not callable(fn):
            return
        if self._locomotion_controller is None:
            from engine.cpg_locomotion import LocomotionController
            self._locomotion_controller = LocomotionController(self.device)

        try:
            nodes = dict(self.agent.graph.nodes)

            # CPG generates base targets
            cpg_targets = self._locomotion_controller.get_joint_targets(nodes, dt=dt)
            cpg_sync = self._locomotion_controller.upper_body_cpg_sync()

            # Phase D: Motor Cortex blending
            mc = self._ensure_motor_cortex()
            final_targets = dict(cpg_targets)
            if mc is not None and len(mc.programs) > 0:
                obs_now = dict(self.agent.env.observe())
                posture_now = float(obs_now.get(
                    "posture_stability", obs_now.get("phys_posture_stability", 0.5)
                ))
                com_z_now = float(obs_now.get("com_z", obs_now.get("phys_com_z", 0.5)))

                # Select active programs by situation
                if com_z_now < 0.35 or fallen:
                    active_progs = ["recovery"]
                elif posture_now < 0.58:
                    active_progs = ["balance", "recovery"]
                else:
                    active_progs = ["walk", "balance"]
                active_progs = [p for p in active_progs if p in mc.programs]

                if active_progs:
                    cortex_targets = mc.infer(nodes, active_progs)
                    final_targets = mc.blend_targets(cpg_targets, cortex_targets)
                    # Expose CPG weight to locomotion controller for diagnostics
                    self._locomotion_controller.cpg_weight = mc.cpg_weight

            obs_before_env = dict(self.agent.env.observe())
            fn(final_targets, cpg_sync=cpg_sync)
            obs = dict(self.agent.env.observe())

            self._sync_motor_state(obs, source="cpg+mc", tick=self.tick)
            self._log_motor_command(
                source="cpg+mc",
                joint_targets=final_targets,
                intents=getattr(self._locomotion_controller, "_last_motor_state", None),
                obs=self._motor_obs_payload(obs),
            )
            self._record_motor_burst_causal(
                obs_before_env=obs_before_env,
                obs_after_env=dict(obs),
                intents=dict(getattr(self._locomotion_controller, "_last_motor_state", {}) or {}),
            )

            # Extract metrics
            posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
            foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
            foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
            com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
            com_x = float(obs.get("com_x", obs.get("phys_com_x", 0.5)))

            # Phase D: train motor cortex programs + anneal CPG
            if mc is not None:
                reward = (
                    posture * 2.0
                    + min(foot_l, foot_r) * 1.5
                    + com_z * 1.0
                    + max(0.0, com_x - 0.46) * 1.5   # forward bonus
                    - (3.0 if fallen else 0.0)
                )
                mc.push_and_train(nodes, cpg_targets, reward, posture, foot_l, foot_r)
                mc.anneal_step(posture, foot_l, foot_r, fallen, self.tick)

                # Inject abstract nodes once when first program spawned
                if not self._mc_abstract_nodes_injected and len(mc.programs) > 0:
                    added = mc.inject_abstract_nodes_into_graph(self.agent.graph)
                    if added > 0:
                        self._mc_abstract_nodes_injected = True
                        self._add_event(
                            f"🧠 MotorCortex: +{added} abstract nodes (mc_walk_drive, mc_balance_signal, …)",
                            "#ff88ff", "phase"
                        )
                mc.sync_abstract_nodes_to_graph(self.agent.graph)

            self._locomotion_controller.learn_from_reward(
                com_z, com_x, fallen, motor_obs=self._motor_obs_payload(obs)
            )
            # Track for embodied reward and motor cortex
            self._mc_posture_window.append(posture)
            self._mc_fallen_count_window.append(1 if fallen else 0)
        except Exception as ex:
            import traceback
            print(f"[Simulation] CPG+MC locomotion error: {ex}")
            if os.environ.get("RKK_DEBUG_CPG"):
                traceback.print_exc()

    def _rsi_full_enabled(self) -> bool:
        v = os.environ.get("RKK_RSI_FULL", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _ensure_motor_cortex(self):
        """Phase D: ленивая инициализация MotorCortexLibrary."""
        if not _MOTOR_CORTEX_AVAILABLE:
            return None
        env_flag = os.environ.get("RKK_MOTOR_CORTEX", "1").strip().lower()
        if env_flag in ("0", "false", "no", "off"):
            return None
        if self.current_world != "humanoid" or self._fixed_root_active:
            return None
        if self._motor_cortex is None:
            self._motor_cortex = _MotorCortexLibrary(self.device)
        return self._motor_cortex

    def _ensure_reward_coord(self) -> None:
        """Lazy-init RewardCoordinator; rebuild CuriosityICM when graph dimension changes."""
        if not _REWARD_COORD_AVAILABLE:
            return
        if not hasattr(self, "agent") or self.agent is None:
            return
        graph = getattr(self.agent, "graph", None)
        if graph is None:
            return
        nids = list(getattr(graph, "_node_ids", []) or [])
        d = len(nids) if nids else int(getattr(graph, "_d", 30) or 30)

        if self._reward_coord is not None:
            cur = getattr(self._reward_coord.curiosity, "d", -1)
            if cur != d:
                self._reward_coord = None
                self._reward_X_prev = []
                self._reward_a_prev = []

        if self._reward_coord is not None:
            return

        self._reward_coord = RewardCoordinator(d=d, device=self.device)
        print(f"[Simulation] RewardCoordinator init d={d}")
