"""Simulation mixin: CPG, motor cortex (обучение — intrinsic objective)."""
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
            from engine.features.humanoid.constants import LEG_VARS

            nodes = dict(self.agent.graph.nodes)

            cb = self._ensure_cerebellum()
            use_cb = cb is not None and cb.ready_for_control()

            # CPG всегда обновляет фазу и upper_body_cpg_sync
            cpg_targets = self._locomotion_controller.get_joint_targets(nodes, dt=dt)
            cpg_sync = self._locomotion_controller.upper_body_cpg_sync()

            mc = self._ensure_motor_cortex()
            final_targets = dict(cpg_targets)

            if use_cb:
                obs_now = dict(self.agent.env.observe())
                obs_r = {**nodes}
                for k, v in obs_now.items():
                    try:
                        obs_r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                leg_cmd = cb.get_joint_commands(obs_r)
                for k in LEG_VARS:
                    if k in leg_cmd:
                        final_targets[k] = leg_cmd[k]
                motor_src = "cerebellum+cpg_sync"
            else:
                # Legacy MLP programmes (RKK_MOTOR_CORTEX_LEGACY_SPECS=1) — иначе только CPG+рефлекс
                if (
                    self._motor_cortex_legacy_specs_enabled()
                    and mc is not None
                    and len(mc.programs) > 0
                ):
                    obs_now = dict(self.agent.env.observe())
                    posture_now = float(obs_now.get(
                        "posture_stability", obs_now.get("phys_posture_stability", 0.5)
                    ))
                    com_z_now = float(obs_now.get("com_z", obs_now.get("phys_com_z", 0.5)))

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
                        self._locomotion_controller.cpg_weight = mc.cpg_weight

                rs = self._ensure_reflex_stabilizer()
                if rs is not None:
                    try:
                        obs_now = dict(self.agent.env.observe())
                        obs_r = {**nodes}
                        for k, v in obs_now.items():
                            try:
                                obs_r[k] = float(v)
                            except (TypeError, ValueError):
                                pass
                        final_targets = rs.step(obs_r, final_targets)
                    except Exception:
                        pass
                motor_src = "cpg+mc"

            final_targets = self._blend_causal_executor_targets(nodes, final_targets)

            self._last_joint_cmd_applied = {
                k: float(final_targets[k]) for k in LEG_VARS if k in final_targets
            }

            obs_before_env = dict(self.agent.env.observe())
            fn(final_targets, cpg_sync=cpg_sync)
            obs = dict(self.agent.env.observe())

            self._sync_motor_state(obs, source=motor_src, tick=self.tick)
            self._log_motor_command(
                source=motor_src,
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

            # Track for embodied reward and motor cortex (mc_* sync → mixin_tick после observe)
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

    def _ensure_reflex_stabilizer(self):
        """Быстрый онлайн-корректор целей ног (RKK_REFLEX_STABILIZER=1)."""
        from engine.features.humanoid.constants import LEG_VARS
        from engine.reflex_stabilizer import ReflexStabilizer, reflex_stabilizer_enabled

        if not reflex_stabilizer_enabled():
            return None
        if self.current_world != "humanoid" or self._fixed_root_active:
            return None
        if self._reflex_stabilizer is not None:
            return self._reflex_stabilizer
        self._reflex_stabilizer = ReflexStabilizer(
            joint_keys=list(LEG_VARS), device=self.device
        )
        if not self._reflex_stabilizer_logged:
            self._reflex_stabilizer_logged = True
            self._add_event(
                "🦿 ReflexStabilizer: online leg corrections (RKK_REFLEX_STABILIZER=1)",
                "#88ccff",
                "phase",
            )
        return self._reflex_stabilizer

    def _ensure_cerebellum(self):
        """Инверсная динамика ног (RKK_CEREBELLUM=1); обучение в agent-тике."""
        from engine.cerebellum import Cerebellum, cerebellum_enabled

        if not cerebellum_enabled():
            return None
        if self.current_world != "humanoid" or self._fixed_root_active:
            return None
        if self._cerebellum is not None:
            return self._cerebellum
        self._cerebellum = Cerebellum(device=self.device)
        if not self._cerebellum_logged:
            self._cerebellum_logged = True
            self._add_event(
                "🧠 Cerebellum: inverse dynamics online (RKK_CEREBELLUM=1)",
                "#aaddff",
                "phase",
            )
        return self._cerebellum

    def _motor_cortex_legacy_specs_enabled(self) -> bool:
        return os.environ.get("RKK_MOTOR_CORTEX_LEGACY_SPECS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _ensure_causal_motor_executor(self):
        from engine.causal_motor_executor import CausalMotorExecutor, causal_motor_executor_enabled

        if not causal_motor_executor_enabled():
            return None
        if self.current_world != "humanoid" or self._fixed_root_active:
            return None
        if self._causal_motor_executor is None:
            self._causal_motor_executor = CausalMotorExecutor()
        return self._causal_motor_executor

    def _blend_causal_executor_targets(
        self, nodes: dict, targets: dict[str, float]
    ) -> dict[str, float]:
        from engine.causal_motor_executor import causal_motor_executor_enabled

        if not causal_motor_executor_enabled():
            return targets
        exe = self._ensure_causal_motor_executor()
        if exe is None or getattr(self.agent.graph, "_core", None) is None:
            return targets
        intents: dict[str, float] = {}
        for k, v in nodes.items():
            sk = str(k)
            if not sk.startswith("intent_"):
                continue
            try:
                intents[sk] = float(v)
            except (TypeError, ValueError):
                continue
        causal = exe.execute(self.agent.graph, intents)
        if not causal:
            return targets
        out = dict(targets)
        mc = getattr(self, "_motor_cortex", None)
        if mc is None:
            try:
                mc = self._ensure_motor_cortex()
            except Exception:
                mc = None
        if mc is not None and len(getattr(mc, "programs", {})) > 0:
            w_cpg = float(mc.cpg_weight)
        else:
            try:
                w_cpg = float(os.environ.get("RKK_CAUSAL_BLEND_CPG", "0.65"))
            except ValueError:
                w_cpg = 0.65
            w_cpg = float(np.clip(w_cpg, 0.0, 1.0))
        for k, v in causal.items():
            cpg_v = out.get(k, 0.5)
            out[k] = float(np.clip(w_cpg * cpg_v + (1.0 - w_cpg) * v, 0.05, 0.95))
        return out
