"""Simulation mixin: L1 motor queue, burst causal."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationMotorPipelineMixin:
    def _locomotion_reward_ema(self) -> float:
        lc = self._locomotion_controller
        if lc is None or not lc._reward_history:
            return 0.0
        w = min(32, len(lc._reward_history))
        return float(np.mean(lc._reward_history[-w:]))

    def _motor_obs_payload(self, obs: dict) -> dict[str, float]:
        keys = (
            "gait_phase_l",
            "gait_phase_r",
            "foot_contact_l",
            "foot_contact_r",
            "support_bias",
            "motor_drive_l",
            "motor_drive_r",
            "posture_stability",
        )
        return {k: float(obs.get(k, 0.5)) for k in keys}

    def _sync_motor_state(self, obs: dict, *, source: str, tick: int | None = None) -> None:
        with self._motor_state_lock:
            self._motor_state.update_from_observation(obs, tick=tick if tick is not None else self.tick, source=source)

    def _log_motor_command(
        self,
        *,
        source: str,
        joint_targets: dict[str, float] | None = None,
        obs: dict | None = None,
        intents: dict[str, float] | None = None,
    ) -> None:
        with self._motor_state_lock:
            self._motor_state.update_from_command(
                tick=self.tick,
                source=source,
                intents=intents,
                joint_targets=joint_targets,
                obs=obs,
            )

    def _motor_state_snapshot(self) -> dict:
        with self._motor_state_lock:
            return self._motor_state.snapshot()

    def _enqueue_l1_motor_command(
        self,
        *,
        source: str,
        joint_targets: dict[str, float],
        intents: dict[str, float] | None = None,
        dt: float,
        cpg_sync: dict[str, float] | None = None,
    ) -> None:
        payload: dict = {
            "tick": int(self.tick),
            "source": str(source),
            "joint_targets": {k: float(v) for k, v in joint_targets.items()},
            "intents": {k: float(v) for k, v in (intents or {}).items()},
            "dt": float(dt),
        }
        if cpg_sync:
            payload["cpg_sync"] = {k: float(v) for k, v in cpg_sync.items()}
        self._l1_motor_q.put(payload)
        self._l1_last_cmd_tick = self.tick

    def _record_motor_burst_causal(
        self,
        *,
        obs_before_env: dict[str, float],
        obs_after_env: dict[str, float],
        intents: dict[str, float],
    ) -> None:
        """
        Low-level motor burst as record_intervention-like event in main writer.
        """
        self.agent.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.agent.graph.snapshot_vec_dict()
        self.agent.graph.apply_env_observation(obs_after_env)
        obs_after_full = self.agent.graph.snapshot_vec_dict()
        self.agent.graph.record_observation(obs_before_full)
        self.agent.graph.record_observation(obs_after_full)
        # Все значимые intent в burst (до 4 сильнейших по |v−0.5|).
        sig: list[tuple[float, str, float]] = []
        for k, v in (intents or {}).items():
            if k not in self.agent.graph.nodes:
                continue
            fv = float(v)
            d = abs(fv - 0.5)
            if d > 0.08:
                sig.append((d, k, fv))
        sig.sort(key=lambda t: -t[0])
        for _d, k, fv in sig[:4]:
            self.agent.graph.record_intervention(k, fv, obs_before_full, obs_after_full)

    def _drain_l1_motor_commands(self) -> None:
        base = self._unwrap_base_env(self.agent.env)
        fn = getattr(base, "apply_cpg_leg_targets", None)
        if not callable(fn):
            return
        latest = None
        while True:
            try:
                latest = self._l1_motor_q.get_nowait()
            except Exception:
                break
        if latest is None:
            return
        targets = dict(latest.get("joint_targets") or {})
        intents = dict(latest.get("intents") or {})
        cpg_sync = latest.get("cpg_sync")
        try:
            credit_every = int(os.environ.get("RKK_MOTOR_CREDIT_EVERY", "4"))
        except ValueError:
            credit_every = 4
        credit_every = max(1, min(credit_every, 64))
        strong_intent = any(abs(float(v) - 0.5) > 0.12 for v in intents.values())
        should_credit = strong_intent and ((self.tick - self._l1_last_credit_tick) >= credit_every)
        obs_before = dict(self.agent.env.observe()) if should_credit else None
        if cpg_sync:
            fn(targets, cpg_sync=dict(cpg_sync))
        else:
            fn(targets)
        obs_after = dict(self.agent.env.observe())
        self._sync_motor_state(obs_after, source=str(latest.get("source", "cpg")), tick=self.tick)
        self._log_motor_command(
            source=str(latest.get("source", "cpg")),
            joint_targets=targets,
            intents=intents or None,
            obs=self._motor_obs_payload(obs_after),
        )
        if should_credit and obs_before is not None:
            self._record_motor_burst_causal(
                obs_before_env=obs_before,
                obs_after_env=obs_after,
                intents=intents,
            )
            self._l1_last_credit_tick = self.tick
        self._l1_last_apply_tick = self.tick
