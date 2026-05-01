"""Simulation mixin: падение и recovery."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationFallMixin:
    def _try_reset_pose_after_fall(self) -> bool:
        """Сброс позы гуманоида (база PyBullet), чтобы выйти из ловушки fallen + VL block."""
        env = self.agent.env
        fn = getattr(env, "reset_stance", None)
        if not callable(fn):
            return False
        if self.tick - self._last_fall_reset_tick < 4:
            return False
        fn()
        self.agent.graph._obs_buffer.clear()
        self.agent.graph._int_buffer.clear()
        self._last_fall_reset_tick = self.tick
        self._add_event("🔄 Сброс позы после падения", "#44aaff", "value")
        return True

    @staticmethod
    def _fall_recovery_score(obs: dict) -> float:
        cz = float(obs.get("com_z", obs.get("phys_com_z", 0.0)))
        posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.0)))
        foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.0)))
        foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.0)))
        return 0.45 * cz + 0.35 * posture + 0.20 * min(foot_l, foot_r)

    def _clear_fall_recovery(self) -> None:
        self._fall_recovery_active = False
        self._fall_recovery_start_tick = 0
        self._fall_recovery_last_progress_tick = 0
        self._fall_recovery_best_score = 0.0

    def _maybe_recover_or_reset_after_fall(self, obs: dict) -> bool:
        """
        Recovery-first policy:
        - give the agent time to stand up on its own,
        - hard-reset only when recovery stalls for too long.
        Returns True if a hard reset was performed.
        """
        score = self._fall_recovery_score(obs)
        try:
            max_ticks = int(os.environ.get("RKK_FALL_RECOVERY_TICKS", "200"))
        except ValueError:
            max_ticks = 40
        try:
            stall_ticks = int(os.environ.get("RKK_FALL_RECOVERY_STALL_TICKS", "12"))
        except ValueError:
            stall_ticks = 12
        try:
            min_gain = float(os.environ.get("RKK_FALL_RECOVERY_MIN_GAIN", "0.02"))
        except ValueError:
            min_gain = 0.02
        max_ticks = max(8, min(max_ticks, 600))
        stall_ticks = max(4, min(stall_ticks, max_ticks))
        min_gain = float(np.clip(min_gain, 0.0, 0.25))

        if not self._fall_recovery_active:
            self._fall_recovery_active = True
            self._fall_recovery_start_tick = self.tick
            self._add_event("🦿 Recovery window after fall (INF)", "#ffbb66", "value")

        # Агент предоставлен сам себе на полу. Никаких автоматических подъемов!
        return False

