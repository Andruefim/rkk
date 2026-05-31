"""Simulation mixin: падение и recovery."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationFallMixin:
    def _reset_tick_obs_caches(self) -> None:
        self._tick_env_obs = None
        self._tick_phys_state_dict = None
        self._tick_phys_state_tick = -1
        self._bt_snap_tick = -1
        self._bt_snap_payload = None
        self._tick_graph_vec = None

    def _graph_vec_cached(self) -> dict[str, float]:
        """One snapshot_vec_dict per sim tick (invalidated after agent.step)."""
        cached = getattr(self, "_tick_graph_vec", None)
        if cached is not None:
            return cached
        g = self.agent.graph
        g.set_snapshot_vec_engine_tick(int(self.tick))
        vec = g.snapshot_vec_dict()
        self._tick_graph_vec = vec
        return vec

    def _env_observe_cached(self) -> dict:
        """Один env.observe() на тик (PyBullet)."""
        obs = getattr(self, "_tick_env_obs", None)
        if obs is None:
            obs = dict(self.agent.env.observe())
            self._tick_env_obs = obs
        return obs

    def _invalidate_env_observe_cache(self) -> None:
        self._tick_env_obs = None

    def _tick_phys_state(self) -> dict | None:
        """Один sim.get_state() на тик для com_x / com_y."""
        if getattr(self, "_tick_phys_state_tick", -1) == int(self.tick):
            return getattr(self, "_tick_phys_state_dict", None)
        self._tick_phys_state_tick = int(self.tick)
        self._tick_phys_state_dict = None
        base = self._unwrap_base_env(self.agent.env)
        sim = getattr(base, "_sim", None)
        if sim is not None and callable(getattr(sim, "get_state", None)):
            try:
                st = sim.get_state()
                if isinstance(st, dict):
                    self._tick_phys_state_dict = st
            except Exception:
                pass
        return self._tick_phys_state_dict

    def _refresh_behavioral_snapshot_cache(self) -> dict | None:
        bt = getattr(self, "behavioral_tracker", None)
        if bt is None:
            return None
        snap = bt.snapshot()
        self._bt_snap_tick = int(self.tick)
        self._bt_snap_payload = snap
        return snap

    def _behavioral_snapshot_cached(self) -> dict | None:
        if getattr(self, "_bt_snap_tick", -1) == int(self.tick):
            payload = getattr(self, "_bt_snap_payload", None)
            if payload is not None:
                return payload
        bt = getattr(self, "behavioral_tracker", None)
        if bt is None:
            return None
        snap = bt.snapshot()
        self._bt_snap_tick = int(self.tick)
        self._bt_snap_payload = snap
        return snap

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

    def _genome_fall_recovery_enabled(self) -> bool:
        raw = os.environ.get("RKK_GENOME_FALL_RECOVERY")
        if raw is not None:
            return raw.strip().lower() not in ("0", "false", "no", "off")
        try:
            from engine.system2.controller import (
                _s2_learned_recovery_enabled,
                system2_enabled,
            )

            if system2_enabled() and _s2_learned_recovery_enabled():
                return False
        except Exception:
            pass
        return True

    @staticmethod
    def _fallen_signal_for_s2(obs: dict, *, fallen_debounced: bool) -> bool:
        """S2 streak: debounced fallen OR low posture/com_z (survives reset_stance)."""
        if fallen_debounced:
            return True
        ps = float(
            obs.get("posture_stability", obs.get("phys_posture_stability", 1.0))
        )
        cz = float(obs.get("com_z", obs.get("phys_com_z", 1.0)))
        try:
            ps_th = float(os.environ.get("RKK_S2_FALLEN_POSTURE_TH", "0.42"))
            cz_th = float(os.environ.get("RKK_S2_FALLEN_COM_Z_TH", "0.38"))
        except ValueError:
            ps_th, cz_th = 0.42, 0.38
        return ps < ps_th or cz < cz_th

    def _raw_com_x_m(self) -> float | None:
        st = self._tick_phys_state()
        if isinstance(st, dict) and "com_x" in st:
            try:
                return float(st["com_x"])
            except (TypeError, ValueError):
                pass
        return None

    def _raw_com_forward_m(self) -> float | None:
        """World forward axis (PyBullet com[1] / com_y)."""
        st = self._tick_phys_state()
        if isinstance(st, dict) and "com_y" in st:
            try:
                return float(st["com_y"])
            except (TypeError, ValueError):
                pass
        return None

    def _maybe_recover_or_reset_after_fall(self, obs: dict) -> bool:
        """
        Recovery-first policy:
        - give the agent time to stand up on its own,
        - hard-reset only when recovery stalls for too long.
        Returns True if a hard reset was performed.
        """
        score = self._fall_recovery_score(obs)
        try:
            max_ticks = int(os.environ.get("RKK_FALL_RECOVERY_TICKS", "120"))
        except ValueError:
            max_ticks = 120
        try:
            stall_ticks = int(os.environ.get("RKK_FALL_RECOVERY_STALL_TICKS", "72"))
        except ValueError:
            stall_ticks = 72
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
            self._fall_recovery_best_score = score
            self._fall_recovery_last_progress_tick = self.tick
            self._genome_stand_phase = 0
            self._genome_stand_phase_tick = self.tick
            try:
                from engine.genome.priors import get_stand_program
                self._genome_stand_program = get_stand_program()
            except Exception:
                self._genome_stand_program = []
            self._add_event(
                f"🦿 Recovery: genome stand program ({len(self._genome_stand_program)} phases)",
                "#ffbb66", "value",
            )

        # Execute genome stand program phases
        prog = getattr(self, "_genome_stand_program", [])
        phase_idx = getattr(self, "_genome_stand_phase", 0)
        if prog and phase_idx < len(prog):
            phase = prog[phase_idx]
            phase_elapsed = self.tick - getattr(self, "_genome_stand_phase_tick", self.tick)
            if phase_elapsed >= phase["ticks"]:
                self._genome_stand_phase = phase_idx + 1
                self._genome_stand_phase_tick = self.tick
                phase_idx = self._genome_stand_phase
            if phase_idx < len(prog):
                base = self._unwrap_base_env(self.agent.env)
                burst_fn = getattr(base, "intervene_burst", None)
                if callable(burst_fn):
                    pairs = [(k, v) for k, v in prog[phase_idx]["intents"].items()]
                    try:
                        burst_fn(pairs, count_intervention=False)
                    except Exception:
                        pass

        elapsed = self.tick - self._fall_recovery_start_tick
        if score > self._fall_recovery_best_score + min_gain:
            self._fall_recovery_best_score = score
            self._fall_recovery_last_progress_tick = self.tick

        stalled = (self.tick - self._fall_recovery_last_progress_tick) >= stall_ticks
        timed_out = elapsed >= max_ticks

        if stalled or timed_out:
            self._clear_fall_recovery()
            return self._try_reset_pose_after_fall()

        return False

