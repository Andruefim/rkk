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

    def _apply_pose_reset_after_fall(self, *, event_label: str, event_color: str) -> bool:
        """Shared reset_stance path (task/OWM state preserved on env side)."""
        env = self.agent.env
        fn = getattr(env, "reset_stance", None)
        if not callable(fn):
            return False
        if self.tick - self._last_fall_reset_tick < 4:
            return False
        fn()
        lc = getattr(self, "_locomotion_controller", None)
        if lc is not None:
            reset_fn = getattr(lc, "reset_cpg_phases", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass
        self.agent.graph._obs_buffer.clear()
        self.agent.graph._int_buffer.clear()
        self._last_fall_reset_tick = self.tick
        sync_fn = getattr(self, "_sync_graph_intents_to_defaults", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                pass
        hold_fn = getattr(self, "_arm_post_reset_motor_hold", None)
        if callable(hold_fn):
            try:
                hold_fn()
            except Exception:
                pass
        self._add_event(event_label, event_color, "value")
        return True

    def _try_reset_pose_after_fall(self) -> bool:
        """Сброс позы гуманоида (база PyBullet), чтобы выйти из ловушки fallen + VL block."""
        from engine.task_binding import human_task_embodiment_protected

        if human_task_embodiment_protected(self):
            return False
        return self._apply_pose_reset_after_fall(
            event_label="🔄 Сброс позы после падения",
            event_color="#44aaff",
        )

    def _fall_assist_allowed_for_stage(self) -> bool:
        """Assist teleport only during early locomotion stages — not reach/verify."""
        try:
            from engine.task_executive import active_tree_stage_kind

            stage = str(active_tree_stage_kind(self) or "").strip()
        except Exception:
            return False
        if not stage:
            return False
        return stage in ("resolve_target", "approach", "approach_target")

    def _try_task_fall_assist_reset(self) -> bool:
        """One assist reset during protected human task (preserves task/OWM)."""
        if not self._fall_assist_allowed_for_stage():
            return False
        if self._apply_pose_reset_after_fall(
            event_label="task_fall_assist_reset",
            event_color="#66ccff",
        ):
            self._task_fall_assist_used = True
            self._task_fallen_after_assist_ticks = 0
            try:
                from engine.task_logger import task_log_event

                task_log_event(
                    "task_fall_assist_reset",
                    tick=int(getattr(self, "tick", 0)),
                )
            except Exception:
                pass
            unlock_fn = getattr(self, "_owm_unlock_after_teleport", None)
            if callable(unlock_fn):
                try:
                    unlock_fn(int(getattr(self, "tick", 0)), reason="task_fall_assist_reset")
                except Exception:
                    pass
            return True
        return False

    def _locked_contact_target_xy(self) -> tuple[float, float] | None:
        """World XY of the locked contact body (registry / live PyBullet)."""
        bid = getattr(self, "_task_locked_body_id", None)
        if bid is None:
            return None
        row_fn = getattr(self, "_static_registry_row_for_body", None)
        row = row_fn(int(bid)) if callable(row_fn) else None
        if isinstance(row, dict):
            try:
                return float(row.get("x", 0.0)), float(row.get("y", 0.0))
            except (TypeError, ValueError):
                pass
        base = None
        phys_fn = getattr(self, "_humanoid_physics_sim", None)
        if callable(phys_fn):
            base = phys_fn()
        if base is None:
            unwrap = getattr(self, "_unwrap_base_env", None)
            env = getattr(getattr(self, "agent", None), "env", None)
            if callable(unwrap) and env is not None:
                base = unwrap(env)
                base = getattr(base, "_sim", base)
        if base is None:
            return None
        try:
            import pybullet as pb

            client = getattr(base, "client", None)
            lock = getattr(base, "_physics_lock", None)
            if lock is not None:
                lock.acquire()
            try:
                p, _ = pb.getBasePositionAndOrientation(int(bid), physicsClientId=client)
                return float(p[0]), float(p[1])
            finally:
                if lock is not None:
                    lock.release()
        except Exception:
            return None

    def _try_task_face_lift_toward_locked(self) -> bool:
        """
        In-place face+lift toward locked body — preserves approach XY progress.

        Spawn teleport would erase closing distance; fallen body yaw is garbage
        so crawl orbits. Re-orient upright facing the locked cylinder, then
        resume WM/AI / crawl navigation.
        """
        if not self._fall_assist_allowed_for_stage():
            return False
        # Do not erase near-goal closing distance.
        near_fn = getattr(self, "_task_fall_assist_near_goal", None)
        if callable(near_fn) and near_fn():
            return False
        phys_fn = getattr(self, "_physics_range_to_locked_body", None)
        if callable(phys_fn):
            try:
                phys = phys_fn()
                if phys is not None and float(phys) <= 1.05:
                    return False
            except Exception:
                pass
        target = self._locked_contact_target_xy()
        if target is None:
            return False
        tick = int(getattr(self, "tick", 0))
        try:
            every = int(os.environ.get("RKK_TASK_FACE_LIFT_EVERY", "90"))
        except ValueError:
            every = 90
        every = max(16, min(every, 600))
        last = int(getattr(self, "_task_face_lift_tick", -10_000))
        if tick - last < every:
            return False
        if tick - int(getattr(self, "_last_fall_reset_tick", -999)) < 4:
            return False

        env = getattr(getattr(self, "agent", None), "env", None)
        fn = getattr(env, "face_target_and_lift", None) if env is not None else None
        if not callable(fn):
            # Fallback through physics sim if HumanoidEnv wrapper missing.
            base = None
            phys_fn = getattr(self, "_humanoid_physics_sim", None)
            if callable(phys_fn):
                base = phys_fn()
            fn = getattr(base, "face_target_and_lift", None) if base is not None else None
        if not callable(fn):
            return False
        try:
            out = fn(target)
        except Exception:
            return False
        if not bool((out or {}).get("ok", True)):
            return False

        self._task_face_lift_tick = tick
        self._last_fall_reset_tick = tick
        self._task_fallen_after_assist_ticks = 0
        # Do NOT consume the one-shot spawn assist — face-lift may repeat.
        lc = getattr(self, "_locomotion_controller", None)
        if lc is not None:
            reset_fn = getattr(lc, "reset_cpg_phases", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass
        graph = getattr(getattr(self, "agent", None), "graph", None)
        if graph is not None:
            try:
                graph._obs_buffer.clear()
                graph._int_buffer.clear()
            except Exception:
                pass
        sync_fn = getattr(self, "_sync_graph_intents_to_defaults", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                pass
        hold_fn = getattr(self, "_arm_post_reset_motor_hold", None)
        if callable(hold_fn):
            try:
                hold_fn()
            except Exception:
                pass
        unlock_fn = getattr(self, "_owm_unlock_after_teleport", None)
        if callable(unlock_fn):
            try:
                unlock_fn(tick, reason="task_face_lift")
            except Exception:
                pass
        self._add_event("task_face_lift", "#88e0a8", "value")
        try:
            from engine.task_logger import task_log_event

            task_log_event(
                "task_face_lift",
                tick=tick,
                target_x=round(float(target[0]), 4),
                target_y=round(float(target[1]), 4),
                x=round(float((out or {}).get("x", 0.0)), 4),
                y=round(float((out or {}).get("y", 0.0)), 4),
                yaw=round(float((out or {}).get("yaw", 0.0)), 4),
            )
        except Exception:
            pass
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

    def _maybe_recover_or_reset_after_fall(
        self, obs: dict, *, apply_genome_program: bool = True
    ) -> bool:
        """
        Recovery-first policy:
        - give the agent time to stand up on its own,
        - hard-reset only when recovery stalls for too long.
        When apply_genome_program=False (S2 learned recovery), only stall watchdog + reset.
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
            if apply_genome_program:
                self._genome_stand_phase = 0
                self._genome_stand_phase_tick = self.tick
                try:
                    from engine.genome.priors import get_stand_program
                    self._genome_stand_program = get_stand_program()
                except Exception:
                    self._genome_stand_program = []
                self._add_event(
                    f"🦿 Recovery: genome stand program ({len(self._genome_stand_program)} phases)",
                    "#ffbb66",
                    "value",
                )
            else:
                self._genome_stand_program = []
                self._add_event(
                    "🦿 Recovery: stall watchdog (hard reset if no progress)",
                    "#ffbb66",
                    "value",
                )

        if apply_genome_program:
            prog = getattr(self, "_genome_stand_program", [])
            phase_idx = getattr(self, "_genome_stand_phase", 0)
            if prog and phase_idx < len(prog):
                phase = prog[phase_idx]
                phase_elapsed = self.tick - getattr(
                    self, "_genome_stand_phase_tick", self.tick
                )
                if phase_elapsed >= phase["ticks"]:
                    self._genome_stand_phase = phase_idx + 1
                    self._genome_stand_phase_tick = self.tick
                    phase_idx = self._genome_stand_phase
                if phase_idx < len(prog):
                    base = self._unwrap_base_env(self.agent.env)
                    burst_fn = getattr(base, "intervene_burst", None)
                    if callable(burst_fn):
                        pairs = [
                            (k, v) for k, v in prog[phase_idx]["intents"].items()
                        ]
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
            from engine.task_binding import human_task_embodiment_protected

            if human_task_embodiment_protected(self):
                try:
                    assist_threshold = int(
                        os.environ.get("RKK_TASK_FALL_ASSIST_TICKS", "180")
                    )
                except ValueError:
                    assist_threshold = 180
                assist_threshold = max(8, min(assist_threshold, 2000))
                fallen_ticks = int(getattr(self, "_task_fallen_ticks", 0))
                stall_count = int(getattr(self, "_task_fall_protected_stall_ticks", 0)) + 1
                self._task_fall_protected_stall_ticks = stall_count
                at_threshold = (
                    fallen_ticks >= assist_threshold
                    or stall_count >= assist_threshold
                )
                if (
                    at_threshold
                    and not bool(getattr(self, "_task_fall_assist_used", False))
                    and self._fall_assist_allowed_for_stage()
                ):
                    progress_fn = getattr(
                        self, "_task_fall_assist_progress_blocks_reset", None
                    )
                    if callable(progress_fn) and progress_fn():
                        # Spawn teleport would erase closing distance — face+lift
                        # in place so crawl yaw matches the locked body.
                        face_fn = getattr(self, "_try_task_face_lift_toward_locked", None)
                        if callable(face_fn) and face_fn():
                            self._clear_fall_recovery()
                            return True
                        self._add_event(
                            "task_fall_assist_skipped_progress", "#66ccff", "value"
                        )
                        try:
                            from engine.task_logger import task_log_event

                            task_log_event(
                                "task_fall_assist_skipped_progress",
                                tick=int(getattr(self, "tick", 0)),
                                fallen_ticks=int(fallen_ticks),
                                stall_count=int(stall_count),
                            )
                        except Exception:
                            pass
                        return False
                    self._clear_fall_recovery()
                    return self._try_task_fall_assist_reset()
                # Even after the one-shot spawn assist was used/skipped, keep
                # re-orienting in place so fallen crawl does not orbit forever.
                if (
                    at_threshold
                    and self._fall_assist_allowed_for_stage()
                ):
                    face_fn = getattr(self, "_try_task_face_lift_toward_locked", None)
                    if callable(face_fn) and face_fn():
                        self._clear_fall_recovery()
                        return True
                # Genome recovery continues; no hard reset until assist threshold.
                return False
            self._clear_fall_recovery()
            return self._try_reset_pose_after_fall()

        return False

