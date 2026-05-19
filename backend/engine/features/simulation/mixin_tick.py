"""Simulation mixin: tick_step, один шаг агента."""
from __future__ import annotations

import json
from pathlib import Path

from engine.features.simulation.mixin_imports import *

# #region agent log
_DBG_LOG_F7 = Path(__file__).resolve().parents[4] / "debug-f7a777.log"


def _dbg_tick(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    try:
        with _DBG_LOG_F7.open("a", encoding="utf-8") as _df:
            _df.write(
                json.dumps(
                    {
                        "sessionId": "f7a777",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "timestamp": int(time.time() * 1000),
                        "runId": "pre-fix",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion


class SimulationTickMixin:
    """Humanoid tick orchestration.

    Phase C₁ (temporal contracts): reflex / CPG / stabilizers run without invoking imagination,
    LLM loops, or L3 goal-planning; those live in ``_run_agent_or_skill_step`` and async teachers.
    Leg commands owned by CPG must not receive conflicting high-rate ``do()`` on the same joints
    (enforced in locomotion / EIG paths — see ``mixin_locomotion``).
    """

    def _sync_temporal_blankets_to_graph(self) -> None:
        """Rebuild TemporalBlankets when |graph nodes| changes (inner_voice, concepts, neurogenesis)."""
        from engine.temporal import TemporalBlankets

        g_d = len(self.agent.graph._node_ids)
        if g_d <= 0:
            return
        tb = self.agent.temporal
        if tb.d_input == g_d:
            return
        self.agent.temporal = TemporalBlankets(d_input=g_d, device=self.device)

    def _maybe_post_release_stabilize_intents(self) -> None:
        """После снятия fixed_root — в окне stabilize_until усилить recover/support (плавный decay)."""
        if self.current_world != "humanoid" or self._fixed_root_active:
            return
        if not getattr(self, "_curriculum_auto_fr_released", False):
            return
        t0 = int(getattr(self, "_post_fr_last_release_tick", -1))
        if t0 < 0:
            return
        until = int(getattr(self, "_curriculum_stabilize_until", 0) or 0)
        if until <= 0:
            return
        if self.tick > until:
            return
        span = max(1, until - t0)
        age = max(0, int(self.tick) - t0)
        if age > span:
            return
        base = self._unwrap_base_env(self.agent.env)
        fn = getattr(base, "apply_motor_intent_residuals", None)
        if not callable(fn):
            return
        try:
            d_rec = float(os.environ.get("RKK_POST_FR_STOP_RECOVER_DELTA", "0.07"))
        except ValueError:
            d_rec = 0.07
        try:
            d_sup = float(os.environ.get("RKK_POST_FR_SUPPORT_DELTA", "0.06"))
        except ValueError:
            d_sup = 0.06
        decay = max(0.12, 1.0 - float(age) / float(span))
        scale = float(np.clip(decay, 0.2, 1.0))
        fn(
            {
                "intent_stop_recover": d_rec * scale,
                "intent_support_left": d_sup * scale,
                "intent_support_right": d_sup * scale,
            }
        )

    def _fr_curriculum_finalize_release(self, *, reason: str) -> None:
        """Снять fixed_root в симуляции + VL, выставить окно stabilize (после мягкой физики)."""
        self._curriculum_auto_fr_released = True
        self.disable_fixed_root()
        self._fr_soft_release_deadline = 0
        self._fr_soft_release_start = 0
        self._fr_soft_release_initial_ratio = 1.0
        try:
            stab = int(os.environ.get("RKK_POST_FR_STABILIZE_TICKS", "120"))
        except ValueError:
            stab = 120
        self._curriculum_stabilize_until = self.tick + max(0, stab)
        self._add_event(
            f"📌 fixed_root OFF ({reason}) tick {self.tick}, "
            f"stabilize until {self._curriculum_stabilize_until}",
            "#66ccaa",
            "phase",
        )

    def _maybe_damp_motor_intents_blind_fixed_root(self) -> None:
        """При fixed_root и низком compression_gain подтягивать intent_* к дефолтам среды (меньше слепого дрейфа)."""
        if self.current_world != "humanoid" or not self._fixed_root_active:
            return
        if os.environ.get("RKK_FR_BLIND_MOTOR_DAMP", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return
        try:
            cg_abs_max = float(os.environ.get("RKK_FR_BLIND_CG_ABS_MAX", "0.04"))
        except ValueError:
            cg_abs_max = 0.04
        cg_abs_max = float(max(0.005, min(0.25, cg_abs_max)))
        if abs(float(self.agent.compression_gain)) > cg_abs_max:
            return
        try:
            excursion_scale = float(
                os.environ.get("RKK_FR_BLIND_INTENT_EXCURSION_SCALE", "0.92")
            )
        except ValueError:
            excursion_scale = 0.92
        excursion_scale = float(np.clip(excursion_scale, 0.55, 0.999))
        base = self._unwrap_base_env(self.agent.env)
        ms = getattr(base, "_motor_state", None)
        if not isinstance(ms, dict):
            return
        try:
            from engine.features.humanoid.constants import MOTOR_INTENT_DEFAULTS
        except Exception:
            MOTOR_INTENT_DEFAULTS = {}
        raw_skip = os.environ.get("RKK_FR_BLIND_DAMP_SKIP", "")
        skip: set[str] = {x.strip() for x in raw_skip.split(",") if x.strip()}
        changed = False
        for sk in list(ms.keys()):
            if not str(sk).startswith("intent_"):
                continue
            if str(sk) in skip:
                continue
            prev = float(ms.get(sk, 0.5))
            anchor = float(MOTOR_INTENT_DEFAULTS.get(str(sk), 0.5))
            new_v = float(anchor + (prev - anchor) * excursion_scale)
            new_v = float(np.clip(new_v, 0.05, 0.95))
            if abs(new_v - prev) > 1e-6:
                ms[str(sk)] = new_v
                changed = True
        if not changed:
            return
        if not getattr(base, "_intero_control_lost", False):
            fn = getattr(base, "_apply_motor_intents", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        try:
            obs = dict(self.agent.env.observe())
            self.agent.graph.apply_env_observation(obs, engine_tick=self.tick)
            self._sync_motor_state(obs, source="fr_blind_damp", tick=self.tick)
        except Exception:
            pass

    def _fr_posture_and_bias_from_graph(self) -> tuple[float, float]:
        g = self.agent.graph.nodes
        posture = float(
            g.get("posture_stability", g.get("phys_posture_stability", 0.5))
        )
        bias = float(g.get("support_bias", g.get("phys_support_bias", 0.5)))
        return posture, bias

    def _fr_update_release_tracking(self) -> None:
        if not self._fixed_root_active:
            return
        try:
            posture_min = float(os.environ.get("RKK_FR_RELEASE_POSTURE_MIN", "0.85"))
        except ValueError:
            posture_min = 0.85
        posture, bias = self._fr_posture_and_bias_from_graph()
        self._fr_support_bias_hist.append(bias)
        if posture >= posture_min:
            self._fr_posture_streak += 1
        else:
            self._fr_posture_streak = 0

    def _fr_early_release_ready(self) -> tuple[bool, str]:
        try:
            rel_mq = float(os.environ.get("RKK_FIXED_ROOT_RELEASE_MASTERY", "0.72"))
            rel_min = max(
                1, int(os.environ.get("RKK_FIXED_ROOT_RELEASE_MIN_TICKS", "400"))
            )
            streak_need = max(
                10, int(os.environ.get("RKK_FR_RELEASE_POSTURE_STREAK", "40"))
            )
            bias_need = float(os.environ.get("RKK_FR_RELEASE_BIAS_RANGE", "0.15"))
            hist_min = max(20, int(os.environ.get("RKK_FR_RELEASE_BIAS_WINDOW", "30")))
        except ValueError:
            rel_mq, rel_min, streak_need, bias_need, hist_min = (
                0.72,
                400,
                40,
                0.15,
                30,
            )
        if self.tick < rel_min or self.tick < self._fr_release_blocked_until:
            return False, ""
        if not hasattr(self.agent, "_prog_scope"):
            return False, ""
        if float(self.agent._prog_scope.mastery_quality) < rel_mq:
            return False, ""
        if self._fr_posture_streak < streak_need:
            return False, ""
        hist = list(self._fr_support_bias_hist)
        if len(hist) < hist_min:
            return False, ""
        if (max(hist) - min(hist)) < bias_need:
            return False, ""
        return True, "mastery+posture+bias"

    def _fr_try_reattach_after_fall(self, obs: dict) -> None:
        if not self._curriculum_auto_fr_released or self._fixed_root_active:
            return
        if self._fr_reattach_active:
            return
        try:
            min_fallen = max(
                0, int(os.environ.get("RKK_FR_REATTACH_MIN_FALLEN_TICKS", "0"))
            )
        except ValueError:
            min_fallen = 0
        if min_fallen > 0 and int(getattr(self, "_fr_fallen_ticks_accum", 0)) < min_fallen:
            return
        try:
            max_n = max(1, int(os.environ.get("RKK_POST_FR_REATTACH_MAX", "3")))
            dur = max(40, int(os.environ.get("RKK_POST_FR_REATTACH_TICKS", "150")))
        except ValueError:
            max_n, dur = 3, 150
        if self._fr_reattach_count >= max_n:
            return
        self._fr_reattach_count += 1
        self._fr_reattach_active = True
        self._fr_reattach_until = self.tick + dur
        self._fr_release_blocked_until = self._fr_reattach_until + dur
        self._fr_posture_streak = 0
        r = self.enable_fixed_root()
        if not r.get("error"):
            fn = getattr(self.agent.env, "reset_stance", None)
            if callable(fn):
                fn()
            self.agent.graph._obs_buffer.clear()
            self.agent.graph._int_buffer.clear()
            self._add_event(
                f"📌 fixed_root RE-ATTACH ({dur} ticks, "
                f"fall #{self._fr_reattach_count}/{max_n})",
                "#ffaa66",
                "phase",
            )

    def _fr_maybe_end_reattach(self) -> None:
        if not self._fr_reattach_active or self.tick < self._fr_reattach_until:
            return
        base = self._unwrap_base_env(self.agent.env)
        z_fn = getattr(base, "_fallen_z_below_threshold", None)
        if callable(z_fn):
            fallen = bool(z_fn())
        else:
            is_fn = getattr(self.agent.env, "is_fallen", None)
            fallen = is_fn() if callable(is_fn) else False
        posture, _ = self._fr_posture_and_bias_from_graph()
        try:
            posture_ok = float(os.environ.get("RKK_FR_REATTACH_POSTURE_MIN", "0.82"))
        except ValueError:
            posture_ok = 0.82
        if fallen or posture < posture_ok:
            return
        self._fr_reattach_active = False
        self.disable_fixed_root()
        try:
            stab = int(os.environ.get("RKK_POST_FR_STABILIZE_TICKS", "120"))
        except ValueError:
            stab = 120
        self._curriculum_stabilize_until = self.tick + max(0, stab)
        self._add_event(
            f"📌 fixed_root re-release after reattach (tick {self.tick})",
            "#66ccaa",
            "phase",
        )

    # ── Tick ──────────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        hz = _agent_loop_hz_from_env()
        # #region agent log
        _t_tick = time.perf_counter()
        # #endregion
        if hz > 0.0:
            self._bg.ensure_rkk_agent_loop()
            with self._sim_step_lock:
                cached = self._agent_step_response
            if cached is not None:
                # Не deepcopy: снимок десятки KB+ — при 15–20 Hz это раздувает RAM на гигабайты
                # и тормозит event loop; фоновый поток каждый цикл пишет новый dict.
                # #region agent log
                _dbg_tick(
                    "H3",
                    "mixin_tick.tick_step",
                    "return_cached",
                    {"hz": hz, "total_ms": (time.perf_counter() - _t_tick) * 1000},
                )
                # #endregion
                return cached
            # Не вызываем public_state() сразу: это второй полный snapshot+PyBullet, пока
            # rkk-agent-loop держит lock на первом тике — минутные зависания и «1 тик / 15 с».
            try:
                max_wait = float(os.environ.get("RKK_WS_AGENT_CACHE_WAIT_SEC", "90"))
            except ValueError:
                max_wait = 90.0
            max_wait = max(0.25, min(300.0, max_wait))
            deadline = time.perf_counter() + max_wait
            # #region agent log
            _spin_start = time.perf_counter()
            _spin_iters = 0
            # #endregion
            while time.perf_counter() < deadline:
                time.sleep(0.04)
                # #region agent log
                _spin_iters += 1
                # #endregion
                with self._sim_step_lock:
                    cached = self._agent_step_response
                if cached is not None:
                    # #region agent log
                    _dbg_tick(
                        "H3",
                        "mixin_tick.tick_step",
                        "cache_after_spin",
                        {
                            "hz": hz,
                            "spin_ms": (time.perf_counter() - _spin_start) * 1000,
                            "spin_iters": _spin_iters,
                            "total_ms": (time.perf_counter() - _t_tick) * 1000,
                        },
                    )
                    # #endregion
                    return cached
            # #region agent log
            _ps0 = time.perf_counter()
            # #endregion
            out = self.public_state()
            # #region agent log
            _dbg_tick(
                "H3",
                "mixin_tick.tick_step",
                "public_state_fallback",
                {
                    "hz": hz,
                    "public_state_ms": (time.perf_counter() - _ps0) * 1000,
                    "spin_ms": (_ps0 - _spin_start) * 1000,
                    "spin_iters": _spin_iters,
                    "total_ms": (time.perf_counter() - _t_tick) * 1000,
                },
            )
            # #endregion
            return out
        # #region agent log
        _t_sync = time.perf_counter()
        # #endregion
        with self._sim_step_lock:
            _inner = self._run_single_agent_timestep_inner()
        # #region agent log
        _dbg_tick(
            "H3",
            "mixin_tick.tick_step",
            "sync_path_inner",
            {"hz": 0.0, "inner_ms": (time.perf_counter() - _t_sync) * 1000},
        )
        # #endregion
        return _inner

    def advance_agent_steps(self, n: int) -> None:
        """Синхронно выполнить n логических тиков агента (bootstrap при RKK_AGENT_LOOP_HZ>0)."""
        n = max(0, int(n))
        if n == 0:
            return
        with self._sim_step_lock:
            for _ in range(n):
                self._agent_step_response = self._run_single_agent_timestep_inner()

    def _run_single_agent_timestep_inner(self) -> dict:
        # #region agent log
        _t_inner0 = time.perf_counter()
        # #endregion
        self.tick += 1
        if self.current_world != "humanoid":
            self._hai_prev_com_x = None
            self._hai_pe_fwd_ema = 0.0
            self._hai_pe_vert_ema = 0.0
            self._hai_pe_lat_ema = 0.0
            self._hai_pe_ema = 0.0
        self._apply_pending_llm_bundle()
        self._ensure_phase2()

        # Humanoid curriculum: фаза 1 — fixed_root с тика 1; снятие после RKK_AUTO_FIXED_ROOT_TICKS.
        try:
            auto_fr_ticks = int(os.environ.get("RKK_AUTO_FIXED_ROOT_TICKS", "0"))
        except ValueError:
            auto_fr_ticks = 0
        if auto_fr_ticks > 0 and self.current_world == "humanoid":
            try:
                fr_retry_max = int(os.environ.get("RKK_CURRICULUM_FIXED_ROOT_RETRY_MAX", "16"))
            except ValueError:
                fr_retry_max = 16
            fr_retry_max = max(1, fr_retry_max)
            if self.tick <= fr_retry_max and not self._fixed_root_active:
                r = self.enable_fixed_root()
                if r.get("fixed_root") and not r.get("error") and self._fixed_root_active:
                    self._add_event(
                        "📌 Curriculum: fixed_root ON (phase 1, arms→cubes)",
                        "#66ccaa",
                        "phase",
                    )
            self._fr_maybe_end_reattach()

            if (
                self._fixed_root_active
                and not self._curriculum_auto_fr_released
            ):
                self._fr_update_release_tracking()
                release_window = 200
                soft_deadline = int(getattr(self, "_fr_soft_release_deadline", 0) or 0)
                if soft_deadline > 0 and self.tick < soft_deadline:
                    start = int(getattr(self, "_fr_soft_release_start", 0) or 0)
                    span = max(1, soft_deadline - start)
                    progress = float(self.tick - start) / float(span)
                    init_r = float(
                        getattr(self, "_fr_soft_release_initial_ratio", 1.0) or 1.0
                    )
                    init_r = float(np.clip(init_r, 0.0, 1.0))
                    soft_ratio = float(max(0.0, init_r * (1.0 - progress)))
                    self.set_fixed_root_force(soft_ratio)
                elif soft_deadline > 0 and self.tick >= soft_deadline:
                    reason = str(getattr(self, "_fr_soft_release_reason", "") or "?")
                    self._fr_curriculum_finalize_release(reason=reason)
                else:
                    if self.tick >= auto_fr_ticks - release_window:
                        ratio = max(
                            0.0,
                            float(auto_fr_ticks - self.tick) / float(release_window),
                        )
                        self.set_fixed_root_force(ratio)

                    early_release, rel_reason = self._fr_early_release_ready()
                    time_release = self.tick >= auto_fr_ticks

                    if time_release or early_release:
                        if early_release and not time_release:
                            reason = rel_reason
                        elif time_release and not early_release:
                            reason = f"tick≥{auto_fr_ticks}"
                        else:
                            reason = f"{rel_reason}+tick≥{auto_fr_ticks}"
                        try:
                            n_soft = int(
                                os.environ.get("RKK_FR_SOFT_RELEASE_TICKS", "40")
                            )
                        except ValueError:
                            n_soft = 40
                        n_soft = max(0, min(120, n_soft))
                        if n_soft > 0:
                            init_r = float(
                                np.clip(
                                    float(auto_fr_ticks - self.tick)
                                    / float(max(1, release_window)),
                                    0.0,
                                    1.0,
                                )
                            )
                            if not time_release:
                                init_r = max(init_r, 0.35)
                            # At tick==auto_fr_ticks the 200-tick window ratio is 0; without a floor
                            # soft-release would decay from zero and never unload stored constraint stress.
                            init_r = max(init_r, 0.12)
                            self._fr_soft_release_start = self.tick
                            self._fr_soft_release_deadline = self.tick + n_soft
                            self._fr_soft_release_initial_ratio = init_r
                            self._fr_soft_release_reason = reason
                            soft_ratio0 = float(max(0.0, init_r))
                            self.set_fixed_root_force(soft_ratio0)
                            self._add_event(
                                f"📌 fixed_root SOFT-RELEASE {n_soft} ticks (physics) → "
                                f"init_force_ratio={init_r:.2f}",
                                "#66ccaa",
                                "phase",
                            )
                        else:
                            self._fr_curriculum_finalize_release(reason=reason)

        # Fallen check + автосброс физики (иначе VL и block_rate залипают)
        fallen = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            fallen = is_fn()
            prev_f = bool(getattr(self, "_prev_fallen", False))
            fallen_edge = bool(fallen and not prev_f)
            if self._fall_recovery_active and not fallen:
                self._clear_fall_recovery()
            if fallen:
                self._fr_fallen_ticks_accum += 1
                obs_fall = dict(self.agent.env.observe())
                if fallen_edge:
                    self._fall_count += 1
                    try:
                        pend = dict(obs_fall)
                        base_b = self._unwrap_base_env(self.agent.env)
                        sm = getattr(base_b, "_sim", None)
                        if sm is not None and callable(getattr(sm, "get_state", None)):
                            st = sm.get_state()
                            if isinstance(st, dict) and "com_z" in st:
                                pend["com_z_raw_m"] = float(st["com_z"])
                        self._pending_fall_obs_for_memory = pend
                    except Exception:
                        self._pending_fall_obs_for_memory = None
                if (
                    self._curriculum_auto_fr_released
                    and self.tick > self._curriculum_stabilize_until
                ):
                    self._fr_try_reattach_after_fall(obs_fall)
                if self._maybe_recover_or_reset_after_fall(obs_fall):
                    obs = self.agent.env.observe()
                    self._sync_motor_state(obs, source="reset", tick=self.tick)
                    for nid in self.agent.graph._node_ids:
                        if nid in obs:
                            self.agent.graph.nodes[nid] = obs[nid]
                    self.agent.graph.record_observation(obs)
                    self.agent.temporal.step(obs)
                    fallen = is_fn()
                if not fallen:
                    self._pending_fall_obs_for_memory = None
                    self._fr_fallen_ticks_accum = 0
                if fallen_edge and self._fall_count % 20 == 1:
                    self._add_event(
                        f"💀 [FALLEN] Nova упал! (×{self._fall_count})",
                        "#ff2244", "value"
                    )
            else:
                self._fr_fallen_ticks_accum = 0
            self._prev_fallen = bool(fallen)
        else:
            self._prev_fallen = False
            self._pending_fall_obs_for_memory = None
            self._fr_fallen_ticks_accum = 0

        # Фаза 12: передаём GNN prediction в visual env (не каждый тик — см. VISION_GNN_FEED_EVERY)
        if self._visual_mode and self._visual_env is not None:
            self._vision_ticks += 1
            if self._vision_ticks % VISION_GNN_FEED_EVERY == 0:
                self._feed_gnn_prediction_to_visual()
            # Фаза 3: Топологическое Я
            if self._vision_ticks % 10 == 0:
                self._apply_topological_self_priors()

        # Фаза 3: annealing teacher_weight; VL-overlay только пока не истёк TTL и weight>0
        try:
            tmax = int(os.environ.get("RKK_TEACHER_T_MAX", "140"))
        except ValueError:
            tmax = 140
        tmax = max(1, tmax)
        tw = max(0.0, 1.0 - (self.agent._total_interventions / tmax))
        self.agent.set_teacher_state(self._phase3_teacher_rules, tw)
        ov = self._phase3_vl_overlay
        if ov is not None and self.tick <= ov.expires_at_tick and tw > 0:
            self.agent.value_layer.set_teacher_vl_overlay(ov)
        else:
            self.agent.value_layer.set_teacher_vl_overlay(None)

        # Phase C₁: reflex path (CPG + reflex stabilizer) stays fast and LLM-free; τ2/τ3 run later.
        # CPG runs BEFORE agent step so legs are stabilized before high-level exploration
        self._ensure_cpg_background_loop()
        self._drain_l1_motor_commands()
        # Re-use ``fallen`` from the early check (after optional recovery): no extra
        # ``is_fallen()`` here — duplicate calls would double-advance debounce streak.
        fallen_pre = bool(fallen)
        self._maybe_apply_cpg_locomotion(fallen_pre)
        self._publish_cpg_node_snapshot()
        if self.current_world == "humanoid" and not self._fixed_root_active:
            self._maybe_post_release_stabilize_intents()
        if self.current_world == "humanoid":
            base_env = self._unwrap_base_env(self.agent.env)
            cpg_on = bool(getattr(base_env, "cpg_owns_legs", False))
            loco_r = (
                self._locomotion_reward_ema()
                if self._locomotion_controller is not None
                else 0.0
            )
            self.agent.graph.set_locomotion_train_context(
                reward_ema=loco_r,
                cpg_active=cpg_on,
                fallen=bool(fallen_pre),
            )
        self.agent.other_agents_phi = []
        self._maybe_step_hierarchical_l1()
        self._sync_temporal_blankets_to_graph()

        if self.current_world == "humanoid":
            try:
                from engine.system2.controller import system2_enabled

                if system2_enabled():
                    if getattr(self, "_system2", None) is None:
                        from engine.system2 import System2Controller

                        self._system2 = System2Controller()
                    obs_s2 = dict(self.agent.graph.snapshot_vec_dict())
                    self._system2_last = self._system2.tick(
                        sim_tick=self.tick,
                        agent=self.agent,
                        obs=obs_s2,
                        sim=self,
                        fallen=bool(fallen),
                    )
                    fn_ctx = getattr(self._system2, "planning_context_for_wm", None)
                    if callable(fn_ctx):
                        self.agent.set_s2_planning_context(
                            fn_ctx(fallen=bool(fallen), sim_tick=int(self.tick))
                        )
                    else:
                        self.agent.set_s2_planning_context(None)
                else:
                    self._system2_last = None
                    self.agent.set_s2_planning_context(None)
            except Exception:
                self._system2_last = {"enabled": False, "error": "system2_tick"}
                self.agent.set_s2_planning_context(None)
        else:
            self._system2_last = None
            self.agent.set_s2_planning_context(None)

        # Controlled perturbations during fixed_root to teach active balance
        if self.current_world == "humanoid" and self._fixed_root_active:
            if self.tick % 40 == 0:
                fn_perturb = getattr(self.agent.env, "apply_random_perturbation", None)
                if callable(fn_perturb):
                    # Start gentle, increase force
                    force = 60.0 + min(100.0, self.tick * 0.2)
                    fn_perturb(max_force=force)

        obs_pre_rssm = dict(self.agent.graph.snapshot_vec_dict())
        _t_phase = time.perf_counter()
        result = self._run_agent_or_skill_step(engine_tick=self.tick)
        self._inner_phase_ms = getattr(self, "_inner_phase_ms", {})
        self._inner_phase_ms["agent"] = round((time.perf_counter() - _t_phase) * 1000.0, 2)

        # Track action for episodic memory
        self._record_last_action(result)
        if self.current_world == "humanoid" and self._fixed_root_active:
            self._maybe_damp_motor_intents_blind_fixed_root()

        _obs_for_d_e: dict = {}
        try:
            _obs_for_d_e = dict(self.agent.graph.snapshot_vec_dict())
        except Exception:
            pass
        _posture_now = float(
            _obs_for_d_e.get(
                "posture_stability",
                _obs_for_d_e.get("phys_posture_stability", 0.5),
            )
        )

        rs = getattr(self, "_reflex_stabilizer", None)
        if rs is None:
            try:
                from engine.reflex_stabilizer import reflex_stabilizer_enabled

                if reflex_stabilizer_enabled():
                    rs = self._ensure_reflex_stabilizer()
            except Exception:
                rs = None
        if rs is not None and self.current_world == "humanoid" and not self._fixed_root_active:
            try:
                train_every = int(os.environ.get("RKK_REFLEX_TRAIN_EVERY", "3"))
            except ValueError:
                train_every = 3
            train_every = max(1, train_every)
            if self.tick % train_every == 0:
                rs.train_on_outcome(self._reflex_posture_prev, _posture_now)
            self._reflex_posture_prev = _posture_now

        cb_cereb = getattr(self, "_cerebellum", None)
        if cb_cereb is None:
            try:
                from engine.cerebellum import cerebellum_enabled

                if cerebellum_enabled():
                    cb_cereb = self._ensure_cerebellum()
            except Exception:
                cb_cereb = None
        if cb_cereb is not None and self.current_world == "humanoid" and not self._fixed_root_active:
            if self._cerebellum_obs_prev is not None and self._last_joint_cmd_applied:
                cb_cereb.record_transition(
                    self._cerebellum_obs_prev,
                    self._last_joint_cmd_applied,
                    _obs_for_d_e,
                )
            try:
                train_every_cb = int(os.environ.get("RKK_CEREBELLUM_TRAIN_EVERY", "5"))
            except ValueError:
                train_every_cb = 5
            train_every_cb = max(1, train_every_cb)
            if self.tick % train_every_cb == 0:
                cb_cereb.train_step()
            intents = {}
            for k, v in self.agent.graph.nodes.items():
                sk = str(k)
                if not sk.startswith("intent_"):
                    continue
                try:
                    intents[sk] = float(v)
                except (TypeError, ValueError):
                    continue
            cb_cereb.set_desired_from_graph(dict(self.agent.graph.nodes), intents)
            self._cerebellum_obs_prev = dict(_obs_for_d_e)

        if (
            _MOTOR_CORTEX_AVAILABLE
            and self.current_world == "humanoid"
            and not self._fixed_root_active
        ):
            mc_fb = self._ensure_motor_cortex()
            if mc_fb is not None and len(mc_fb.programs) > 0:
                fl_mc = float(
                    _obs_for_d_e.get(
                        "foot_contact_l",
                        _obs_for_d_e.get("phys_foot_contact_l", 0.5),
                    )
                )
                fr_mc = float(
                    _obs_for_d_e.get(
                        "foot_contact_r",
                        _obs_for_d_e.get("phys_foot_contact_r", 0.5),
                    )
                )
                loco_r = self._locomotion_reward_ema()
                cpg_cmd: dict = {}
                if self._locomotion_controller is not None:
                    cpg_cmd = dict(
                        getattr(self._locomotion_controller, "_last_command", {}) or {}
                    )
                mc_fb.push_and_train(
                    nodes=dict(self.agent.graph.nodes),
                    cpg_targets=cpg_cmd,
                    reward=loco_r,
                    posture=_posture_now,
                    foot_l=fl_mc,
                    foot_r=fr_mc,
                )
                mc_fb.anneal_step(_posture_now, fl_mc, fr_mc, fallen, self.tick)
                if not self._mc_abstract_nodes_injected:
                    added = mc_fb.inject_abstract_nodes_into_graph(self.agent.graph)
                    if added > 0:
                        self._mc_abstract_nodes_injected = True
                        self._add_event(
                            "🧠 MotorCortex: +mc_* feedback nodes for GNN",
                            "#ff88ff",
                            "phase",
                        )
                mc_fb.sync_abstract_nodes_to_graph(self.agent.graph)

        # Level 3-I: Multi-scale time tick (first consumer of post-step obs)
        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            self._timescale.tick(self.tick, _obs_for_d_e)
            if self.current_world != "humanoid":
                motor_intents = self._timescale.get_intents(LEVEL_MOTOR)
                for var, val in motor_intents.items():
                    if var.startswith("intent_"):
                        try:
                            self.agent.env.intervene(var, float(val), count_intervention=False)
                        except Exception:
                            pass

        # Phase J: Inner Voice (τ2, fast, no LLM)
        self._tick_inner_voice(self.tick)

        # Phase J: LLM teacher (τ3, async)
        self._tick_llm_teacher(self.tick)

        # Level 3-G: Proprioception update (after CPG + agent step; fresh obs)
        _proprio_anomaly = 0.0
        _proprio_emp_reward = 0.0
        if _PROPRIO_AVAILABLE and self._proprio is not None and self.current_world == "humanoid":
            self._proprio.update(
                tick=self.tick,
                obs=_obs_for_d_e,
                graph=self.agent.graph if hasattr(self.agent, "graph") else None,
                agent=self.agent,
            )
            _proprio_anomaly = self._proprio.anomaly_score
            _proprio_emp_reward = self._proprio.get_empowerment_reward()
            # Feed empowerment reward into CPG so it's incentivized to create
            # diverse, high-influence actions (not just stand still)
            if self._locomotion_controller is not None and _proprio_emp_reward > 0:
                self._locomotion_controller._reward_history.append(
                    float(_proprio_emp_reward) * 0.3
                )

            if _TIMESCALE_AVAILABLE and self._timescale is not None:
                if self._timescale.should_run(LEVEL_REFLEX, self.tick):
                    self._timescale.mark_ran(LEVEL_REFLEX, self.tick)

        # Problem 3: hierarchical PE — stride prior vs com_x drift → low-level intent residuals
        self._hai_last_diag = None
        if self.current_world == "humanoid":
            if (
                not self._fixed_root_active
                and not fallen
                and _obs_for_d_e
            ):
                from engine.hierarchical_active_inference import run_hierarchical_pe_tick

                self._hai_last_diag = run_hierarchical_pe_tick(self, _obs_for_d_e)
            elif self._fixed_root_active or fallen:
                self._hai_prev_com_x = None
                self._hai_pe_fwd_ema = 0.0
                self._hai_pe_vert_ema = 0.0
                self._hai_pe_lat_ema = 0.0
                self._hai_pe_ema = 0.0

        # Phase K: Sleep Controller
        _t_sleep = time.perf_counter()
        if (
            _PHASE_K_AVAILABLE
            and self._sleep_ctrl is not None
            and self.current_world == "humanoid"
        ):
            if fallen and not self._was_fallen_last_tick:
                self._sleep_ctrl.notify_fall()
            self._was_fallen_last_tick = fallen

            _total_falls = (
                getattr(self._episodic_memory, "total_falls_recorded", 0)
                if self._episodic_memory
                else 0
            )
            _sleep_reason = self._sleep_ctrl.check_trigger(
                self.tick, _total_falls,
                intrinsic_objective=getattr(self, "_intrinsic", None),
            )

            if _sleep_reason and not self._sleep_ctrl.is_sleeping:
                self._sleep_ctrl.begin_sleep(self.tick, _sleep_reason, sim=self)
                self._add_event(
                    f"😴 Sleep: {_sleep_reason} (falls={self._sleep_ctrl._falls_since_sleep})",
                    "#9988ff",
                    "sleep",
                )

            if self._sleep_ctrl.is_sleeping:
                self._sleep_ctrl.tick(self.tick, self)
                if not self._sleep_ctrl.is_sleeping:
                    self._add_event(
                        f"🌅 Woke up (sleep #{self._sleep_ctrl.sleep_count})",
                        "#ffff88",
                        "sleep",
                    )
        self._inner_phase_ms["sleep"] = round((time.perf_counter() - _t_sleep) * 1000.0, 2)

        # Phase K: Physical curriculum when scheduler runs low on stages ahead
        if (
            _PHASE_K_AVAILABLE
            and self._physical_curriculum is not None
            and self._curriculum is not None
        ):
            try:
                phys_every = max(
                    1, int(os.environ.get("RKK_PHYS_CURRICULUM_INJECT_EVERY", "50"))
                )
            except ValueError:
                phys_every = 50
            if self.tick % phys_every == 0:
                self._physical_curriculum.inject_into_scheduler(self._curriculum)

        # Phase L: Verbal Action (async in background thread)
        if _VERBAL_AVAILABLE and self._verbal is not None:
            self._schedule_verbal_tick(fallen)

        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            if self._timescale.should_run(LEVEL_MOTOR, self.tick):
                self._timescale.mark_ran(LEVEL_MOTOR, self.tick)
            if self._timescale.should_run(LEVEL_COGNIT, self.tick):
                self._timescale.mark_ran(LEVEL_COGNIT, self.tick)

        # Level 2-D: Episodic Memory
        self._update_episodic_memory(self.tick, _obs_for_d_e, fallen, _posture_now)

        # Phase I: PEARL-style rolling observation posterior (flag ``RKK_PEARL_CONTEXT``)
        try:
            from engine.context_posterior import RollingObservationPosterior, pearl_context_enabled

            if pearl_context_enabled() and self.current_world == "humanoid":
                try:
                    pearl_every = max(1, int(os.environ.get("RKK_PEARL_PUSH_EVERY", "4")))
                except ValueError:
                    pearl_every = 4
                if self.tick % pearl_every == 0:
                    nids = list(self.agent.graph._node_ids)
                    d_g = len(nids)
                    if self._context_posterior is None:
                        self._context_posterior = RollingObservationPosterior(nids)
                    else:
                        self._context_posterior.remap_node_ids(nids)
                    self._context_posterior_d = d_g
                    _phy_ctx: dict[str, float] = {}
                    try:
                        _gdp = getattr(self.agent.env, "get_dynamics_params", None)
                        if callable(_gdp):
                            _phy_ctx = dict(_gdp())
                    except Exception:
                        _phy_ctx = {}
                    self._context_posterior.push(
                        dict(self.agent.graph.snapshot_vec_dict()),
                        _phy_ctx,
                    )
        except Exception:
            pass

        # Living Memory: непрерывная временная шкала (humanoid), до curriculum-тика
        if self.current_world == "humanoid" and self._episodic_memory is not None:
            _cn = None
            if self._curriculum is not None:
                try:
                    _cn = self._curriculum.current_stage.name
                except Exception:
                    _cn = None
            self._episodic_memory.append_timeline_tick(
                self.tick, _obs_for_d_e, fallen, _posture_now, _cn
            )

        # Level 2-E: Curriculum
        self._tick_curriculum(self.tick, _obs_for_d_e, fallen)

        # Level 2-F: RSSM upgrade + training
        self._maybe_upgrade_rssm(self.tick)
        if not result.get("blocked") and not result.get("skipped"):
            _var = str(result.get("variable", ""))
            _val = float(result.get("value", 0.5))
            obs_post = dict(self.agent.graph.snapshot_vec_dict())
            self._rssm_train_step(obs_pre_rssm, _var, _val, obs_post)

        # Фаза 2 ч.3: L4 concept mining (sync fallback или async worker + single-writer apply)
        if self._visual_env is not None and self.tick % self._concept_inject_every == 0:
            vis = self._visual_env.get_slot_visualization()
            slot_vecs = self._visual_env._last_slot_vecs
            if slot_vecs is not None:
                full_obs = dict(self._visual_env.observe())
                phys_obs = {
                    k: float(v)
                    for k, v in full_obs.items()
                    if not str(k).startswith("slot_")
                }
                if _l4_worker_enabled():
                    self._enqueue_l4_task(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                    )
                elif self._concept_store is not None:
                    new_concepts = self._concept_store.update(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                        tick=self.tick,
                        graph_node_ids=list(self.agent.graph._node_ids),
                    )
                    if new_concepts:
                        added = self._concept_store.inject_into_graph(self.agent.graph)
                        c0 = new_concepts[0]
                        self._add_event(
                            f"Concept formed: {c0.cid[:4]}, slot_{c0.slot_idx}, +{added} nodes",
                            "#EF9F27",
                            "phase",
                        )
        if _l4_worker_enabled():
            self._drain_l4_results()

        self._log_step(result, fallen)
        self._rolling_block_bits.append(1 if result.get("blocked") else 0)

        _t_snap = time.perf_counter()
        snap = self.agent.snapshot()
        self._inner_phase_ms["snapshot"] = round((time.perf_counter() - _t_snap) * 1000.0, 2)
        snap["fallen"]     = fallen
        snap["fall_count"] = self._fall_count
        cp = getattr(self, "_context_posterior", None)
        if cp is not None:
            try:
                zm = cp.mean_z()
                snap["pearl_context_z_dim"] = int(zm.size)
                snap["pearl_context_z_head"] = [float(x) for x in zm[:16]]
                te = cp.task_embedding()
                snap["pearl_context_task_dim"] = int(te.size)
                snap["pearl_context_task_head"] = [float(x) for x in te[:16]]
                snap["physics_context_keys"] = list(cp.last_physics_context().keys())[:24]
            except Exception:
                pass
        self._last_snapshot = snap

        if self._rsi_full_enabled():
            from engine.rsi_full import RSIController

            if self._rsi_full is None:
                sup = (
                    self._ensure_skill_library
                    if self._skill_library_enabled()
                    else None
                )
                self._rsi_full = RSIController(
                    self.agent,
                    self._locomotion_controller,
                    skill_library_supplier=sup,
                    motor_cortex_supplier=self._ensure_motor_cortex,
                )
            rsi_ev = self._rsi_full.tick(
                snap,
                self._locomotion_reward_ema(),
                tick=self.tick,
                locomotion_ctrl=self._locomotion_controller,
            )
            if rsi_ev is not None:
                t = rsi_ev.get("type", "?")
                self._add_event(f"🔧 RSI [{t}]", "#66ccaa", "phase")

        # Phase D: Motor Cortex RSI check (every 50 ticks)
        if self.tick % 50 == 0:
            mc = self._ensure_motor_cortex()
            if mc is not None:
                posture_mean = (
                    float(np.mean(self._mc_posture_window))
                    if self._mc_posture_window else 0.0
                )
                fallen_rate = (
                    float(np.mean(self._mc_fallen_count_window))
                    if self._mc_fallen_count_window else 0.0
                )
                loco_r = self._locomotion_reward_ema()
                new_progs = mc.rsi_check_and_spawn(
                    self.tick, posture_mean, loco_r, fallen_rate
                )
                for prog_name in new_progs:
                    self._add_event(
                        f"🧠 MC-RSI: spawned '{prog_name}' "
                        f"(posture={posture_mean:.2f}, cpg_w={mc.cpg_weight:.2f})",
                        "#ff88ff", "phase"
                    )

        dr = float(snap.get("discovery_rate", 0.0))
        self._tick_discovery_plateau(dr)
        if dr > self._best_discovery_rate + 1e-5:
            self._best_discovery_rate = dr
            self._last_dr_gain_tick = self.tick

        # Level 1-B: Visual Body Grounding
        self._maybe_run_visual_grounding()

        # Phase M: slot labels + attention → visual concepts / verbal context
        if _PHASE_M_AVAILABLE:
            try:
                _pm_every = max(1, int(os.environ.get("RKK_PHASE_M_EVERY", "5")))
            except ValueError:
                _pm_every = 5
            if self.tick % _pm_every == 0:
                self._phase_m_sync_from_vision()

        if _WORLD_BRIDGE_AVAILABLE and self._world_bridge is not None:
            try:
                self._world_bridge.on_tick(self, tick_obs=_obs_for_d_e)
            except Exception as e:
                print(f"[Simulation] world_bridge.on_tick: {e}")

        # Level 1-C: Standalone reconstruction training (warm up decoder early)
        if (
            self._visual_mode
            and self._visual_env is not None
            and self.tick % 5 == 0
            and hasattr(self._visual_env, "cortex")
        ):
            cortex = self._visual_env.cortex
            if (
                hasattr(cortex, "train_reconstruction_only")
                and hasattr(self._visual_env, "_last_frame")
                and self._visual_env._last_frame is not None
                and cortex.n_train == 0  # only during warmup phase
            ):
                try:
                    cortex.train_reconstruction_only(self._visual_env._last_frame)
                except Exception:
                    pass

        # Demon
        if self.demon._last_action is not None:
            pe = 0.0
            if not result.get("blocked") and not result.get("skipped"):
                pe = float(result.get("prediction_error", 0))
            self.demon.learn(pe, self.demon._last_action_complexity, [snap])
        self._step_demon(snap)

        smoothed = self._update_phase(snap)

        graph_deltas = {}
        cnt = self.agent.graph.edge_count
        if cnt != self._prev_edge_count:
            # Не материализовать десятки тысяч Edge — тот же capped payload, что и в agent.snapshot (WS/UI).
            _, el_list = self.agent._snapshot_edges_payload()
            graph_deltas[0] = el_list
            self._prev_edge_count = cnt

        # Neurogenesis
        # Structural ASI: Neurogenesis
        if self.current_world == "humanoid" and not self._fixed_root_active:
            neuro_event = self.neuro_engine.scan_and_grow(self.agent, self.tick)
            if neuro_event is not None:
                self._add_event(
                    f"🧬 Neurogenesis: {neuro_event['new_node']} allocated", 
                    "#ff44cc", 
                    "phase"
                )
                self._sync_temporal_blankets_to_graph()

        # Scene
        scene_fn = getattr(self.agent.env, "get_full_scene", None)
        scene    = scene_fn() if callable(scene_fn) else {}

        # Vision state (кэш для /vision/slots endpoint)
        if self._visual_mode and self._visual_env is not None:
            try:
                self._last_vision_state = self._visual_env.get_slot_visualization()
            except Exception:
                pass

        self._maybe_schedule_llm_loop(result, snap)

        self._maybe_refresh_concepts_cache()
        self._maybe_autosave_memory()

        try:
            from engine.memory_diag import log_sim_memory, memory_diag_enabled

            _mem_iv = int(os.environ.get("RKK_MEMORY_DIAG_INTERVAL", "0") or "0")
            if (
                memory_diag_enabled()
                and _mem_iv > 0
                and self.current_world == "humanoid"
                and self.tick % _mem_iv == 0
            ):
                log_sim_memory(self, f"tick={self.tick}")
        except Exception:
            pass

        # #region agent log
        _inner_ms = (time.perf_counter() - _t_inner0) * 1000.0
        _dbg_tick(
            "H4",
            "mixin_tick._run_single_agent_timestep_inner",
            "timestep_inner_done",
            {"tick": self.tick, "ms": _inner_ms},
        )
        # #endregion
        try:
            from engine.tick_run_logger import record_sim_tick

            record_sim_tick(
                self,
                result=result,
                snap=snap,
                inner_ms=_inner_ms,
                obs=_obs_for_d_e if _obs_for_d_e else None,
                fallen=fallen,
                posture=_posture_now,
            )
        except Exception as e:
            print(f"[TickRunLog] hook: {e}")
        return self._build_snapshot(snap, graph_deltas, smoothed, scene)

    def _apply_topological_self_priors(self) -> None:
        """Фаза 3: Топологическое Я. Если найден [EGO] слот, добавляем замороженные/запрещенные связи."""
        from engine.environment_humanoid import HUMANOID_KINEMATIC_EDGE_PRIORS
        if not getattr(self, "_visual_mode", False) or self._visual_env is None:
            return
            
        ego_slot = None
        for slot_id, meta in self._visual_env._slot_lexicon.items():
            if "[EGO]" in meta.get("label", ""):
                ego_slot = slot_id
                break
                
        frozen = list(HUMANOID_KINEMATIC_EDGE_PRIORS)
        forbidden = []
        nids = self.agent.graph._node_ids
        
        for nid in nids:
            if str(nid).startswith("intent_") or str(nid).startswith("phys_intent_"):
                # intent_* -> EGO_slot (frozen prior)
                if ego_slot and ego_slot in nids:
                    frozen.append((nid, ego_slot))
                # Запрещаем slot_k -> intent_* (мир не управляет намерениями)
                for k in range(self._visual_env.n_slots):
                    slot_name = f"slot_{k}"
                    if slot_name in nids:
                        forbidden.append((slot_name, nid))
                        
        self.agent.graph.freeze_kinematic_priors(frozen)
        self.agent.graph.freeze_forbidden_priors(forbidden)
