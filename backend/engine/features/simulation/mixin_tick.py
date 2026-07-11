"""Simulation mixin: tick_step, один шаг агента."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from engine.features.simulation.mixin_imports import *

# #region agent log
_DBG_LOG_F7 = Path(__file__).resolve().parents[4] / "debug-f7a777.log"


def _dbg_tick(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    if os.environ.get("RKK_DBG_AGENT", "0").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return
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

    def _prof_mark(self, name: str, t0: list[float]) -> None:
        """Record elapsed ms since previous mark (RKK_TICK_PROFILE)."""
        from engine.tick_profiler import get_tick_profiler

        p = get_tick_profiler()
        now = time.perf_counter()
        if p.enabled():
            p.record(name, (now - t0[0]) * 1000.0)
        t0[0] = now

    def _human_task_motor_active(self) -> bool:
        try:
            from engine.task_binding import task_binding_enabled
            from engine.task_tree import task_tree_enabled

            if not task_binding_enabled():
                return False
            if task_tree_enabled():
                tt = getattr(self, "_task_tree_ctrl", None)
                if tt is not None and tt.is_active:
                    return True
            tb = getattr(self, "_task_binding", None)
            ht = tb.active_task if tb is not None else None
            return ht is not None and str(getattr(ht, "status", "active")) == "active"
        except Exception:
            return False

    def _motor_substrate_suppressed(self) -> bool:
        arb = getattr(self, "_motor_arbiter", None)
        return arb is not None and arb.should_suppress_substrate()

    def _arm_post_reset_motor_hold(self) -> None:
        try:
            from engine.motor_arbiter import task_motor_hold_ticks

            n = task_motor_hold_ticks()
        except Exception:
            n = 60
        self._post_reset_motor_hold_until = int(self.tick) + max(0, n)

    def _sync_graph_intents_to_defaults(self) -> None:
        try:
            from engine.features.humanoid.constants import (
                MOTOR_INTENT_DEFAULTS,
                MOTOR_INTENT_VARS,
            )
        except Exception:
            return
        try:
            nodes = self.agent.graph.nodes
            for k in MOTOR_INTENT_VARS:
                if k in nodes:
                    nodes[k] = float(MOTOR_INTENT_DEFAULTS.get(k, 0.5))
            for k in ("intent_reach_right", "intent_reach_left", "intent_grasp"):
                if k in nodes:
                    nodes[k] = float(MOTOR_INTENT_DEFAULTS.get(k, 0.5))
        except Exception:
            pass

    @staticmethod
    def _canonical_motor_intent_key(k: str) -> str:
        sk = str(k)
        if sk.startswith("phys_intent_"):
            return "intent_" + sk[len("phys_intent_") :]
        return sk

    def _register_task_executive_motor_intents(self, *, fallen: bool = False) -> None:
        """Register human_task + S2 WM planner intents for arbiter finalize."""
        arb = getattr(self, "_motor_arbiter", None)
        if arb is None or not arb.human_task_active():
            return

        hold_until = int(getattr(self, "_post_reset_motor_hold_until", 0) or 0)
        if hold_until > 0 and int(self.tick) < hold_until:
            return
        if fallen:
            return
        try:
            obs = self._env_observe_cached()
            posture = float(
                obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
            )
            if posture < 0.55:
                return
        except Exception:
            pass

        targets: dict[str, float] = {}
        if hasattr(self, "task_tree_motor_targets"):
            tree_mt = self.task_tree_motor_targets()
            if tree_mt:
                targets.update(tree_mt)

        ic = getattr(self, "_intention_state", None)
        if ic is not None:
            primary = getattr(ic, "primary", None)
            if primary is not None:
                for k, v in (getattr(primary, "intent_targets", None) or {}).items():
                    ck = self._canonical_motor_intent_key(k)
                    if ck.startswith("intent_"):
                        targets[ck] = float(v)
                pvar = self._canonical_motor_intent_key(
                    str(getattr(primary, "var_id", ""))
                )
                if pvar.startswith("intent_"):
                    targets[pvar] = float(getattr(primary, "target_val", 0.5))
            ctx = getattr(ic, "_last_context", None)
            if ctx is not None:
                for k, v in (getattr(ctx, "intent_residuals", None) or {}).items():
                    ck = self._canonical_motor_intent_key(k)
                    if ck.startswith("intent_"):
                        targets[ck] = float(np.clip(0.5 + float(v), 0.06, 0.94))

        tb = getattr(self, "_task_binding", None)
        ht = tb.active_task if tb is not None else None
        if ht is not None and getattr(ht, "expected_state", None):
            for k, v in ht.expected_state.items():
                ck = self._canonical_motor_intent_key(k)
                if ck.startswith("intent_"):
                    targets[ck] = float(v)

        if targets:
            try:
                from engine.motor_arbiter import filter_human_task_targets

                targets = filter_human_task_targets(targets)
            except Exception:
                pass
            if targets:
                arb.register_from_dict("human_task", targets)

        s2 = getattr(self, "_system2", None)
        plan = getattr(s2, "_last_wm_plan", None) if s2 is not None else None
        if isinstance(plan, dict):
            var = self._canonical_motor_intent_key(str(plan.get("var", "")))
            if var.startswith("intent_"):
                arb.register_from_dict(
                    "s2_wm",
                    {var: float(plan.get("val", 0.5))},
                )

    def _sync_temporal_blankets_to_graph(self) -> None:
        """Rebuild TemporalBlankets when |graph nodes| changes (inner_voice, concepts, neurogenesis)."""
        from engine.graph_perf import is_large_graph, temporal_rebuild_min_interval
        from engine.temporal import TemporalBlankets

        g_d = len(self.agent.graph._node_ids)
        if g_d <= 0:
            return
        tb = self.agent.temporal
        if tb.d_input == g_d:
            return
        if is_large_graph(self.agent.graph):
            every = temporal_rebuild_min_interval()
            if int(self.tick) % every != 0:
                return
        self.agent.temporal = TemporalBlankets(d_input=g_d, device=self.device)

    def _maybe_post_release_stabilize_intents(self) -> None:
        """После снятия fixed_root — в окне stabilize_until усилить recover/support (плавный decay)."""
        if not is_humanoid_topology(self.current_world) or self._fixed_root_active:
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
        alt = (int(self.tick) // 30) % 2
        if alt == 0:
            sup_l = d_sup * scale
            sup_r = d_sup * scale * 0.4
        else:
            sup_l = d_sup * scale * 0.4
            sup_r = d_sup * scale
        fn(
            {
                "intent_stop_recover": d_rec * scale,
                "intent_support_left": sup_l,
                "intent_support_right": sup_r,
            }
        )

    def _apply_hardcoded_reflexes(self, is_fallen: bool) -> None:
        """Apply genome-based spinal reflexes: fast reactive balance corrections."""
        base = self._unwrap_base_env(self.agent.env)
        fn = getattr(base, "apply_motor_intent_residuals", None)
        if not callable(fn) or getattr(base, "_intero_control_lost", False):
            return

        obs = dict(self.agent.env.observe())
        ms = dict(getattr(base, "_motor_state", {}))

        try:
            from engine.genome.priors import apply_reflexes
            updated = apply_reflexes(obs, ms)
        except Exception:
            return

        residuals = {}
        for k, v in updated.items():
            if k in ms and abs(v - ms[k]) > 0.01:
                residuals[k] = v - ms[k]

        if residuals:
            fn(residuals)

    def _genome_walk_obs_state(self) -> dict:
        obs = dict(self.agent.env.observe())
        st = {**obs}
        for k, v in list(obs.items()):
            if isinstance(k, str) and k.startswith("phys_"):
                st.setdefault(k[5:], v)
        return st

    def _genome_walk_active(self, is_fallen: bool) -> bool:
        if is_fallen or not is_humanoid_topology(self.current_world):
            return False
        if self._fixed_root_active:
            try:
                from engine.genome.priors import genome_walk_during_fixed_root_enabled

                if not genome_walk_during_fixed_root_enabled():
                    return False
            except Exception:
                return False
        if getattr(self, "_fall_recovery_active", False):
            return False
        s2 = getattr(self, "_system2", None)
        if s2 is not None and getattr(s2, "_s2_override_active", False):
            return False
        try:
            from engine.genome.priors import genome_walk_eligible
        except Exception:
            return False
        st = self._genome_walk_obs_state()
        posture = float(st.get("posture_stability", st.get("phys_posture_stability", 0.5)))
        if posture >= 0.62:
            self._genome_walk_stand_streak = int(getattr(self, "_genome_walk_stand_streak", 0)) + 1
        else:
            self._genome_walk_stand_streak = 0
        try:
            warm = int(os.environ.get("RKK_GENOME_WALK_WARMUP_TICKS", "24"))
        except ValueError:
            warm = 24
        warm = max(0, min(warm, 200))
        if warm > 0 and self._genome_walk_stand_streak < warm:
            return False
        goal = self._skill_goal_hint(st)
        return genome_walk_eligible(
            st,
            goal_walk=(goal == "walk"),
            is_fallen=is_fallen,
            fixed_root=bool(self._fixed_root_active),
        )

    def _apply_genome_walk_intents(self, is_fallen: bool) -> None:
        if not self._genome_walk_active(is_fallen):
            self._genome_walk_active_tick = False
            return
        self._genome_walk_active_tick = True
        try:
            from engine.genome.priors import compute_walk_residuals, walk_intents_at_tick
        except Exception:
            return
        base = self._unwrap_base_env(self.agent.env)
        if getattr(base, "_intero_control_lost", False):
            return
        ms = getattr(base, "_motor_state", None)
        if not isinstance(ms, dict):
            return
        targets = walk_intents_at_tick(self.tick)
        residuals = compute_walk_residuals(ms, self.tick)
        fn = getattr(base, "apply_motor_intent_residuals", None)
        arb = getattr(self, "_motor_arbiter", None)
        suppress = arb is not None and arb.should_suppress_substrate()
        if callable(fn) and residuals and not suppress:
            try:
                fn(residuals)
            except Exception:
                pass
        if arb is not None:
            arb.register_from_dict("genome", dict(targets), precision=0.42)
        if suppress:
            return
        for k, v in targets.items():
            if k in self.agent.graph.nodes:
                self.agent.graph.nodes[k] = float(
                    getattr(base, "_motor_state", {}).get(k, v)
                )

    def _fr_curriculum_finalize_release(self, *, reason: str) -> None:
        """Снять fixed_root в симуляции + VL, выставить окно stabilize (после мягкой физики)."""
        try:
            from engine.curriculum_eval_gate import maybe_gate_fr_release

            if not maybe_gate_fr_release(self):
                self._add_event(
                    "⛔ FR release blocked by curriculum eval gate",
                    "#ff8844",
                    "phase",
                )
                return
        except Exception:
            pass
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
        if not is_humanoid_topology(self.current_world) or not self._fixed_root_active:
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
        from engine.json_util import sanitize_for_json

        for _ in range(n):
            with self._sim_step_lock:
                raw = self._run_single_agent_timestep_inner()
            payload = sanitize_for_json(raw)
            if isinstance(payload, dict):
                payload["_json_sanitized"] = True
            with self._sim_step_lock:
                self._agent_step_response = payload
                self._public_state_cache = payload
                self._public_state_cache_at = time.monotonic()

    def _run_single_agent_timestep_inner(self) -> dict:
        from engine.tick_profiler import get_tick_profiler, tick_profile

        # #region agent log
        _t_inner0 = time.perf_counter()
        # #endregion
        self.tick += 1
        _prof = get_tick_profiler()
        _prof.begin_tick(self.tick)
        try:
            return self._run_single_agent_timestep_inner_profiled(_t_inner0)
        finally:
            _prof.end_tick()

    def _run_single_agent_timestep_inner_profiled(self, _t_inner0: float) -> dict:
        _pt = [time.perf_counter()]
        self._reset_tick_obs_caches()
        if not is_humanoid_topology(self.current_world):
            self._hai_prev_com_x = None
            self._hai_pe_fwd_ema = 0.0
            self._hai_pe_vert_ema = 0.0
            self._hai_pe_lat_ema = 0.0
            self._hai_pe_ema = 0.0
        self._ensure_phase2()

        # Humanoid curriculum: фаза 1 — fixed_root с тика 1; снятие после RKK_AUTO_FIXED_ROOT_TICKS.
        try:
            auto_fr_ticks = int(os.environ.get("RKK_AUTO_FIXED_ROOT_TICKS", "0"))
        except ValueError:
            auto_fr_ticks = 0
        if auto_fr_ticks > 0 and is_humanoid_topology(self.current_world):
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
        fallen_raw = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            fallen_raw = bool(is_fn())
            fallen = fallen_raw
            prev_f = bool(getattr(self, "_prev_fallen", False))
            fallen_edge = bool(fallen and not prev_f)
            if self._fall_recovery_active and not fallen:
                self._clear_fall_recovery()
            if fallen:
                self._fr_fallen_ticks_accum += 1
                obs_fall = self._env_observe_cached()
                if fallen_edge:
                    self._fall_count += 1
                    try:
                        pend = dict(obs_fall)
                        st = self._tick_phys_state()
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
                s2 = getattr(self, "_system2", None)
                defer_fall_reset = (
                    s2 is not None
                    and callable(getattr(s2, "defer_sim_fall_hard_reset", None))
                    and s2.defer_sim_fall_hard_reset()
                )
                use_genome_recovery = self._genome_fall_recovery_enabled()
                if defer_fall_reset:
                    if self._fall_recovery_active:
                        self._clear_fall_recovery()
                elif self._maybe_recover_or_reset_after_fall(
                    obs_fall,
                    apply_genome_program=use_genome_recovery,
                ):
                    self._invalidate_env_observe_cache()
                    obs = self.agent.env.observe()
                    self._sync_motor_state(obs, source="reset", tick=self.tick)
                    self._sync_graph_intents_to_defaults()
                    self._arm_post_reset_motor_hold()
                    for nid in self.agent.graph._node_ids:
                        if nid in obs:
                            self.agent.graph.nodes[nid] = obs[nid]
                    self.agent.graph.record_observation(obs)
                    self.agent.temporal.step(obs)
                    fallen = is_fn()
                if not fallen:
                    self._pending_fall_obs_for_memory = None
                    if not (s2 is not None and getattr(s2, "fallen_override_active", False)):
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

        obs_s2_fall = self._env_observe_cached()
        if self._fixed_root_active:
            fallen_for_s2 = False
        else:
            fallen_for_s2 = self._fallen_signal_for_s2(
                obs_s2_fall, fallen_debounced=bool(fallen_raw or fallen)
            )
            if getattr(self, "_fall_recovery_active", False):
                fallen_for_s2 = True

        self._prof_mark("sim.fall_curriculum", _pt)
        # Фаза 12: передаём GNN prediction в visual env (не каждый тик — RKK_VISION_GNN_FEED_EVERY)
        if self._visual_mode and self._visual_env is not None:
            from engine.core.constants import (
                topological_self_every_from_env,
                vision_gnn_feed_every_from_env,
            )

            self._vision_ticks += 1
            _gnn_every = vision_gnn_feed_every_from_env()
            if self._vision_ticks % _gnn_every == 0:
                self._feed_gnn_prediction_to_visual()
            # Фаза 3: Топологическое Я
            _topo_every = topological_self_every_from_env()
            if self._vision_ticks % _topo_every == 0:
                self._apply_topological_self_priors()

        self.agent.set_teacher_state([], 0.0)
        self.agent.value_layer.set_teacher_vl_overlay(None)

        self._prof_mark("sim.teacher_vision", _pt)
        # Phase C₁: intention → CPG legs → agent (high-level planning after motor context is set).
        self._ensure_cpg_background_loop()
        self._drain_l1_motor_commands()
        # Re-use ``fallen`` from the early check (after optional recovery): no extra
        # ``is_fallen()`` here — duplicate calls would double-advance debounce streak.
        fallen_pre = bool(fallen)
        if is_humanoid_topology(self.current_world):
            arb_ht = getattr(self, "_motor_arbiter", None)
            if arb_ht is not None:
                arb_ht.begin_tick()
                arb_ht.set_human_task_active(self._human_task_motor_active())
            try:
                from engine.neuro_symbolic.motor_sync import feed_distance_to_human

                feed_distance_to_human(self)
            except Exception:
                pass
            try:
                self._tick_intention_pre_system2(fallen=bool(fallen_for_s2))
            except Exception as _ic_ex:
                logging.getLogger(__name__).warning(
                    "intention_cortex pre-system2 failed at tick %s: %s",
                    self.tick,
                    _ic_ex,
                    exc_info=True,
                )
            try:
                self._tick_neuro_symbolic_slow(fallen=bool(fallen_for_s2))
            except Exception as _ns_ex:
                logging.getLogger(__name__).warning(
                    "neuro_symbolic slow loop failed at tick %s: %s",
                    self.tick,
                    _ns_ex,
                    exc_info=True,
                )
            try:
                self._tick_grounded_language(fallen=bool(fallen_for_s2))
            except Exception as _gl_ex:
                logging.getLogger(__name__).warning(
                    "grounded_language tick failed at tick %s: %s",
                    self.tick,
                    _gl_ex,
                    exc_info=True,
                )
            try:
                self._tick_human_task(fallen=bool(fallen_for_s2))
            except Exception as _ht_ex:
                logging.getLogger(__name__).warning(
                    "human_task tick failed at tick %s: %s",
                    self.tick,
                    _ht_ex,
                    exc_info=True,
                )
            try:
                from engine.neuro_symbolic.motor_sync import sync_ns_motor_every_tick

                self._ns_fast_applied = sync_ns_motor_every_tick(self)
            except Exception:
                self._ns_fast_applied = {}
            self._apply_genome_walk_intents(fallen_pre)
        self._maybe_apply_cpg_locomotion(fallen_pre)
        self._publish_cpg_node_snapshot()
        if is_humanoid_topology(self.current_world) and not self._fixed_root_active:
            self._maybe_post_release_stabilize_intents()
            self._apply_hardcoded_reflexes(fallen_pre)
        if is_humanoid_topology(self.current_world):
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

        self._prof_mark("sim.cpg_prep", _pt)
        if is_humanoid_topology(self.current_world):
            try:
                from engine.eval_mode import eval_mode_enabled, eval_skip_system2
                from engine.system2.controller import system2_enabled

                if system2_enabled() and not (
                    eval_mode_enabled() and eval_skip_system2()
                ):
                    if getattr(self, "_system2", None) is None:
                        from engine.system2 import System2Controller

                        self._system2 = System2Controller()
                        self._system2._rkk_sim = self
                    obs_s2 = dict(self._graph_vec_cached())
                    self._system2_last = self._system2.tick(
                        sim_tick=self.tick,
                        agent=self.agent,
                        obs=obs_s2,
                        sim=self,
                        fallen=bool(fallen_for_s2),
                    )
                    macro_vl = str((self._system2_last or {}).get("macro") or "")
                    self.agent.value_layer.set_context({"macro": macro_vl})
                    self._system2.note_autonomy_sample(self.tick)
                    if isinstance(self._system2_last, dict):
                        self._system2_last.update(self._system2.autonomy_fields())
                    fn_ctx = getattr(self._system2, "planning_context_for_wm", None)
                    if callable(fn_ctx):
                        self.agent.set_s2_planning_context(
                            fn_ctx(
                                fallen=bool(fallen_for_s2),
                                sim_tick=int(self.tick),
                                sim=self,
                            )
                        )
                    else:
                        self.agent.set_s2_planning_context(None)
                else:
                    self._system2_last = None
                    self.agent.set_s2_planning_context(None)
            except Exception as _s2_ex:
                logging.getLogger(__name__).warning(
                    "system2_tick failed at tick %s: %s",
                    self.tick,
                    _s2_ex,
                    exc_info=True,
                )
                self._system2_last = {
                    "enabled": False,
                    "error": "system2_tick",
                    "error_detail": f"{type(_s2_ex).__name__}: {_s2_ex}",
                }
                self.agent.set_s2_planning_context(None)
        else:
            self._system2_last = None
            self.agent.set_s2_planning_context(None)
        self._prof_mark("sim.system2", _pt)

        # Controlled perturbations during fixed_root to teach active balance
        _skip_perturb = False
        try:
            from engine.eval_mode import transfer_bench_enabled

            _skip_perturb = transfer_bench_enabled()
        except ImportError:
            pass
        if (
            not _skip_perturb
            and is_humanoid_topology(self.current_world)
            and self._fixed_root_active
        ):
            if self.tick % 40 == 0:
                fn_perturb = getattr(self.agent.env, "apply_random_perturbation", None)
                if callable(fn_perturb):
                    # Start gentle, increase force
                    force = 60.0 + min(100.0, self.tick * 0.2)
                    fn_perturb(max_force=force)
        elif (
            not _skip_perturb
            and is_humanoid_topology(self.current_world)
            and not self._fixed_root_active
        ):
            from engine.features.simulation.snapshot import humanoid_curriculum_step

            _cur_step, _ = humanoid_curriculum_step(self)
            if _cur_step >= 3 and self.tick % 50 == 0 and not fallen_pre:
                base_p = self._unwrap_base_env(self.agent.env)
                fn_perturb = getattr(base_p, "apply_random_perturbation", None)
                if callable(fn_perturb):
                    fn_perturb(max_force=85.0 + min(140.0, max(0, self.tick - 900) * 0.1))

        self.agent.graph.set_snapshot_vec_engine_tick(int(self.tick))
        try:
            from engine.post_fr import post_fr_wm_lr_scale

            self.agent.graph._post_fr_wm_lr_mult = post_fr_wm_lr_scale(self)
        except Exception:
            pass
        _t_phase = time.perf_counter()
        result = self._run_agent_or_skill_step(engine_tick=self.tick)
        self._inner_phase_ms = getattr(self, "_inner_phase_ms", {})
        self._inner_phase_ms["agent"] = round((time.perf_counter() - _t_phase) * 1000.0, 2)
        self._prof_mark("sim.agent_step", _pt)
        self.agent.graph.invalidate_snapshot_vec_cache()
        self._tick_graph_vec = None

        # Track action for episodic memory
        self._record_last_action(result)
        if is_humanoid_topology(self.current_world) and self._fixed_root_active:
            self._maybe_damp_motor_intents_blind_fixed_root()

        _obs_for_d_e: dict = {}
        try:
            self.agent.graph.set_snapshot_vec_engine_tick(int(self.tick))
            _obs_for_d_e = dict(self.agent.graph.snapshot_vec_dict())
            self._tick_graph_vec = _obs_for_d_e
        except Exception:
            pass
        _posture_now = float(
            _obs_for_d_e.get(
                "posture_stability",
                _obs_for_d_e.get("phys_posture_stability", 0.5),
            )
        )
        self._prof_mark("sim.post_vec_snap", _pt)

        rs = getattr(self, "_reflex_stabilizer", None)
        if rs is None:
            try:
                from engine.reflex_stabilizer import reflex_stabilizer_enabled

                if reflex_stabilizer_enabled():
                    rs = self._ensure_reflex_stabilizer()
            except Exception:
                rs = None
        if rs is not None and is_humanoid_topology(self.current_world) and not self._fixed_root_active:
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
        if (
            cb_cereb is not None
            and is_humanoid_topology(self.current_world)
            and not self._fixed_root_active
            and not fallen
        ):
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
        self._prof_mark("sim.post_reflex_cereb", _pt)

        if (
            _MOTOR_CORTEX_AVAILABLE
            and is_humanoid_topology(self.current_world)
            and not self._fixed_root_active
            and not fallen
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
        self._prof_mark("sim.post_motor_cortex", _pt)

        # Level 3-I: Multi-scale time tick (first consumer of post-step obs)
        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            self._timescale.tick(self.tick, _obs_for_d_e)
            if not is_humanoid_topology(self.current_world):
                motor_intents = self._timescale.get_intents(LEVEL_MOTOR)
                for var, val in motor_intents.items():
                    if var.startswith("intent_"):
                        try:
                            self.agent.env.intervene(var, float(val), count_intervention=False)
                        except Exception:
                            pass

        self._tick_inner_voice(self.tick)

        # Level 3-G: Proprioception update (after CPG + agent step; fresh obs)
        _proprio_anomaly = 0.0
        _proprio_emp_reward = 0.0
        if (
            _PROPRIO_AVAILABLE
            and self._proprio is not None
            and is_humanoid_topology(self.current_world)
            and not fallen
        ):
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
        if is_humanoid_topology(self.current_world):
            if (
                not self._fixed_root_active
                and not fallen
                and _obs_for_d_e
                and not self._motor_substrate_suppressed()
            ):
                from engine.hierarchical_active_inference import run_hierarchical_pe_tick

                self._hai_last_diag = run_hierarchical_pe_tick(self, _obs_for_d_e)
            elif self._fixed_root_active or fallen:
                self._hai_prev_com_x = None
                self._hai_pe_fwd_ema = 0.0
                self._hai_pe_vert_ema = 0.0
                self._hai_pe_lat_ema = 0.0
                self._hai_pe_ema = 0.0
        arb = getattr(self, "_motor_arbiter", None)
        if arb is not None and is_humanoid_topology(self.current_world):
            self._register_task_executive_motor_intents(fallen=bool(fallen))
            arb.finalize(self)
        sg_obs = getattr(self, "_scene_graph", None)
        if sg_obs is not None and is_humanoid_topology(self.current_world):
            try:
                sg_obs.update_gnn(self)
            except Exception:
                pass
        le = getattr(self, "_locomotion_eval", None)
        bt = getattr(self, "behavioral_tracker", None)
        if le is not None and bt is not None and is_humanoid_topology(self.current_world):
            bs = bt.snapshot()
            ms = self._motor_state_snapshot()
            intents = ms.get("intents") or {}
            sl = float(intents.get("intent_support_left", 0.5))
            sr = float(intents.get("intent_support_right", 0.5))
            bs["support_asymmetry"] = abs(sl - sr)
            bs["pe_fwd_ema"] = float(getattr(self, "_hai_pe_fwd_ema", 0.0))
            bs["coupling_motor"] = float(intents.get("intent_gait_coupling", 0.5))
            le.record_tick(bs)
            if self.tick % 30 == 0:
                le.evaluate()
        self._prof_mark("sim.post_cognition", _pt)

        # Phase K: Sleep Controller
        _t_sleep = time.perf_counter()
        if (
            _PHASE_K_AVAILABLE
            and self._sleep_ctrl is not None
            and is_humanoid_topology(self.current_world)
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
                sim=self,
            )

            if _sleep_reason and not self._sleep_ctrl.is_sleeping:
                self._sleep_attach_fixed_root()
                self._sleep_ctrl.begin_sleep(self.tick, _sleep_reason, sim=self)
                self._add_event(
                    f"😴 Sleep: {_sleep_reason} (falls={self._sleep_ctrl._falls_since_sleep})",
                    "#9988ff",
                    "sleep",
                )

            if self._sleep_ctrl.is_sleeping:
                self._sleep_ensure_fixed_root_while_sleeping()
                self._sleep_ctrl.tick(self.tick, self)
                if not self._sleep_ctrl.is_sleeping:
                    self._sleep_detach_fixed_root()
                    self._add_event(
                        f"🌅 Woke up (sleep #{self._sleep_ctrl.sleep_count})",
                        "#ffff88",
                        "sleep",
                    )
        self._inner_phase_ms["sleep"] = round((time.perf_counter() - _t_sleep) * 1000.0, 2)

        self._prof_mark("sim.post_sleep", _pt)

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

            if pearl_context_enabled() and is_humanoid_topology(self.current_world):
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

        # Living Memory: непрерывная временная шкала (humanoid)
        if is_humanoid_topology(self.current_world) and self._episodic_memory is not None:
            self._episodic_memory.append_timeline_tick(
                self.tick, _obs_for_d_e, fallen, _posture_now, None
            )

        self._prof_mark("sim.post_episodic", _pt)

        # Фаза 2 ч.3: L4 concept mining (sync fallback или async worker + single-writer apply)
        if (
            not getattr(self, "_fixed_root_active", False)
            and self._visual_env is not None
            and self.tick % self._concept_inject_every == 0
        ):
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
        self._prof_mark("sim.post_l4", _pt)

        self._log_step(result, fallen)
        self._rolling_block_bits.append(1 if result.get("blocked") else 0)
        self._prof_mark("sim.post_tick_log", _pt)

        _t_snap = time.perf_counter()
        snap = self.agent.snapshot()
        self._inner_phase_ms["snapshot"] = round((time.perf_counter() - _t_snap) * 1000.0, 2)
        self._prof_mark("sim.agent_snapshot", _pt)
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
        self._prof_mark("sim.post_rsi", _pt)

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

        try:
            self.agent.graph.tick_compositional_structure(
                int(self.tick),
                fixed_root=bool(getattr(self, "_fixed_root_active", False)),
            )
        except Exception:
            pass

        lc = getattr(self, "_latent_confounder", None)
        if lc is not None:
            try:
                from engine.latent_confounder import collect_language_context, compute_cluster_pe
                from engine.genome.learned_roles import c5_enabled, try_promote_all_passed

                obs_lc = dict(_obs_for_d_e if _obs_for_d_e else {})
                cluster_pe = compute_cluster_pe(self.agent.graph, obs_lc)
                pe_lc = float(result.get("prediction_error", 0) or 0)
                self._latent_confounder_last = lc.tick(
                    self.agent.graph,
                    engine_tick=int(self.tick),
                    prediction_error=pe_lc,
                    obs=obs_lc,
                    cluster_pe=cluster_pe,
                    lang_text=collect_language_context(self),
                    world_id=str(self.current_world),
                )
                if c5_enabled():
                    try_promote_all_passed(
                        lc.records_passed_ttl(),
                        self.agent.graph,
                        world_id=str(self.current_world),
                    )
            except Exception as _lc_ex:
                logging.getLogger(__name__).debug(
                    "latent_confounder tick: %s", _lc_ex, exc_info=True
                )

        self._prof_mark("sim.post_visual_ui", _pt)

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
            edge_delta = cnt - self._prev_edge_count
            self._edge_delta_hist.append(int(edge_delta))
            try:
                max_delta = max(1, int(os.environ.get("RKK_MAX_EDGE_DELTA_PER_WINDOW", "200")))
                window = max(1, int(os.environ.get("RKK_EDGE_DELTA_WINDOW", "100")))
            except ValueError:
                max_delta, window = 200, 100
            recent = list(self._edge_delta_hist)[-window:]
            self._edge_growth_blocked = (
                len(recent) >= window and sum(int(x) for x in recent) > max_delta
            )
            try:
                max_edges = int(os.environ.get("RKK_MAX_EDGE_COUNT", "8000"))
            except ValueError:
                max_edges = 8000
            if cnt > max_edges:
                try:
                    self.agent.graph.prune_weak_W()
                except Exception:
                    pass
            _, el_list = self.agent._snapshot_edges_payload()
            graph_deltas[0] = el_list
            self._prev_edge_count = cnt

        if is_humanoid_topology(self.current_world):
            from engine.features.simulation.snapshot import humanoid_curriculum_step

            cur_step, _ = humanoid_curriculum_step(self)
            _skip_neuro = False
            try:
                from engine.eval_mode import eval_mode_enabled, transfer_bench_enabled

                _skip_neuro = eval_mode_enabled() or transfer_bench_enabled()
            except ImportError:
                pass
            if cur_step >= 3 and not _skip_neuro:
                self.neuro_coordinator.note_step3_entry(self.tick)
                bt = getattr(self, "behavioral_tracker", None)
                if bt is not None:
                    bt.note_step3_entry(self.tick)
            neuro_event = (
                None
                if _skip_neuro
                else self.neuro_coordinator.request_or_apply(self, tick=self.tick)
            )
            if neuro_event is not None:
                if neuro_event.get("type") == "structural_asi_growth":
                    self._add_event(
                        f"🧬 Neurogenesis: {neuro_event['new_node']} allocated",
                        "#ff44cc",
                        "phase",
                    )
                    self._sync_temporal_blankets_to_graph()
                    self.agent._wm_warmup_until = int(getattr(self, "_wm_warmup_until", 0))
                elif neuro_event.get("type") == "neurogenesis_pending":
                    self._neuro_pending = True
        self._prof_mark("sim.post_neuro", _pt)

        # Scene: тяжёлый get_full_scene реже; skeleton/динамика — каждый agent-тик (RKK_SKELETON_EVERY).
        scene_every = self._scene_cache_every()
        scene_fn = getattr(self.agent.env, "get_full_scene", None)
        scene_stale = (
            not hasattr(self, "_cached_scene")
            or self._cached_scene_tick < 0
            or (self.tick - int(self._cached_scene_tick)) >= scene_every
        )
        if scene_stale:
            try:
                self._cached_scene = scene_fn() if callable(scene_fn) else {}
            except Exception:
                # Keep last good scene; light patch below retries skeleton and
                # its failure streak re-invalidates the cache for another attempt.
                self._cached_scene = dict(getattr(self, "_cached_scene", {}) or {})
            self._cached_scene_tick = int(self.tick)
            self._cached_skeleton_tick = int(self.tick)
            from engine.tick_profiler import tick_profile

            with tick_profile("sim.scene_patch"):
                self._patch_scene_skeleton_light(self._cached_scene)
        else:
            sk_every = self._skeleton_cache_every()
            sk_stale = (
                not hasattr(self, "_cached_skeleton_tick")
                or int(self._cached_skeleton_tick) < 0
                or (self.tick - int(self._cached_skeleton_tick)) >= sk_every
            )
            if sk_stale and getattr(self, "_cached_scene", None):
                from engine.tick_profiler import tick_profile

                with tick_profile("sim.scene_skeleton_patch"):
                    self._patch_scene_skeleton_light(self._cached_scene)
                self._cached_skeleton_tick = int(self.tick)
        scene = dict(getattr(self, "_cached_scene", {}) or {})

        # Vision state (кэш для /vision/slots; не на каждый agent-тик)
        if self._visual_mode and self._visual_env is not None:
            try:
                vis_every = max(1, int(os.environ.get("RKK_VISION_STATE_EVERY", "6")))
            except ValueError:
                vis_every = 6
            if (
                not getattr(self, "_last_vision_state", None)
                or self.tick % vis_every == 0
            ):
                try:
                    self._last_vision_state = self._visual_env.get_slot_visualization()
                except Exception:
                    pass
        self._prof_mark("sim.post_scene_vision", _pt)

        if not fallen:
            self._maybe_refresh_concepts_cache()
            self._maybe_autosave_memory()
        self._prof_mark("sim.post_persist", _pt)

        try:
            from engine.memory_diag import log_sim_memory, memory_diag_enabled

            _mem_iv = int(os.environ.get("RKK_MEMORY_DIAG_INTERVAL", "0") or "0")
            if (
                memory_diag_enabled()
                and _mem_iv > 0
                and is_humanoid_topology(self.current_world)
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
            self._tick_phase6(snap)
        except Exception:
            pass
        try:
            self._tick_phase5(snap)
        except Exception:
            pass

        bt = getattr(self, "behavioral_tracker", None)
        if bt is not None and is_humanoid_topology(self.current_world):
            from engine.features.simulation.snapshot import humanoid_curriculum_step

            cur_step, _ = humanoid_curriculum_step(self)
            loco_r = 0.0
            intr = getattr(self, "_intrinsic", None)
            if intr is not None:
                loco_r = float(intr.recent_reward(8))
            s2 = getattr(self, "_system2_last", None) or {}
            learned_ok = None
            if s2.get("override_recovered") and s2.get("motor_owner") == "s1_learned":
                learned_ok = True
            obs_bt = dict(_obs_for_d_e if _obs_for_d_e else {})
            rf = self._raw_com_forward_m()
            if rf is not None:
                obs_bt["com_forward_raw_m"] = rf
            rx = self._raw_com_x_m()
            if rx is not None:
                obs_bt["com_x_raw_m"] = rx
            bt.record_tick(
                tick=self.tick,
                obs=obs_bt,
                fallen=fallen,
                locomotion_reward=loco_r,
                recovery_learned_success=learned_ok,
                in_step3=cur_step >= 3,
            )
            self._refresh_behavioral_snapshot_cache()
            bs = self._behavioral_snapshot_cached()
            if bs is not None:
                snap["behavioral_score"] = bs.get("behavioral_score")
        self._prof_mark("sim.build_response", _pt)
        return self._build_snapshot(snap, graph_deltas, smoothed, scene)

    def _scene_cache_every(self) -> int:
        try:
            return max(1, int(os.environ.get("RKK_SCENE_CACHE_EVERY", "4")))
        except ValueError:
            return 4

    def _skeleton_cache_every(self) -> int:
        try:
            return max(1, int(os.environ.get("RKK_SKELETON_EVERY", "1")))
        except ValueError:
            return 1

    def _patch_scene_skeleton_light(self, scene: dict) -> None:
        """Лёгкий per-tick патч: skeleton, ankleQuats, динамические объекты (без rebuild static_geometry)."""
        env = getattr(self.agent, "env", None)
        if env is None:
            return
        _log = logging.getLogger(__name__)
        try:
            sk_fn = getattr(env, "get_joint_positions_world", None)
            if callable(sk_fn):
                scene["skeleton"] = sk_fn()
            streak = int(getattr(self, "_skel_patch_fail_streak", 0))
            if streak:
                self._skel_patch_fail_streak = 0
        except Exception as exc:
            streak = int(getattr(self, "_skel_patch_fail_streak", 0)) + 1
            self._skel_patch_fail_streak = streak
            last_warn = int(getattr(self, "_skel_patch_last_warn_tick", -200))
            if int(self.tick) - last_warn >= 100:
                self._skel_patch_last_warn_tick = int(self.tick)
                _log.warning(
                    "skeleton light-patch failed (streak=%d tick=%d): %s",
                    streak,
                    int(self.tick),
                    exc,
                )
            if streak >= 3:
                self._cached_scene_tick = -1
                self._skel_patch_fail_streak = 0
        sim = getattr(env, "_sim", None)
        try:
            aq_fn = getattr(sim, "get_ankle_quaternions_three_js", None)
            if callable(aq_fn):
                scene["ankleQuats"] = aq_fn()
        except Exception:
            pass
        try:
            cp_fn = getattr(env, "get_cube_positions", None)
            if callable(cp_fn):
                scene["cubes"] = cp_fn()
        except Exception:
            pass
        try:
            fallen_fn = getattr(env, "_fallen_z_below_threshold", None)
            if callable(fallen_fn):
                scene["fallen"] = bool(fallen_fn())
        except Exception:
            pass
        try:
            extras_fn = getattr(sim, "get_sandbox_scene_extras", None)
            if callable(extras_fn):
                extras = extras_fn()
                for key in ("ball", "lever", "delivery_target", "props"):
                    val = extras.get(key)
                    if val is not None:
                        scene[key] = val
        except Exception:
            pass
        try:
            if not getattr(env, "_fixed_root", False):
                obs_fn = getattr(env, "observe", None)
                if callable(obs_fn):
                    scene["com_z"] = float(obs_fn().get("com_z", scene.get("com_z", 0.5)))
        except Exception:
            pass

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
