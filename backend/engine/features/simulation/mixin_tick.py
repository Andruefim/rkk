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
            if self.tick == 1 and not self._fixed_root_active:
                self.enable_fixed_root()
                self._add_event(
                    "📌 Curriculum: fixed_root ON (phase 1, arms→cubes)",
                    "#66ccaa",
                    "phase",
                )
            if (
                self._fixed_root_active
                and self.tick >= auto_fr_ticks
                and not self._curriculum_auto_fr_released
            ):
                self._curriculum_auto_fr_released = True
                self.disable_fixed_root()
                try:
                    stab = int(os.environ.get("RKK_POST_FR_STABILIZE_TICKS", "80"))
                except ValueError:
                    stab = 80
                self._curriculum_stabilize_until = self.tick + max(0, stab)
                self._add_event(
                    f"📌 Auto fixed_root OFF at tick {self.tick}, stabilize until {self._curriculum_stabilize_until}",
                    "#66ccaa",
                    "phase",
                )

        # Fallen check + автосброс физики (иначе VL и block_rate залипают)
        fallen = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            fallen = is_fn()
            if self._fall_recovery_active and not fallen:
                self._clear_fall_recovery()
            if fallen:
                self._fall_count += 1
                obs_fall = dict(self.agent.env.observe())
                if self._maybe_recover_or_reset_after_fall(obs_fall):
                    obs = self.agent.env.observe()
                    self._sync_motor_state(obs, source="reset", tick=self.tick)
                    for nid in self.agent.graph._node_ids:
                        if nid in obs:
                            self.agent.graph.nodes[nid] = obs[nid]
                    self.agent.graph.record_observation(obs)
                    self.agent.temporal.step(obs)
                    fallen = is_fn()
                if self._fall_count % 20 == 1:
                    self._add_event(
                        f"💀 [FALLEN] Nova упал! (×{self._fall_count})",
                        "#ff2244", "value"
                    )

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

        # CPG runs BEFORE agent step so legs are stabilized before high-level exploration
        self._ensure_cpg_background_loop()
        self._drain_l1_motor_commands()
        fallen_pre = False
        is_fn_pre = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn_pre) and not self._fixed_root_active:
            fallen_pre = is_fn_pre()
        self._maybe_apply_cpg_locomotion(fallen_pre)
        self._publish_cpg_node_snapshot()
        self.agent.other_agents_phi = []
        self._maybe_step_hierarchical_l1()
        self._sync_temporal_blankets_to_graph()
        obs_pre_rssm = dict(self.agent.graph.snapshot_vec_dict())
        result = self._run_agent_or_skill_step(engine_tick=self.tick)

        # Track action for episodic memory
        self._record_last_action(result)

        _obs_for_d_e: dict = {}
        try:
            _obs_for_d_e = dict(self.agent.env.observe())
        except Exception:
            pass
        _posture_now = float(
            _obs_for_d_e.get(
                "posture_stability",
                _obs_for_d_e.get("phys_posture_stability", 0.5),
            )
        )

        # Level 3-I: Multi-scale time tick (first consumer of post-step obs)
        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            self._timescale.tick(self.tick, _obs_for_d_e)
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

        # Phase K: Physical curriculum when scheduler runs low on stages ahead
        if (
            _PHASE_K_AVAILABLE
            and self._physical_curriculum is not None
            and self._curriculum is not None
            and self.tick % 1000 == 0
        ):
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

        snap = self.agent.snapshot()
        snap["fallen"]     = fallen
        snap["fall_count"] = self._fall_count
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
            graph_deltas[0] = [e.as_dict() for e in self.agent.graph.edges]
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
        _dbg_tick(
            "H4",
            "mixin_tick._run_single_agent_timestep_inner",
            "timestep_inner_done",
            {"tick": self.tick, "ms": (time.perf_counter() - _t_inner0) * 1000},
        )
        # #endregion
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
