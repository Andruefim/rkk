"""Simulation mixin: skill library, agent/skill step."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationSkillsMixin:
    def _human_task_suppresses_autonomous_locomotion(self) -> bool:
        try:
            from engine.task_executive import human_task_suppresses_autonomous_locomotion

            return human_task_suppresses_autonomous_locomotion(self)
        except Exception:
            return False

    def _skill_library_enabled(self) -> bool:
        v = os.environ.get("RKK_SKILL_LIBRARY", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _skill_start_prob(self) -> float:
        try:
            p = float(os.environ.get("RKK_SKILL_LIBRARY_PROB", "0.1"))
        except ValueError:
            p = 0.1
        if is_humanoid_topology(self.current_world) and not self._fixed_root_active:
            obs = self.agent.env.observe()
            posture = float(
                obs.get(
                    "posture_stability",
                    obs.get("phys_posture_stability", 0.5),
                )
            )
            # Чем нестабильнее — тем выше доля скиллов (меньше сырого EIG).
            adaptive = 0.80 - posture * 0.30  # posture=0 → 0.80, posture=1 → 0.50
            p = max(p, adaptive)
        return float(np.clip(p, 0.0, 1.0))

    def _ensure_skill_library(self):
        if self._skill_library is None:
            from engine.skill_library import SkillLibrary

            self._skill_library = SkillLibrary()
        return self._skill_library

    @staticmethod
    def _skill_state_dict(obs: dict) -> dict:
        out = dict(obs)
        for k, v in list(obs.items()):
            if isinstance(k, str) and k.startswith("phys_"):
                out.setdefault(k[5:], v)
        return out

    def _skill_chain_max_depth(self) -> int:
        try:
            return max(1, int(os.environ.get("RKK_SKILL_CHAIN_MAX_DEPTH", "4")))
        except ValueError:
            return 4

    def _skill_chain_pe_max(self) -> float:
        try:
            return float(os.environ.get("RKK_SKILL_CHAIN_PE_MAX", "0.25"))
        except ValueError:
            return 0.25

    def _skill_chain_depth(self) -> int:
        pack = self._skill_exec
        if pack is None:
            return len(getattr(self, "_skill_chain", []) or [])
        return int(pack.get("chain_depth", len(getattr(self, "_skill_chain", []) or [])))

    def _maybe_chain_next_skill(
        self,
        skill,
        st_after: dict,
        *,
        prediction_error: float,
        obs_before_init: dict,
    ) -> bool:
        """C1: queue next macro if PE gate passes and chain depth < max."""
        depth = self._skill_chain_depth()
        if depth >= self._skill_chain_max_depth():
            return False
        if float(prediction_error) > self._skill_chain_pe_max():
            return False
        try:
            if not skill.postcondition(st_after):
                return False
        except Exception:
            return False
        goal = self._skill_goal_hint(st_after)
        nxt = self._ensure_skill_library().select_skill(st_after, goal)
        if nxt is None or nxt.name == skill.name:
            return False
        chain = list(getattr(self, "_skill_chain", []) or [])
        chain.append(skill.name)
        self._skill_chain = chain
        self._skill_exec = {
            "skill": nxt,
            "index": 0,
            "obs_before": dict(obs_before_init),
            "chain_depth": depth + 1,
        }
        return True

    def _start_skill_if_due(self, engine_tick: int) -> bool:
        if self._human_task_suppresses_autonomous_locomotion():
            return False
        if not self._skill_library_enabled():
            return False
        if self._skill_exec is not None:
            return False
        import random

        if random.random() > self._skill_start_prob():
            return False
        obs = dict(self.agent.env.observe())
        st = self._skill_state_dict(obs)
        goal = self._skill_goal_hint(st)
        sk = self._ensure_skill_library().select_skill(st, goal)
        if sk is None:
            return False
        self._skill_chain = []
        self._skill_exec = {
            "skill": sk,
            "index": 0,
            "obs_before": st,
            "chain_depth": 0,
        }
        return True

    def _skill_ns_blend_weight(self, st: dict) -> float:
        """NS prior override: full weight when stable-gate LOCOMOTE is active."""
        if self._executive_macro_hint() in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            try:
                from engine.locomote_gate import stable_locomote_ready

                if stable_locomote_ready(st):
                    return float(os.environ.get("RKK_SKILL_NS_BLEND_STABLE", "1.0"))
            except ImportError:
                pass
        try:
            return float(os.environ.get("RKK_SKILL_NS_BLEND", "0.85"))
        except ValueError:
            return 0.85

    def _executive_macro_hint(self) -> str:
        s2 = getattr(self, "_system2", None)
        if s2 is not None:
            macro = str(getattr(s2, "_active_macro", "") or "").strip().upper()
            if macro:
                return macro
        ctx = getattr(self, "_intention_state", None)
        if ctx is not None:
            return str(getattr(ctx, "macro_hint", "") or "").strip().upper()
        last = getattr(self, "_system2_last", None) or {}
        if isinstance(last, dict):
            return str(last.get("macro") or "").strip().upper()
        return ""

    def _skill_goal_hint(self, st: dict) -> str:
        from engine.locomote_gate import stable_locomote_ready

        cz = float(st.get("com_z", st.get("phys_com_z", 0.5)))
        posture = float(st.get("posture_stability", st.get("phys_posture_stability", 0.5)))
        foot_l = float(st.get("foot_contact_l", st.get("phys_foot_contact_l", 0.5)))
        foot_r = float(st.get("foot_contact_r", st.get("phys_foot_contact_r", 0.5)))
        macro = self._executive_macro_hint()
        if macro in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            return "walk"
        if (
            is_humanoid_topology(self.current_world)
            and not self._fixed_root_active
            and stable_locomote_ready(st)
        ):
            return "walk"
        if cz < 0.36:
            return "stand"
        if posture < 0.68 or min(foot_l, foot_r) < 0.54:
            return "stand"
        try:
            walk_min = int(os.environ.get("RKK_CURRICULUM_WALK_MIN_TICK", "2000"))
        except ValueError:
            walk_min = 2000
        if (
            walk_min > 0
            and is_humanoid_topology(self.current_world)
            and not self._fixed_root_active
            and self.tick < walk_min
            and not stable_locomote_ready(st)
        ):
            return "stand"
        bt = getattr(self, "behavioral_tracker", None)
        if bt is not None:
            snap = self._behavioral_snapshot_cached() or {}
            vel = float(snap.get("com_x_vel_ema", 0.0))
            vel_min = float(os.environ.get("RKK_STEP3_COM_X_VEL_MIN", "0.08"))
            if vel < vel_min and not stable_locomote_ready(st):
                return "stand"
        g = os.environ.get("RKK_SKILL_GOAL", "walk").strip().lower()
        return g if g else "walk"

    def _abort_stance_skill_for_locomote(self) -> None:
        pack = self._skill_exec
        if pack is None:
            return
        skill = pack.get("skill")
        if skill is None:
            return
        if str(getattr(skill, "name", "")) != "hold_stance":
            return
        if self._executive_macro_hint() in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            self._skill_exec = None
            self._skill_chain = []

    def _sim_env_intervene(
        self, var: str, val: float, *, count_intervention: bool
    ) -> dict:
        from engine.graph_constants import is_read_only_macro_var

        if is_read_only_macro_var(var):
            return dict(self.agent.env.observe())
        env = self.agent.env
        fn = getattr(env, "intervene", None)
        if not callable(fn):
            return {}
        try:
            return fn(var, val, count_intervention=count_intervention)
        except TypeError:
            return fn(var, val)

    @staticmethod
    def _skill_step_to_pairs(step) -> list[tuple[str, float]]:
        if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str):
            return [(str(step[0]), float(step[1]))]
        if isinstance(step, list):
            return [(str(a), float(b)) for a, b in step]
        return []

    def _execute_skill_frame(self) -> dict:
        from engine.graph_constants import is_read_only_macro_var

        pack = self._skill_exec
        if pack is None:
            return self.agent.step(engine_tick=self.tick)
        skill = pack["skill"]
        idx: int = pack["index"]
        obs_before_init: dict = pack["obs_before"]
        step = skill.action_sequence[idx]
        pairs = [
            (v, x)
            for v, x in self._skill_step_to_pairs(step)
            if not is_read_only_macro_var(v)
        ]
        if pairs and self._executive_macro_hint() in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            try:
                from engine.neuro_symbolic.motor_sync import collect_motor_targets

                ns_t = collect_motor_targets(self)
                st = self._skill_state_dict(obs_before_init)
                w = self._skill_ns_blend_weight(st)
                blended: list[tuple[str, float]] = []
                for v, x in pairs:
                    ck = v[5:] if v.startswith("phys_") else v
                    if ck.startswith("intent_") and ck in ns_t:
                        blended.append((v, float(x) * (1.0 - w) + float(ns_t[ck]) * w))
                    else:
                        blended.append((v, x))
                pairs = blended
            except Exception:
                pass
        var0, val0 = (pairs[0] if pairs else ("", 0.5))

        obs_before_env = dict(self.agent.env.observe())
        self.agent.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.agent.graph.snapshot_vec_dict()

        if not pairs:
            idx += 1
            done = idx >= len(skill.action_sequence)
            if done:
                obs = dict(self.agent.env.observe())
                st = self._skill_state_dict(obs)
                cz_a = float(st.get("com_z", st.get("phys_com_z", 0.5)))
                cz_b = float(
                    obs_before_init.get(
                        "com_z", obs_before_init.get("phys_com_z", 0.5)
                    )
                )
                self._ensure_skill_library().record_outcome(
                    skill, st, cz_a - cz_b
                )
                if not self._maybe_chain_next_skill(
                    skill,
                    st,
                    prediction_error=0.0,
                    obs_before_init=obs_before_init,
                ):
                    self._skill_exec = None
                    self._skill_chain = []
            else:
                self._skill_exec = {
                    "skill": skill,
                    "index": idx,
                    "obs_before": obs_before_init,
                    "chain_depth": int(pack.get("chain_depth", 0)),
                }
            return {
                "blocked": False,
                "skipped": True,
                "hierarchy": "skill",
                "skill": skill.name,
                "skill_step": idx,
                "skill_done": done,
                "variable": "",
                "value": 0.5,
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        burst = len(pairs) > 1

        if not burst:
            var, val = pairs[0]
            check = self.agent.value_layer.check_action(
                variable=var,
                value=float(val),
                current_nodes=dict(self.agent.graph.nodes),
                graph=self.agent.graph,
                temporal=self.agent.temporal,
                current_phi=self.agent.phi_approx(),
                other_agents_phi=self.agent.other_agents_phi,
                engine_tick=self.tick,
                imagination_horizon=0,
            )
            if not check.allowed:
                return {
                    "blocked": True,
                    "blocked_count": 1,
                    "reason": check.reason.value,
                    "variable": var,
                    "value": float(val),
                    "updated_edges": [],
                    "compression_delta": 0.0,
                    "prediction_error": 0.0,
                    "cf_predicted": {},
                    "cf_observed": {},
                    "goal_planned": False,
                    "hierarchy": "skill",
                    "skill": skill.name,
                    "skill_step": idx,
                    "skill_done": False,
                }
            obs_after = self._sim_env_intervene(var, val, count_intervention=True)
        else:
            burst_fn = getattr(self.agent.env, "intervene_burst", None)
            if callable(burst_fn):
                obs_after = dict(burst_fn(pairs, count_intervention=True))
            else:
                obs_after = {}
                for var, val in pairs:
                    obs_after = self._sim_env_intervene(
                        var, val, count_intervention=False
                    )
                if not obs_after:
                    obs_after = dict(self.agent.env.observe())

        if not obs_after:
            obs_after = dict(self.agent.env.observe())
        st_after = self._skill_state_dict(obs_after)
        self._sync_motor_state(obs_after, source="skill", tick=self.tick)
        try:
            from engine.neuro_symbolic.motor_sync import enforce_sticky_locomote_priors

            enforce_sticky_locomote_priors(self)
        except Exception:
            pass
        intents_log = {
            v: float(x) for v, x in pairs if str(v).startswith("intent_")
        }
        self._log_motor_command(
            source="skill",
            intents=intents_log if intents_log else None,
            obs=self._motor_obs_payload(obs_after),
        )
        self.agent.graph.apply_env_observation(obs_after)
        obs_after_full = self.agent.graph.snapshot_vec_dict()
        self.agent.graph.record_observation(obs_before_full)
        self.agent.graph.record_observation(obs_after_full)
        for var, val in pairs:
            if var in self.agent.graph.nodes:
                self.agent.graph.record_intervention(
                    var, float(val), obs_before_full, obs_after_full
                )
        self.agent.temporal.step(obs_after)

        idx += 1
        done = idx >= len(skill.action_sequence)
        pe = float(
            np.mean(
                [
                    abs(
                        float(self.agent.graph.nodes.get(nid, 0.5))
                        - float(obs_after_full.get(nid, 0.5))
                    )
                    for nid in self.agent.graph._node_ids[:24]
                ]
            )
            if self.agent.graph._node_ids
            else 0.0
        )
        if done:
            cz_a = float(st_after.get("com_z", st_after.get("phys_com_z", 0.5)))
            cz_b = float(
                obs_before_init.get(
                    "com_z", obs_before_init.get("phys_com_z", 0.5)
                )
            )
            reward = cz_a - cz_b
            self._ensure_skill_library().record_outcome(skill, st_after, reward)
            if not self._maybe_chain_next_skill(
                skill,
                st_after,
                prediction_error=pe,
                obs_before_init=obs_before_init,
            ):
                self._skill_exec = None
                self._skill_chain = []
        else:
            self._skill_exec = {
                "skill": skill,
                "index": idx,
                "obs_before": obs_before_init,
                "chain_depth": int(pack.get("chain_depth", 0)),
            }

        return {
            "blocked": False,
            "skipped": True,
            "hierarchy": "skill",
            "skill": skill.name,
            "skill_chain_depth": self._skill_chain_depth(),
            "skill_chain": list(getattr(self, "_skill_chain", []) or []),
            "skill_step": idx,
            "skill_done": done,
            "variable": var0,
            "value": float(val0),
            "updated_edges": [],
            "compression_delta": 0.0,
            "prediction_error": 0.0,
            "cf_predicted": {},
            "cf_observed": {},
            "goal_planned": False,
        }

    def _ensure_homeostatic_ctrl(self):
        if not hasattr(self, "_homeostatic_ctrl"):
            from engine.active_inference import HomeostaticController
            import torch
            device = getattr(self.agent.graph._core, "device", torch.device("cpu"))
            self._homeostatic_ctrl = HomeostaticController(device=device, learning_rate=0.1, max_iters=10)
        return self._homeostatic_ctrl

    @staticmethod
    def _graph_intent_to_env_var(nid: str) -> str | None:
        """Имя узла графа → переменная motor intent в HumanoidEnvironment (или None)."""
        from engine.features.humanoid.constants import MOTOR_INTENT_VARS

        s = str(nid)
        if s in MOTOR_INTENT_VARS:
            return s
        if s.startswith("phys_intent_"):
            suf = s[len("phys_intent_") :]
            if suf in MOTOR_INTENT_VARS:
                return suf
        return None

    def _intent_pairs_for_env(self, actions: dict[str, float]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for gid, val in actions.items():
            ev = self._graph_intent_to_env_var(gid)
            if ev is not None:
                out.append((ev, float(val)))
        return out

    def _run_active_inference_step(self, engine_tick: int) -> dict:
        """Один шаг Active Inference: минимизация Free Energy (дивергенции с target_priors)."""
        ctrl = self._ensure_homeostatic_ctrl()
        
        obs_before_env = dict(self.agent.env.observe())
        self.agent.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.agent.graph.snapshot_vec_dict()
        
        # 1. Применение возмущений (Perturbations) в режиме fixed_root
        if self._fixed_root_active and engine_tick % 50 == 0:
            import random
            from engine.graph_constants import is_read_only_macro_var

            perturb_val = random.uniform(0.1, 0.9)
            candidates = [
                n
                for n in self.agent.graph._node_ids
                if not is_read_only_macro_var(n) and self._graph_intent_to_env_var(n)
            ]
            if candidates:
                ev = self._graph_intent_to_env_var(random.choice(candidates))
                if ev:
                    self._sim_env_intervene(ev, perturb_val, count_intervention=False)
            obs_before_env = dict(self.agent.env.observe())
            self.agent.graph.apply_env_observation(obs_before_env)
            obs_before_full = self.agent.graph.snapshot_vec_dict()

        # 2. Вычисляем компенсирующие действия через Active Inference
        goal = self._skill_goal_hint(obs_before_full)
        if goal == "walk":
            # Если мы стабильны, добавляем приор на движение вперед (com_x_vel)
            target_priors = {
                "phys_posture_stability": 1.0, 
                "phys_com_z": 0.82,
                "phys_com_x_vel": 0.35  # Целевая скорость вперед
            }
        else:
            # Если мы падаем или нестабильны, фокусируемся только на балансе
            target_priors = {
                "phys_posture_stability": 1.0, 
                "phys_com_z": 0.82
            }
        
        # Добавляем интринсик (любопытство) если включено
        if getattr(self, "_intrinsic", None) and hasattr(self._intrinsic, "get_target_priors"):
            intrinsic_priors = self._intrinsic.get_target_priors(obs_before_full)
            target_priors.update(intrinsic_priors)

        # Living Memory: поправки приоров из эпизодической памяти (паттерны падений / окна)
        em = getattr(self, "_episodic_memory", None)
        if em is not None:
            try:
                mem_adj = em.retrieve_prior_adjustments(obs_before_full)
                for k, v in mem_adj.items():
                    target_priors[k] = float(v)
            except Exception:
                pass
        
        # Инверсия модели: какие действия приведут к target_priors?
        actions = ctrl.optimize_action(obs_before_full, self.agent.graph, target_priors)
        
        if not actions:
            # Motor Babbling: если модель еще не обучена (градиенты нулевые),
            # добавляем случайные микродвижения по motor intent (имена как в среде).
            import random

            for nid in self.agent.graph._node_ids:
                ev = self._graph_intent_to_env_var(nid)
                if ev and random.random() < 0.35:
                    actions[nid] = random.uniform(0.15, 0.85)

        pairs_graph = list(actions.items())
        pairs = self._intent_pairs_for_env(actions)
        if actions:
            ranked = sorted(
                pairs,
                key=lambda kv: abs(kv[1] - 0.5),
                reverse=True,
            )
            top = ranked[0] if ranked else None
            print(
                f"[ACTIVE INFERENCE] Tick {engine_tick}: Generated {len(actions)} graph intents, "
                f"{len(pairs)} env intents. Top delta: {top}"
            )

        var0, val0 = (pairs[0] if pairs else ("", 0.5))
        
        # 3. Применяем действия (только имена, понятные HumanoidEnvironment)
        if not pairs:
            obs_after = dict(self.agent.env.observe())
        else:
            burst_fn = getattr(self.agent.env, "intervene_burst", None)
            if callable(burst_fn):
                obs_after = dict(burst_fn(pairs, count_intervention=True))
            else:
                for var, val in pairs:
                    obs_after = self._sim_env_intervene(var, val, count_intervention=False)
                if not obs_after:
                    obs_after = dict(self.agent.env.observe())
                    
        # 4. Логирование и обучение графа
        if not obs_after:
            obs_after = dict(self.agent.env.observe())
            
        self._sync_motor_state(obs_after, source="active_inference", tick=self.tick)
        self._log_motor_command(
            source="active_inference",
            intents=actions,
            obs=self._motor_obs_payload(obs_after),
        )
        
        self.agent.graph.apply_env_observation(obs_after)
        obs_after_full = self.agent.graph.snapshot_vec_dict()
        
        self.agent.graph.record_observation(obs_before_full)
        self.agent.graph.record_observation(obs_after_full)
        
        for var, val in pairs_graph:
            if var in self.agent.graph.nodes:
                self.agent.graph.record_intervention(
                    var, float(val), obs_before_full, obs_after_full
                )
                
        self.agent.temporal.step(obs_after)
        
        return {
            "blocked": False,
            "skipped": False,
            "hierarchy": "active_inference",
            "skill": "homeostasis",
            "skill_step": 0,
            "skill_done": True,
            "variable": var0,
            "value": float(val0),
            "updated_edges": [],
            "compression_delta": 0.0,
            "prediction_error": 0.0,
            "cf_predicted": {},
            "cf_observed": {},
            "goal_planned": False,
        }

    def _run_agent_or_skill_step(self, engine_tick: int) -> dict:
        """Curiosity-driven exploration; optional Active Inference every K ticks."""
        self._abort_stance_skill_for_locomote()
        if self._human_task_suppresses_autonomous_locomotion():
            self._skill_exec = None
            self._skill_chain = []
        if self._skill_exec is not None:
            return self._execute_skill_frame()
        ai_on = os.environ.get("RKK_ACTIVE_INFERENCE", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        s2_strict = os.environ.get("RKK_S2_WM_GATE_STRICT", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if ai_on and not s2_strict:
            skip_global_ai = self._human_task_suppresses_autonomous_locomotion()
            if not skip_global_ai:
                try:
                    from engine.task_executive import human_task_executive_active

                    skip_global_ai = human_task_executive_active(self)
                except Exception:
                    pass
            if not skip_global_ai:
                try:
                    every = max(1, int(os.environ.get("RKK_ACTIVE_INFERENCE_EVERY", "4")))
                except ValueError:
                    every = 4
                if engine_tick % every == 0:
                    try:
                        return self._run_active_inference_step(engine_tick)
                    except Exception:
                        pass
        if self._start_skill_if_due(engine_tick):
            return self._execute_skill_frame()
        fallen = False
        if is_humanoid_topology(self.current_world) and not self._fixed_root_active:
            is_fn = getattr(self.agent.env, "is_fallen", None)
            fallen = bool(is_fn()) if callable(is_fn) else False
        return self.agent.step(
            engine_tick=engine_tick,
            enable_l3=self._l3_planning_due(),
            fallen=fallen,
        )
