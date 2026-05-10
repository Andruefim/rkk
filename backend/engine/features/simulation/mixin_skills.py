"""Simulation mixin: skill library, agent/skill step."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationSkillsMixin:
    def _skill_library_enabled(self) -> bool:
        v = os.environ.get("RKK_SKILL_LIBRARY", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _skill_start_prob(self) -> float:
        try:
            p = float(os.environ.get("RKK_SKILL_LIBRARY_PROB", "0.1"))
        except ValueError:
            p = 0.1
        if self.current_world == "humanoid" and not self._fixed_root_active:
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

    def _skill_goal_hint(self, st: dict) -> str:
        cz = float(st.get("com_z", st.get("phys_com_z", 0.5)))
        posture = float(st.get("posture_stability", st.get("phys_posture_stability", 0.5)))
        foot_l = float(st.get("foot_contact_l", st.get("phys_foot_contact_l", 0.5)))
        foot_r = float(st.get("foot_contact_r", st.get("phys_foot_contact_r", 0.5)))
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
            and self.current_world == "humanoid"
            and not self._fixed_root_active
            and self.tick < walk_min
        ):
            return "stand"
        g = os.environ.get("RKK_SKILL_GOAL", "walk").strip().lower()
        return g if g else "walk"

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
                self._skill_exec = None
            else:
                self._skill_exec = {
                    "skill": skill,
                    "index": idx,
                    "obs_before": obs_before_init,
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
        if done:
            cz_a = float(st_after.get("com_z", st_after.get("phys_com_z", 0.5)))
            cz_b = float(
                obs_before_init.get(
                    "com_z", obs_before_init.get("phys_com_z", 0.5)
                )
            )
            reward = cz_a - cz_b
            self._ensure_skill_library().record_outcome(skill, st_after, reward)
            self._skill_exec = None
        else:
            self._skill_exec = {
                "skill": skill,
                "index": idx,
                "obs_before": obs_before_init,
            }

        return {
            "blocked": False,
            "skipped": True,
            "hierarchy": "skill",
            "skill": skill.name,
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
        """Curiosity-driven exploration for all environments including humanoid."""
        return self.agent.step(
            engine_tick=engine_tick,
            enable_l3=self._l3_planning_due(),
        )
