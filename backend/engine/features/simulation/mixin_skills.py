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

    def _run_agent_or_skill_step(self, engine_tick: int) -> dict:
        """L3 внутри agent.step; L2 skill — один шаг последовательности за тик."""
        if (
            self.current_world == "humanoid"
            and not self._fixed_root_active
            and self._curriculum_stabilize_until > 0
            and engine_tick <= self._curriculum_stabilize_until
        ):
            if self._skill_exec is not None:
                return self._execute_skill_frame()
            if self._skill_library_enabled():
                lib = self._ensure_skill_library()
                obs = self.agent.env.observe()
                obs_st = self._skill_state_dict(obs)
                sk = lib.select_skill(obs_st, "stand")
                if sk is not None:
                    self._skill_exec = {
                        "skill": sk,
                        "index": 0,
                        "obs_before": dict(obs_st),
                    }
                    return self._execute_skill_frame()
            return self.agent.step(engine_tick=engine_tick, enable_l3=False)

        # Humanoid: при нестабильной позе не даём EIG выбирать сырые суставы — только скиллы / stand.
        if self.current_world == "humanoid" and not self._fixed_root_active:
            obs = self.agent.env.observe()
            posture = float(
                obs.get(
                    "posture_stability", obs.get("phys_posture_stability", 0.5)
                )
            )
            if posture < 0.65:
                if self._skill_exec is not None:
                    return self._execute_skill_frame()
                if self._skill_library_enabled():
                    lib = self._ensure_skill_library()
                    obs_st = self._skill_state_dict(obs)
                    sk = lib.select_skill(obs_st, "stand")
                    if sk is not None:
                        self._skill_exec = {
                            "skill": sk,
                            "index": 0,
                            "obs_before": dict(obs_st),
                        }
                        return self._execute_skill_frame()
        if (
            self._skill_library_enabled()
            and self.current_world == "humanoid"
            and not self._fixed_root_active
        ):
            if self._skill_exec is not None:
                return self._execute_skill_frame()
            if self._skill_start_prob() > 0.0 and np.random.random() < self._skill_start_prob():
                obs = self.agent.env.observe()
                st = self._skill_state_dict(obs)
                goal = self._skill_goal_hint(st)
                sk = self._ensure_skill_library().select_skill(st, goal)
                if sk is not None:
                    self._skill_exec = {
                        "skill": sk,
                        "index": 0,
                        "obs_before": dict(st),
                    }
                    return self._execute_skill_frame()
        return self.agent.step(
            engine_tick=engine_tick,
            enable_l3=self._l3_planning_due(),
        )
