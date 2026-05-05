"""Simulation mixin: смена мира, vision, fixed_root."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationWorldMixin:
    def _humanoid_base_env(self):
        """
        Реальный EnvironmentHumanoid под agent.env (слоты / EnvironmentVisual).
        Без этого isinstance(agent.env, Humanoid) ломается и curriculum fixed_root не включается.
        """
        try:
            from engine.environment_humanoid import EnvironmentHumanoid
        except ImportError:
            return None
        ref = getattr(self, "_base_env_ref", None)
        if isinstance(ref, EnvironmentHumanoid):
            return ref
        env = getattr(self.agent, "env", None)
        if isinstance(env, EnvironmentHumanoid):
            return env
        inner = getattr(env, "base_env", None)
        if isinstance(inner, EnvironmentHumanoid):
            return inner
        return None

    def switch_world(self, new_world: str) -> dict:
        """Только humanoid; переключение сцен отключено."""
        if new_world not in WORLDS:
            return {"error": f"unknown world: {new_world}", "switched": False}
        return {
            "switched": False,
            "world": "humanoid",
            "current_world": self.current_world,
        }

    # ── Фаза 12: Visual mode ──────────────────────────────────────────────────
    def enable_visual(self, n_slots: int = 8, mode: str = "hybrid") -> dict:
        """
        Включаем Causal Visual Cortex.
        Текущая среда оборачивается в EnvironmentVisual.
        По умолчанию mode="hybrid": слоты для наблюдения + phys_* моторы — иначе VL/физика
        блокируют do(slot_k) без реального сустава.
        """
        if self._visual_mode:
            return {"visual": True, "already_enabled": True}

        try:
            from engine.environment_visual import EnvironmentVisual
        except ImportError:
            return {"error": "causal_vision module not available (install: opencv-python, scipy)"}

        with self._sim_step_lock:
            if self._visual_mode:
                return {"visual": True, "already_enabled": True}

            # Сохраняем оригинальный env
            self._base_env_ref = self.agent.env

            # Оборачиваем
            vis_env = EnvironmentVisual(
                self._base_env_ref,
                device=self.device,
                n_slots=n_slots,
                mode=mode,
            )
            self._visual_env = vis_env

            # Меняем среду агента: граф только под variable_ids обёртки (не 26+K узлов)
            new_vars  = list(vis_env.variable_ids)
            init_obs  = vis_env.observe()
            self.agent.graph.rebind_variables(new_vars, init_obs)

            self.agent.env = vis_env

            # Пересоздаём Temporal для нового d
            from engine.temporal import TemporalBlankets
            new_d = len(new_vars)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)

            self.agent.temporal.step(init_obs)
            self.agent.graph.record_observation(init_obs)

            # Инжектируем слабые seeds между слотами
            seeds = vis_env.hardcoded_seeds()
            self.agent.inject_text_priors(seeds)

            self._visual_mode = True
            self._vision_ticks = 0

            self._add_event(
                f"👁 Visual Cortex ENABLED: {n_slots} slots · {mode} mode",
                "#44ffcc", "phase"
            )
            cd = str(next(vis_env.cortex.parameters()).device)
            print(
                f"[Simulation] Visual mode ON: {n_slots} slots, d={self.agent.graph._d}, "
                f"cortex={cd}"
            )

            return {
                "visual": True,
                "n_slots": n_slots,
                "mode": mode,
                "new_vars": new_vars,
                "gnn_d": self.agent.graph._d,
                "cortex_device": cd,
            }

    def _disable_visual_internal(self):
        """Внутреннее отключение без event."""
        if not self._visual_mode:
            return
        if self._base_env_ref is not None:
            self.agent.env = self._base_env_ref
            base_ids = list(self._base_env_ref.variable_ids)
            base_obs = self._base_env_ref.observe()
            self.agent.graph.rebind_variables(base_ids, base_obs)
            from engine.temporal import TemporalBlankets
            new_d = len(base_ids)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)
            self.agent.temporal.step(base_obs)
            self.agent.graph.record_observation(base_obs)
        self._visual_mode = False
        self._visual_env  = None
        self._concept_store = None
        self._stop_l4_worker()
        self._l3_next_due_ts = 0.0

    def disable_visual(self) -> dict:
        """Отключаем Visual Cortex, возвращаемся к ручным переменным."""
        if not self._visual_mode:
            return {"visual": False, "was_enabled": False}
        with self._sim_step_lock:
            if not self._visual_mode:
                return {"visual": False, "was_enabled": False}
            self._disable_visual_internal()
            self._add_event("👁 Visual Cortex DISABLED", "#cc44ff", "phase")
        return {"visual": False, "was_enabled": True}

    def enable_fixed_root(self) -> dict:
        """
        Включаем fixed_root mode:
          1. PyBullet JOINT_FIXED constraint фиксирует базу
          2. variable_ids → FIXED_BASE_VARS (+ sandbox vars)
          3. GNN rebind; при visual — через EnvironmentVisual.set_fixed_root
          4. HomeostaticBounds → for_fixed_root()
          5. inject fixed_root_seeds() (для slot-графа большинство рёбер может быть skipped)
        """
        from engine.environment_humanoid import fixed_root_seeds

        humanoid = self._humanoid_base_env()
        if humanoid is None:
            return {"error": "fixed_root требует humanoid world"}

        if humanoid.fixed_root and self._fixed_root_active:
            return {"fixed_root": True, "already_enabled": True}

        with self._sim_step_lock:
            if self._visual_mode and self._visual_env is not None:
                self._visual_env.set_fixed_root(True)
            else:
                fn = getattr(self.agent.env, "set_fixed_root", None)
                if callable(fn) and self.agent.env is not humanoid:
                    fn(True)
                else:
                    humanoid.set_fixed_root(True)

            env = self.agent.env
            new_vars = list(env.variable_ids)
            init_obs = env.observe()
            self.agent.graph.rebind_variables(new_vars, init_obs)

            from engine.temporal import TemporalBlankets
            new_d = len(new_vars)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)
            self.agent.temporal.step(init_obs)
            self.agent.graph.record_observation(init_obs)

            from engine.value_layer import HomeostaticBounds
            self.agent.value_layer.bounds = HomeostaticBounds.for_fixed_root()

            seeds = fixed_root_seeds()
            result = self.agent.inject_text_priors(seeds)

            self._fixed_root_active = True
            self._fall_count = 0
            self._stop_cpg_background_loop()
            self.agent._post_fr_explore_until = 0
            self.agent._post_fr_vl_relax_until = 0

            self._add_event(
                f"📌 FIXED ROOT ON: d={self.agent.graph._d}, "
                f"{len(new_vars)} vars, +{result.get('injected',0)} seeds",
                "#ffcc44", "phase"
            )
            print(
                f"[Simulation] fixed_root ON: vars={len(new_vars)}, "
                f"d={self.agent.graph._d}, seeds={result.get('injected',0)}"
            )
            return {
                "fixed_root": True,
                "gnn_d":      self.agent.graph._d,
                "new_vars":   new_vars,
                "seeds_injected": result.get("injected", 0),
            }

    def disable_fixed_root(self) -> dict:
        """
        Отключаем fixed_root mode:
          1. Снимаем JOINT_FIXED constraint
          2. variable_ids → полные VAR_NAMES
          3. GNN rebind; при visual — через EnvironmentVisual.set_fixed_root(False)
          4. HomeostaticBounds → default (строгие, но с warmup)
        """
        from engine.environment_humanoid import humanoid_hardcoded_seeds

        humanoid = self._humanoid_base_env()
        if humanoid is None:
            return {"error": "не humanoid world"}

        if not self._fixed_root_active and not humanoid.fixed_root:
            return {"fixed_root": False, "was_enabled": False}

        with self._sim_step_lock:
            if self._visual_mode and self._visual_env is not None:
                self._visual_env.set_fixed_root(False)
            else:
                fn = getattr(self.agent.env, "set_fixed_root", None)
                if callable(fn) and self.agent.env is not humanoid:
                    fn(False)
                else:
                    humanoid.set_fixed_root(False)

            env = self.agent.env
            new_vars = list(env.variable_ids)
            init_obs = env.observe()
            self.agent.graph.rebind_variables(new_vars, init_obs)

            from engine.temporal import TemporalBlankets
            new_d = len(new_vars)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)
            self.agent.temporal.step(init_obs)
            self.agent.graph.record_observation(init_obs)

            from engine.value_layer import HomeostaticBounds
            self.agent.value_layer.bounds = HomeostaticBounds(
                var_min=0.05, var_max=0.95,
                phi_min=0.005,
                h_slow_max=18.0,
                env_entropy_max_delta=1.2,
                s1_penalty=-0.2,
                warmup_ticks=800,
                blend_ticks=400,
                phi_min_steady=0.03,
                env_entropy_max_delta_steady=0.65,
                h_slow_max_steady=14.0,
                predict_band_edge_steady=0.015,
                fixed_root_mode=False,
            )

            result = self.agent.inject_text_priors(humanoid_hardcoded_seeds())

            self._fixed_root_active = False

            try:
                n_exp = int(os.environ.get("RKK_POST_FR_EXPLORE_TICKS", "300"))
            except ValueError:
                n_exp = 300
            try:
                n_vl = int(os.environ.get("RKK_POST_FR_VL_RELAX_TICKS", "300"))
            except ValueError:
                n_vl = 300
            tcur = int(getattr(self, "tick", 0))
            self.agent._post_fr_explore_until = tcur + max(0, n_exp)
            self.agent._post_fr_vl_relax_until = tcur + max(0, n_vl)
            try:
                self.agent.system1.buffer.clear()
            except Exception:
                pass
            print(
                f"[Curriculum] System1 buffer cleared on fixed_root release "
                f"(tick={tcur}, explore_until={self.agent._post_fr_explore_until}, "
                f"vl_relax_until={self.agent._post_fr_vl_relax_until})",
                flush=True,
            )

            self._add_event(
                f"📌 FIXED ROOT OFF: d={self.agent.graph._d}, {len(new_vars)} vars",
                "#cc44ff", "phase"
            )
            return {
                "fixed_root": False,
                "gnn_d":      self.agent.graph._d,
                "new_vars":   new_vars,
                "seeds_injected": result.get("injected", 0),
            }

    # ── Seeds ─────────────────────────────────────────────────────────────────
    def agent_seed_context(self, agent_id: int) -> dict | None:
        """
        Контекст для POST /bootstrap/llm, GET /variables/{id}, RAG auto-seed:
        preset (имя мира) и список имён переменных текущей среды агента.
        """
        if agent_id != 0:
            return None
        try:
            vars_ = list(self.agent.env.variable_ids)
        except Exception:
            return None
        if not vars_:
            return None
        return {
            "preset": self.current_world,
            "variables": vars_,
            "agent_id": agent_id,
        }

    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        with self._sim_step_lock:
            result = self.agent.inject_text_priors(edges)
            n = result.get("injected", 0)
            self._add_event(f"💉 Seeds → Nova: {n} edges (α=0.05)", "#886600", "discovery")
        return {"injected": n, "agent": self.AGI_NAME,
                "skipped": result.get("skipped", []),
                "node_ids": result.get("node_ids", [])}

