"""
simulation_singleton_v2.py — Singleton AGI с гуманоидом (Фаза 11/12).

Фаза 12 добавляет:
  - visual_mode toggle: enable_visual() / disable_visual()
  - EnvironmentVisual wrapper активируется без перезапуска агента
  - Predictive coding loop: GNN prediction → visual cortex feedback
  - /vision/slots endpoint data через get_vision_state()
  - vision_stats в snapshot
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque

from engine.agent       import RKKAgent
from engine.demon       import AdversarialDemon
from engine.value_layer import HomeostaticBounds
from engine.environment import Environment

PHASE_THRESHOLDS = [0.0, 0.15, 0.30, 0.50, 0.70, 0.88]
PHASE_HOLD_TICKS = 12
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]

WORLDS = {
    "humanoid":  {"label": "Humanoid",          "color": "#cc44ff"},
    "robot":     {"label": "Robot Arm",          "color": "#aa22dd"},
    "pybullet":  {"label": "3D Physics",         "color": "#ff44aa"},
    "physics":   {"label": "Thermodynamics",     "color": "#00ff99"},
    "chemistry": {"label": "Chemical Kinetics",  "color": "#0099ff"},
    "logic":     {"label": "Logic Gates",        "color": "#ff9900"},
}


def _make_env(world: str, device: torch.device):
    if world == "humanoid":
        from engine.environment_humanoid import EnvironmentHumanoid
        return EnvironmentHumanoid(device=device)
    if world == "robot":
        from engine.environment_robot import EnvironmentRobot
        return EnvironmentRobot(device=device)
    if world == "pybullet":
        from engine.environment_pybullet import EnvironmentPyBullet
        return EnvironmentPyBullet(n_objects=3, device=device, use_pybullet=True)
    return Environment(world, device)


def _default_bounds() -> HomeostaticBounds:
    return HomeostaticBounds(
        var_min=0.05, var_max=0.95,
        phi_min=0.01,
        h_slow_max=14.0,
        env_entropy_max_delta=0.96,
        warmup_ticks=1500,
        blend_ticks=500,
        phi_min_steady=0.04,
        env_entropy_max_delta_steady=0.60,
        h_slow_max_steady=11.0,
        predict_band_edge_steady=0.02,
    )


class WorldSwitcher:
    def __init__(self, agent: RKKAgent, device: torch.device):
        self.agent   = agent
        self.device  = device
        self.history: list[dict] = []

    def switch(self, new_world: str) -> dict:
        old = self.agent.env.preset
        if old == new_world:
            return {"switched": False, "world": new_world}

        new_env = _make_env(new_world, self.device)

        old_nodes = set(self.agent.graph.nodes.keys())
        init_obs  = new_env.observe()
        new_vars  = new_env.variable_ids
        new_nodes = [v for v in new_vars if v not in old_nodes]

        for var_id in new_vars:
            self.agent.graph.set_node(var_id, init_obs.get(var_id, 0.5))

        self.agent.env = new_env

        from engine.temporal import TemporalBlankets
        new_d = len(new_vars)
        if self.agent.temporal.d_input != new_d:
            self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)

        self.agent.temporal.step(init_obs)
        self.agent.graph.record_observation(init_obs)

        rec = {
            "from_world":  old,
            "to_world":    new_world,
            "new_nodes":   new_nodes,
            "total_nodes": len(self.agent.graph.nodes),
            "gnn_d":       self.agent.graph._d,
        }
        self.history.append(rec)
        print(f"[WorldSwitch] {old} → {new_world} | +{len(new_nodes)} nodes | d={self.agent.graph._d}")
        return {"switched": True, **rec}


# ─── Simulation ───────────────────────────────────────────────────────────────
class Simulation:
    AGI_NAME  = "Nova"
    AGI_COLOR = "#cc44ff"

    def __init__(self, device_str: str = "cuda", start_world: str = "humanoid"):
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self.current_world = start_world
        print(f"[Singleton v2] Device: {self.device} | World: {start_world}")

        env    = _make_env(start_world, self.device)
        bounds = _default_bounds()

        self.agent = RKKAgent(
            agent_id=0, name=self.AGI_NAME,
            env=env, device=self.device, bounds=bounds,
        )

        self.switcher    = WorldSwitcher(self.agent, self.device)
        self.demon       = AdversarialDemon(n_agents=1, device=self.device)

        self.tick        = 0
        self.phase       = 1
        self.max_phase   = 1

        self._phase_hold_counter = 0
        self._candidate_phase    = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events:    deque[dict]   = deque(maxlen=24)
        self._prev_edge_count = 0
        self._last_snapshot: dict = {}

        self._fall_count  = 0
        self._stand_ticks = 0

        # ── Фаза 12: Visual Cortex ────────────────────────────────────────────
        self._visual_mode   = False     # выкл по умолчанию
        self._visual_env    = None      # EnvironmentVisual instance
        self._base_env_ref  = None      # оригинальный env (до visual wrap)
        self._vision_ticks  = 0         # тиков с включённым зрением
        self._last_vision_state: dict = {}

    # ── World switch ──────────────────────────────────────────────────────────
    def switch_world(self, new_world: str) -> dict:
        if new_world not in WORLDS:
            return {"error": f"unknown world: {new_world}"}

        # Если visual mode — сначала отключаем
        was_visual = self._visual_mode
        if was_visual:
            self._disable_visual_internal()

        result = self.switcher.switch(new_world)
        if result.get("switched"):
            self.current_world = new_world
            winfo = WORLDS[new_world]
            self._add_event(
                f"🌍 → {winfo['label']} "
                f"(+{len(result.get('new_nodes',[]))} vars, d={result.get('gnn_d')})",
                winfo["color"], "phase"
            )

        # Восстанавливаем visual mode если был
        if was_visual and result.get("switched"):
            self.enable_visual()

        return result

    # ── Фаза 12: Visual mode ──────────────────────────────────────────────────
    def enable_visual(self, n_slots: int = 8, mode: str = "visual") -> dict:
        """
        Включаем Causal Visual Cortex.
        Текущая среда оборачивается в EnvironmentVisual.
        GNN перестраивается под slot_0...slot_N переменные.
        """
        if self._visual_mode:
            return {"visual": True, "already_enabled": True}

        try:
            from engine.environment_visual import EnvironmentVisual
        except ImportError:
            return {"error": "causal_vision module not available (install: opencv-python, scipy)"}

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
        print(f"[Simulation] Visual mode ON: {n_slots} slots, d={self.agent.graph._d}")

        return {
            "visual": True,
            "n_slots": n_slots,
            "mode": mode,
            "new_vars": new_vars,
            "gnn_d": self.agent.graph._d,
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

    def disable_visual(self) -> dict:
        """Отключаем Visual Cortex, возвращаемся к ручным переменным."""
        if not self._visual_mode:
            return {"visual": False, "was_enabled": False}
        self._disable_visual_internal()
        self._add_event("👁 Visual Cortex DISABLED", "#cc44ff", "phase")
        return {"visual": False, "was_enabled": True}

    # ── Seeds ─────────────────────────────────────────────────────────────────
    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        result = self.agent.inject_text_priors(edges)
        n = result.get("injected", 0)
        self._add_event(f"💉 Seeds → Nova: {n} edges (α=0.05)", "#886600", "discovery")
        return {"injected": n, "agent": self.AGI_NAME,
                "skipped": result.get("skipped", []),
                "node_ids": result.get("node_ids", [])}

    def agent_seed_context(self, agent_id: int = 0) -> dict | None:
        return {
            "name":      self.AGI_NAME,
            "preset":    self.current_world,
            "variables": list(self.agent.graph.nodes.keys()),
        }

    # ── Tick ──────────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1

        # Fallen check
        fallen = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn):
            fallen = is_fn()
            if fallen:
                self._fall_count += 1
                if self._fall_count % 20 == 1:
                    self._add_event(
                        f"💀 [FALLEN] Nova упал! (×{self._fall_count})",
                        "#ff2244", "value"
                    )

        # Фаза 12: передаём GNN prediction в visual env перед шагом
        if self._visual_mode and self._visual_env is not None:
            self._vision_ticks += 1
            self._feed_gnn_prediction_to_visual()

        self.agent.other_agents_phi = []
        result = self.agent.step(engine_tick=self.tick)
        self._log_step(result, fallen)

        snap = self.agent.snapshot()
        snap["fallen"]     = fallen
        snap["fall_count"] = self._fall_count
        self._last_snapshot = snap

        # Demon
        if self.demon._last_action is not None:
            pe = 0.0
            if not result.get("blocked") and not result.get("skipped"):
                pe = float(result.get("prediction_error", 0))
            self.demon.learn(pe, self.demon._last_action_complexity, [snap])
        self._step_demon(snap)

        smoothed = self._update_phase(snap)

        graph_deltas = {}
        cnt = len(self.agent.graph.edges)
        if cnt != self._prev_edge_count:
            graph_deltas[0] = [e.as_dict() for e in self.agent.graph.edges]
            self._prev_edge_count = cnt

        # Scene
        scene_fn = getattr(self.agent.env, "get_full_scene", None)
        scene    = scene_fn() if callable(scene_fn) else {}

        # Vision state (кэш для /vision/slots endpoint)
        if self._visual_mode and self._visual_env is not None:
            try:
                self._last_vision_state = self._visual_env.get_slot_visualization()
            except Exception:
                pass

        return self._build_snapshot(snap, graph_deltas, smoothed, scene)

    def _feed_gnn_prediction_to_visual(self):
        """Передаём текущий GNN-прогноз в visual env для predictive coding."""
        if self._visual_env is None or self.agent.graph._core is None:
            return
        try:
            current_obs = self._visual_env.observe()
            node_ids    = self.agent.graph._node_ids
            slot_ids    = [f"slot_{k}" for k in range(self._visual_env.n_slots)]
            values_list = [current_obs.get(sid, 0.5) for sid in slot_ids]
            current_t   = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            # Прогоняем текущие значения через GNN → получаем предсказания
            with torch.no_grad():
                full_state = torch.tensor(
                    [self.agent.graph.nodes.get(n, 0.5) for n in node_ids],
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                pred_full = self.agent.graph._core(full_state).squeeze(0)
            # Выбираем только slot_ переменные
            slot_pred = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            for i, sid in enumerate(slot_ids):
                if sid in node_ids:
                    idx = node_ids.index(sid)
                    if idx < len(pred_full):
                        slot_pred[i] = pred_full[idx]
            self._visual_env.set_gnn_prediction(slot_pred)
        except Exception:
            pass

    def _step_demon(self, snap: dict):
        try:
            action = self.demon.step([snap], 1 - snap.get("peak_discovery_rate", 0))
        except RuntimeError as e:
            print(f"[Singleton] Demon step skipped: {e}")
            return
        if action is None:
            return
        corrupted = self.agent.demon_disrupt()
        self._add_event(
            f"⚠ Demon [{action.get('mode','?')}] → Nova: {corrupted}",
            "#ff2244", "demon"
        )

    def _update_phase(self, snap: dict) -> float:
        dr = snap.get("discovery_rate", 0)
        self._dr_window.append(dr)
        smoothed = float(np.mean(self._dr_window))

        potential = 1
        for i, t in enumerate(PHASE_THRESHOLDS):
            if smoothed >= t:
                potential = i + 1
        potential = min(potential, 5)

        if potential > self.max_phase:
            if potential == self._candidate_phase:
                self._phase_hold_counter += 1
            else:
                self._candidate_phase    = potential
                self._phase_hold_counter = 1
            if self._phase_hold_counter >= PHASE_HOLD_TICKS:
                self.max_phase = potential
                self.phase     = potential
                self._phase_hold_counter = 0
                self._add_event(f"⬆ Phase {potential}: {PHASE_NAMES[potential]}", "#ffcc00", "phase")
        else:
            self._candidate_phase    = self.max_phase
            self._phase_hold_counter = 0
        self.phase = self.max_phase
        return smoothed

    def _log_step(self, result: dict, fallen: bool):
        if result.get("blocked"):
            self._add_event(
                f"🛡 [BLOCKED] Nova: {result.get('reason','?')}",
                "#884400", "value"
            )
        elif result.get("updated_edges"):
            cg  = result.get("compression_delta", 0)
            var = result.get("variable", "?")
            val = result.get("value", 0)
            color = WORLDS.get(self.current_world, {}).get("color", "#cc44ff")
            if self._visual_mode:
                color = "#44ffcc"
            self._add_event(
                f"Nova: do({var}={val:.2f}) CG{'+' if cg>=0 else ''}{cg:.3f}",
                color, "discovery"
            )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})

    # ── Camera / Scene ────────────────────────────────────────────────────────
    def get_camera_frame(self, view: str = "diag") -> str | None:
        fn = getattr(self.agent.env, "get_frame_base64", None)
        return fn(view) if callable(fn) else None

    def get_vision_state(self) -> dict:
        """Данные для /vision/slots endpoint."""
        if not self._visual_mode or self._visual_env is None:
            return {"visual_mode": False}
        state = dict(self._last_vision_state)
        state["visual_mode"]  = True
        state["n_slots"]      = self._visual_env.n_slots
        state["vision_ticks"] = self._vision_ticks
        state["cortex"]       = self._visual_env.cortex.snapshot()
        return state

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _build_snapshot(self, snap: dict, graph_deltas: dict,
                        smoothed: float, scene: dict) -> dict:
        winfo = WORLDS.get(self.current_world, {"color": "#cc44ff", "label": self.current_world})

        # Visual cortex summary
        vision_summary = None
        if self._visual_mode and self._visual_env is not None:
            vision_summary = self._visual_env.cortex.snapshot()

        return {
            "tick":          self.tick,
            "phase":         self.phase,
            "max_phase":     self.max_phase,
            "entropy":       round((1 - snap.get("peak_discovery_rate", 0)) * 100, 1),
            "smoothed_dr":   round(smoothed, 3),
            "agents":        [snap],
            "n_agents":      1,
            "demon":         self.demon.snapshot,
            "tom_links":     [],
            "events":        list(self.events),
            "graph_deltas":  graph_deltas,
            "value_layer":   {
                "total_blocked_all": snap.get("total_blocked", 0),
                "block_rates": [round(snap.get("value_layer", {}).get("block_rate", 0), 3)],
            },
            "byzantine":     None,
            "motif":         None,
            "multiprocess":  False,
            "singleton":     True,
            "current_world": self.current_world,
            "world_label":   winfo["label"],
            "world_color":   winfo["color"],
            "worlds":        WORLDS,
            "switch_history":self.switcher.history[-5:],
            "gnn_d":         self.agent.graph._d,
            "fallen":        snap.get("fallen", False),
            "fall_count":    snap.get("fall_count", 0),
            "scene":         scene,
            # Фаза 12
            "visual_mode":   self._visual_mode,
            "vision_ticks":  self._vision_ticks,
            "vision":        vision_summary,
        }

    def public_state(self) -> dict:
        snap     = self._last_snapshot or self.agent.snapshot()
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        fn       = getattr(self.agent.env, "get_full_scene", None)
        scene    = fn() if callable(fn) else {}
        return self._build_snapshot(snap, {}, smoothed, scene)

    def shutdown(self):
        pass