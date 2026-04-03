"""
simulation_singleton.py — Singleton AGI (Фаза 11).

Один агент вместо четырёх. Без Byzantine consensus (нет пиров).

Что убрано:
  - Мультиагентный пул (AgentPool / multiprocessing)
  - Byzantine consensus / MotifTransfer
  - ToM между агентами

Что добавлено:
  - WorldSwitcher: смена среды без сброса агента (GNN resize_to)
  - RobotEnv: 3-звенный манипулятор в PyBullet
  - Camera endpoint: кадр из PyBullet
  - MiniLLM bootstrap: генерация начальных гипотез через Ollama
  - ImaginationBuffer: хранение N-step прогнозов (заготовка для Фазы 13)

Архитектура (Singleton):
  ┌─────────────────────────────────────┐
  │  SingletonAGI (один процесс, GPU)   │
  │  ┌───────────┐  ┌─────────────────┐ │
  │  │ System 1  │  │ CausalGNNCore   │ │
  │  │ (MLP)     │  │ (d × d, W-grad) │ │
  │  └───────────┘  └─────────────────┘ │
  │  ┌───────────┐  ┌─────────────────┐ │
  │  │ Temporal  │  │  Value Layer    │ │
  │  │ (SSM f/s) │  │  (homeostasis)  │ │
  │  └───────────┘  └─────────────────┘ │
  └─────────────────────────────────────┘
           │  do(variable=value)
           ▼
  ┌─────────────────────────────────────┐
  │   World (active environment)        │
  │   physics / chemistry / logic /     │
  │   pybullet / robot  ← switch!       │
  └─────────────────────────────────────┘
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque

from engine.agent       import RKKAgent
from engine.demon       import AdversarialDemon
from engine.value_layer import HomeostaticBounds
from engine.environment import Environment

PHASE_THRESHOLDS = [0.0, 0.18, 0.38, 0.58, 0.76, 0.92]
PHASE_HOLD_TICKS = 15
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]


# ─── Доступные миры ───────────────────────────────────────────────────────────
WORLDS = {
    "physics":   {"label": "Thermodynamics",  "color": "#00ff99"},
    "chemistry": {"label": "Chemical Kinetics","color": "#0099ff"},
    "logic":     {"label": "Logic Gates",      "color": "#ff9900"},
    "robot":     {"label": "Robot Arm",        "color": "#cc44ff"},
    "pybullet":  {"label": "3D Physics",       "color": "#ff44aa"},
}


def _make_env(world: str, device: torch.device):
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
        h_slow_max=12.0,
        env_entropy_max_delta=0.95,
        warmup_ticks=1500,
        blend_ticks=500,
        phi_min_steady=0.05,
        env_entropy_max_delta_steady=0.55,
        h_slow_max_steady=10.0,
        predict_band_edge_steady=0.02,
    )


# ─── WorldSwitcher ────────────────────────────────────────────────────────────
class WorldSwitcher:
    """
    Переключает активный мир без сброса агента.

    При смене мира:
    1. CausalGraph сохраняет веса W для старых узлов
    2. Новые переменные добавляются (GNN resize_to автоматически)
    3. System 1 сохраняет веса MLP (абстрактная интуиция переносится)
    4. Temporal Blankets накапливают историю из нового мира
    5. Injecting new-world seeds (необязательно) через RAG

    Философия: AGI не «забывает» физику когда учит химию.
    Граф расширяется, а не перезаписывается.
    """

    def __init__(self, agent: RKKAgent, device: torch.device):
        self.agent   = agent
        self.device  = device
        self._history: list[dict] = []

    def switch(self, new_world: str) -> dict:
        old_preset = self.agent.env.preset
        if old_preset == new_world:
            return {"switched": False, "world": new_world}

        # Создаём новую среду
        new_env = _make_env(new_world, self.device)

        # Добавляем новые узлы (GNN resize_to срабатывает автоматически в set_node)
        old_nodes = set(self.agent.graph.nodes.keys())
        new_vars  = new_env.variable_ids
        new_nodes = [v for v in new_vars if v not in old_nodes]

        # Начальные наблюдения новой среды
        init_obs = new_env.observe()

        # Добавляем новые узлы (CausalGraph._rebuild_core → GNN.resize_to)
        for var_id in new_vars:
            self.agent.graph.set_node(var_id, init_obs.get(var_id, 0.5))

        # Обновляем среду агента
        self.agent.env = new_env

        # Пересоздаём TemporalBlankets для нового d_input
        from engine.temporal import TemporalBlankets
        new_d = len(new_vars)
        if self.agent.temporal.d_input != new_d:
            self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)

        # Начальный шаг наблюдения
        self.agent.temporal.step(init_obs)
        self.agent.graph.record_observation(init_obs)

        record = {
            "from_world":  old_preset,
            "to_world":    new_world,
            "new_nodes":   new_nodes,
            "total_nodes": len(self.agent.graph.nodes),
            "gnn_d":       self.agent.graph._d,
        }
        self._history.append(record)
        print(f"[WorldSwitch] {old_preset} → {new_world} | "
              f"+{len(new_nodes)} nodes | total_d={self.agent.graph._d}")
        return {"switched": True, **record}

    @property
    def history(self) -> list[dict]:
        return self._history


# ─── SingletonSimulation ─────────────────────────────────────────────────────
class Simulation:
    """
    Singleton AGI симуляция.

    Публичный интерфейс совместим с simulation_v4.py:
      tick_step() → dict
      inject_seeds() → dict
      agent_seed_context() → dict
      public_state() → dict
    """

    AGI_NAME  = "Nova"
    AGI_COLOR = "#00ff99"

    def __init__(
        self,
        device_str:  str = "cuda",
        start_world: str = "robot",
    ):
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self.current_world = start_world
        print(f"[Singleton] Device: {self.device} | World: {start_world}")

        # Создаём начальную среду и агента
        env    = _make_env(start_world, self.device)
        bounds = _default_bounds()

        self.agent = RKKAgent(
            agent_id=0,
            name=self.AGI_NAME,
            env=env,
            device=self.device,
            bounds=bounds,
        )

        # WorldSwitcher
        self.switcher = WorldSwitcher(self.agent, self.device)

        # Demon (один, против одного агента)
        self.demon = AdversarialDemon(n_agents=1, device=self.device)

        self.tick      = 0
        self.phase     = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase    = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events: deque[dict]      = deque(maxlen=20)
        self._prev_edge_count = 0

        # Кэш снапшота агента
        self._last_snapshot: dict = {}

    # ── World switching ───────────────────────────────────────────────────────
    def switch_world(self, new_world: str) -> dict:
        if new_world not in WORLDS:
            return {"error": f"unknown world: {new_world}"}
        result = self.switcher.switch(new_world)
        if result.get("switched"):
            self.current_world = new_world
            self._add_event(
                f"🌍 World switch: → {WORLDS[new_world]['label']} "
                f"(+{len(result.get('new_nodes', []))} nodes, d={result.get('gnn_d')})",
                WORLDS[new_world]["color"], "phase"
            )
        return result

    # ── Seed injection ────────────────────────────────────────────────────────
    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        result = self.agent.inject_text_priors(edges)
        n = result.get("injected", 0)
        self._add_event(
            f"💉 Seeds → {self.AGI_NAME}: {n} edges (α=0.05)",
            "#886600", "discovery"
        )
        return {
            "injected": n,
            "agent":    self.AGI_NAME,
            "skipped":  result.get("skipped", []),
            "node_ids": result.get("node_ids", []),
        }

    def agent_seed_context(self, agent_id: int = 0) -> dict | None:
        return {
            "name":      self.AGI_NAME,
            "preset":    self.current_world,
            "variables": list(self.agent.graph.nodes.keys()),
        }

    # ── Один тик ──────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1

        # Шаг агента
        self.agent.other_agents_phi = []   # singleton, нет других
        result = self.agent.step(engine_tick=self.tick)
        self._log_step_result(result)

        # Demon feedback
        if self.demon._last_action is not None:
            pe = 0.0
            if not result.get("blocked") and not result.get("skipped"):
                pe = float(result.get("prediction_error", 0))
            snap = self.agent.snapshot()
            self.demon.learn(pe, self.demon._last_action_complexity, [snap])

        # Demon step
        snap = self.agent.snapshot()
        self._last_snapshot = snap
        self._step_demon(snap)

        # Phase progression
        smoothed_dr = self._update_phase(snap)

        # Graph deltas
        graph_deltas = {}
        cnt = len(self.agent.graph.edges)
        if cnt != self._prev_edge_count:
            graph_deltas[0] = [e.as_dict() for e in self.agent.graph.edges]
            self._prev_edge_count = cnt

        return self._snapshot(snap, graph_deltas, smoothed_dr)

    def _step_demon(self, snap: dict):
        demon_action = self.demon.step([snap], 1 - snap.get("peak_discovery_rate", 0))
        if demon_action is None:
            return
        corrupted = self.agent.demon_disrupt()
        self._add_event(
            f"⚠ Demon [{demon_action.get('mode','?')}] → {self.AGI_NAME}: {corrupted}",
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
                self._add_event(
                    f"⬆ Phase {potential}: {PHASE_NAMES[potential]}",
                    "#ffcc00", "phase"
                )
        else:
            self._candidate_phase    = self.max_phase
            self._phase_hold_counter = 0
        self.phase = self.max_phase
        return smoothed

    def _log_step_result(self, result: dict):
        if result.get("blocked"):
            self._add_event(
                f"🛡 [BLOCKED] {self.AGI_NAME}: {result.get('reason','?')}",
                "#884400", "value"
            )
        elif result.get("updated_edges"):
            cg = result.get("compression_delta", 0)
            var = result.get("variable", "?")
            val = result.get("value", 0)
            self._add_event(
                f"{self.AGI_NAME}: do({var}={val:.2f}) CG{'+' if cg>=0 else ''}{cg:.3f}",
                self.AGI_COLOR, "discovery"
            )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})

    # ── Camera ────────────────────────────────────────────────────────────────
    def get_camera_frame(self) -> str | None:
        """Кадр из текущей среды (если поддерживает)."""
        fn = getattr(self.agent.env, "get_frame_base64", None)
        return fn() if callable(fn) else None

    def get_robot_skeleton(self) -> list[dict] | None:
        """Joint positions для Three.js визуализации скелетона."""
        fn = getattr(self.agent.env, "get_joint_positions_world", None)
        return fn() if callable(fn) else None

    def get_robot_target(self) -> dict | None:
        fn = getattr(self.agent.env, "get_target", None)
        return fn() if callable(fn) else None

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _snapshot(self, snap: dict, graph_deltas: dict, smoothed_dr: float) -> dict:
        return {
            "tick":         self.tick,
            "phase":        self.phase,
            "max_phase":    self.max_phase,
            "entropy":      round((1 - snap.get("peak_discovery_rate", 0)) * 100, 1),
            "smoothed_dr":  round(smoothed_dr, 3),
            "agents":       [snap],   # массив из 1 элемента — совместимость с UI
            "n_agents":     1,
            "demon":        self.demon.snapshot,
            "tom_links":    [],
            "events":       list(self.events),
            "graph_deltas": graph_deltas,
            "value_layer":  {
                "total_blocked_all": snap.get("total_blocked", 0),
                "block_rates": [
                    round(snap.get("value_layer", {}).get("block_rate", 0), 3)
                ],
            },
            "byzantine":    None,
            "motif":        None,
            "multiprocess": False,
            "singleton":    True,
            "current_world":    self.current_world,
            "world_label":      WORLDS.get(self.current_world, {}).get("label", ""),
            "world_color":      WORLDS.get(self.current_world, {}).get("color", "#00ff99"),
            "worlds":           WORLDS,
            "switch_history":   self.switcher.history[-5:],
            "gnn_d":            self.agent.graph._d,
            "robot_skeleton":   self.get_robot_skeleton(),
            "robot_target":     self.get_robot_target(),
            "pybullet":    {
                "phi":             snap.get("phi", 0),
                "discovery_rate":  snap.get("discovery_rate", 0),
                "interventions":   snap.get("total_interventions", 0),
                "node_count":      snap.get("node_count", 0),
                "edge_count":      snap.get("edge_count", 0),
                "h_W":             snap.get("h_W", 0),
                "compression_gain":snap.get("compression_gain", 0),
                "objects":         [],
            },
        }

    def public_state(self) -> dict:
        snap = self._last_snapshot or self.agent.snapshot()
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        return self._snapshot(snap, {}, smoothed)

    def shutdown(self):
        pass