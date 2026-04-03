"""
simulation_singleton_v2.py — Singleton AGI с гуманоидом (Фаза 11).

Ключевые изменения:
  - start_world="humanoid" по умолчанию
  - WORLDS расширен (humanoid включён)
  - tick_step() передаёт is_fallen() агенту через контекст
  - full_scene() для UI (skeleton + cubes + camera)
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

        self.switcher = WorldSwitcher(self.agent, self.device)
        self.demon    = AdversarialDemon(n_agents=1, device=self.device)

        self.tick      = 0
        self.phase     = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase    = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events:    deque[dict]   = deque(maxlen=24)
        self._prev_edge_count = 0
        self._last_snapshot: dict     = {}

        # Статистика падений
        self._fall_count  = 0
        self._stand_ticks = 0

    # ── World switch ──────────────────────────────────────────────────────────
    def switch_world(self, new_world: str) -> dict:
        if new_world not in WORLDS:
            return {"error": f"unknown world: {new_world}"}
        result = self.switcher.switch(new_world)
        if result.get("switched"):
            self.current_world = new_world
            winfo = WORLDS[new_world]
            self._add_event(
                f"🌍 → {winfo['label']} "
                f"(+{len(result.get('new_nodes',[]))} vars, d={result.get('gnn_d')})",
                winfo["color"], "phase"
            )
        return result

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

        # Проверяем падение (гуманоид-специфично)
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

        self.agent.other_agents_phi = []
        result = self.agent.step(engine_tick=self.tick)
        self._log_step(result, fallen)

        snap = self.agent.snapshot()
        snap["fallen"] = fallen
        snap["fall_count"] = self._fall_count
        self._last_snapshot = snap

        # Demon feedback
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

        # Сцена (skeleton + cubes)
        scene_fn = getattr(self.agent.env, "get_full_scene", None)
        scene    = scene_fn() if callable(scene_fn) else {}

        return self._build_snapshot(snap, graph_deltas, smoothed, scene)

    def _step_demon(self, snap: dict):
        try:
            action = self.demon.step([snap], 1 - snap.get("peak_discovery_rate", 0))
        except RuntimeError as e:
            # Несовпадение размерности политики (например старый чекпойнт / другой n_agents)
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
            self._add_event(
                f"Nova: do({var}={val:.2f}) CG{'+' if cg>=0 else ''}{cg:.3f}",
                WORLDS.get(self.current_world, {}).get("color", "#cc44ff"),
                "discovery"
            )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})

    # ── Camera ────────────────────────────────────────────────────────────────
    def get_camera_frame(self, view: str = "diag") -> str | None:
        fn = getattr(self.agent.env, "get_frame_base64", None)
        return fn(view) if callable(fn) else None

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _build_snapshot(self, snap: dict, graph_deltas: dict, smoothed: float, scene: dict) -> dict:
        winfo = WORLDS.get(self.current_world, {"color": "#cc44ff", "label": self.current_world})
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
        }

    def public_state(self) -> dict:
        snap     = self._last_snapshot or self.agent.snapshot()
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        fn       = getattr(self.agent.env, "get_full_scene", None)
        scene    = fn() if callable(fn) else {}
        return self._build_snapshot(snap, {}, smoothed, scene)

    def shutdown(self):
        pass