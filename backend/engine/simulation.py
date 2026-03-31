"""
Simulation — Python оркестратор РКК v5.
"""
from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from collections import deque

from engine.environment import Environment
from engine.agent       import RKKAgent
from engine.demon       import AdversarialDemon

PHASE_THRESHOLDS = [0.0, 0.20, 0.40, 0.62, 0.80, 0.95]
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]
AGENT_COLORS     = ["#00ff99", "#0099ff", "#ff9900"]
AGENT_NAMES      = ["Nova", "Aether", "Lyra"]
ENV_PRESETS      = ["physics", "chemistry", "logic"]


@dataclass
class SimEvent:
    tick:  int
    text:  str
    color: str
    type:  str


class Simulation:
    def __init__(self, device_str: str = "cuda"):
        # HIP на Windows: torch.cuda → HIP автоматически
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        print(f"[RKK] Device: {self.device}")

        self.envs   = [Environment(p, self.device) for p in ENV_PRESETS]
        self.agents = [RKKAgent(i, AGENT_NAMES[i], self.envs[i], self.device)
                       for i in range(3)]
        self.demon  = AdversarialDemon(3, self.device)

        self.tick   = 0
        self.phase  = 1
        self.events: deque[SimEvent] = deque(maxlen=14)

        # ToM матрица [3×3]
        self.tom: list[list[float]] = [[0.0] * 3 for _ in range(3)]

        # Graph delta tracking (улучшение 1: WebSocket отправляет только дельты)
        self._prev_edge_counts = [0, 0, 0]

    # ── Один тик ──────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1

        # 1. Каждый агент делает шаг
        for agent in self.agents:
            result = agent.step()
            if result and result.get("updated_edges"):
                cg   = result["compression_delta"]
                sign = "+" if cg >= 0 else ""
                self._add_event(
                    f'{agent.name}: {agent._last_do} → CG {sign}{cg:.3f}',
                    AGENT_COLORS[agent.id],
                    "discovery"
                )
            # Обучаем Демона на ошибке предсказания
            if result:
                self.demon.learn(
                    result.get("prediction_error", 0),
                    self.demon._last_action_complexity
                )

        # 2. Шаг Демона
        snapshots    = [a.snapshot() for a in self.agents]
        mean_dr      = np.mean([s["discovery_rate"] for s in snapshots])
        env_entropy  = 1 - mean_dr
        demon_action = self.demon.step(snapshots, env_entropy)

        if demon_action is not None:
            target_id = demon_action["target_agent"]
            corrupted = self.agents[target_id].demon_disrupt()
            self._add_event(
                f'⚠ Demon [{demon_action["action_complexity"]:.2f} complexity] '
                f'→ {self.agents[target_id].name}: {corrupted}',
                "#ff2244", "demon"
            )

            # Phase IV: другие агенты защищают (ΔAutonomy ≥ 0)
            if self.phase >= 4:
                for i, agent in enumerate(self.agents):
                    if i != target_id:
                        strongest = sorted(
                            self.agents[target_id].graph.edges,
                            key=lambda e: e.alpha_trust, reverse=True
                        )[:2]
                        for e in strongest:
                            e.alpha_trust = min(0.98, e.alpha_trust + 0.05)
                self._add_event(
                    f'🛡 Phase IV: agents protect {self.agents[target_id].name}',
                    "#00ccff", "value"
                )

        # 3. Theory of Mind (каждые 25 тиков, Фаза III+)
        if self.tick % 25 == 0 and self.phase >= 3:
            self._update_tom()

        # 4. Phi recording
        for agent in self.agents:
            agent.record_phi(agent.discovery_rate)

        # 5. Phase progression
        new_phase = 1
        for threshold in PHASE_THRESHOLDS:
            if mean_dr >= threshold:
                new_phase += 1
        new_phase = min(new_phase - 1, 5)

        if new_phase > self.phase:
            self.phase = new_phase
            self._add_event(
                f'⬆ Phase {new_phase}: {PHASE_NAMES[new_phase]} unlocked',
                "#ffcc00", "phase"
            )

        # 6. Graph deltas
        graph_deltas = {}
        for i, agent in enumerate(self.agents):
            current_count = len(agent.graph.edges)
            if current_count != self._prev_edge_counts[i]:
                graph_deltas[i] = [
                    {"from_": e.from_, "to": e.to,
                     "weight": round(e.weight, 3), "alpha_trust": round(e.alpha_trust, 3),
                     "intervention_count": e.intervention_count}
                    for e in agent.graph.edges
                ]
                self._prev_edge_counts[i] = current_count

        return self._snapshot(graph_deltas)

    # ── ToM update ────────────────────────────────────────────────────────────
    def _update_tom(self):
        for i in range(3):
            for j in range(i + 1, 3):
                phi_i = self.agents[i].phi_approx()
                phi_j = self.agents[j].phi_approx()
                strength = (phi_i + phi_j) / 2
                self.tom[i][j] = strength
                self.tom[j][i] = strength
                if strength > 0.5 and np.random.rand() < 0.25:
                    self._add_event(
                        f'ToM: {AGENT_NAMES[i]} ↔ {AGENT_NAMES[j]} '
                        f'Φ-sync {strength*100:.0f}%',
                        "#003388", "tom"
                    )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft(SimEvent(tick=self.tick, text=text, color=color, type=type_))

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _snapshot(self, graph_deltas: dict) -> dict:
        snapshots = [a.snapshot() for a in self.agents]
        mean_dr   = np.mean([s["discovery_rate"] for s in snapshots])

        tom_links = []
        for i in range(3):
            for j in range(i + 1, 3):
                if self.tom[i][j] > 0.25:
                    tom_links.append({"a": i, "b": j, "strength": round(self.tom[i][j], 3)})

        return {
            "tick":          self.tick,
            "phase":         self.phase,
            "entropy":       round((1 - mean_dr) * 100, 1),
            "agents":        snapshots,
            "demon":         self.demon.snapshot,
            "tom_links":     tom_links,
            "events":        [
                {"tick": e.tick, "text": e.text, "color": e.color, "type": e.type}
                for e in self.events
            ],
            "graph_deltas":  graph_deltas,
        }
