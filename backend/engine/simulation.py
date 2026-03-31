"""
Simulation — Python оркестратор РКК v5.

Исправления:
  - Фаза монотонно растёт (max_phase никогда не снижается)
  - Гистерезис: фаза повышается только при устойчивом discovery_rate
  - peak_discovery_rate у агентов не уменьшается
"""
from __future__ import annotations
import torch
import numpy as np
from collections import deque

from engine.environment import Environment
from engine.agent       import RKKAgent
from engine.demon       import AdversarialDemon

# Порог для повышения фазы — нужно удержать N тиков подряд
PHASE_THRESHOLDS = [0.0, 0.18, 0.38, 0.58, 0.76, 0.92]
PHASE_HOLD_TICKS = 15   # сколько тиков подряд держать порог перед повышением

PHASE_NAMES  = ["", "Causal Crib", "Robotic Explorer",
                "Social Sandbox", "Value Lock", "Open Reality"]
AGENT_COLORS = ["#00ff99", "#0099ff", "#ff9900"]
AGENT_NAMES  = ["Nova", "Aether", "Lyra"]
ENV_PRESETS  = ["physics", "chemistry", "logic"]


class Simulation:
    def __init__(self, device_str: str = "cuda"):
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        print(f"[RKK] Device: {self.device}")

        self.envs   = [Environment(p, self.device) for p in ENV_PRESETS]
        self.agents = [RKKAgent(i, AGENT_NAMES[i], self.envs[i], self.device)
                       for i in range(3)]
        self.demon  = AdversarialDemon(3, self.device)

        self.tick      = 0
        self.phase     = 1
        self.max_phase = 1  # ← никогда не снижается

        # Гистерезис: считаем тики выше порога перед повышением
        self._phase_hold_counter = 0
        self._candidate_phase    = 1

        # Скользящее среднее discovery_rate (сглаживает флуктуации)
        self._dr_window: deque[float] = deque(maxlen=20)

        self.events: deque[dict] = deque(maxlen=14)
        self.tom: list[list[float]] = [[0.0] * 3 for _ in range(3)]
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
            if result:
                self.demon.learn(
                    result.get("prediction_error", 0),
                    self.demon._last_action_complexity
                )

        # 2. Шаг Демона
        snapshots   = [a.snapshot() for a in self.agents]
        # Используем peak discovery rate для entropy (не флуктуирует)
        mean_peak   = np.mean([s["peak_discovery_rate"] for s in snapshots])
        mean_cur    = np.mean([s["discovery_rate"]      for s in snapshots])
        env_entropy = 1 - mean_peak

        demon_action = self.demon.step(snapshots, env_entropy)
        if demon_action is not None:
            target_id = demon_action["target_agent"]
            corrupted = self.agents[target_id].demon_disrupt()
            self._add_event(
                f'⚠ Demon [C={demon_action["action_complexity"]:.2f}] '
                f'→ {self.agents[target_id].name}: {corrupted}',
                "#ff2244", "demon"
            )
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

        # 3. Theory of Mind
        if self.tick % 25 == 0 and self.phase >= 3:
            self._update_tom()

        # 4. Phi recording
        for agent in self.agents:
            agent.record_phi(agent.discovery_rate)

# 5. Phase progression — монотонная с гистерезисом
        self._dr_window.append(mean_cur)
        smoothed_dr = float(np.mean(self._dr_window))

        # Определяем потенциальную фазу по сглаженному discovery rate
        potential_target = 1
        for i, threshold in enumerate(PHASE_THRESHOLDS):
            if smoothed_dr >= threshold:
                potential_target = i + 1
        potential_target = min(potential_target, 5)

        # ЛОГИКА ГИСТЕРЕЗИСА:
        # Если потенциальная фаза выше той, что мы уже достигли
        if potential_target > self.max_phase:
            if potential_target == self._candidate_phase:
                # Мы продолжаем удерживать порог новой фазы
                self._phase_hold_counter += 1
            else:
                # Мы переключились на нового кандидата (или только начали расти)
                self._candidate_phase = potential_target
                self._phase_hold_counter = 1

            # Если продержались достаточно долго — повышаем навсегда
            if self._phase_hold_counter >= PHASE_HOLD_TICKS:
                self.max_phase = potential_target
                self.phase     = potential_target # Синхронизируем
                self._phase_hold_counter = 0
                self._add_event(
                    f'⬆ Phase {potential_target}: {PHASE_NAMES[potential_target]} unlocked '
                    f'(DR={smoothed_dr:.2f})',
                    "#ffcc00", "phase"
                )
        else:
            # Если текущий DR упал ниже порога max_phase, мы просто сбрасываем 
            # попытку дальнейшего роста, но САМУ ФАЗУ НЕ СНИЖАЕМ.
            self._candidate_phase = self.max_phase
            self._phase_hold_counter = 0
            
        # Гарантируем, что визуальная фаза всегда равна максимальной достигнутой
        self.phase = self.max_phase

        # 6. Graph deltas
        graph_deltas = {}
        for i, agent in enumerate(self.agents):
            current_count = len(agent.graph.edges)
            if current_count != self._prev_edge_counts[i]:
                graph_deltas[i] = [
                    {
                        "from_":             e.from_,
                        "to":                e.to,
                        "weight":            round(e.weight, 3),
                        "alpha_trust":       round(e.alpha_trust, 3),
                        "intervention_count": e.intervention_count,
                    }
                    for e in agent.graph.edges
                ]
                self._prev_edge_counts[i] = current_count

        return self._snapshot(graph_deltas, smoothed_dr)

    # ── ToM ───────────────────────────────────────────────────────────────────
    def _update_tom(self):
        for i in range(3):
            for j in range(i + 1, 3):
                phi_i    = self.agents[i].phi_approx()
                phi_j    = self.agents[j].phi_approx()
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
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _snapshot(self, graph_deltas: dict, smoothed_dr: float = 0.0) -> dict:
        snapshots = [a.snapshot() for a in self.agents]
        mean_peak = np.mean([s["peak_discovery_rate"] for s in snapshots])

        tom_links = [
            {"a": i, "b": j, "strength": round(self.tom[i][j], 3)}
            for i in range(3) for j in range(i + 1, 3)
            if self.tom[i][j] > 0.25
        ]

        return {
            "tick":         self.tick,
            "phase":        self.phase,
            "max_phase":    self.max_phase,
            "entropy":      round((1 - mean_peak) * 100, 1),
            "smoothed_dr":  round(smoothed_dr, 3),
            "agents":       snapshots,
            "demon":        self.demon.snapshot,
            "tom_links":    tom_links,
            "events":       list(self.events),
            "graph_deltas": graph_deltas,
        }