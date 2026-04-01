"""
simulation_v2.py — оркестратор с Value Layer.

Изменения:
  - Передаём other_agents_phi каждому агенту перед step()
  - Логируем [BLOCKED BY VALUE LAYER] события
  - inject_seeds() для LLM/RAG прiorов
  - value_layer stats в snapshot
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque

from engine.environment  import Environment
from engine.agent        import RKKAgent       # agent_v4
from engine.demon        import AdversarialDemon
from engine.value_layer  import HomeostaticBounds

PHASE_THRESHOLDS = [0.0, 0.18, 0.38, 0.58, 0.76, 0.92]
PHASE_HOLD_TICKS = 15
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]
AGENT_COLORS     = ["#00ff99", "#0099ff", "#ff9900"]
AGENT_NAMES      = ["Nova", "Aether", "Lyra"]
ENV_PRESETS      = ["physics", "chemistry", "logic"]


class Simulation:
    def __init__(self, device_str: str = "cuda"):
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        print(f"[RKK] Device: {self.device}")

        # Гомеостатические ограничения: прогрев (мягко) → blend → рабочий режим (steady).
        # Пороги по тику: см. effective_vl_state() в value_layer.py
        bounds = HomeostaticBounds(
            var_min=0.05, var_max=0.95,
            phi_min=0.01,
            h_slow_max=12.0,
            env_entropy_max_delta=0.95,
            warmup_ticks=2000,
            blend_ticks=600,
            phi_min_steady=0.05,
            env_entropy_max_delta_steady=0.55,
            h_slow_max_steady=10.0,
            predict_band_edge_steady=0.02,
        )

        self.envs   = [Environment(p, self.device) for p in ENV_PRESETS]
        self.agents = [
            RKKAgent(i, AGENT_NAMES[i], self.envs[i], self.device, bounds)
            for i in range(3)
        ]
        self.demon  = AdversarialDemon(3, self.device)

        self.tick      = 0
        self.phase     = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase    = 1
        self._dr_window: deque[float] = deque(maxlen=20)

        self.events: deque[dict] = deque(maxlen=16)
        self.tom: list[list[float]] = [[0.0] * 3 for _ in range(3)]
        self._prev_edge_counts = [0, 0, 0]

    # ── LLM/RAG seed injection ────────────────────────────────────────────────
    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        """
        Загружаем text priors от LLM/RAG в агента.

        edges: [{"from_": "Temp", "to": "Pressure", "weight": 0.8}, ...]

        Возвращает результат инъекции.
        """
        if agent_id >= len(self.agents):
            return {"error": "invalid agent_id"}

        result = self.agents[agent_id].inject_text_priors(edges)
        n = result["injected"]
        self._add_event(
            f"💉 Seeds injected → {AGENT_NAMES[agent_id]}: {n} edges (α=0.05)",
            "#886600", "discovery"
        )
        return {
            "injected": n,
            "agent":    AGENT_NAMES[agent_id],
            "skipped":  result["skipped"],
            "node_ids": result["node_ids"],
        }

    # ── Один тик ──────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1

        # 0. Передаём phi других агентов каждому агенту (для ΔΦ≥0)
        phis = [a.phi_approx() for a in self.agents]
        for i, agent in enumerate(self.agents):
            agent.other_agents_phi = [p for j, p in enumerate(phis) if j != i]

        # 1. Шаги агентов (тик симуляции — для фазы прогрева Value Layer)
        for agent in self.agents:
            result = agent.step(engine_tick=self.tick)

            if result.get("blocked"):
                # Все кандидаты заблокированы Value Layer
                reason = result.get("reason", "unknown")
                self._add_event(
                    f"🛡 [BLOCKED] {agent.name}: do({result.get('variable','?')}="
                    f"{result.get('value', 0):.2f}) — {reason} "
                    f"(tried {result.get('blocked_count', 0)})",
                    "#884400", "value"
                )
            elif result.get("blocked_count", 0) > 0:
                # Были блокировки, но в итоге нашли допустимое действие
                self._add_event(
                    f"⚡ {agent.name}: {agent._last_do} "
                    f"(skipped {result['blocked_count']} unsafe actions)",
                    AGENT_COLORS[agent.id], "discovery"
                )
            elif result.get("updated_edges"):
                cg   = result["compression_delta"]
                sign = "+" if cg >= 0 else ""
                self._add_event(
                    f"{agent.name}: {agent._last_do} → CG {sign}{cg:.3f}",
                    AGENT_COLORS[agent.id], "discovery"
                )

            if result and not result.get("blocked") and not result.get("skipped"):
                self.demon.learn(
                    result.get("prediction_error", 0),
                    self.demon._last_action_complexity
                )

        # 2. Шаг Демона
        snapshots    = [a.snapshot() for a in self.agents]
        mean_peak    = np.mean([s["peak_discovery_rate"] for s in snapshots])
        env_entropy  = 1 - mean_peak

        demon_action = self.demon.step(snapshots, env_entropy)
        if demon_action is not None:
            tid      = demon_action["target_agent"]
            corrupted = self.agents[tid].demon_disrupt()
            self._add_event(
                f"⚠ Demon [C={demon_action['action_complexity']:.2f}] "
                f"→ {AGENT_NAMES[tid]}: {corrupted}",
                "#ff2244", "demon"
            )
            # Phase IV: защита (ΔΦ≥0)
            if self.phase >= 4:
                for i, agent in enumerate(self.agents):
                    if i != tid:
                        strongest = sorted(
                            self.agents[tid].graph.edges,
                            key=lambda e: e.alpha_trust, reverse=True
                        )[:2]
                        for e in strongest:
                            e.alpha_trust = min(0.98, e.alpha_trust + 0.05)
                self._add_event(
                    f"🛡 Phase IV: agents protect {AGENT_NAMES[tid]} (ΔΦ≥0)",
                    "#00ccff", "value"
                )

        # 3. Theory of Mind
        if self.tick % 25 == 0 and self.phase >= 3:
            self._update_tom()

        # 4. Phase progression
        mean_cur = np.mean([s["discovery_rate"] for s in snapshots])
        self._dr_window.append(mean_cur)
        smoothed_dr = float(np.mean(self._dr_window))

        potential = 1
        for i, t in enumerate(PHASE_THRESHOLDS):
            if smoothed_dr >= t:
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
                    f"⬆ Phase {potential}: {PHASE_NAMES[potential]} unlocked",
                    "#ffcc00", "phase"
                )
        else:
            self._candidate_phase    = self.max_phase
            self._phase_hold_counter = 0

        self.phase = self.max_phase

        # 5. Graph deltas
        graph_deltas = {}
        for i, agent in enumerate(self.agents):
            cnt = len(agent.graph.edges)
            if cnt != self._prev_edge_counts[i]:
                graph_deltas[i] = [e.as_dict() for e in agent.graph.edges]
                self._prev_edge_counts[i] = cnt

        return self._snapshot(graph_deltas, smoothed_dr)

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
                        f"ToM: {AGENT_NAMES[i]} ↔ {AGENT_NAMES[j]} Φ-sync {strength*100:.0f}%",
                        "#003388", "tom"
                    )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})

    def _snapshot(self, graph_deltas: dict, smoothed_dr: float = 0.0) -> dict:
        snapshots = [a.snapshot() for a in self.agents]
        mean_peak = np.mean([s["peak_discovery_rate"] for s in snapshots])

        # Суммарная статистика Value Layer по всем агентам
        total_blocked = sum(s.get("total_blocked", 0) for s in snapshots)
        vl_stats = {
            "total_blocked_all": total_blocked,
            "block_rates": [
                round(s.get("value_layer", {}).get("block_rate", 0), 3)
                for s in snapshots
            ],
        }

        tom_links = [
            {"a": i, "b": j, "strength": round(self.tom[i][j], 3)}
            for i in range(3) for j in range(i+1, 3)
            if self.tom[i][j] > 0.25
        ]

        return {
            "tick":          self.tick,
            "phase":         self.phase,
            "max_phase":     self.max_phase,
            "entropy":       round((1 - mean_peak) * 100, 1),
            "smoothed_dr":   round(smoothed_dr, 3),
            "agents":        snapshots,
            "demon":         self.demon.snapshot,
            "tom_links":     tom_links,
            "events":        list(self.events),
            "graph_deltas":  graph_deltas,
            "value_layer":   vl_stats,
        }