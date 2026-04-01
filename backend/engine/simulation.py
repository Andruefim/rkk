"""
simulation_v3.py — оркестратор с Byzantine консенсусом и Motif Transfer.

Два режима работы (переключается в __init__):

  SINGLE_PROCESS (по умолчанию):
    Все агенты в одном процессе — как раньше.
    Byzantine и Motif Transfer работают напрямую через объекты.

  MULTI_PROCESS (опционально):
    AgentPool запускает 3 дочерних процесса.
    Byzantine консенсус работает через JSON-обмен W-матрицами.
    Включается через Simulation(multiprocess=True).
    Требует Windows spawn-context.

В обоих режимах публичный интерфейс одинаков:
  sim.tick_step() → dict
  sim.inject_seeds(agent_id, edges) → dict
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque

from engine.environment  import Environment
from engine.agent        import RKKAgent
from engine.demon        import AdversarialDemon
from engine.value_layer  import HomeostaticBounds
from engine.byzantine    import (
    ByzantineConsensus, MotifTransfer,
    gather_consensus_data, CONSENSUS_EVERY, MOTIF_EVERY,
)

PHASE_THRESHOLDS = [0.0, 0.18, 0.38, 0.58, 0.76, 0.92]
PHASE_HOLD_TICKS = 15
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]
AGENT_COLORS     = ["#00ff99", "#0099ff", "#ff9900"]
AGENT_NAMES      = ["Nova", "Aether", "Lyra"]
ENV_PRESETS      = ["physics", "chemistry", "logic"]


def _default_bounds() -> HomeostaticBounds:
    return HomeostaticBounds(
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


class Simulation:
    def __init__(self, device_str: str = "cuda", multiprocess: bool = True):
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self.multiprocess = multiprocess
        print(f"[RKK] Device: {self.device} | Multiprocess: {multiprocess}")

        bounds = _default_bounds()

        # ── Single-process агенты ────────────────────────────────────────────
        self._pool: "AgentPool | None" = None

        if multiprocess:
            from engine.agent_worker import AgentPool
            import dataclasses
            bounds_dict = dataclasses.asdict(bounds)
            self._pool = AgentPool(device_str=device_str, bounds_dict=bounds_dict)
            ok = self._pool.start()
            if not ok:
                print("[RKK] Multiprocess start failed, falling back to single-process")
                self.multiprocess = False
                self._pool = None

        if not self.multiprocess:
            self.envs   = [Environment(p, self.device) for p in ENV_PRESETS]
            self.agents = [
                RKKAgent(i, AGENT_NAMES[i], self.envs[i], self.device, bounds)
                for i in range(3)
            ]

        self.demon = AdversarialDemon(3, self.device)

        # ── Byzantine консенсус ──────────────────────────────────────────────
        self.byzantine = ByzantineConsensus(n_agents=3)
        self.motif     = MotifTransfer(n_agents=3)

        self.tick      = 0
        self.phase     = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase    = 1
        self._dr_window: deque[float] = deque(maxlen=20)

        self.events: deque[dict] = deque(maxlen=18)
        self.tom: list[list[float]] = [[0.0] * 3 for _ in range(3)]
        self._prev_edge_counts = [0, 0, 0]

        # Кэш снапшотов для multi-process режима
        self._mp_snapshots: list[dict] = [{} for _ in range(3)]

    # ── Seed injection ────────────────────────────────────────────────────────
    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        if agent_id >= 3:
            return {"error": "invalid agent_id"}

        if self.multiprocess and self._pool:
            result = self._pool.inject_seeds(agent_id, edges)
        else:
            result = self.agents[agent_id].inject_text_priors(edges)

        n = result.get("injected", 0)
        self._add_event(
            f"💉 Seeds → {AGENT_NAMES[agent_id]}: {n} edges (α=0.05)",
            "#886600", "discovery"
        )
        return {
            "injected": n,
            "agent":    AGENT_NAMES[agent_id],
            "skipped":  result.get("skipped", []),
            "node_ids": result.get("node_ids", []),
        }

    # ── Один тик ──────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1

        if self.multiprocess:
            return self._tick_multiprocess()
        else:
            return self._tick_single()

    # ── Single-process тик ────────────────────────────────────────────────────
    def _tick_single(self) -> dict:
        # Phi других агентов для ΔΦ≥0
        phis = [a.phi_approx() for a in self.agents]
        for i, agent in enumerate(self.agents):
            agent.other_agents_phi = [p for j, p in enumerate(phis) if j != i]

        # Шаги агентов
        for agent in self.agents:
            result = agent.step(engine_tick=self.tick)
            self._log_step_result(agent, result)
            if result and not result.get("blocked") and not result.get("skipped"):
                self.demon.learn(
                    result.get("prediction_error", 0),
                    self.demon._last_action_complexity,
                )

        # Demon
        snapshots   = [a.snapshot() for a in self.agents]
        self._step_demon(snapshots)

        # ToM
        if self.tick % 25 == 0 and self.phase >= 3:
            self._update_tom_single()

        # Byzantine консенсус (каждые CONSENSUS_EVERY тиков)
        if self.tick % CONSENSUS_EVERY == 0:
            self._run_byzantine_single()

        # Motif Transfer (каждые MOTIF_EVERY тиков)
        if self.tick % MOTIF_EVERY == 0:
            self._run_motif_single()

        # Phase progression
        snapshots   = [a.snapshot() for a in self.agents]
        smoothed_dr = self._update_phase(snapshots)

        # Graph deltas
        graph_deltas = self._compute_deltas_single()

        return self._snapshot(snapshots, graph_deltas, smoothed_dr)

    # ── Multi-process тик ─────────────────────────────────────────────────────
    def _tick_multiprocess(self) -> dict:
        # Шаги всех агентов параллельно
        results = self._pool.step_all(tick=self.tick)
        snapshots = []
        for msg in results:
            aid  = msg.get("agent_id", 0)
            snap = msg.get("snapshot", {})
            self._mp_snapshots[aid] = snap
            snapshots.append(snap)
            # Логируем
            result = msg.get("result", {})
            self._log_step_result_mp(aid, result)

        # Demon (в главном процессе)
        self._step_demon(snapshots)

        # Byzantine (каждые CONSENSUS_EVERY тиков)
        if self.tick % CONSENSUS_EVERY == 0:
            self._run_byzantine_multiprocess()

        # Motif Transfer (каждые MOTIF_EVERY тиков)
        if self.tick % MOTIF_EVERY == 0:
            self._run_motif_multiprocess()

        # Phase
        smoothed_dr = self._update_phase(snapshots)

        # Graph deltas — берём из снапшотов (уже содержат edges)
        graph_deltas = {}
        for i, snap in enumerate(snapshots):
            edge_cnt = snap.get("edge_count", 0)
            if edge_cnt != self._prev_edge_counts[i]:
                graph_deltas[i] = snap.get("edges", [])
                self._prev_edge_counts[i] = edge_cnt

        return self._snapshot(snapshots, graph_deltas, smoothed_dr)

    # ── Byzantine (single) ────────────────────────────────────────────────────
    def _run_byzantine_single(self):
        data   = gather_consensus_data(self.agents)
        result = self.byzantine.run(self.agents, data, self.device)

        if result["type"] == "consensus_ok":
            n  = result["updates_applied"]
            md = result["mean_deviance"]
            self._add_event(
                f"🗳 Byzantine round {result['round']}: "
                f"{n} edges updated · dev={md:.4f}",
                "#004466", "tom"
            )
        elif result["type"] == "consensus_skip":
            self._add_event(
                f"🗳 Byzantine skip: {result['reason']}",
                "#223344", "tom"
            )

    # ── Byzantine (multi) ─────────────────────────────────────────────────────
    def _run_byzantine_multiprocess(self):
        raw_data = self._pool.get_consensus_data()
        if len(raw_data) < 2:
            return

        # Строим Byzantine данные из JSON
        eligible = [d for d in raw_data if d.get("phi", 0) > 0.05 and d.get("W")]
        if len(eligible) < 2:
            return

        phi_total = sum(d["phi"] for d in eligible)
        total_updates = 0

        for target in eligible:
            peers = [d for d in eligible if d["agent_id"] != target["agent_id"]]
            if not peers:
                continue

            target_ids = target["node_ids"]
            target_W   = torch.tensor(target["W"])

            for my_i, nid in enumerate(target_ids):
                for my_j, nid2 in enumerate(target_ids):
                    consensus_w   = 0.0
                    consensus_phi = 0.0

                    for peer in peers:
                        peer_ids = peer["node_ids"]
                        if nid in peer_ids and nid2 in peer_ids:
                            pi = peer_ids.index(nid)
                            pj = peer_ids.index(nid2)
                            consensus_w   += peer["W"][pi][pj] * peer["phi"]
                            consensus_phi += peer["phi"]

                    if consensus_phi < 1e-6:
                        continue

                    consensus_w /= consensus_phi
                    deviance = abs(target_W[my_i, my_j].item() - consensus_w)

                    if deviance > 0.25:
                        self._pool.apply_alpha_decay(
                            target["agent_id"], nid, nid2, decay=0.12
                        )
                        total_updates += 1

        self._add_event(
            f"🗳 Byzantine MP round {self.byzantine.round}: {total_updates} edges",
            "#004466", "tom"
        )
        self.byzantine.round += 1

    # ── Motif Transfer (single) ───────────────────────────────────────────────
    def _run_motif_single(self):
        data   = gather_consensus_data(self.agents)
        result = self.motif.run(self.agents, data, self.device)

        if result["type"] == "motif_transfer":
            donor_name = AGENT_NAMES[result["donor_id"]]
            self._add_event(
                f"🧬 Motif Transfer: {donor_name} → others "
                f"(CG={result['donor_cg']:.4f}, EMA={result['ema']})",
                "#224466", "tom"
            )

    # ── Motif Transfer (multi) ────────────────────────────────────────────────
    def _run_motif_multiprocess(self):
        raw_data = self._pool.get_consensus_data()
        eligible = [d for d in raw_data
                    if d.get("cg", 0) > 0
                    and d.get("block_rate", 1) < 0.5]
        if not eligible:
            return

        donor = max(eligible, key=lambda d: d["cg"])
        donor_snap = self._mp_snapshots[donor["agent_id"]]
        # s1 state_dict недоступен напрямую в multi — пропускаем
        # (в полной версии нужно добавить cmd "get_s1_weights")
        self._add_event(
            f"🧬 Motif Transfer MP: {AGENT_NAMES[donor['agent_id']]} → others",
            "#224466", "tom"
        )

    # ── Demon step ────────────────────────────────────────────────────────────
    def _step_demon(self, snapshots: list[dict]):
        mean_peak   = np.mean([s.get("peak_discovery_rate", 0) for s in snapshots])
        env_entropy = 1 - mean_peak

        demon_action = self.demon.step(snapshots, env_entropy)
        if demon_action is None:
            return

        tid = demon_action["target_agent"]

        if self.multiprocess:
            corrupted = self._pool.demon_disrupt(tid)
        else:
            corrupted = self.agents[tid].demon_disrupt()

        self._add_event(
            f"⚠ Demon [C={demon_action['action_complexity']:.2f}] "
            f"→ {AGENT_NAMES[tid]}: {corrupted}",
            "#ff2244", "demon"
        )

        # Phase IV: ΔΦ≥0 защита
        if self.phase >= 4:
            if not self.multiprocess:
                for i, agent in enumerate(self.agents):
                    if i != tid:
                        for edge in sorted(
                            self.agents[tid].graph.edges,
                            key=lambda e: e.alpha_trust, reverse=True
                        )[:2]:
                            edge.alpha_trust = min(0.98, edge.alpha_trust + 0.05)
            self._add_event(
                f"🛡 Phase IV: ΔΦ≥0 protection → {AGENT_NAMES[tid]}",
                "#00ccff", "value"
            )

    # ── ToM (single) ──────────────────────────────────────────────────────────
    def _update_tom_single(self):
        for i in range(3):
            for j in range(i + 1, 3):
                phi_i    = self.agents[i].phi_approx()
                phi_j    = self.agents[j].phi_approx()
                strength = (phi_i + phi_j) / 2
                self.tom[i][j] = strength
                self.tom[j][i] = strength
                if strength > 0.5 and np.random.rand() < 0.25:
                    self._add_event(
                        f"ToM: {AGENT_NAMES[i]} ↔ {AGENT_NAMES[j]} "
                        f"Φ-sync {strength*100:.0f}%",
                        "#003388", "tom"
                    )

    # ── Phase progression ─────────────────────────────────────────────────────
    def _update_phase(self, snapshots: list[dict]) -> float:
        mean_cur = np.mean([s.get("discovery_rate", 0) for s in snapshots])
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
        return smoothed_dr

    # ── Graph deltas (single) ─────────────────────────────────────────────────
    def _compute_deltas_single(self) -> dict:
        deltas = {}
        for i, agent in enumerate(self.agents):
            cnt = len(agent.graph.edges)
            if cnt != self._prev_edge_counts[i]:
                deltas[i] = [e.as_dict() for e in agent.graph.edges]
                self._prev_edge_counts[i] = cnt
        return deltas

    # ── Логирование ───────────────────────────────────────────────────────────
    def _log_step_result(self, agent, result: dict):
        if result.get("blocked"):
            reason = result.get("reason", "?")
            self._add_event(
                f"🛡 [BLOCKED] {agent.name}: {reason} "
                f"(tried {result.get('blocked_count', 0)})",
                "#884400", "value"
            )
        elif result.get("blocked_count", 0) > 0:
            self._add_event(
                f"⚡ {agent.name}: {agent._last_do} "
                f"(skipped {result['blocked_count']})",
                AGENT_COLORS[agent.id], "discovery"
            )
        elif result.get("updated_edges"):
            cg = result["compression_delta"]
            self._add_event(
                f"{agent.name}: {agent._last_do} → CG {'+' if cg>=0 else ''}{cg:.3f}",
                AGENT_COLORS[agent.id], "discovery"
            )

    def _log_step_result_mp(self, agent_id: int, result: dict):
        name = AGENT_NAMES[agent_id]
        col  = AGENT_COLORS[agent_id]
        if result.get("blocked"):
            self._add_event(
                f"🛡 [BLOCKED] {name}: {result.get('reason','?')}",
                "#884400", "value"
            )
        elif result.get("updated_edges"):
            cg = result.get("compression_delta", 0)
            self._add_event(
                f"{name}: {result.get('variable','?')}={result.get('value',0):.2f} "
                f"CG {'+' if cg>=0 else ''}{cg:.3f}",
                col, "discovery"
            )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({
            "tick": self.tick, "text": text, "color": color, "type": type_
        })

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _snapshot(
        self,
        snapshots:    list[dict],
        graph_deltas: dict,
        smoothed_dr:  float,
    ) -> dict:
        mean_peak     = np.mean([s.get("peak_discovery_rate", 0) for s in snapshots])
        total_blocked = sum(s.get("total_blocked", 0) for s in snapshots)

        tom_links = [
            {"a": i, "b": j, "strength": round(self.tom[i][j], 3)}
            for i in range(3) for j in range(i+1, 3)
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
            "value_layer":  {
                "total_blocked_all": total_blocked,
                "block_rates": [
                    round(s.get("value_layer", {}).get("block_rate", 0), 3)
                    for s in snapshots
                ],
            },
            "byzantine": self.byzantine.snapshot(),
            "motif":     self.motif.snapshot(),
            "multiprocess": self.multiprocess,
        }

    def shutdown(self):
        """Корректно останавливаем пул (если multiprocess)."""
        if self._pool:
            self._pool.stop()