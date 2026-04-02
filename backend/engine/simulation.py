"""
simulation_v3.py — оркестратор: только multiprocess (AgentPool).

Три воркера-агента, Byzantine / Motif / Demon в главном процессе.
Требует Windows spawn + успешный старт пула.
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque

from engine.environment  import Environment
from engine.demon        import AdversarialDemon
from engine.value_layer  import HomeostaticBounds
from engine.byzantine    import (
    ByzantineConsensus, MotifTransfer,
    CONSENSUS_EVERY, MOTIF_EVERY,
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
    def __init__(self, device_str: str = "cuda"):
        self.device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        print(f"[RKK] Device: {self.device} | AgentPool (multiprocess only)")

        bounds = _default_bounds()
        import dataclasses
        from engine.agent_worker import AgentPool

        bounds_dict = dataclasses.asdict(bounds)
        self._pool = AgentPool(device_str=device_str, bounds_dict=bounds_dict)
        if not self._pool.start():
            raise RuntimeError(
                "AgentPool failed to start. Check GPU/worker logs; single-process mode removed."
            )

        self.demon = AdversarialDemon(3, self.device)

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

        self._mp_snapshots: list[dict] = [{} for _ in range(3)]

    # ── Seed injection ────────────────────────────────────────────────────────
    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        if agent_id >= 3:
            return {"error": "invalid agent_id"}

        result = self._pool.inject_seeds(agent_id, edges)

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

    def agent_seed_context(self, agent_id: int) -> dict | None:
        """Имя, пресет и переменные среды (как у воркеров)."""
        if agent_id < 0 or agent_id >= 3:
            return None
        preset = ENV_PRESETS[agent_id]
        name   = AGENT_NAMES[agent_id]
        env    = Environment(preset, self.device)
        return {
            "name":      name,
            "preset":    preset,
            "variables": list(env.variable_ids),
        }

    # ── Один тик ──────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1

        results = self._pool.step_all(tick=self.tick)
        snapshots = []
        for msg in results:
            aid  = msg.get("agent_id", 0)
            snap = msg.get("snapshot", {})
            self._mp_snapshots[aid] = snap
            snapshots.append(snap)
            result = msg.get("result", {})
            self._log_step_result_mp(aid, result)

        # Упорядоченные снапшоты [0,1,2] для демона и Value Layer
        ordered_snaps = [self._mp_snapshots[i] for i in range(3)]

        # Обратная связь демона за прошлый тик (всегда снимаем _last_action)
        if self.demon._last_action is not None:
            tid = self.demon._last_action["target_agent"]
            pe  = 0.0
            res = None
            for msg in results:
                if msg.get("status") != "step_done":
                    continue
                if msg.get("agent_id") != tid:
                    continue
                res = msg.get("result", {})
                break
            if res is not None:
                if not res.get("blocked") and not res.get("skipped"):
                    pe = float(res.get("prediction_error", 0))
                self.demon.learn(
                    pe,
                    self.demon._last_action_complexity,
                    ordered_snaps,
                )

        self._step_demon(ordered_snaps)

        if self.tick % 25 == 0 and self.phase >= 3:
            self._update_tom_from_snapshots(ordered_snaps)

        if self.tick % CONSENSUS_EVERY == 0:
            self._run_byzantine_multiprocess()

        if self.tick % MOTIF_EVERY == 0:
            self._run_motif_multiprocess()

        smoothed_dr = self._update_phase(ordered_snaps)

        graph_deltas = {}
        for i, snap in enumerate(ordered_snaps):
            edge_cnt = snap.get("edge_count", 0)
            if edge_cnt != self._prev_edge_counts[i]:
                graph_deltas[i] = snap.get("edges", [])
                self._prev_edge_counts[i] = edge_cnt

        return self._snapshot(ordered_snaps, graph_deltas, smoothed_dr)

    # ── Byzantine (multi) ─────────────────────────────────────────────────────
    def _run_byzantine_multiprocess(self):
        raw_data = self._pool.get_consensus_data()
        if len(raw_data) < 2:
            return

        eligible = [d for d in raw_data if d.get("phi", 0) > 0.05 and d.get("W")]
        if len(eligible) < 2:
            return

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

    # ── Motif Transfer (multi) ────────────────────────────────────────────────
    def _run_motif_multiprocess(self):
        raw_data = self._pool.get_consensus_data()
        eligible = [d for d in raw_data
                    if d.get("cg", 0) > 0
                    and d.get("block_rate", 1) < 0.5]
        if not eligible:
            return

        donor = max(eligible, key=lambda d: d["cg"])
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
        corrupted = self._pool.demon_disrupt(tid)

        self._add_event(
            f"⚠ Demon [C={demon_action['action_complexity']:.2f}] "
            f"→ {AGENT_NAMES[tid]}: {corrupted}",
            "#ff2244", "demon"
        )

        if self.phase >= 4:
            self._add_event(
                f"🛡 Phase IV: ΔΦ≥0 protection → {AGENT_NAMES[tid]}",
                "#00ccff", "value"
            )

    # ── ToM из снапшотов воркеров ─────────────────────────────────────────────
    def _update_tom_from_snapshots(self, snapshots: list[dict]):
        for i in range(3):
            for j in range(i + 1, 3):
                phi_i    = snapshots[i].get("phi", 0.1)
                phi_j    = snapshots[j].get("phi", 0.1)
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
            "byzantine":    self.byzantine.snapshot(),
            "motif":        self.motif.snapshot(),
            "multiprocess": True,
        }

    def public_state(self) -> dict:
        """Снимок для GET /state без выполнения тика."""
        snaps = [self._mp_snapshots[i] for i in range(3)]
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        return self._snapshot(snaps, {}, smoothed)

    def shutdown(self):
        if self._pool:
            self._pool.stop()
