"""
simulation_v4.py — оркестратор с 4 агентами (+ Ignis/PyBullet).

Изменения:
  - 4-й агент Ignis с env_preset="pybullet"
  - agent_seed_context() → запрашивает переменные у воркера напрямую
    (PyBullet имеет 18 переменных, раньше мы делали re-init Environment)
  - AGENT_COLORS / AGENT_NAMES / ENV_PRESETS расширены до 4
  - Demon атакует 4 агентов; Byzantine консенсус с 4 участниками
  - В snapshot добавлено поле pybullet_stats
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque

from engine.demon       import AdversarialDemon
from engine.value_layer import HomeostaticBounds
from engine.byzantine   import (
    ByzantineConsensus, MotifTransfer,
    CONSENSUS_EVERY, MOTIF_EVERY,
)

PHASE_THRESHOLDS = [0.0, 0.18, 0.38, 0.58, 0.76, 0.92]
PHASE_HOLD_TICKS = 15
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]

AGENT_COLORS = ["#00ff99", "#0099ff", "#ff9900", "#cc44ff"]
AGENT_NAMES  = ["Nova",    "Aether",   "Lyra",    "Ignis"]
ENV_PRESETS  = ["physics", "chemistry","logic",   "pybullet"]

N_AGENTS = 4


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
        print(f"[RKK] Device: {self.device} | 4-agent pool (Nova/Aether/Lyra/Ignis)")

        bounds = _default_bounds()
        import dataclasses
        from engine.agent_worker import AgentPool

        bounds_dict = dataclasses.asdict(bounds)
        self._pool = AgentPool(
            device_str=device_str,
            bounds_dict=bounds_dict,
            n_agents=N_AGENTS,
        )
        if not self._pool.start():
            raise RuntimeError("AgentPool failed to start.")

        self.demon     = AdversarialDemon(N_AGENTS, self.device)
        self.byzantine = ByzantineConsensus(n_agents=N_AGENTS)
        self.motif     = MotifTransfer(n_agents=N_AGENTS)

        self.tick      = 0
        self.phase     = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase    = 1
        self._dr_window: deque[float] = deque(maxlen=20)

        self.events: deque[dict] = deque(maxlen=20)
        self.tom: list[list[float]] = [[0.0] * N_AGENTS for _ in range(N_AGENTS)]
        self._prev_edge_counts = [0] * N_AGENTS

        self._mp_snapshots: list[dict] = [{} for _ in range(N_AGENTS)]

        # Кэш переменных воркеров (заполняется при первом запросе)
        self._worker_var_cache: dict[int, dict] = {}

    # ── Seed injection ────────────────────────────────────────────────────────
    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        if agent_id < 0 or agent_id >= N_AGENTS:
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
        """
        Имя, пресет и переменные среды агента.
        Для PyBullet запрашиваем переменные у воркера (18 штук),
        для остальных — используем кэш или запрос.
        """
        if agent_id < 0 or agent_id >= N_AGENTS:
            return None

        # Кэш
        if agent_id in self._worker_var_cache:
            cached = self._worker_var_cache[agent_id]
            return {
                "name":      AGENT_NAMES[agent_id],
                "preset":    cached["preset"],
                "variables": cached["variables"],
            }

        # Запрашиваем у воркера
        resp = self._pool.get_variable_ids(agent_id)
        self._worker_var_cache[agent_id] = resp

        return {
            "name":      AGENT_NAMES[agent_id],
            "preset":    resp.get("preset", ENV_PRESETS[agent_id]),
            "variables": resp.get("variables", []),
        }

    # ── Один тик ──────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1

        results = self._pool.step_all(tick=self.tick)
        ordered_snaps = [{}] * N_AGENTS

        for msg in results:
            aid  = msg.get("agent_id", 0)
            snap = msg.get("snapshot", {})
            self._mp_snapshots[aid] = snap
            ordered_snaps[aid]      = snap
            self._log_step_result_mp(aid, msg.get("result", {}))

        # Обратная связь демона
        if self.demon._last_action is not None:
            tid = self.demon._last_action["target_agent"]
            pe  = 0.0
            for msg in results:
                if msg.get("agent_id") == tid and msg.get("status") == "step_done":
                    res = msg.get("result", {})
                    if not res.get("blocked") and not res.get("skipped"):
                        pe = float(res.get("prediction_error", 0))
                    break
            self.demon.learn(pe, self.demon._last_action_complexity, ordered_snaps)

        self._step_demon(ordered_snaps)

        if self.tick % 25 == 0 and self.phase >= 3:
            self._update_tom(ordered_snaps)

        if self.tick % CONSENSUS_EVERY == 0:
            self._run_byzantine()

        if self.tick % MOTIF_EVERY == 0:
            self._run_motif()

        smoothed_dr = self._update_phase(ordered_snaps)

        graph_deltas = {}
        for i, snap in enumerate(ordered_snaps):
            edge_cnt = snap.get("edge_count", 0)
            if edge_cnt != self._prev_edge_counts[i]:
                graph_deltas[i] = snap.get("edges", [])
                self._prev_edge_counts[i] = edge_cnt

        return self._snapshot(ordered_snaps, graph_deltas, smoothed_dr)

    # ── Byzantine ─────────────────────────────────────────────────────────────
    def _run_byzantine(self):
        raw_data = self._pool.get_consensus_data()
        eligible = [d for d in raw_data if d.get("phi", 0) > 0.05 and d.get("W")]
        if len(eligible) < 2:
            return

        total_updates = 0
        for target in eligible:
            peers      = [d for d in eligible if d["agent_id"] != target["agent_id"]]
            target_ids = target["node_ids"]
            target_W   = torch.tensor(target["W"])

            for mi, nid in enumerate(target_ids):
                for mj, nid2 in enumerate(target_ids):
                    cw = cp = 0.0
                    for peer in peers:
                        pids = peer["node_ids"]
                        if nid in pids and nid2 in pids:
                            pi, pj = pids.index(nid), pids.index(nid2)
                            cw += peer["W"][pi][pj] * peer["phi"]
                            cp += peer["phi"]
                    if cp < 1e-6:
                        continue
                    deviance = abs(target_W[mi, mj].item() - cw / cp)
                    if deviance > 0.25:
                        self._pool.apply_alpha_decay(target["agent_id"], nid, nid2, 0.12)
                        total_updates += 1

        self._add_event(
            f"🗳 Byzantine R{self.byzantine.round}: {total_updates} edges",
            "#004466", "tom"
        )
        self.byzantine.round += 1

    # ── Motif Transfer ────────────────────────────────────────────────────────
    def _run_motif(self):
        raw_data = self._pool.get_consensus_data()
        eligible = [d for d in raw_data
                    if d.get("cg", 0) > 0 and d.get("block_rate", 1) < 0.5]
        if not eligible:
            return
        donor = max(eligible, key=lambda d: d["cg"])
        self.motif.last_donor = donor["agent_id"]
        self.motif.round += 1
        self._add_event(
            f"🧬 Motif: {AGENT_NAMES[donor['agent_id']]} → others "
            f"(CG={donor['cg']:.4f})",
            "#224466", "tom"
        )

    # ── Demon ─────────────────────────────────────────────────────────────────
    def _step_demon(self, snapshots: list[dict]):
        mean_peak   = np.mean([s.get("peak_discovery_rate", 0) for s in snapshots])
        demon_action = self.demon.step(snapshots, 1 - mean_peak)
        if demon_action is None:
            return

        tid       = demon_action["target_agent"]
        corrupted = self._pool.demon_disrupt(tid)
        mode      = demon_action.get("mode", "?")

        self._add_event(
            f"⚠ Demon [{mode}·C={demon_action['action_complexity']:.2f}] "
            f"→ {AGENT_NAMES[tid]}: {corrupted}",
            "#ff2244", "demon"
        )
        if self.phase >= 4:
            self._add_event(
                f"🛡 Phase IV: ΔΦ≥0 → {AGENT_NAMES[tid]}",
                "#00ccff", "value"
            )

    # ── ToM ───────────────────────────────────────────────────────────────────
    def _update_tom(self, snapshots: list[dict]):
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                phi_i    = snapshots[i].get("phi", 0.1)
                phi_j    = snapshots[j].get("phi", 0.1)
                strength = (phi_i + phi_j) / 2
                self.tom[i][j] = strength
                self.tom[j][i] = strength
                if strength > 0.5 and np.random.rand() < 0.2:
                    self._add_event(
                        f"ToM: {AGENT_NAMES[i]} ↔ {AGENT_NAMES[j]} {strength*100:.0f}%",
                        "#003388", "tom"
                    )

    # ── Phase ─────────────────────────────────────────────────────────────────
    def _update_phase(self, snapshots: list[dict]) -> float:
        # Фаза определяется по среднему DR трёх табличных агентов (0-2)
        # Ignis/PyBullet исключается (у него другая шкала discovery_rate)
        table_snaps = snapshots[:3]
        mean_cur = np.mean([s.get("discovery_rate", 0) for s in table_snaps])
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
                    f"⬆ Phase {potential}: {PHASE_NAMES[potential]}",
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
                f"{name}: do({result.get('variable','?')}="
                f"{result.get('value',0):.2f}) CG{'+' if cg>=0 else ''}{cg:.3f}",
                col, "discovery"
            )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _snapshot(self, snapshots: list[dict], graph_deltas: dict, smoothed_dr: float) -> dict:
        mean_peak     = np.mean([s.get("peak_discovery_rate", 0) for s in snapshots[:3]])
        total_blocked = sum(s.get("total_blocked", 0) for s in snapshots)

        tom_links = [
            {"a": i, "b": j, "strength": round(self.tom[i][j], 3)}
            for i in range(N_AGENTS) for j in range(i+1, N_AGENTS)
            if self.tom[i][j] > 0.25
        ]

        # PyBullet специфичная статистика (последний агент)
        ignis_snap = snapshots[3] if len(snapshots) > 3 else {}
        pb_stats = {
            "phi":            ignis_snap.get("phi", 0),
            "discovery_rate": ignis_snap.get("discovery_rate", 0),
            "interventions":  ignis_snap.get("total_interventions", 0),
            "node_count":     ignis_snap.get("node_count", 0),
            "edge_count":     ignis_snap.get("edge_count", 0),
            "h_W":            ignis_snap.get("h_W", 0),
            "compression_gain": ignis_snap.get("compression_gain", 0),
        }

        return {
            "tick":           self.tick,
            "phase":          self.phase,
            "max_phase":      self.max_phase,
            "entropy":        round((1 - mean_peak) * 100, 1),
            "smoothed_dr":    round(smoothed_dr, 3),
            "agents":         snapshots,
            "demon":          self.demon.snapshot,
            "tom_links":      tom_links,
            "events":         list(self.events),
            "graph_deltas":   graph_deltas,
            "value_layer": {
                "total_blocked_all": total_blocked,
                "block_rates": [
                    round(s.get("value_layer", {}).get("block_rate", 0), 3)
                    for s in snapshots
                ],
            },
            "byzantine":      self.byzantine.snapshot(),
            "motif":          self.motif.snapshot(),
            "multiprocess":   True,
            "n_agents":       N_AGENTS,
            "pybullet":       pb_stats,
        }

    def public_state(self) -> dict:
        snaps    = [self._mp_snapshots[i] for i in range(N_AGENTS)]
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        return self._snapshot(snaps, {}, smoothed)

    def shutdown(self):
        if self._pool:
            self._pool.stop()