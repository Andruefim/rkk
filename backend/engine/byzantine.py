"""
byzantine.py — Byzantine Majority Vote (РКК v5, Шаг Б).

Два механизма:

1. Φ-взвешенный консенсус матриц W.
   Каждые CONSENSUS_EVERY тиков агенты обмениваются матрицами W.
   Консенсус: W_ij_consensus = Σ(W_ij(k) * Φ(k)) / Σ Φ(k)
   Если W_ij агента сильно отклоняется от консенсуса → alpha_trust ↓
   Интерпретация: «Демон галлюцинирует мне — общество не согласно».

2. Causal Motif Transfer (кросс-доменное обучение).
   Агенты обучают РАЗНЫЕ среды → у них разные переменные → нельзя
   голосовать за рёбра напрямую.
   Решение: передаём веса System 1 (MLP epistemic scorer) от агента
   с наивысшим Compression Gain остальным через EMA (экспоненциальное
   скользящее среднее).
   Интерпретация: «У Aether прорыв в химии — его интуиция лучше,
   обновите ваши MLP».

Безопасность:
   - Голосование взвешивается Φ: «больной» агент (низкое Φ) влияет меньше.
   - Если все три агента вышли из строя (Φ < PHI_QUORUM) — консенсус
     пропускается (нет кворума).
   - Motif transfer блокируется если донор имеет block_rate > 0.5
     (его интуиция ненадёжна).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

CONSENSUS_EVERY  = 50    # тиков между голосованиями
MOTIF_EVERY      = 100   # тиков между передачей весов System1
PHI_QUORUM       = 0.05  # минимальный Φ для участия в голосовании
ALPHA_DECAY_RATE = 0.12  # насколько снижаем alpha_trust при отклонении от консенсуса
MOTIF_EMA        = 0.15  # коэффициент EMA при смешивании весов S1
DEVIANCE_THRESH  = 0.25  # порог отклонения W_ij от консенсуса → снижаем alpha


# ─── Snapshot агента для передачи между процессами ───────────────────────────
@dataclass
class AgentConsensusData:
    agent_id:        int
    phi:             float
    W:               torch.Tensor          # матрица смежности (d, d), CPU
    node_ids:        list[str]             # упорядоченный список переменных
    compression_gain: float
    block_rate:      float
    # System 1 state dict (только параметры, не оптимизатор)
    s1_state_dict:   dict | None = None


def gather_consensus_data(agents: list) -> list[AgentConsensusData]:
    """Собираем данные от всех агентов для консенсуса."""
    result = []
    for agent in agents:
        W = None
        node_ids = []
        if agent.graph._core is not None:
            W        = agent.graph._core.W_masked().detach().cpu().clone()
            node_ids = list(agent.graph._node_ids)

        s1_state = None
        try:
            s1_state = {k: v.detach().cpu().clone()
                        for k, v in agent.system1.net.state_dict().items()}
        except Exception:
            pass

        result.append(AgentConsensusData(
            agent_id=agent.id,
            phi=agent.phi_approx(),
            W=W,
            node_ids=node_ids,
            compression_gain=agent.compression_gain,
            block_rate=agent.value_layer.block_rate,
            s1_state_dict=s1_state,
        ))
    return result


# ─── Φ-взвешенный консенсус ──────────────────────────────────────────────────
class ByzantineConsensus:
    """
    Φ-weighted Byzantine voting на матрицах W.

    Алгоритм:
      1. Проверяем кворум (хотя бы 2 агента с Φ > PHI_QUORUM)
      2. Для каждого агента-участника вычисляем W_consensus
      3. Сравниваем локальный W с консенсусом
      4. Если |W_ij - W_consensus_ij| > DEVIANCE_THRESH → понижаем alpha_trust
    """

    def __init__(self, n_agents: int = 3):
        self.n_agents   = n_agents
        self.round      = 0
        self.last_consensus: dict | None = None  # {agent_id: W_consensus}
        self.deviations: list[float]     = []    # история отклонений

    def run(
        self,
        agents:     list,
        data:       list[AgentConsensusData],
        device:     torch.device,
    ) -> dict:
        """
        Запускаем один раунд голосования.
        Возвращает событие-словарь для логирования.
        """
        self.round += 1

        # 1. Кворум
        eligible = [d for d in data if d.phi > PHI_QUORUM and d.W is not None]
        if len(eligible) < 2:
            return {
                "type":    "consensus_skip",
                "round":   self.round,
                "reason":  f"quorum failed ({len(eligible)}/3 eligible)",
            }

        # 2. Суммарный Φ
        phi_sum = sum(d.phi for d in eligible)
        if phi_sum < 1e-6:
            return {"type": "consensus_skip", "round": self.round, "reason": "phi_sum=0"}

        # 3. Для каждого участника — Φ-взвешенный консенсус по другим
        #    (исключаем самого агента из его собственного консенсуса,
        #     чтобы не «самоподтверждать» свои галлюцинации)
        results = {}
        total_deviance = 0.0
        updates_applied = 0

        for agent_data in eligible:
            peers = [d for d in eligible if d.agent_id != agent_data.agent_id]
            if not peers:
                continue

            peer_phi_sum = sum(d.phi for d in peers)
            if peer_phi_sum < 1e-6:
                continue

            # Консенсус матрицы: только по рёбрам которые есть у обоих агентов
            # (пересечение node_ids)
            my_ids   = agent_data.node_ids
            my_W     = agent_data.W   # (d_self, d_self)

            # Строим консенсус-матрицу в пространстве данного агента
            W_consensus = torch.zeros_like(my_W)
            W_weights   = torch.zeros_like(my_W)

            for peer in peers:
                peer_ids = peer.node_ids
                peer_W   = peer.W

                # Находим общие переменные (они редко совпадают из-за разных сред,
                # но могут совпадать по имени, напр. "Temp" есть и в physics, и chemistry)
                common_my   = []
                common_peer = []
                for my_i, nid in enumerate(my_ids):
                    if nid in peer_ids:
                        p_i = peer_ids.index(nid)
                        for my_j, nid2 in enumerate(my_ids):
                            if nid2 in peer_ids:
                                p_j = peer_ids.index(nid2)
                                common_my.append((my_i, my_j))
                                common_peer.append((p_i, p_j))

                for (mi, mj), (pi, pj) in zip(common_my, common_peer):
                    w_peer = peer_W[pi, pj].item()
                    W_consensus[mi, mj] += w_peer * peer.phi
                    W_weights[mi, mj]   += peer.phi

            # Нормализуем
            mask = W_weights > 1e-6
            W_consensus[mask] /= W_weights[mask]

            # 4. Сравниваем с локальным W
            deviance = (my_W - W_consensus).abs()
            high_dev = (deviance > DEVIANCE_THRESH) & mask

            if high_dev.any():
                # Понижаем alpha_trust для отклоняющихся рёбер
                agent = agents[agent_data.agent_id]
                if agent.graph._core is not None:
                    for ei, (from_, to) in enumerate(
                        [(agent.graph._node_ids[r], agent.graph._node_ids[c])
                         for r in range(my_W.shape[0])
                         for c in range(my_W.shape[1])
                         if high_dev[r, c]]
                    ):
                        for edge in agent.graph.edges:
                            if edge.from_ == from_ and edge.to == to:
                                edge.alpha_trust = max(
                                    0.02,
                                    edge.alpha_trust - ALPHA_DECAY_RATE
                                )
                                break
                    agent.graph._invalidate_cache()

            n_deviant   = int(high_dev.sum().item())
            total_deviance += deviance[mask].mean().item() if mask.any() else 0.0
            updates_applied += n_deviant

            results[agent_data.agent_id] = {
                "n_deviant":     n_deviant,
                "mean_deviance": round(deviance[mask].mean().item(), 4) if mask.any() else 0.0,
            }

        self.deviations.append(total_deviance / max(len(eligible), 1))
        if len(self.deviations) > 50:
            self.deviations.pop(0)

        return {
            "type":             "consensus_ok",
            "round":            self.round,
            "eligible":         len(eligible),
            "phi_weighted":     True,
            "updates_applied":  updates_applied,
            "mean_deviance":    round(total_deviance / max(len(eligible), 1), 4),
            "per_agent":        results,
        }

    @property
    def mean_deviance(self) -> float:
        if not self.deviations:
            return 0.0
        return float(np.mean(self.deviations[-10:]))

    def snapshot(self) -> dict:
        return {
            "round":         self.round,
            "mean_deviance": round(self.mean_deviance, 4),
        }


# ─── Causal Motif Transfer ────────────────────────────────────────────────────
class MotifTransfer:
    """
    Передаём веса System 1 от лучшего агента (по CG) остальным через EMA.

    Почему это работает кросс-доменно:
      System 1 предсказывает «насколько сюрпризна эта пара переменных»
      на основе 9 абстрактных признаков (uncertainty, alpha, h_W_norm и т.д.)
      — они не специфичны для конкретной среды.
      Если Aether (Химия) нашёл хорошую интуицию, её абстрактный MLP
      работает и для Nova (Физика) и для Lyra (Логика).

    Защита:
      - Донор должен иметь block_rate < 0.5 (его интуиция надёжна)
      - Донор должен иметь CG > 0 (реально учится)
      - EMA = 0.15 (мягкое смешивание, не перезапись)
    """

    def __init__(self, n_agents: int = 3):
        self.n_agents = n_agents
        self.round    = 0
        self.last_donor: int | None = None

    def run(
        self,
        agents: list,
        data:   list[AgentConsensusData],
        device: torch.device,
    ) -> dict:
        self.round += 1

        # Выбираем донора: максимальный CG, block_rate < 0.5
        eligible = [d for d in data
                    if d.compression_gain > 0
                    and d.block_rate < 0.5
                    and d.s1_state_dict is not None]

        if not eligible:
            return {"type": "motif_skip", "reason": "no eligible donor"}

        donor = max(eligible, key=lambda d: d.compression_gain)
        self.last_donor = donor.agent_id

        transferred = 0
        for agent in agents:
            if agent.id == donor.agent_id:
                continue

            try:
                # EMA смешивание весов (не перезаписываем — сохраняем специализацию)
                target_state = agent.system1.net.state_dict()
                donor_state  = donor.s1_state_dict

                new_state = {}
                for key in target_state:
                    if key in donor_state:
                        t = target_state[key].float()
                        d = donor_state[key].to(device).float()
                        if t.shape == d.shape:
                            new_state[key] = (1 - MOTIF_EMA) * t + MOTIF_EMA * d
                        else:
                            new_state[key] = t
                    else:
                        new_state[key] = target_state[key]

                agent.system1.net.load_state_dict(new_state, strict=False)
                transferred += 1
            except Exception as e:
                pass

        return {
            "type":        "motif_transfer",
            "round":       self.round,
            "donor_id":    donor.agent_id,
            "donor_cg":    round(donor.compression_gain, 4),
            "transferred": transferred,
            "ema":         MOTIF_EMA,
        }

    def snapshot(self) -> dict:
        return {
            "round":       self.round,
            "last_donor":  self.last_donor,
        }
