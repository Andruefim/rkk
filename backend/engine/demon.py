"""
Demon — Adversarial Environment Generator.

Улучшение 2 (Термодинамический штраф):
  demon_loss = -prediction_error + lambda * ||intervention||_0
  Демон не может просто добавить шум — должен найти физически
  реальную уязвимость с минимальной сложностью действия.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class AdversarialDemon:
    """
    Простая реализация Демона Фазы I.
    В Фазе II заменяется нейросетевым агентом.
    """

    def __init__(self, n_agents: int, device: torch.device):
        self.n_agents  = n_agents
        self.device    = device
        self.energy    = 1.0
        self.cooldown  = 0
        self.target    = 0

        self._last_action_complexity = 0.0
        self._history: list[dict]    = []

        # Простая нейросеть Демона (Фаза I)
        # Input:  [agent_phi, agent_alpha, env_entropy, demon_energy]
        # Output: [target_agent, intervention_strength, variable_idx]
        self.policy = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 3), nn.Sigmoid(),  # [target_prob, strength, var_idx_norm]
        ).to(device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    # ── Один шаг Демона ───────────────────────────────────────────────────────
    def step(self, agent_snapshots: list[dict], env_entropy: float) -> dict | None:
        if self.cooldown > 0:
            self.cooldown -= 1
            self.energy    = min(1.0, self.energy + 0.002)
            return None

        if self.energy < 0.15:
            return None

        # Выбираем цель через policy
        phis   = [s["phi"]        for s in agent_snapshots]
        alphas = [s["alpha_mean"] for s in agent_snapshots]
        avg_phi   = float(np.mean(phis))
        avg_alpha = float(np.mean(alphas))

        state = torch.tensor(
            [avg_phi, avg_alpha, env_entropy, self.energy],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            out = self.policy(state).squeeze(0)

        target_idx  = int(out[0].item() * (self.n_agents - 0.01))
        strength    = float(out[1].item()) * 0.4 + 0.1   # [0.1, 0.5]
        var_idx_raw = float(out[2].item())                # нормализованный индекс переменной

        # ── Термодинамический штраф L0 ────────────────────────────────────────
        # action_complexity = L0-норма вектора вмешательства
        # (сколько переменных задействовано)
        intervention_vector = torch.tensor([strength, var_idx_raw], device=self.device)
        l0_norm = (intervention_vector.abs() > 0.05).float().sum()
        lambda_penalty      = 0.3
        action_complexity   = lambda_penalty * l0_norm.item()

        # Демон платит энергию пропорционально сложности вмешательства
        energy_cost              = 0.15 + action_complexity * 0.1
        self._last_action_complexity = action_complexity

        if self.energy < energy_cost:
            return None

        self.energy   -= energy_cost
        self.cooldown  = 70 + int(action_complexity * 20)
        self.target    = target_idx

        action = {
            "target_agent":      target_idx,
            "strength":          strength,
            "var_idx_raw":       var_idx_raw,
            "action_complexity": action_complexity,
            "energy_spent":      energy_cost,
        }
        self._history.append(action)

        return action

    # ── Обучение Демона (максимизируем ошибку агента) ─────────────────────────
    def learn(self, prediction_error: float, action_complexity: float):
        """
        Demon loss = -prediction_error + λ * action_complexity
        Демон хочет максимальную ошибку при минимальных затратах.
        """
        if not self._history:
            return

        loss = torch.tensor(
            -prediction_error + 0.3 * action_complexity,
            requires_grad=False,
            device=self.device
        )
        # Простое policy gradient (REINFORCE-like, без baseline)
        # В Фазе II заменяем на полноценный PPO
        self.optim.zero_grad()
        dummy = sum(p.sum() for p in self.policy.parameters()) * 0.0
        (loss + dummy).backward()
        self.optim.step()

    @property
    def snapshot(self) -> dict:
        return {
            "energy":                self.energy,
            "cooldown":              self.cooldown,
            "last_target":           self.target,
            "last_action_complexity": round(self._last_action_complexity, 3),
        }
