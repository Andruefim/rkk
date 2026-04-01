"""
system1.py — Amortized Epistemic Scorer (System 1).

Задача: предсказать E[IG] для пары (from_var, to_var, test_value)
        БЕЗ запуска полного каузального поиска.

Архитектура: MLP с residual connection + BatchNorm.
Вход (9 признаков):
  [w_ij, alpha_ij, val_from, val_to, uncertainty,
   h_W_norm, grad_norm_ij, intervention_count_norm, discovery_rate]
Выход: E[IG] ∈ [0, 1]

Обучение: онлайн, после каждого do()-эксперимента.
  actual_ig = |predicted_outcome - observed_outcome|  (интервенционная жёсткость)
  loss = MSE(E[IG]_pred, actual_ig) + entropy_bonus

Seed Diversity (улучшение 5):
  Agent 0: ReLU  — жёсткие логические связи
  Agent 1: GELU  — вероятностные мягкие связи
  Agent 2: Tanh  — насыщаемые связи
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass


# ─── Experience Buffer ────────────────────────────────────────────────────────
@dataclass
class S1Experience:
    features:  list[float]   # 9 входных признаков
    actual_ig: float         # реальный info gain после do()


class ExperienceBuffer:
    """Кольцевой буфер опыта для батч-обучения System 1."""

    def __init__(self, maxlen: int = 512):
        self._buf: deque[S1Experience] = deque(maxlen=maxlen)

    def push(self, exp: S1Experience):
        self._buf.append(exp)

    def sample(self, batch_size: int) -> list[S1Experience]:
        if len(self._buf) < batch_size:
            return list(self._buf)
        idxs = np.random.choice(len(self._buf), batch_size, replace=False)
        return [self._buf[i] for i in idxs]

    def __len__(self):
        return len(self._buf)


# ─── System 1 Network ─────────────────────────────────────────────────────────
class System1Net(nn.Module):
    """
    MLP с residual block.
    Input dim: 9
    Hidden: 64 → 64 (residual) → 32 → 1
    """

    INPUT_DIM = 9

    def __init__(self, activation: str = "relu"):
        super().__init__()

        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        Act = act_map.get(activation, nn.ReLU)

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 64),
            nn.BatchNorm1d(64),
            Act(),
        )

        # Residual block
        self.res = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            Act(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        self.res_act = Act()

        # Head
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            Act(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 9) → (B, 1)"""
        h = self.stem(x)
        h = self.res_act(h + self.res(h))   # residual
        return self.head(h)


# ─── System 1 (публичный интерфейс) ──────────────────────────────────────────
class System1:
    """
    Оберётка над System1Net.
    Управляет буфером опыта, онлайн-обучением и feature-инжинирингом.
    """

    BATCH_SIZE   = 32
    TRAIN_EVERY  = 4   # обучаемся каждые N push()-ов
    ENTROPY_COEF = 0.01

    def __init__(self, activation: str = "relu", device: torch.device = None):
        self.device     = device or torch.device("cpu")
        self.activation = activation
        self.net        = System1Net(activation).to(self.device)
        self.optim      = torch.optim.Adam(self.net.parameters(), lr=3e-4, weight_decay=1e-5)
        self.buffer     = ExperienceBuffer(maxlen=512)
        self._push_count = 0
        self.train_losses: list[float] = []

    def build_features(
        self,
        w_ij:               float,
        alpha_ij:           float,
        val_from:           float,
        val_to:             float,
        uncertainty:        float,
        h_W_norm:           float,
        grad_norm_ij:       float,
        intervention_count: int,
        discovery_rate:     float,
    ) -> list[float]:
        """Собираем вектор признаков из 9 чисел."""
        return [
            float(np.tanh(w_ij)),                              # нормализуем вес
            float(np.clip(alpha_ij, 0, 1)),
            float(np.clip(val_from, 0, 1)),
            float(np.clip(val_to,   0, 1)),
            float(np.clip(uncertainty, 0, 1)),
            float(np.clip(h_W_norm, 0, 1)),
            float(np.tanh(grad_norm_ij)),                      # нормализуем норму градиента
            float(np.clip(intervention_count / 100.0, 0, 1)), # нормализуем счётчик
            float(np.clip(discovery_rate, 0, 1)),
        ]

    def score(self, features_batch: list[list[float]]) -> list[float]:
        """
        Предсказываем E[IG] для батча пар (from_var, to_var).
        Возвращает список скоров [0, 1].
        """
        if not features_batch:
            return []
        self.net.eval()
        x = torch.tensor(features_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores = self.net(x).squeeze(-1)
        return scores.cpu().tolist()

    def push_experience(self, features: list[float], actual_ig: float):
        """Записываем опыт и запускаем обучение по расписанию."""
        self.buffer.push(S1Experience(features=features, actual_ig=actual_ig))
        self._push_count += 1
        if self._push_count % self.TRAIN_EVERY == 0 and len(self.buffer) >= self.BATCH_SIZE:
            self._train_step()

    def _train_step(self):
        """Один шаг обучения System 1."""
        batch = self.buffer.sample(self.BATCH_SIZE)
        if not batch:
            return

        self.net.train()
        X = torch.tensor([e.features  for e in batch], dtype=torch.float32, device=self.device)
        y = torch.tensor([e.actual_ig for e in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        self.optim.zero_grad()
        pred = self.net(X)
        loss = F.mse_loss(pred, y)

        # Entropy bonus: поощряем разнообразие предсказаний
        entropy = -(pred * pred.log().clamp(-10) + (1-pred) * (1-pred).log().clamp(-10))
        loss    = loss - self.ENTROPY_COEF * entropy.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()

        self.train_losses.append(loss.item())
        if len(self.train_losses) > 200:
            self.train_losses.pop(0)

    @property
    def mean_loss(self) -> float:
        if not self.train_losses:
            return 0.0
        return float(np.mean(self.train_losses[-20:]))
