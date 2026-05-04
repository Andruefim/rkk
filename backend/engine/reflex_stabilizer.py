"""
Быстрый онлайн-корректор целей ног (аналог мозжечковой коррекции):
маленькая сеть, inference в CPG-потоке (~60 Hz), обучение в agent-потоке на сигнале posture.

Вкл.: RKK_REFLEX_STABILIZER=1
"""
from __future__ import annotations

import os
import threading

import numpy as np
import torch
import torch.nn as nn


def reflex_stabilizer_enabled() -> bool:
    return os.environ.get("RKK_REFLEX_STABILIZER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class ReflexStabilizer:
    """
    Вход: posture, com_z, foot_l, foot_r + ошибки суставов (obs − CPG target) по joint_keys.
    Выход: дельты к целям CPG для тех же суставов (tanh, масштабируются).
    """

    def __init__(
        self,
        joint_keys: list[str],
        hidden: int = 32,
        device: torch.device | None = None,
    ) -> None:
        self.joint_keys = list(joint_keys)
        self.n_joints = len(self.joint_keys)
        self.device = device or torch.device("cpu")
        d_in = 4 + self.n_joints
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.n_joints),
            nn.Tanh(),
        ).to(self.device)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)
        try:
            lr = float(os.environ.get("RKK_REFLEX_LR", "3e-4"))
        except ValueError:
            lr = 3e-4
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._lock = threading.Lock()
        self._last_x: np.ndarray | None = None
        self._last_correction: np.ndarray | None = None
        self._posture_ema = 0.5
        try:
            self._ema_alpha = float(os.environ.get("RKK_REFLEX_EMA", "0.9"))
        except ValueError:
            self._ema_alpha = 0.9
        self._ema_alpha = float(np.clip(self._ema_alpha, 0.0, 0.999))
        self.n_updates = 0

    def _obs_to_tensor(self, obs: dict, cpg_targets: dict) -> torch.Tensor:
        posture = float(
            obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
        )
        com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        foot_l = float(
            obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5))
        )
        foot_r = float(
            obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5))
        )
        errs: list[float] = []
        for k in self.joint_keys:
            cur = float(obs.get(k, obs.get(f"phys_{k}", 0.5)))
            tgt = float(cpg_targets.get(k, cur))
            errs.append(cur - tgt)
        vec = [posture, com_z, foot_l, foot_r] + errs
        return torch.tensor(vec, dtype=torch.float32, device=self.device)

    def step(self, obs: dict, cpg_targets: dict[str, float]) -> dict[str, float]:
        """Коррекция целей CPG; потокобезопасно кеширует состояние для train_on_outcome."""
        with self._lock:
            x = self._obs_to_tensor(obs, cpg_targets)
            with torch.no_grad():
                corrections = self.net(x).detach().cpu().numpy().astype(np.float64)
            self._last_x = x.detach().cpu().numpy().copy()
            self._last_correction = corrections.copy()

        try:
            scale_max = float(os.environ.get("RKK_REFLEX_SCALE_MAX", "0.15"))
        except ValueError:
            scale_max = 0.15
        scale_base = float(os.environ.get("RKK_REFLEX_SCALE_BASE", "0.02"))
        scale_grow = float(os.environ.get("RKK_REFLEX_SCALE_GROW", "1e-5"))
        scale = min(scale_max, scale_base + self.n_updates * scale_grow)

        out = dict(cpg_targets)
        for i, k in enumerate(self.joint_keys):
            if k not in out:
                continue
            out[k] = float(np.clip(out[k] + scale * corrections[i], 0.05, 0.95))
        return out

    def train_on_outcome(self, posture_before: float, posture_after: float) -> float:
        """Один шаг онлайн-обучения (вызывать из agent-потока)."""
        self._posture_ema = (
            self._ema_alpha * self._posture_ema
            + (1.0 - self._ema_alpha) * posture_after
        )
        reward = posture_after - self._posture_ema
        if abs(reward) < 1e-5:
            return 0.0

        try:
            boost = float(os.environ.get("RKK_REFLEX_REWARD_GAIN", "3.0"))
        except ValueError:
            boost = 3.0

        with self._lock:
            if self._last_x is None or self._last_correction is None:
                return 0.0
            x = torch.tensor(self._last_x, dtype=torch.float32, device=self.device)
            corrections = torch.tensor(
                self._last_correction, dtype=torch.float32, device=self.device
            )
            target = corrections * (1.0 + float(reward) * boost)
            self.opt.zero_grad()
            pred = self.net(x)
            loss = ((pred - target.detach()) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
            self.n_updates += 1
            return float(loss.item())
