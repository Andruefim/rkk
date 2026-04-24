"""
mocap_loader.py — Уровень 0 (CMU MoCap Library) & Уровень 1 (MediaPipe).

Этот модуль отвечает за парсинг реальных данных человеческих движений (BVH, JSON).
Он нормализует сырые углы суставов в переменные PyBullet (0.05 - 0.95), чтобы 
_seq_buffer GNN питался чистой, физически корректной реальностью.
"""
from __future__ import annotations

import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Any

# TODO: Для полной поддержки BVH без зависимостей можно реализовать рекурсивный
# спуск по дереву костей. Пока мы используем заглушку, которая читает JSON/NPZ 
# (например отработанный через MediaPipe) или использует fallback-манифолд, 
# если директория пуста.

class MoCapDataLoader:
    def __init__(self, data_dir: str = "mocap_data"):
        self.data_dir = data_dir
        self.clips: list[np.ndarray] = []
        self._load_all_clips()

    def _load_all_clips(self):
        """Парсит все файлы в data_dir."""
        os.makedirs(self.data_dir, exist_ok=True)
        files = glob.glob(os.path.join(self.data_dir, "*.npz"))
        
        for f in files:
            try:
                data = np.load(f)
                # Ожидается массив формы [T, J], где J - суставы гуманоида
                if "angles" in data:
                    self.clips.append(data["angles"])
            except Exception as e:
                print(f"[MoCap] Failed to load {f}: {e}")
                
        if not self.clips:
            print(f"[MoCap] No real data in {self.data_dir}. Generating synthetic fallback CMU-like clip.")
            self.clips.append(self._generate_synthetic_clip(steps=300))
        else:
            print(f"[MoCap] Loaded {len(self.clips)} real motion clips.")

    def _generate_synthetic_clip(self, steps: int = 300) -> np.ndarray:
        """Fallback, если нет скачанных BVH/MediaPipe (генерируем идеальную ходьбу)."""
        # Порядок суставов (условно): 0:lhip, 1:rhip, 2:lknee, 3:rknee, 4:lankle, 5:rankle, 6:com_z, 7:posture
        clip = np.zeros((steps, 8))
        for t in range(steps):
            phase = (t / 20.0) * (2 * math.pi)
            clip[t, 0] = 0.5 + 0.25 * math.sin(phase)           # lhip
            clip[t, 1] = 0.5 + 0.25 * math.sin(phase + math.pi) # rhip
            clip[t, 2] = 0.5 + 0.15 * math.sin(phase)           # lknee
            clip[t, 3] = 0.5 + 0.15 * math.sin(phase + math.pi) # rknee
            clip[t, 4] = 0.5 + 0.10 * math.sin(phase)           # lankle
            clip[t, 5] = 0.5 + 0.10 * math.sin(phase + math.pi) # rankle
            clip[t, 6] = 0.85                                   # com_z (high)
            clip[t, 7] = 0.95                                   # posture (stable)
        return clip

    def sample_clip(self, length: int) -> np.ndarray:
        """Возвращает случайный кусок MoCap."""
        clip = self.clips[np.random.randint(0, len(self.clips))]
        if len(clip) <= length:
            return clip
        start = np.random.randint(0, len(clip) - length)
        return clip[start : start + length]


class MotionDiscriminator(nn.Module):
    """
    Уровень 2: Adversarial Motion Prior (AMP-style).
    Оценивает, насколько 'человекоподобна' текущая физическая поза/переход.
    Обучается offline на MoCap данных отличать реальные движения от фейковых/случайных.
    """
    def __init__(self, state_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # Без сигмоиды для стабильности LSGAN/WGAN loss
        )
        
        # Пытаемся загрузить веса, если они есть
        weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mocap_data", "weights", "amp_discriminator.pth"))
        if os.path.exists(weights_path):
            try:
                # Используем weights_only=True для безопасности, как требует PyTorch
                self.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
                print(f"[AMP] Загружены веса дискриминатора из {weights_path}")
            except Exception as e:
                print(f"[AMP] Ошибка загрузки весов: {e}")
        else:
            print("[AMP] Веса дискриминатора не найдены. Он инициализирован случайно. Запустите scripts/train_amp.py")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Возвращает logits. Больше = более похоже на MoCap."""
        return self.net(state)

    def compute_amp_reward(self, state: torch.Tensor) -> float:
        """Intrinsic reward для агента."""
        with torch.no_grad():
            logits = self.forward(state)
            # В оригинальном AMP: r = 1 - 0.25 * (D(s) - 1)^2 (LSGAN)
            # Упрощенно: пропускаем через сигмоиду.
            prob = torch.sigmoid(logits).item()
            # Награда от 0 до 1
            return max(0.0, min(1.0, prob))

