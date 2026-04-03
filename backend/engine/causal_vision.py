"""
causal_vision.py — Causal Visual Cortex (Фаза 12).

Архитектура: SlotAttention Lite
  raw_frame (H×W×3)
      ↓ CNN Encoder (3 conv, lightweight)
  feature_map (P×F)  [P = spatial positions, F = feature dim]
      ↓ Slot Attention (K slots, T iterations)
  slots (K×D)        [K independent object representations]
      ↓ SlotProjector (D → 1 scalar)
  slot_values (K,)   → GNN nodes ["slot_0"..."slot_K-1"]

Ключевая идея:
  Агент не получает готовые переменные (com_x, lhip...).
  Он сам открывает объекты из пикселей.
  Prediction error от GNN пробрасывается назад в SlotAttention —
  зрение «затачивается» под каузально-значимые объекты.

Slot Continuity:
  Слоты могут менять порядок между кадрами.
  Используем Hungarian matching по cosine similarity для
  отслеживания идентичности слотов через время.

References:
  Locatello et al. (2020) "Object-Centric Learning with Slot Attention"
  Adapt: no reconstruction decoder (saves compute), pure predictive coding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─── Конфигурация ──────────────────────────────────────────────────────────────
@dataclass
class VisionConfig:
    frame_h:     int   = 64      # входной размер (ресайз)
    frame_w:     int   = 64
    cnn_channels: list = field(default_factory=lambda: [16, 32, 32])
    feat_dim:    int   = 32      # F: размер фичи на позицию
    n_slots:     int   = 8       # K: количество слотов (объектов)
    slot_dim:    int   = 64      # D: размер вектора слота
    n_iters:     int   = 3       # итерации SlotAttention
    lr:          float = 3e-4
    recon_weight: float = 0.1    # вес reconstruction loss (опционально)


# ─── CNN Encoder ──────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    """
    Лёгкий CNN: (B, 3, H, W) → (B, P, F)
    P = число spatial positions, F = feat_dim
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        ch = [3] + cfg.cnn_channels
        layers = []
        for i in range(len(ch) - 1):
            # stride=2 только в первых двух слоях → spatial compression
            stride = 2 if i < 2 else 1
            layers += [
                nn.Conv2d(ch[i], ch[i+1], kernel_size=5, stride=stride, padding=2),
                nn.GroupNorm(max(1, ch[i+1]//8), ch[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.backbone = nn.Sequential(*layers)

        # Position embeddings: добавляем xy-сетку как отдельные каналы
        self.pos_embed = nn.Sequential(
            nn.Conv2d(ch[-1] + 2, cfg.feat_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.feat_dim, cfg.feat_dim, kernel_size=1),
        )

        # Вычисляем spatial size после strided convs
        h_out = cfg.frame_h // 4
        w_out = cfg.frame_w // 4
        self.n_positions = h_out * w_out
        self._h_out = h_out
        self._w_out = w_out

        # Создаём фиксированные pos_grid при инициализации
        ys = torch.linspace(-1, 1, h_out)
        xs = torch.linspace(-1, 1, w_out)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        # (1, 2, H', W')
        self.register_buffer("pos_grid", torch.stack([grid_x, grid_y], dim=0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        → (B, P, F)  где P = H'×W', F = feat_dim
        """
        feat = self.backbone(x)                            # (B, C, H', W')
        B    = feat.shape[0]
        pos  = self.pos_grid.expand(B, -1, -1, -1)        # (B, 2, H', W')
        feat = torch.cat([feat, pos], dim=1)               # (B, C+2, H', W')
        feat = self.pos_embed(feat)                        # (B, F, H', W')
        feat = feat.flatten(2).permute(0, 2, 1)           # (B, P, F)
        return feat


# ─── Slot Attention ───────────────────────────────────────────────────────────
class SlotAttention(nn.Module):
    """
    Iterative Slot Attention.
    inputs: (B, P, F)  — feature map
    → slots: (B, K, D) — slot representations
    → attn:  (B, K, P) — attention masks (used for visualization)
    """
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.K    = cfg.n_slots
        self.D    = cfg.slot_dim
        self.F    = cfg.feat_dim
        self.iters = cfg.n_iters
        eps        = 1e-8

        # Slot initializer: learned mean + noise
        self.slots_mu    = nn.Parameter(torch.randn(1, 1, cfg.slot_dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, cfg.slot_dim) * 0.1)

        # Attention projections
        self.q = nn.Linear(cfg.slot_dim, cfg.slot_dim, bias=False)
        self.k = nn.Linear(cfg.feat_dim, cfg.slot_dim, bias=False)
        self.v = nn.Linear(cfg.feat_dim, cfg.slot_dim, bias=False)

        # Slot update: GRU
        self.gru = nn.GRUCell(cfg.slot_dim, cfg.slot_dim)

        # Feed-forward residual
        self.ff = nn.Sequential(
            nn.LayerNorm(cfg.slot_dim),
            nn.Linear(cfg.slot_dim, cfg.slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.slot_dim * 2, cfg.slot_dim),
        )
        self.norm_input = nn.LayerNorm(cfg.feat_dim)
        self.norm_slots = nn.LayerNorm(cfg.slot_dim)
        self.scale = cfg.slot_dim ** -0.5
        self._eps  = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, P, F)
        → slots (B, K, D), attn (B, K, P)
        """
        B, P, _ = x.shape
        x_norm  = self.norm_input(x)        # (B, P, F)
        k       = self.k(x_norm)            # (B, P, D)
        v       = self.v(x_norm)            # (B, P, D)

        # Init slots with noise
        noise  = torch.randn(B, self.K, self.D, device=x.device)
        slots  = self.slots_mu + self.slots_sigma.abs() * noise  # (B, K, D)

        last_attn = None
        for _ in range(self.iters):
            slots_prev = slots
            q  = self.q(self.norm_slots(slots))          # (B, K, D)
            # Attention: (B, K, P)
            dots = torch.einsum("bkd,bpd->bkp", q, k) * self.scale
            attn = dots.softmax(dim=1) + self._eps       # competition across slots dim=1
            attn = attn / attn.sum(dim=-1, keepdim=True) # normalize across positions
            last_attn = attn
            # Weighted mean of values
            updates = torch.einsum("bkp,bpd->bkd", attn, v)   # (B, K, D)
            # GRU update (per slot)
            slots = self.gru(
                updates.reshape(B * self.K, self.D),
                slots_prev.reshape(B * self.K, self.D)
            ).reshape(B, self.K, self.D)
            slots = slots + self.ff(slots)

        return slots, last_attn   # (B,K,D), (B,K,P)


# ─── Slot Projector ───────────────────────────────────────────────────────────
class SlotProjector(nn.Module):
    """Slot vector → scalar в коридоре Value Layer [0.05, 0.95] (гомеостаз)."""
    def __init__(self, slot_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """slots: (B, K, D) → values (B, K) строго внутри [0.05, 0.95]."""
        raw = self.trunk(slots).squeeze(-1)
        return torch.sigmoid(raw) * 0.9 + 0.05


# ─── CausalVisualCortex ───────────────────────────────────────────────────────
class CausalVisualCortex(nn.Module):
    """
    Полный модуль: frame → slot_values (для GNN) + attention masks (для UI).

    Обучение через predictive coding:
      - GNN предсказывает slot_values после do()
      - Реальные slot_values из нового кадра
      - MSE loss → backprop через SlotProjector → SlotAttention → CNN
      Зрение учится выделять каузально-значимые объекты.
    """

    def __init__(self, cfg: VisionConfig, device: torch.device):
        super().__init__()
        self.cfg    = cfg
        self.device = device

        self.encoder   = CNNEncoder(cfg)
        self.attention = SlotAttention(cfg)
        self.projector = SlotProjector(cfg.slot_dim)

        self.to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        # Статистика
        self.train_losses:  deque = deque(maxlen=100)
        self.n_encode       = 0
        self.n_train        = 0

        # Предыдущие слоты для Hungarian matching
        self._prev_slot_vecs: torch.Tensor | None = None
        self._slot_order: list[int] = list(range(cfg.n_slots))

        # История slot values для обнаружения каузальных изменений
        self._slot_history: deque[torch.Tensor] = deque(maxlen=32)

    # ── Frame preprocessing ────────────────────────────────────────────────────
    def preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """
        frame_rgb: (H, W, 3) uint8
        → (1, 3, cfg.frame_h, cfg.frame_w) float32 на device
        """
        import cv2
        resized = cv2.resize(frame_rgb, (self.cfg.frame_w, self.cfg.frame_h),
                             interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).float().div_(255.0)   # (H,W,3)
        t = t.permute(2, 0, 1).unsqueeze(0)                 # (1,3,H,W)
        return t.to(self.device)

    # ── Encode ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, frame_rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        frame_rgb: (H, W, 3) uint8
        → slot_values (K,), slot_vecs (K,D), attn_masks (K, H', W')
        """
        self.eval()
        x = self.preprocess(frame_rgb)                              # (1,3,H,W)
        feats  = self.encoder(x)                                    # (1,P,F)
        slots, attn = self.attention(feats)                         # (1,K,D), (1,K,P)
        values = self.projector(slots).squeeze(0)                   # (K,)

        # Reshape attn to spatial
        H_out = self.cfg.frame_h // 4
        W_out = self.cfg.frame_w // 4
        attn_spatial = attn.squeeze(0).reshape(
            self.cfg.n_slots, H_out, W_out
        )   # (K, H', W')

        slot_vecs = slots.squeeze(0)   # (K, D)

        # Hungarian matching для стабильности slot order
        if self._prev_slot_vecs is not None:
            order = self._hungarian_match(slot_vecs, self._prev_slot_vecs)
            slot_vecs   = slot_vecs[order]
            values      = values[order]
            attn_spatial = attn_spatial[order]
            self._slot_order = [self._slot_order[i] for i in order]

        self._prev_slot_vecs = slot_vecs.detach().clone()
        self._slot_history.append(values.detach().clone())
        self.n_encode += 1

        return values.detach(), slot_vecs.detach(), attn_spatial.detach()

    def _hungarian_match(self, curr: torch.Tensor, prev: torch.Tensor) -> list[int]:
        """
        Cosine similarity matching для сохранения slot identity.
        curr, prev: (K, D)
        """
        K = curr.shape[0]
        curr_n = F.normalize(curr.float(), dim=-1).cpu().numpy()
        prev_n = F.normalize(prev.float(), dim=-1).cpu().numpy()
        # Cost matrix: 1 - cosine_sim (хотим максимизировать сходство)
        cost = 1.0 - (curr_n @ prev_n.T)   # (K, K)
        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost)
            # col_ind[i] = best prev-slot for curr-slot i
            # Нам нужен обратный маппинг: для позиции i выбрать слот из prev
            order = [0] * K
            for c, p in zip(col_ind, row_ind):
                order[c] = p
        else:
            # Greedy fallback
            used = set()
            order = list(range(K))
            for i in range(K):
                best_j = -1
                best_s = -2.0
                for j in range(K):
                    if j in used:
                        continue
                    s = float(curr_n[i] @ prev_n[j])
                    if s > best_s:
                        best_s, best_j = s, j
                used.add(best_j)
                order[i] = best_j
        return order

    # ── Train step (predictive coding) ────────────────────────────────────────
    def train_on_prediction_error(
        self,
        frame_before:  np.ndarray,
        frame_after:   np.ndarray,
        gnn_predicted: torch.Tensor,   # (K,) предсказание GNN
    ) -> float:
        """
        Обучаем зрение: slot_values_after должны совпасть с gnn_predicted.
        Это заставляет SlotAttention выделять каузально-значимые объекты.
        """
        self.train()
        x_after  = self.preprocess(frame_after)
        feats    = self.encoder(x_after)
        slots, _ = self.attention(feats)
        values   = self.projector(slots).squeeze(0)   # (K,)

        # L_pred: GNN предсказал values, слоты должны это подтвердить
        l_pred = F.mse_loss(values, gnn_predicted.detach().to(self.device))

        # L_consistency: slot values не должны взрываться (entropy)
        l_ent = -(values * (values + 1e-6).log()
                  + (1 - values) * (1 - values + 1e-6).log()).mean()

        loss = l_pred + 0.05 * l_ent
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()

        v = float(loss.item())
        self.train_losses.append(v)
        self.n_train += 1
        return v

    # ── Variable discovery ────────────────────────────────────────────────────
    def slot_variability(self) -> np.ndarray:
        """
        Насколько каждый слот изменялся — мера "информативности".
        Возвращает (K,) variability scores.
        """
        if len(self._slot_history) < 4:
            return np.ones(self.cfg.n_slots) * 0.5
        hist = torch.stack(list(self._slot_history), dim=0)   # (T, K)
        return hist.std(dim=0).cpu().numpy()

    # ── Attention maps for UI ─────────────────────────────────────────────────
    def get_slot_masks_base64(self, attn_masks: torch.Tensor) -> list[str]:
        """
        attn_masks: (K, H', W') → list of K base64 PNG images (grayscale heatmap).
        Используется для визуализации в UI.
        """
        import base64
        from io import BytesIO
        try:
            from PIL import Image as PILImage
        except ImportError:
            return []

        masks_b64 = []
        K = attn_masks.shape[0]
        attn_np = attn_masks.cpu().float().numpy()

        # Upscale to cfg.frame_h × cfg.frame_w
        import cv2
        for k in range(K):
            mask = attn_np[k]                                      # (H', W')
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            mask_up = cv2.resize(mask, (self.cfg.frame_w, self.cfg.frame_h),
                                 interpolation=cv2.INTER_LINEAR)
            img_arr = (mask_up * 255).astype(np.uint8)
            img = PILImage.fromarray(img_arr, mode="L")
            buf = BytesIO()
            img.save(buf, format="PNG")
            masks_b64.append(base64.b64encode(buf.getvalue()).decode())
        return masks_b64

    # ── Snapshot ─────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        var = self.slot_variability()
        return {
            "n_slots":      self.cfg.n_slots,
            "n_encode":     self.n_encode,
            "n_train":      self.n_train,
            "mean_loss":    float(np.mean(list(self.train_losses))) if self.train_losses else 0.0,
            "variability":  [round(float(v), 4) for v in var],
            "active_slots": int((var > 0.03).sum()),
        }


# ─── Фабрика с умолчаниями для гуманоидной среды ────────────────────────────
def make_visual_cortex(device: torch.device, n_slots: int = 8) -> CausalVisualCortex:
    cfg = VisionConfig(
        frame_h=64, frame_w=64,
        cnn_channels=[16, 32, 32],
        feat_dim=32,
        n_slots=n_slots,
        slot_dim=64,
        n_iters=3,
        lr=3e-4,
    )
    return CausalVisualCortex(cfg, device)
