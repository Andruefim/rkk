"""
causal_vision.py — Causal Visual Cortex (Фаза 12) + Level 1-C: Reconstruction Decoder.

ИЗМЕНЕНИЯ (Level 1-C: Slot Reconstruction Decoder):
  - SlotDecoder: обратное преобразование slots → feature_map
  - L_recon = MSE(decode(slots), encode(frame)) добавлен к predictive coding loss
  - Без reconstruction loss SlotAttention не имеет причины кодировать
    что-то осмысленное — он выучит тривиальное разбиение пространства
  - С L_recon slots вынуждены представлять реальные объекты сцены
  - RKK_VISION_RECON_WEIGHT=0.3 — вес reconstruction loss (default 0.3)
  - RKK_VISION_RECON_ENABLED=1 — включить decoder (default)

Все оригинальные методы сохранены.
"""
from __future__ import annotations

import os
import queue
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
from dataclasses import dataclass, field
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _recon_enabled() -> bool:
    return os.environ.get("RKK_VISION_RECON_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _recon_weight() -> float:
    try:
        return float(np.clip(float(os.environ.get("RKK_VISION_RECON_WEIGHT", "0.3")), 0.0, 1.0))
    except ValueError:
        return 0.3


@dataclass
class VisionConfig:
    frame_h:     int   = 64
    frame_w:     int   = 64
    cnn_channels: list = field(default_factory=lambda: [16, 32, 32])
    feat_dim:    int   = 32
    n_slots:     int   = 8
    slot_dim:    int   = 64
    n_iters:     int   = 2
    lr:          float = 3e-4
    recon_weight: float = 0.1


class CNNEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        ch = [3] + cfg.cnn_channels
        layers = []
        for i in range(len(ch) - 1):
            stride = 2 if i < 2 else 1
            layers += [
                nn.Conv2d(ch[i], ch[i+1], kernel_size=5, stride=stride, padding=2),
                nn.GroupNorm(max(1, ch[i+1]//8), ch[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.backbone = nn.Sequential(*layers)
        self.pos_embed = nn.Sequential(
            nn.Conv2d(ch[-1] + 2, cfg.feat_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.feat_dim, cfg.feat_dim, kernel_size=1),
        )
        h_out = cfg.frame_h // 4
        w_out = cfg.frame_w // 4
        self.n_positions = h_out * w_out
        self._h_out = h_out
        self._w_out = w_out
        ys = torch.linspace(-1, 1, h_out)
        xs = torch.linspace(-1, 1, w_out)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer("pos_grid", torch.stack([grid_x, grid_y], dim=0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        B = feat.shape[0]
        pos = self.pos_grid.expand(B, -1, -1, -1)
        feat = torch.cat([feat, pos], dim=1)
        feat = self.pos_embed(feat)
        feat = feat.flatten(2).permute(0, 2, 1)
        return feat


class SlotAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.K    = cfg.n_slots
        self.D    = cfg.slot_dim
        self.F    = cfg.feat_dim
        self.iters = cfg.n_iters
        eps = 1e-8
        self.slots_mu    = nn.Parameter(torch.randn(1, 1, cfg.slot_dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, cfg.slot_dim) * 0.1)
        self.q = nn.Linear(cfg.slot_dim, cfg.slot_dim, bias=False)
        self.k = nn.Linear(cfg.feat_dim, cfg.slot_dim, bias=False)
        self.v = nn.Linear(cfg.feat_dim, cfg.slot_dim, bias=False)
        self.gru = nn.GRUCell(cfg.slot_dim, cfg.slot_dim)
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
        B, P, _ = x.shape
        x_norm  = self.norm_input(x)
        k = self.k(x_norm)
        v = self.v(x_norm)
        noise = torch.randn(B, self.K, self.D, device=x.device)
        slots = self.slots_mu + self.slots_sigma.abs() * noise
        last_attn = None
        for _ in range(self.iters):
            slots_prev = slots
            q = self.q(self.norm_slots(slots))
            dots = torch.einsum("bkd,bpd->bkp", q, k) * self.scale
            attn = dots.softmax(dim=1) + self._eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            last_attn = attn
            updates = torch.einsum("bkp,bpd->bkd", attn, v)
            slots = self.gru(
                updates.reshape(B * self.K, self.D),
                slots_prev.reshape(B * self.K, self.D)
            ).reshape(B, self.K, self.D)
            slots = slots + self.ff(slots)
        return slots, last_attn


# ── Level 1-C: Slot Reconstruction Decoder ────────────────────────────────────
class SlotDecoder(nn.Module):
    """
    Decoder: slots (K, D) → reconstructed feature_map (P, F).

    Архитектура:
      Каждый slot генерирует маску alpha_k (P,) и вклад r_k (P, F).
      Feature map = sum_k(alpha_k * r_k), где alpha_k — softmax веса.
      
      Это Mixture Decoder из оригинальной SlotAttention paper.
      Reconstruction loss L_recon = MSE(decode(slots), encode(frame))
      заставляет slots представлять реальные объекты.

    Параметры:
      slot_dim: D
      feat_dim: F
      n_positions: P = H'*W'
    """

    def __init__(self, slot_dim: int, feat_dim: int, n_positions: int, hidden: int = 64):
        super().__init__()
        self.n_positions = n_positions
        self.feat_dim = feat_dim

        # Generates (feature_contribution + alpha_logit) per slot per position
        # Input: slot vector (D) + position embedding (2)
        # Output: feat_dim (feature) + 1 (alpha logit)
        self.slot_to_feat = nn.Sequential(
            nn.Linear(slot_dim + 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim + 1),  # feat_dim colors + 1 alpha
        )

        # Learnable position embeddings for decoder
        # (P, 2) — normalized grid coords
        self._n_pos = n_positions

        # Initialize small
        for m in self.slot_to_feat:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    def _make_pos_grid(self, n_positions: int, device: torch.device) -> torch.Tensor:
        """Create normalized position grid (n_positions, 2)."""
        side = int(np.sqrt(n_positions))
        ys = torch.linspace(-1, 1, side, device=device)
        xs = torch.linspace(-1, 1, side, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (P, 2)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        slots: (B, K, D)
        Returns: reconstructed feature_map (B, P, F)
        """
        B, K, D = slots.shape
        P = self.n_positions
        device = slots.device

        # Position grid (P, 2)
        pos_grid = self._make_pos_grid(P, device)  # (P, 2)

        # For each slot and each position: concatenate slot vector with position
        # slots: (B, K, D) → expand to (B, K, P, D)
        slots_exp = slots.unsqueeze(2).expand(B, K, P, D)  # (B, K, P, D)
        # pos_grid: (P, 2) → expand to (B, K, P, 2)
        pos_exp = pos_grid.unsqueeze(0).unsqueeze(0).expand(B, K, P, 2)  # (B, K, P, 2)

        # Concatenate: (B, K, P, D+2)
        x = torch.cat([slots_exp, pos_exp], dim=-1)

        # Reshape for linear: (B*K*P, D+2)
        x_flat = x.reshape(B * K * P, D + 2)
        out = self.slot_to_feat(x_flat)  # (B*K*P, F+1)
        out = out.reshape(B, K, P, self.feat_dim + 1)

        # Split into features and alpha
        feat_k = out[..., :self.feat_dim]   # (B, K, P, F)
        alpha_k = out[..., self.feat_dim:]   # (B, K, P, 1)

        # Softmax over slots dimension → mixture weights
        alpha_k = F.softmax(alpha_k, dim=1)  # (B, K, P, 1)

        # Weighted sum: (B, P, F)
        reconstructed = (alpha_k * feat_k).sum(dim=1)  # (B, P, F)

        return reconstructed


class SlotProjector(nn.Module):
    def __init__(self, slot_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        raw = self.trunk(slots).squeeze(-1)
        return torch.sigmoid(raw) * 0.9 + 0.05


class CausalVisualCortex(nn.Module):
    """
    Полный модуль: frame → slot_values (для GNN) + attention masks (для UI).
    
    Level 1-C additions:
      - SlotDecoder: decode(slots) → feature_map
      - train_on_prediction_error: добавлен L_recon = MSE(decode, encode)
      - recon_loss_history: для мониторинга
    """

    def __init__(self, cfg: VisionConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.encoder = CNNEncoder(cfg)
        self.attention = SlotAttention(cfg)
        self.projector = SlotProjector(cfg.slot_dim)

        # Level 1-C: Reconstruction decoder
        n_positions = self.encoder.n_positions
        if _recon_enabled():
            self.decoder = SlotDecoder(
                slot_dim=cfg.slot_dim,
                feat_dim=cfg.feat_dim,
                n_positions=n_positions,
                hidden=64,
            )
        else:
            self.decoder = None

        self.to(device)

        # Single optimizer for all components (including decoder)
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        self.train_losses: deque = deque(maxlen=100)
        self.recon_losses: deque = deque(maxlen=100)  # Level 1-C
        self.pred_losses: deque = deque(maxlen=100)
        self.n_encode = 0
        self.n_train = 0

        self._prev_slot_vecs: torch.Tensor | None = None
        self._slot_order: list[int] = list(range(cfg.n_slots))
        self._slot_history: deque[torch.Tensor] = deque(maxlen=32)
        self._variability_cache: np.ndarray | None = None
        self._variability_cache_at_encode: int = -1

        # Store last encoded features for reconstruction training
        self._last_encoded_feats: torch.Tensor | None = None

    def preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(frame_rgb)
        x = torch.from_numpy(arr).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).float()
        x = F.interpolate(
            x,
            size=(self.cfg.frame_h, self.cfg.frame_w),
            mode="bilinear",
            align_corners=False,
        )
        return x * (1.0 / 255.0)

    @torch.no_grad()
    def encode(self, frame_rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        x = self.preprocess(frame_rgb)
        feats = self.encoder(x)
        # Store features for reconstruction training
        self._last_encoded_feats = feats.detach()

        slots, attn = self.attention(feats)
        values = self.projector(slots).squeeze(0)

        H_out = self.cfg.frame_h // 4
        W_out = self.cfg.frame_w // 4
        attn_spatial = attn.squeeze(0).reshape(self.cfg.n_slots, H_out, W_out)
        slot_vecs = slots.squeeze(0)

        if self._prev_slot_vecs is not None:
            order = self._hungarian_match(slot_vecs, self._prev_slot_vecs)
            slot_vecs = slot_vecs[order]
            values = values[order]
            attn_spatial = attn_spatial[order]
            self._slot_order = [self._slot_order[i] for i in order]

        self._prev_slot_vecs = slot_vecs.detach().clone()
        self._slot_history.append(values.detach().clone())
        self.n_encode += 1
        self._variability_cache = None

        return values.detach(), slot_vecs.detach(), attn_spatial.detach()

    def _hungarian_match(self, curr: torch.Tensor, prev: torch.Tensor) -> list[int]:
        K = curr.shape[0]
        curr_n = F.normalize(curr.float(), dim=-1).cpu().numpy()
        prev_n = F.normalize(prev.float(), dim=-1).cpu().numpy()
        cost = 1.0 - (curr_n @ prev_n.T)
        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost)
            order = [0] * K
            for c, p in zip(col_ind, row_ind):
                order[c] = p
        else:
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

    def train_on_prediction_error(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        gnn_predicted: torch.Tensor,
    ) -> float:
        """
        Level 1-C: Combined training with predictive coding + reconstruction loss.
        
        L_total = L_pred + recon_weight * L_recon

        L_pred: slot_values_after should match gnn_predicted (causal grounding)
        L_recon: decode(slots_after) should reconstruct encoded features (visual grounding)
        """
        self.train()
        x_after = self.preprocess(frame_after)
        feats_after = self.encoder(x_after)          # (1, P, F)
        slots_after, _ = self.attention(feats_after)  # (1, K, D)
        values_after = self.projector(slots_after).squeeze(0)  # (K,)

        # L_pred: predictive coding loss (original)
        l_pred = F.mse_loss(values_after, gnn_predicted.detach().to(self.device))

        # L_ent: entropy regularization
        l_ent = -(
            values_after * (values_after + 1e-6).log()
            + (1 - values_after) * (1 - values_after + 1e-6).log()
        ).mean()

        # Level 1-C: L_recon — reconstruction loss
        l_recon = torch.tensor(0.0, device=self.device)
        if _recon_enabled() and self.decoder is not None:
            try:
                # Reconstruct feature map from slots
                reconstructed = self.decoder(slots_after)  # (1, P, F)
                # Compare with actual encoded features
                l_recon = F.mse_loss(reconstructed, feats_after.detach())
            except Exception as e:
                pass  # decoder might fail on unusual inputs

        rw = _recon_weight() if _recon_enabled() else 0.0
        loss = l_pred + 0.05 * l_ent + rw * l_recon

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()

        v = float(loss.item())
        self.train_losses.append(v)
        if _recon_enabled():
            self.recon_losses.append(float(l_recon.item()))
        self.pred_losses.append(float(l_pred.item()))
        self.n_train += 1
        return v

    def train_reconstruction_only(self, frame_rgb: np.ndarray) -> float | None:
        """
        Level 1-C: Standalone reconstruction training (no predictive coding target needed).
        Can be called even without GNN predictions, e.g. during early training.
        Used for: warming up decoder before predictive coding starts.
        """
        if not _recon_enabled() or self.decoder is None:
            return None

        self.train()
        x = self.preprocess(frame_rgb)
        feats = self.encoder(x)          # (1, P, F)
        slots, _ = self.attention(feats)  # (1, K, D)

        try:
            reconstructed = self.decoder(slots)  # (1, P, F)
            l_recon = F.mse_loss(reconstructed, feats.detach())
        except Exception:
            return None

        self.optim.zero_grad()
        l_recon.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()

        v = float(l_recon.item())
        self.recon_losses.append(v)
        return v

    def slot_variability(self) -> np.ndarray:
        if self._variability_cache is not None and self._variability_cache_at_encode == self.n_encode:
            return self._variability_cache
        if len(self._slot_history) < 4:
            out = np.ones(self.cfg.n_slots) * 0.5
        else:
            hist = torch.stack(list(self._slot_history), dim=0)
            out = hist.std(dim=0).cpu().numpy()
        self._variability_cache = out
        self._variability_cache_at_encode = self.n_encode
        return out

    def get_slot_masks_base64(self, attn_masks: torch.Tensor) -> list[str]:
        import cv2
        from io import BytesIO
        try:
            from PIL import Image as PILImage
        except ImportError:
            return []

        masks_b64 = []
        K = attn_masks.shape[0]
        attn_np = attn_masks.cpu().float().numpy()
        mw, mh = 48, 48
        for k in range(K):
            mask = attn_np[k]
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            mask_up = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_LINEAR)
            img_arr = (mask_up * 255).astype(np.uint8)
            img = PILImage.fromarray(img_arr, mode="L")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
            masks_b64.append(base64.b64encode(buf.getvalue()).decode())
        return masks_b64

    def snapshot(self) -> dict:
        var = self.slot_variability()
        mean_recon = float(np.mean(list(self.recon_losses))) if self.recon_losses else 0.0
        mean_pred = float(np.mean(list(self.pred_losses))) if self.pred_losses else 0.0
        return {
            "n_slots":       self.cfg.n_slots,
            "n_encode":      self.n_encode,
            "n_train":       self.n_train,
            "mean_loss":     float(np.mean(list(self.train_losses))) if self.train_losses else 0.0,
            "mean_recon_loss": round(mean_recon, 5),  # Level 1-C
            "mean_pred_loss": round(mean_pred, 5),
            "recon_enabled": _recon_enabled(),         # Level 1-C
            "recon_weight":  _recon_weight(),
            "variability":   [round(float(v), 4) for v in var],
            "active_slots":  int((var > 0.03).sum()),
        }


def make_visual_cortex(device: torch.device, n_slots: int = 8) -> CausalVisualCortex:
    cfg = VisionConfig(
        frame_h=64, frame_w=64,
        cnn_channels=[16, 32, 32],
        feat_dim=32,
        n_slots=n_slots,
        slot_dim=64,
        n_iters=2,
        lr=3e-4,
    )
    return CausalVisualCortex(cfg, device)