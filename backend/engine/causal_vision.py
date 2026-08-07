"""causal_vision.py — Causal Visual Cortex (Фаза 12): SlotAttention + predictive coding."""
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


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def recon_training_enabled() -> bool:
    return os.environ.get("RKK_VISION_RECON", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def recon_every() -> int:
    return max(1, _env_int("RKK_VISION_RECON_EVERY", 1))


def recon_batch() -> int:
    # 2 кадра ≈ 250 мс на CPU при 8 слотах — дешевле, чем 1 или 4 (батчинг conv).
    return max(1, _env_int("RKK_VISION_RECON_BATCH", 2))


def recon_steps() -> int:
    return max(1, _env_int("RKK_VISION_RECON_STEPS", 1))


def recon_buffer_size() -> int:
    return max(2, _env_int("RKK_VISION_RECON_BUFFER", 32))


def spatial_slot_init() -> bool:
    return os.environ.get("RKK_VISION_SPATIAL_SLOTS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def mask_entropy_weight() -> float:
    """Штраф за «размазанность» альфа-масок: пиксель должен принадлежать одному слоту."""
    return _env_float("RKK_VISION_MASK_ENTROPY_W", 0.05)


def slot_diversity_weight() -> float:
    """Штраф за схлопывание слотов в один и тот же вектор."""
    return _env_float("RKK_VISION_SLOT_DIVERSITY_W", 0.05)


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
        # Слоты, инициализированные одним и тем же гауссианом, при коротком бюджете
        # обучения схлопываются в одинаковые «средние» слоты и маски выходят
        # равномерными. Якоря по сетке кадра сразу разводят слоты по местам сцены.
        self.spatial_init = spatial_slot_init()
        self.slot_from_feat = nn.Linear(cfg.feat_dim, cfg.slot_dim)

    def _init_slots(self, x_norm: torch.Tensor) -> torch.Tensor:
        B, P, _ = x_norm.shape
        noise = torch.randn(B, self.K, self.D, device=x_norm.device)
        base = self.slots_mu + self.slots_sigma.abs() * noise
        if not self.spatial_init or P < self.K:
            return base
        anchors = torch.linspace(0, P - 1, self.K, device=x_norm.device).round().long()
        seeds = self.slot_from_feat(x_norm[:, anchors, :])
        return seeds + 0.1 * base

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = x.shape
        x_norm  = self.norm_input(x)
        k = self.k(x_norm)
        v = self.v(x_norm)
        slots = self._init_slots(x_norm)
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


class SlotDecoder(nn.Module):
    """
    Spatial broadcast decoder: каждый слот разворачивается в свою RGB-карту и
    альфа-маску, кадр собирается как сумма слотов с softmax-альфой по слотам.

    Именно этот путь заставляет SlotAttention делить сцену на объекты: без него
    кора обучалась только скалярным MSE к прогнозу GNN и маски оставались
    размазанными, из-за чего vision-резолв цели не находил «пиковый» слот.
    """

    def __init__(self, cfg: VisionConfig, hidden: int | None = None, start: int = 8):
        super().__init__()
        hidden = hidden if hidden is not None else _env_int("RKK_VISION_DEC_HIDDEN", 32)
        self.h = cfg.frame_h // 2
        self.w = cfg.frame_w // 2
        self.h0 = max(4, cfg.frame_h // start)
        self.w0 = max(4, cfg.frame_w // start)
        ys = torch.linspace(-1, 1, self.h0)
        xs = torch.linspace(-1, 1, self.w0)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("pos_grid", torch.stack([grid_x, grid_y], dim=0).unsqueeze(0))
        # Апсемплинг с грубой сетки: при декодировании сразу в H/4 мелкие объекты
        # сцены занимают меньше пикселя и реконструкция вырождается в средний фон.
        self.net = nn.Sequential(
            nn.Conv2d(cfg.slot_dim + 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=3, padding=1),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, D = slots.shape
        x = slots.reshape(B * K, D, 1, 1).expand(-1, -1, self.h0, self.w0)
        pos = self.pos_grid.expand(B * K, -1, -1, -1)
        out = self.net(torch.cat([x, pos], dim=1))
        if out.shape[-2:] != (self.h, self.w):
            out = F.interpolate(out, size=(self.h, self.w), mode="bilinear", align_corners=False)
        rgb = out[:, :3].sigmoid().reshape(B, K, 3, self.h, self.w)
        alpha = out[:, 3:].reshape(B, K, 1, self.h, self.w).softmax(dim=1)
        recon = (rgb * alpha).sum(dim=1)
        return recon, alpha.squeeze(2)


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
        self.decoder = SlotDecoder(cfg)

        self.to(device)

        # Single optimizer for all components (including decoder)
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        self.train_losses: deque = deque(maxlen=100)
        self.pred_losses: deque = deque(maxlen=100)
        self.recon_losses: deque = deque(maxlen=100)
        self.n_encode = 0
        self.n_train = 0
        self.n_recon_train = 0

        # Кадры для self-supervised реконструкции: рендер идёт редко
        # (RKK_VISION_ENCODE_EVERY), поэтому учимся мини-батчами по недавним кадрам.
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=recon_buffer_size())
        self._train_lock = threading.Lock()

        self._prev_slot_vecs: torch.Tensor | None = None
        self._slot_order: list[int] = list(range(cfg.n_slots))
        self._slot_history: deque[torch.Tensor] = deque(maxlen=32)
        self._variability_cache: np.ndarray | None = None
        self._variability_cache_at_encode: int = -1

        # Store last encoded features for reconstruction training
        self._last_encoded_feats: torch.Tensor | None = None
        self._last_attn_spatial: torch.Tensor | None = None

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
        self._last_attn_spatial = attn_spatial.detach()
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

    def remember_frame(self, frame_rgb: np.ndarray) -> None:
        """Класть кадр в буфер реконструкции (вызывается на каждом encode)."""
        if not recon_training_enabled():
            return
        try:
            with torch.no_grad():
                self._frame_buffer.append(self.preprocess(frame_rgb).squeeze(0).cpu())
        except Exception:
            pass

    def _recon_target(self, x: torch.Tensor) -> torch.Tensor:
        """Кадр в разрешении декодера."""
        if x.shape[-2:] == (self.decoder.h, self.decoder.w):
            return x
        return F.interpolate(
            x, size=(self.decoder.h, self.decoder.w), mode="bilinear", align_corners=False
        )

    def _decomposition_penalty(self, slots: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Регуляризаторы, из-за отсутствия которых слоты сходились в один усреднённый."""
        total = slots.new_zeros(())
        w_ent = mask_entropy_weight()
        if w_ent > 0.0:
            a = alpha.clamp_min(1e-6)
            total = total + w_ent * (-(a * a.log()).sum(dim=1)).mean()
        w_div = slot_diversity_weight()
        if w_div > 0.0:
            s = F.normalize(slots, dim=-1)
            sim = torch.einsum("bkd,bjd->bkj", s, s)
            eye = torch.eye(s.shape[1], device=s.device).unsqueeze(0)
            total = total + w_div * (sim - eye).pow(2).mean()
        return total

    def train_reconstruction(self, batch_size: int | None = None, steps: int | None = None) -> float | None:
        """
        Self-supervised шаг object-centric обучения: слоты должны совместно
        восстанавливать кадр, конкурируя за пиксели через softmax-альфу.
        Возвращает последний recon loss или None, если данных не хватило.
        """
        if not recon_training_enabled():
            return None
        bs = batch_size or recon_batch()
        n_steps = steps or recon_steps()
        if len(self._frame_buffer) < min(bs, 2):
            return None

        last: float | None = None
        with self._train_lock:
            self.train()
            for _ in range(n_steps):
                idx = np.random.choice(len(self._frame_buffer), min(bs, len(self._frame_buffer)), replace=False)
                x = torch.stack([self._frame_buffer[int(i)] for i in idx]).to(self.device)
                feats = self.encoder(x)
                slots, _ = self.attention(feats)
                recon, alpha = self.decoder(slots)
                l_recon = F.mse_loss(recon, self._recon_target(x))
                loss = l_recon + self._decomposition_penalty(slots, alpha)

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optim.step()

                # В метрику пишем чистый MSE: со штрафами число нечитаемо.
                last = float(l_recon.item())
                self.recon_losses.append(last)
                self.n_recon_train += 1
            self.eval()
        return last

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
        with self._train_lock:
            self.train()
            x_after = self.preprocess(frame_after)
            feats_after = self.encoder(x_after)          # (1, P, F)
            slots_after, _ = self.attention(feats_after)  # (1, K, D)
            values_after = self.projector(slots_after).squeeze(0)  # (K,)

            # L_pred: predictive coding loss (original)
            # BUGFIX: Убрано .detach(), теперь градиент течет обратно в World Model
            l_pred = F.mse_loss(values_after, gnn_predicted.to(self.device))

            # L_ent: entropy regularization
            l_ent = -(
                values_after * (values_after + 1e-6).log()
                + (1 - values_after) * (1 - values_after + 1e-6).log()
            ).mean()

            loss = l_pred + 0.05 * l_ent

            l_recon = None
            if recon_training_enabled():
                recon, _alpha = self.decoder(slots_after)
                l_recon = F.mse_loss(recon, self._recon_target(x_after))
                loss = loss + self.cfg.recon_weight * l_recon

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optim.step()
            self.eval()

        v = float(loss.item())
        self.train_losses.append(v)
        self.pred_losses.append(float(l_pred.item()))
        if l_recon is not None:
            self.recon_losses.append(float(l_recon.item()))
        self.n_train += 1
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

    def mask_peakiness(self) -> float:
        """
        Во сколько раз пик маски выше среднего. Ниже ~1.8 слот считается размазанным,
        и vision-резолв цели его отбрасывает — это прямой индикатор обученности зрения.
        """
        attn = self._last_attn_spatial
        if attn is None:
            return 0.0
        flat = attn.reshape(attn.shape[0], -1).float()
        mean = flat.mean(dim=-1).clamp_min(1e-8)
        return float((flat.max(dim=-1).values / mean).mean())

    def snapshot(self) -> dict:
        var = self.slot_variability()
        mean_pred = float(np.mean(list(self.pred_losses))) if self.pred_losses else 0.0
        mean_recon = float(np.mean(list(self.recon_losses))) if self.recon_losses else 0.0
        return {
            "n_slots":       self.cfg.n_slots,
            "n_encode":      self.n_encode,
            "n_train":       self.n_train,
            "n_recon_train": self.n_recon_train,
            "mean_loss":     float(np.mean(list(self.train_losses))) if self.train_losses else 0.0,
            "mean_pred_loss": round(mean_pred, 5),
            "mean_recon_loss": round(mean_recon, 5),
            "mask_peakiness": round(self.mask_peakiness(), 3),
            "variability":   [round(float(v), 4) for v in var],
            "active_slots":  int((var > 0.03).sum()),
        }


def vision_config_from_env(n_slots: int = 8) -> VisionConfig:
    """
    Дефолты рассчитаны на CPU. На GPU имеет смысл поднять разрешение и ёмкость:
    RKK_VISION_FRAME_H/W=128, RKK_VISION_CNN_CHANNELS=32,64,64, RKK_VISION_ITERS=3.
    Разрешение должно делиться на 4 (энкодер даёт сетку H/4 × W/4).
    """
    raw_channels = os.environ.get("RKK_VISION_CNN_CHANNELS", "16,32,32")
    try:
        channels = [int(c) for c in raw_channels.replace(" ", "").split(",") if c]
    except ValueError:
        channels = [16, 32, 32]
    h = max(16, _env_int("RKK_VISION_FRAME_H", 64) // 4 * 4)
    w = max(16, _env_int("RKK_VISION_FRAME_W", 64) // 4 * 4)
    return VisionConfig(
        frame_h=h,
        frame_w=w,
        cnn_channels=channels or [16, 32, 32],
        feat_dim=_env_int("RKK_VISION_FEAT_DIM", 32),
        n_slots=n_slots,
        slot_dim=_env_int("RKK_VISION_SLOT_DIM", 64),
        n_iters=max(1, _env_int("RKK_VISION_ITERS", 2)),
        lr=_env_float("RKK_VISION_LR", 3e-4),
        recon_weight=_env_float("RKK_VISION_RECON_WEIGHT", 0.1),
    )


def make_visual_cortex(device: torch.device, n_slots: int = 8) -> CausalVisualCortex:
    return CausalVisualCortex(vision_config_from_env(n_slots), device)