"""Object-centric slot embedding dynamics (JEPA) + ego geometry head.

Predicts ``(z_{t+1}, ego_{t+1})`` from ``(z_t, ego_t, a_t)`` for the active
visual target. Used as a residual on OWM odometry during FOV loss / occlusion.

This is *not* the graph JEPA over scalar ``slot_*`` nodes and *not* a Dreamer
RSSM / actor-critic. GNN ``forward_dynamics`` stays the executive SCM.
"""
from __future__ import annotations

import math
import os
from collections import deque
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_DIM = 4  # dtheta, ds, intent_stride, intent_gait_coupling
EGO_DIM = 2
_DEFAULT_HIDDEN = 32
_DEFAULT_SLOT_DIM = 64
_BUFFER_MAX = 256
_TRAIN_MIN = 8
_Z_RES_SCALE = 0.25
_EGO_STEP_MAX = 0.5


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return float(default)


def _ei(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return int(default)


def slot_dynamics_enabled() -> bool:
    raw = os.environ.get("RKK_SLOT_DYNAMICS", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def slot_dyn_blend() -> float:
    return float(max(0.0, min(1.0, _ef("RKK_SLOT_DYN_BLEND", 0.35))))


def slot_dyn_horizon() -> int:
    return max(1, _ei("RKK_SLOT_DYN_HORIZON", 8))


def slot_dyn_train_every() -> int:
    return max(1, _ei("RKK_SLOT_DYN_TRAIN_EVERY", 8))


def slot_dyn_sigma_scale() -> float:
    return float(max(0.05, min(1.0, _ef("RKK_SLOT_DYN_SIGMA_SCALE", 0.4))))


def slot_dyn_agree_m() -> float:
    """Max |ego_hat − ego_odom| (m) to treat predictor as agreeing with odometry."""
    return max(0.02, _ef("RKK_SLOT_DYN_AGREE_M", 0.25))


def slot_dyn_ema_tau() -> float:
    return float(max(0.0, min(1.0, _ef("RKK_SLOT_DYN_EMA_TAU", _ef("RKK_JEPA_EMA_TAU", 0.006)))))


def slot_dyn_lr() -> float:
    return max(1e-5, _ef("RKK_SLOT_DYN_LR", 3e-4))


def pack_action(
    dtheta: float = 0.0,
    ds: float = 0.0,
    intent_stride: float = 0.5,
    intent_gait_coupling: float = 0.5,
) -> tuple[float, float, float, float]:
    return (
        float(dtheta),
        float(ds),
        float(intent_stride),
        float(intent_gait_coupling),
    )


def _as_tensor(x: Any, *, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=dtype).reshape(-1)
    else:
        t = torch.tensor(list(x) if not isinstance(x, (int, float)) else [x], device=device, dtype=dtype)
        t = t.reshape(-1)
    if t.numel() < dim:
        t = F.pad(t, (0, dim - t.numel()))
    return t[:dim]


class SlotDynamics(nn.Module):
    """JEPA predictor: z residual + Δego, EMA target encoder, replay buffer."""

    def __init__(
        self,
        slot_dim: int = _DEFAULT_SLOT_DIM,
        *,
        hidden: int = _DEFAULT_HIDDEN,
        device: torch.device | str | None = None,
        lr: float | None = None,
    ) -> None:
        super().__init__()
        self.slot_dim = int(max(4, slot_dim))
        self.hidden = int(max(8, hidden))
        self.device = torch.device(device or "cpu")
        h = self.hidden
        d = self.slot_dim

        self.node_enc = nn.Sequential(nn.Linear(d, h), nn.Tanh())
        self.ego_enc = nn.Sequential(nn.Linear(EGO_DIM, h), nn.Tanh())
        self.action_enc = nn.Sequential(nn.Linear(ACTION_DIM, h), nn.Tanh())
        self.predictor = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.z_head = nn.Linear(h, d)
        self.ego_head = nn.Linear(h, EGO_DIM)
        # Near-identity at init so untrained predict ≈ odom prior.
        nn.init.zeros_(self.z_head.weight)
        nn.init.zeros_(self.z_head.bias)
        nn.init.zeros_(self.ego_head.weight)
        nn.init.zeros_(self.ego_head.bias)

        self.target_enc = nn.Sequential(nn.Linear(d, h), nn.Tanh())
        self.target_enc.load_state_dict(self.node_enc.state_dict())
        for p in self.target_enc.parameters():
            p.requires_grad = False

        self.to(self.device)
        self.optim = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=float(lr if lr is not None else slot_dyn_lr()),
        )
        self.train_steps = 0
        self.n_predict = 0
        self.n_calls = 0
        self.last_loss: float | None = None
        self._buffer: deque[dict[str, Any]] = deque(maxlen=_BUFFER_MAX)
        self._prev: dict[str, Any] | None = None

    def _enc_h(self, z: torch.Tensor, ego: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.node_enc(z), self.ego_enc(ego), self.action_enc(action)],
            dim=-1,
        )

    def forward(
        self,
        z: torch.Tensor,
        ego: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched: z (B,D), ego (B,2), action (B,4) → z_hat, ego_hat."""
        h = self.predictor(self._enc_h(z, ego, action))
        z_res = torch.tanh(self.z_head(h)) * _Z_RES_SCALE
        d_ego = torch.tanh(self.ego_head(h)) * _EGO_STEP_MAX
        z_hat = z + z_res
        ego_hat = ego + d_ego
        return z_hat, ego_hat

    @torch.no_grad()
    def predict(
        self,
        z: Sequence[float] | torch.Tensor,
        ego: tuple[float, float] | Sequence[float],
        action: Sequence[float] | None = None,
    ) -> tuple[list[float], tuple[float, float]]:
        self.n_predict += 1
        self.n_calls += 1
        zt = _as_tensor(z, dim=self.slot_dim, device=self.device, dtype=torch.float32).unsqueeze(0)
        et = _as_tensor(ego, dim=EGO_DIM, device=self.device, dtype=torch.float32).unsqueeze(0)
        at = _as_tensor(
            action if action is not None else (0.0, 0.0, 0.5, 0.5),
            dim=ACTION_DIM,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.eval()
        z_hat, ego_hat = self.forward(zt, et, at)
        z_list = [float(x) for x in z_hat.squeeze(0).detach().cpu().tolist()]
        ex, ey = float(ego_hat[0, 0].item()), float(ego_hat[0, 1].item())
        return z_list, (ex, ey)

    @torch.no_grad()
    def rollout(
        self,
        z: Sequence[float] | torch.Tensor,
        ego: tuple[float, float] | Sequence[float],
        actions: Sequence[Sequence[float]],
        horizon: int | None = None,
    ) -> list[tuple[list[float], tuple[float, float]]]:
        hz = int(horizon) if horizon is not None else slot_dyn_horizon()
        hz = max(1, min(hz, len(list(actions)) if actions else hz))
        cur_z = list(z) if not isinstance(z, torch.Tensor) else [float(x) for x in z.detach().cpu().reshape(-1).tolist()]
        cur_e = (float(ego[0]), float(ego[1]))
        out: list[tuple[list[float], tuple[float, float]]] = []
        acts = list(actions)
        for i in range(hz):
            a = acts[i] if i < len(acts) else acts[-1] if acts else pack_action()
            cur_z, cur_e = self.predict(cur_z, cur_e, a)
            nrm = math.sqrt(sum(v * v for v in cur_z)) + 1e-8
            cap = math.sqrt(float(self.slot_dim)) * 4.0
            if nrm > cap:
                cur_z = [v * cap / nrm for v in cur_z]
            out.append((cur_z, cur_e))
        return out

    @torch.no_grad()
    def update_target_encoder(self, tau: float | None = None) -> None:
        tau_v = float(tau if tau is not None else slot_dyn_ema_tau())
        tau_v = max(0.0, min(1.0, tau_v))
        for t_p, s_p in zip(self.target_enc.parameters(), self.node_enc.parameters()):
            t_p.data.mul_(1.0 - tau_v).add_(s_p.data, alpha=tau_v)

    def push_pair(
        self,
        *,
        z_t: Sequence[float],
        ego_t: tuple[float, float],
        action: Sequence[float],
        z_next: Sequence[float],
        ego_next: tuple[float, float],
        has_live: bool = True,
    ) -> None:
        self._buffer.append(
            {
                "z_t": [float(x) for x in list(z_t)[: self.slot_dim]],
                "ego_t": (float(ego_t[0]), float(ego_t[1])),
                "action": pack_action(
                    float(action[0]) if len(action) > 0 else 0.0,
                    float(action[1]) if len(action) > 1 else 0.0,
                    float(action[2]) if len(action) > 2 else 0.5,
                    float(action[3]) if len(action) > 3 else 0.5,
                ),
                "z_next": [float(x) for x in list(z_next)[: self.slot_dim]],
                "ego_next": (float(ego_next[0]), float(ego_next[1])),
                "has_live": bool(has_live),
            }
        )

    def remember_prev(
        self,
        *,
        z: Sequence[float],
        ego: tuple[float, float],
        action: Sequence[float],
        entity_id: str = "",
        tick: int = 0,
    ) -> None:
        self._prev = {
            "z": [float(x) for x in list(z)[: self.slot_dim]],
            "ego": (float(ego[0]), float(ego[1])),
            "action": pack_action(*(list(action) + [0.0, 0.0, 0.5, 0.5])[:4]),
            "entity_id": str(entity_id),
            "tick": int(tick),
        }

    def commit_next(
        self,
        *,
        z_next: Sequence[float],
        ego_next: tuple[float, float],
        entity_id: str = "",
        has_live: bool = True,
        tick: int = 0,
    ) -> bool:
        prev = self._prev
        if prev is None:
            return False
        if entity_id and prev.get("entity_id") and str(prev["entity_id"]) != str(entity_id):
            return False
        if int(tick) - int(prev.get("tick") or 0) > 64:
            return False
        self.push_pair(
            z_t=prev["z"],
            ego_t=prev["ego"],
            action=prev["action"],
            z_next=z_next,
            ego_next=ego_next,
            has_live=has_live,
        )
        return True

    def buffer_len(self) -> int:
        return len(self._buffer)

    def train_step(self, batch_size: int = 16) -> dict[str, float] | None:
        n = len(self._buffer)
        if n < _TRAIN_MIN:
            return None
        bs = int(max(1, min(batch_size, n)))
        idx = torch.randint(0, n, (bs,))
        rows = [self._buffer[int(i)] for i in idx.tolist()]

        def _pad_z(v: Sequence[float]) -> list[float]:
            xs = [float(x) for x in v]
            if len(xs) < self.slot_dim:
                xs = xs + [0.0] * (self.slot_dim - len(xs))
            return xs[: self.slot_dim]

        z_t = torch.tensor([_pad_z(r["z_t"]) for r in rows], device=self.device, dtype=torch.float32)
        z_n = torch.tensor([_pad_z(r["z_next"]) for r in rows], device=self.device, dtype=torch.float32)
        ego_t = torch.tensor([list(r["ego_t"]) for r in rows], device=self.device, dtype=torch.float32)
        ego_n = torch.tensor([list(r["ego_next"]) for r in rows], device=self.device, dtype=torch.float32)
        act = torch.tensor([list(r["action"]) for r in rows], device=self.device, dtype=torch.float32)
        live = torch.tensor(
            [1.0 if r.get("has_live") else 0.0 for r in rows],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(-1)

        self.train()
        z_hat, ego_hat = self.forward(z_t, ego_t, act)
        h_pred = self.node_enc(z_hat)
        with torch.no_grad():
            h_tgt = self.target_enc(z_n)
        hp = F.normalize(h_pred, dim=-1, eps=1e-8)
        ht = F.normalize(h_tgt, dim=-1, eps=1e-8)
        l_jepa = (1.0 - (hp * ht).sum(dim=-1)).mean()
        ego_err = F.huber_loss(ego_hat, ego_n, reduction="none").mean(dim=-1, keepdim=True)
        l_ego = (ego_err * live).sum() / live.sum().clamp(min=1.0)
        l_reg = 1e-4 * (z_hat.pow(2).mean() + (ego_hat - ego_t).pow(2).mean())
        # SIGReg-lite: keep predicted z variance from collapsing.
        z_std = z_hat.std(dim=0).mean()
        l_sig = F.relu(0.05 - z_std)
        loss = l_jepa + 0.5 * l_ego + l_reg + 0.05 * l_sig

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()
        self.update_target_encoder()
        self.train_steps += 1
        self.last_loss = float(loss.detach().item())
        return {
            "loss": float(self.last_loss),
            "l_jepa": float(l_jepa.detach().item()),
            "l_ego": float(l_ego.detach().item()),
            "n": float(bs),
        }


def apply_slot_dynamics_hold(
    ent: Any,
    dynamics: Any,
    action: Sequence[float],
    *,
    ego_prev: tuple[float, float] | None = None,
    tick: int,
    sigma_grown: float,
) -> dict[str, Any]:
    """Blend predicted ego into an OWM entity after odometry warp.

    Predicts from pre-warp ``ego_prev``; blends with post-warp odom ego.
    ``sigma_grown`` is the increment just applied by odom process noise; on
    agreement it is scaled back by ``RKK_SLOT_DYN_SIGMA_SCALE``.
    """
    diag: dict[str, Any] = {"slot_dyn": False}
    z = getattr(ent, "latent", None) or []
    if not z or dynamics is None:
        return diag
    live_ref = int(getattr(ent, "last_live_uv_tick", -1) or -1)
    if live_ref < 0:
        live_ref = int(getattr(ent, "last_vision_tick", -1) or -1)
    age = int(tick) - live_ref if live_ref >= 0 else 9999
    horizon = slot_dyn_horizon()
    ego_odom = (float(ent.x_fwd), float(ent.y_right))
    prev = ego_prev if ego_prev is not None else ego_odom
    try:
        z_hat, ego_hat = dynamics.predict(z, prev, action)
    except Exception:
        return diag
    agree_d = math.hypot(float(ego_hat[0]) - ego_odom[0], float(ego_hat[1]) - ego_odom[1])
    agree = agree_d <= slot_dyn_agree_m()
    within = age <= horizon
    beta = slot_dyn_blend() if (within and z_hat) else 0.0
    if beta > 0.0:
        ent.x_fwd = (1.0 - beta) * ego_odom[0] + beta * float(ego_hat[0])
        ent.y_right = (1.0 - beta) * ego_odom[1] + beta * float(ego_hat[1])
        try:
            from engine.object_working_memory import bearing_range_from_ego

            ent.bearing, ent.range_m = bearing_range_from_ego(ent.x_fwd, ent.y_right)
        except Exception:
            pass
    if z_hat:
        z_old = [float(x) for x in z]
        z_new = [float(x) for x in z_hat]
        n = min(len(z_old), len(z_new))
        mixed = [0.7 * z_old[i] + 0.3 * z_new[i] for i in range(n)]
        if len(z_new) > n:
            mixed.extend(z_new[n:])
        ent.latent = mixed
    if within and agree and sigma_grown > 0.0:
        scale = slot_dyn_sigma_scale()
        ent.bearing_sigma = float(
            max(0.02, float(ent.bearing_sigma) - float(sigma_grown) * (1.0 - scale))
        )
    diag.update(
        {
            "slot_dyn": True,
            "slot_dyn_agree": bool(agree),
            "slot_dyn_agree_m": round(float(agree_d), 4),
            "slot_dyn_beta": round(float(beta), 4),
            "slot_dyn_age": int(age),
            "slot_dyn_within_horizon": bool(within),
        }
    )
    return diag


PredictFn = Callable[
    [Sequence[float], tuple[float, float], Sequence[float]],
    tuple[list[float], tuple[float, float]],
]
