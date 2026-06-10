"""
Track F Phase 5: W_meta ensemble — meta-causal self-model over hyperparameters.

Meta nodes: learning_rate_eff, exploration_rate, curriculum_phase, wm_lr_mult → success_rate.
do-calculus is counterfactual-only when RKK_META_DO_SAFE=1 (no live LR mutation).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


META_INPUTS = (
    "learning_rate_eff",
    "exploration_rate",
    "curriculum_phase",
    "wm_lr_mult",
)
META_OUTCOME = "success_rate"
META_NODE_IDS = tuple(META_INPUTS) + (META_OUTCOME,)


def meta_causal_enabled() -> bool:
    return os.environ.get("RKK_META_CAUSAL_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def meta_do_safe() -> bool:
    return os.environ.get("RKK_META_DO_SAFE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ei(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _ensemble_n() -> int:
    try:
        return max(1, min(8, int(os.environ.get("RKK_META_ENSEMBLE_N", "3"))))
    except ValueError:
        return 3


@dataclass
class MetaObservation:
    learning_rate_eff: float = 0.5
    exploration_rate: float = 0.0
    curriculum_phase: float = 0.0
    wm_lr_mult: float = 1.0
    success_rate: float = 0.5
    train_loss_delta: float = 0.0
    discovery_rate: float = 0.0
    prediction_error: float = 0.0
    tick: int = 0


@dataclass
class MetaDoResult:
    variable: str
    value: float
    predicted_success: float
    observed_success: float
    meta_prediction_error: float
    applied_live: bool = False


class WMetaEnsemble(nn.Module):
    """
    Small Bayesian ensemble over meta adjacency W_meta (4 inputs → success).
    """

    def __init__(self, device: torch.device, n: int | None = None):
        super().__init__()
        self.device = device
        self.n = n if n is not None else _ensemble_n()
        d = len(META_NODE_IDS)
        self.d = d
        # Structural mask: inputs → success only (no self-loops on success as parent)
        mask = torch.zeros(d, d, device=device)
        succ_i = META_NODE_IDS.index(META_OUTCOME)
        for j, name in enumerate(META_INPUTS):
            mask[succ_i, META_NODE_IDS.index(name)] = 1.0
        self.register_buffer("mask", mask)
        W_stack = torch.randn(self.n, d, d, device=device) * 0.08
        for k in range(self.n):
            for j, name in enumerate(META_INPUTS):
                W_stack[k, succ_i, META_NODE_IDS.index(name)] = 0.35 + 0.05 * j
        self.register_buffer("W_stack", W_stack)
        self.log_weights = nn.Parameter(torch.zeros(self.n, device=device))
        self._obs_buffer: deque[MetaObservation] = deque(maxlen=256)
        self._pe_history: deque[float] = deque(maxlen=512)
        self._last_do: MetaDoResult | None = None
        self._last_update_tick: int = -9999
        self._success_rate_after_meta_do: float | None = None
        self._suggested_intervention: dict[str, Any] | None = None

    def posterior(self) -> torch.Tensor:
        return F.softmax(self.log_weights, dim=0)

    def posterior_mean_W(self) -> torch.Tensor:
        w = self.posterior()
        return torch.einsum("n,nij->ij", w, self.W_stack * self.mask)

    def _vector_from_obs(self, obs: MetaObservation) -> torch.Tensor:
        vals = [
            float(np.clip(obs.learning_rate_eff, 0.0, 1.0)),
            float(np.clip(obs.exploration_rate, 0.0, 1.0)),
            float(np.clip(obs.curriculum_phase, 0.0, 1.0)),
            float(np.clip(obs.wm_lr_mult / 2.0, 0.0, 1.0)),
            1.0,
        ]
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_success(
        self,
        obs: MetaObservation | None = None,
        *,
        goal_var: str | None = None,
        goal_score: float = 0.0,
    ) -> float:
        """Predict success; optional goal features bias the readout."""
        if obs is None:
            obs = MetaObservation()
        x = self._vector_from_obs(obs)
        W = self.posterior_mean_W()
        succ_i = META_NODE_IDS.index(META_OUTCOME)
        logits = float((W[succ_i] * x).sum().item())
        bias = 0.12 * float(np.clip(goal_score, 0.0, 2.0))
        if goal_var:
            bias += 0.05
        return float(torch.sigmoid(torch.tensor(logits + bias)).item())

    @torch.no_grad()
    def do_intervention(
        self,
        variable: str,
        value: float,
        base_obs: MetaObservation,
    ) -> MetaDoResult:
        """Counterfactual do(variable=value); returns predicted vs observed success."""
        x = self._vector_from_obs(base_obs)
        idx = META_NODE_IDS.index(variable) if variable in META_NODE_IDS else None
        if idx is not None and variable != META_OUTCOME:
            x[idx] = float(np.clip(value, 0.0, 1.0))
        W = self.posterior_mean_W()
        succ_i = META_NODE_IDS.index(META_OUTCOME)
        pred = float(torch.sigmoid((W[succ_i] * x).sum()).item())
        observed = float(np.clip(base_obs.success_rate, 0.0, 1.0))
        pe = abs(pred - observed)
        result = MetaDoResult(
            variable=variable,
            value=float(value),
            predicted_success=pred,
            observed_success=observed,
            meta_prediction_error=pe,
            applied_live=not meta_do_safe(),
        )
        self._last_do = result
        return result

    def observe(self, obs: MetaObservation, *, tick: int) -> float | None:
        """Record observation and optionally run meta do-calculus; returns latest PE."""
        self._obs_buffer.append(obs)
        self._last_update_tick = int(tick)

        pred_actual = self.predict_success(obs)
        pe_actual = abs(pred_actual - obs.success_rate)
        self._pe_history.append(pe_actual)

        pe_out: float | None = None
        every = _ei("RKK_META_UPDATE_EVERY", 50)
        if tick % every != 0:
            return self.meta_prediction_error_rolling()

        # Train ensemble weights from recent prediction error
        if len(self._obs_buffer) >= 4:
            self._update_posterior_from_buffer()

        # Suggest best do among meta inputs (counterfactual grid)
        best_do: MetaDoResult | None = None
        grid = (0.25, 0.5, 0.75)
        for var in META_INPUTS:
            for val in grid:
                dr = self.do_intervention(var, val, obs)
                if best_do is None or dr.predicted_success > best_do.predicted_success:
                    best_do = dr
        if best_do is not None:
            self._suggested_intervention = {
                "variable": best_do.variable,
                "value": best_do.value,
                "predicted_success": round(best_do.predicted_success, 4),
            }
            pe_out = pe_actual
            self._success_rate_after_meta_do = best_do.predicted_success

        return pe_out

    @torch.no_grad()
    def _update_posterior_from_buffer(self) -> None:
        recent = list(self._obs_buffer)[-16:]
        ll = []
        W_mean = self.W_stack * self.mask
        succ_i = META_NODE_IDS.index(META_OUTCOME)
        for k in range(self.n):
            err_sum = 0.0
            for obs in recent:
                x = self._vector_from_obs(obs)
                pred = float(torch.sigmoid((W_mean[k, succ_i] * x).sum()).item())
                err_sum += (pred - obs.success_rate) ** 2
            ll.append(-err_sum)
        try:
            from engine.eval_mode import transfer_bench_enabled

            lr = 0.18 if transfer_bench_enabled() else 0.05
        except ImportError:
            lr = 0.05
        self.log_weights.add_(torch.tensor(ll, device=self.device) * lr)

    def meta_prediction_error_rolling(self, window: int = 500) -> float:
        if not self._pe_history:
            return 0.0
        recent = list(self._pe_history)[-window:]
        try:
            from engine.eval_mode import transfer_bench_enabled

            if transfer_bench_enabled() and len(recent) >= 3:
                ema = float(recent[0])
                for pe in recent[1:]:
                    ema = 0.82 * ema + 0.18 * float(pe)
                return float(ema)
        except ImportError:
            pass
        return float(np.mean(recent))

    def effect_observable(self) -> dict[str, float]:
        """Smoke: each meta input's correlation with success in buffer."""
        if len(self._obs_buffer) < 3:
            return {f"{m}_effect": 0.0 for m in META_INPUTS}
        rows = list(self._obs_buffer)[-32:]
        succ = np.array([r.success_rate for r in rows], dtype=np.float64)
        out: dict[str, float] = {}
        for attr in META_INPUTS:
            xs = np.array([getattr(r, attr) for r in rows], dtype=np.float64)
            if float(xs.std()) < 1e-8 or float(succ.std()) < 1e-8:
                out[f"{attr}_effect"] = 0.0
            else:
                out[f"{attr}_effect"] = float(np.corrcoef(xs, succ)[0, 1])
        return out

    def snapshot(self) -> dict[str, Any]:
        do = self._last_do
        return {
            "enabled": meta_causal_enabled(),
            "ensemble_n": self.n,
            "meta_prediction_error": round(self.meta_prediction_error_rolling(), 4),
            "meta_prediction_error_rolling_500": round(
                self.meta_prediction_error_rolling(500), 4
            ),
            "success_rate_after_meta_do": self._success_rate_after_meta_do,
            "suggested_intervention": self._suggested_intervention,
            "last_do": (
                {
                    "variable": do.variable,
                    "value": do.value,
                    "predicted_success": round(do.predicted_success, 4),
                    "meta_prediction_error": round(do.meta_prediction_error, 4),
                    "applied_live": do.applied_live,
                }
                if do
                else None
            ),
            "effects": self.effect_observable(),
            "obs_buffer_len": len(self._obs_buffer),
            "last_update_tick": self._last_update_tick,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "W_stack": self.W_stack.detach().cpu().tolist(),
            "log_weights": self.log_weights.detach().cpu().tolist(),
            "pe_history": list(self._pe_history)[-64:],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if not data:
            return
        try:
            W = torch.tensor(data["W_stack"], device=self.device, dtype=torch.float32)
            if W.shape == self.W_stack.shape:
                self.W_stack.copy_(W)
            lw = torch.tensor(data["log_weights"], device=self.device, dtype=torch.float32)
            if lw.shape == self.log_weights.shape:
                self.log_weights.copy_(lw)
            for pe in data.get("pe_history") or []:
                self._pe_history.append(float(pe))
        except (KeyError, ValueError, RuntimeError):
            pass


def build_meta_observation(
    agent: Any,
    *,
    tick: int,
    curriculum_step: int = 0,
    success_rate: float | None = None,
) -> MetaObservation:
    """Collect meta features from agent + simulation context."""
    graph = agent.graph
    train_loss = 0.0
    if agent._last_notears_loss:
        train_loss = float(agent._last_notears_loss.get("loss", 0.0))
    lr_eff = float(np.clip(train_loss / 10.0, 0.0, 1.0))
    if graph._optim is not None:
        for pg in graph._optim.param_groups:
            lr_eff = max(lr_eff, float(np.clip(pg.get("lr", 5e-3) * 200.0, 0.0, 1.0)))
            break
    disc = float(getattr(agent, "_disc_rate_val", 0.0) or 0.0)
    try:
        disc = agent.discovery_rate
    except Exception:
        pass
    cur_phase = float(np.clip(curriculum_step / 3.0, 0.0, 1.0))
    wm_mult = float(getattr(graph, "_post_fr_wm_lr_mult", 1.0))
    pe = 0.0
    lr = getattr(agent, "_last_result", None) or {}
    pe = float(lr.get("prediction_error", 0.0))
    if success_rate is None:
        success_rate = float(np.clip(1.0 - pe, 0.0, 1.0))
        bs = lr.get("behavioral_score")
        if bs is not None:
            success_rate = float(np.clip(bs, 0.0, 1.0))
    return MetaObservation(
        learning_rate_eff=lr_eff,
        exploration_rate=float(np.clip(disc, 0.0, 1.0)),
        curriculum_phase=cur_phase,
        wm_lr_mult=wm_mult,
        success_rate=float(np.clip(success_rate, 0.0, 1.0)),
        train_loss_delta=lr_eff,
        discovery_rate=float(np.clip(disc, 0.0, 1.0)),
        prediction_error=float(np.clip(pe, 0.0, 1.0)),
        tick=int(tick),
    )
