"""
Neural ODE для world model (GNN / NOTEARS core).

Вместо одного шага X' = forward_dynamics(X, a) интегрируем по псевдовремени τ:

    dY/dτ = forward_dynamics(Y, a) - Y

При одном шаге Эйлера с Δτ=1: Y + (F(Y,a) - Y) = F(Y,a) — совпадает с прежним одношаговым
прогнозом в начальной точке. Несколько подшагов (dopri5 или плотная сетка по τ) даёт
сглаженную траекторию в «режиме непрерывной динамики», что лучше стыкуется с
переменным шагом физики PyBullet, чем жёсткий один дискретный прыжок.

Включение: RKK_WM_NEURAL_ODE=1 и pip install torchdiffeq

Переменные:
  RKK_WM_ODE_TIME_POINTS — число точек сетки по τ (минимум 2), по умолчанию 17
  RKK_WM_ODE_T0, RKK_WM_ODE_T1 — интервал псевдовремени (по умолчанию 0..1)
  RKK_WM_ODE_METHOD — dopri5 | euler | rk4 | … (см. torchdiffeq)
  RKK_WM_ODE_RTOL, RKK_WM_ODE_ATOL — для адаптивных методов
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint as _odeint

    _TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    _odeint = None
    _TORCHDIFFEQ_AVAILABLE = False


def wm_neural_ode_available() -> bool:
    return _TORCHDIFFEQ_AVAILABLE


def wm_neural_ode_enabled() -> bool:
    if not _TORCHDIFFEQ_AVAILABLE:
        return False
    return os.environ.get("RKK_WM_NEURAL_ODE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _t_span(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    try:
        n = int(os.environ.get("RKK_WM_ODE_TIME_POINTS", "17"))
    except ValueError:
        n = 17
    n = max(2, min(n, 513))
    try:
        t0 = float(os.environ.get("RKK_WM_ODE_T0", "0"))
        t1 = float(os.environ.get("RKK_WM_ODE_T1", "1"))
    except ValueError:
        t0, t1 = 0.0, 1.0
    if t1 <= t0:
        t1 = t0 + 1.0
    return torch.linspace(t0, t1, n, device=device, dtype=dtype)


def _resolve_wm_forward_dynamics(entity: Any):
    """
    CausalGraph.forward_dynamics дополняет активную ширину до MAX_D и обрезает предсказание до _d.
    Сырой CausalGNNCore / NOTEARSCore ожидает уже (B, MAX_D). Передавать graph._core с (B, _d)
    даёт matmul (…,256,…) vs (…,214,…) — см. integrate_world_model_step.
    """
    fd = getattr(entity, "forward_dynamics", None)
    if callable(fd) and hasattr(entity, "_pad"):
        return fd
    return fd


def integrate_world_model_step(
    core: nn.Module,
    y0: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Одна логическая итерация world model: либо odeint по τ, либо один forward_dynamics.

    Первый аргумент: **CausalGraph** (предпочтительно) или сырой _core.
    y0, a: (B, graph._d); возврат (B, graph._d).
    """
    fd: Any = _resolve_wm_forward_dynamics(core)
    if not callable(fd):
        return core(y0 + a) if hasattr(core, "__call__") else y0

    if not wm_neural_ode_enabled():
        return fd(y0, a)

    method = (os.environ.get("RKK_WM_ODE_METHOD", "dopri5") or "dopri5").strip().lower()
    try:
        rtol = float(os.environ.get("RKK_WM_ODE_RTOL", "1e-3"))
        atol = float(os.environ.get("RKK_WM_ODE_ATOL", "1e-4"))
    except ValueError:
        rtol, atol = 1e-3, 1e-4

    t = _t_span(y0.device, y0.dtype)

    def odefunc(_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return fd(y, a) - y

    kwargs: dict[str, Any] = {"method": method}
    if method in ("dopri5", "adaptive_heun", "bdf"):
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol

    try:
        if _odeint is None:
            return fd(y0, a)
        traj = _odeint(odefunc, y0, t, **kwargs)
    except Exception:
        return fd(y0, a)
    return traj[-1]
