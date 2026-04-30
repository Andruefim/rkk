"""
Hierarchical Active Inference (Problem 3): descending + ascending PE, GNN prior, multi-channel PE.

Descending: prediction error → low-level motor residuals (and optional CPG parameter nudge).
Ascending: smoothed forward PE → multiplicative calibration of intent_stride (belief update).

Prior for forward Δcom_x:
  RKK_HAI_PRIOR_FROM_GNN=1: expected step from agent.graph.forward_dynamics(X, A) at com_x.
  Fallback: tanh(intent_stride / intent_torso_forward) * RKK_HAI_EXPECT_GAIN.

Channels:
  pe_fwd  (expected Δcom_x − actual)   → gait_coupling (+ gated); drives ascending stride.
  pe_vert (com_z_target − com_z)      → symmetric support L/R.
  pe_lat  (EMA of com_y − 0.5)        → asymmetric support L/R.

Enable: RKK_HIERARCHICAL_AI=1. See .env.example for RKK_HAI_*.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


def hierarchical_pe_enabled() -> bool:
    return os.environ.get("RKK_HIERARCHICAL_AI", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _env_f(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_b(key: str, default: bool) -> bool:
    return os.environ.get(key, "1" if default else "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _fallback_expected_delta(stride: float, tf: float) -> float:
    k_exp = _env_f("RKK_HAI_EXPECT_GAIN", 0.028)
    return float(
        k_exp * np.tanh(4.0 * (stride - 0.5))
        + 0.4 * k_exp * np.tanh(2.5 * (tf - 0.5))
    )


def _gnn_prior_com_x_delta(graph: Any, obs: dict[str, float]) -> float | None:
    """World-model one-step implied Δcom_x; None if disabled or unavailable."""
    if not _env_b("RKK_HAI_PRIOR_FROM_GNN", True):
        return None
    if graph is None or getattr(graph, "_core", None) is None:
        return None
    ids = list(getattr(graph, "_node_ids", []) or [])
    if "com_x" not in ids:
        return None
    d = int(getattr(graph, "_d", 0) or 0)
    if d < 1:
        return None
    vec = [float(obs.get(nid, graph.nodes.get(nid, 0.5))) for nid in ids]
    dev = getattr(graph, "device", torch.device("cpu"))
    try:
        X = torch.tensor([vec], dtype=torch.float32, device=dev)
        A = torch.zeros(1, d, dtype=torch.float32, device=dev)
        for i, nid in enumerate(ids):
            if nid.startswith("intent_"):
                A[0, i] = float(obs.get(nid, vec[i]))
        with torch.no_grad():
            pred = graph.forward_dynamics(X, A)
        ix = ids.index("com_x")
        pred_cx = float(pred[0, ix].item())
        cur_cx = float(vec[ix])
        delta = pred_cx - cur_cx
        if not np.isfinite(delta):
            return None
        return float(delta)
    except Exception:
        return None


def _hai_reset_pe_state(sim: Any) -> None:
    sim._hai_prev_com_x = None
    sim._hai_pe_fwd_ema = 0.0
    sim._hai_pe_vert_ema = 0.0
    sim._hai_pe_lat_ema = 0.0
    sim._hai_pe_ema = 0.0


def _maybe_cpg_pe_nudge(sim: Any, u_fwd: float) -> None:
    """Optional direct nudge to Hopf CPG parameters (logit space), not only intent_gait_coupling."""
    if not _env_b("RKK_HAI_CPG_DIRECT", False):
        return
    lc = getattr(sim, "_locomotion_controller", None)
    cpg = getattr(lc, "cpg", None) if lc is not None else None
    if cpg is None:
        return
    kf = _env_f("RKK_HAI_CPG_FREQ_RES", 0.035)
    ka = _env_f("RKK_HAI_CPG_AMP_RES", 0.035)
    u = float(np.clip(u_fwd, -1.0, 1.0))
    try:
        with torch.no_grad():
            cpg.frequency.data.add_(u * kf)
            cpg.amplitude.data.add_(u * ka)
    except Exception:
        pass


def run_hierarchical_pe_tick(sim: Any, obs: dict[str, float]) -> dict[str, Any] | None:
    """
    One tick: GNN/fallback prior → multi-channel PE → descending residuals + ascending stride;
    optional CPG nudge. Mutates base env _motor_state and graph.nodes.
    """
    if not hierarchical_pe_enabled():
        return None

    if sim.current_world != "humanoid":
        _hai_reset_pe_state(sim)
        return None
    if getattr(sim, "_fixed_root_active", False):
        return None

    base = sim._unwrap_base_env(sim.agent.env)
    fn = getattr(base, "apply_motor_intent_residuals", None)
    if not callable(fn):
        return None

    graph = getattr(getattr(sim, "agent", None), "graph", None)

    cx = float(obs.get("com_x", 0.5))
    prev = getattr(sim, "_hai_prev_com_x", None)
    sim._hai_prev_com_x = cx

    cz = float(obs.get("com_z", 0.5))
    cy = float(obs.get("com_y", 0.5))
    stride0 = float(base._motor_state.get("intent_stride", 0.5))
    tf = float(obs.get("intent_torso_forward", 0.5))

    if prev is None:
        return {"enabled": True, "warmup": True}

    actual_delta = float(cx - prev)

    gnn_delta = _gnn_prior_com_x_delta(graph, obs)
    if gnn_delta is not None:
        expected_fwd = float(gnn_delta)
        prior_source = "gnn"
    else:
        expected_fwd = _fallback_expected_delta(stride0, tf)
        prior_source = "fallback"

    pe_fwd_raw = float(expected_fwd - actual_delta)

    beta = _env_f("RKK_HAI_PE_EMA", 0.25)
    beta = float(np.clip(beta, 0.01, 1.0))
    pe_fwd_prev = float(getattr(sim, "_hai_pe_fwd_ema", getattr(sim, "_hai_pe_ema", 0.0)))
    pe_fwd_ema = (1.0 - beta) * pe_fwd_prev + beta * pe_fwd_raw
    sim._hai_pe_fwd_ema = pe_fwd_ema
    sim._hai_pe_ema = pe_fwd_ema  # backward compat for external readers

    cz_t = _env_f("RKK_HAI_COM_Z_TARGET", 0.82)
    pe_vert_prev = float(getattr(sim, "_hai_pe_vert_ema", 0.0))
    pe_vert_ema = (1.0 - beta) * pe_vert_prev + beta * float(cz_t - cz)
    sim._hai_pe_vert_ema = pe_vert_ema

    pe_lat_prev = float(getattr(sim, "_hai_pe_lat_ema", 0.0))
    pe_lat_ema = (1.0 - beta) * pe_lat_prev + beta * float(cy - 0.5)
    sim._hai_pe_lat_ema = pe_lat_ema

    dead_fwd = _env_f("RKK_HAI_PE_DEAD", 0.004)
    dead_vert = _env_f("RKK_HAI_PE_VERT_DEAD", 0.03)
    dead_lat = _env_f("RKK_HAI_PE_LAT_DEAD", 0.04)

    kp = _env_f("RKK_HAI_PE_GAIN", 0.055)
    kp_vert = _env_f("RKK_HAI_SUPPORT_GAIN", 0.045)
    kp_lat = _env_f("RKK_HAI_LAT_GAIN", 0.04)
    lat_sign = _env_f("RKK_HAI_LAT_SIGN", 1.0)

    residuals: dict[str, float] = {}

    # --- Descending: forward PE → gait coupling (gated by stride intent; avoids blind boost at rest)
    if abs(pe_fwd_ema) >= dead_fwd:
        u_fwd = float(np.tanh(3.5 * pe_fwd_ema))
        stride_gate = float(np.clip((stride0 - 0.48) / 0.34, 0.12, 1.0))
        residuals["intent_gait_coupling"] = kp * u_fwd * stride_gate

    # --- Descending: vertical PE → symmetric support
    if abs(pe_vert_ema) >= dead_vert:
        u_v = float(np.tanh(3.0 * pe_vert_ema))
        dv = kp_vert * u_v
        residuals["intent_support_left"] = residuals.get("intent_support_left", 0.0) + dv
        residuals["intent_support_right"] = residuals.get("intent_support_right", 0.0) + dv

    # --- Descending: lateral PE → asymmetric support
    if abs(pe_lat_ema) >= dead_lat:
        u_y = float(np.tanh(4.0 * pe_lat_ema))
        dk = kp_lat * u_y * lat_sign
        residuals["intent_support_left"] = residuals.get("intent_support_left", 0.0) - dk
        residuals["intent_support_right"] = residuals.get("intent_support_right", 0.0) + dk

    # --- Ascending: belief update on intent_stride from sustained forward PE
    if _env_b("RKK_HAI_ASCEND", True) and abs(pe_fwd_ema) >= dead_fwd:
        alpha = _env_f("RKK_HAI_BELIEF_LR", 0.018)
        mult = 1.0 + alpha * float(np.tanh(2.0 * pe_fwd_ema))
        stride1 = float(np.clip(stride0 * mult, 0.05, 0.95))
        residuals["intent_stride"] = residuals.get("intent_stride", 0.0) + (stride1 - stride0)

    applied = bool(residuals)
    if applied:
        fn(residuals)
        if graph is not None:
            for k in residuals:
                if k in graph.nodes and k in getattr(base, "_motor_state", {}):
                    graph.nodes[k] = float(base._motor_state.get(k, graph.nodes[k]))

    u_cpg = 0.0
    if abs(pe_fwd_ema) >= dead_fwd:
        u_cpg = float(np.tanh(2.5 * pe_fwd_ema))
    if abs(u_cpg) > 1e-6:
        _maybe_cpg_pe_nudge(sim, u_cpg)

    return {
        "enabled": True,
        "prior_source": prior_source,
        "expected_fwd": round(expected_fwd, 5),
        "actual_delta": round(actual_delta, 5),
        "pe_fwd_raw": round(pe_fwd_raw, 5),
        "pe_fwd_ema": round(pe_fwd_ema, 5),
        "pe_vert_ema": round(pe_vert_ema, 5),
        "pe_lat_ema": round(pe_lat_ema, 5),
        "applied": applied,
        "residual_keys": list(residuals.keys()),
    }
