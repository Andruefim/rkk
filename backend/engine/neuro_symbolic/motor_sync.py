"""
Fast-path motor prior sync: NS / intention → graph.nodes + env._motor_state (every tick).
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

# Exo / third-person camera — proxy for human observer position (see pybullet_humanoid).
_EXO_CAMERA_POS = (2.2, -2.2, 1.6)


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _canonical_intent_key(k: str) -> str:
    sk = str(k)
    if sk.startswith("phys_intent_"):
        return "intent_" + sk[len("phys_intent_") :]
    if sk.startswith("intent_"):
        return sk
    return sk


def _clip_dist_norm(v: float) -> float:
    return float(np.clip(float(v), 0.0, 1.0))


def _human_slot_distance(sim: Any) -> float | None:
    """If a vision slot is labeled human/person, map activation → proximity."""
    if not getattr(sim, "_visual_mode", False):
        return None
    vis_env = getattr(sim, "_visual_env", None)
    if vis_env is None:
        return None
    slots = getattr(vis_env, "_last_slots", None)
    lex = getattr(vis_env, "_slot_lexicon", None) or {}
    if slots is None or not lex:
        return None
    try:
        vals = slots.detach().cpu().numpy()
    except Exception:
        return None
    n = min(len(vals), int(getattr(vis_env, "n_slots", len(vals))))
    for k in range(n):
        meta = lex.get(f"slot_{k}") or {}
        label = str(meta.get("label") or "").lower()
        if any(tok in label for tok in ("human", "person", "operator", "viewer")):
            act = float(vals[k])
            return _clip_dist_norm(1.0 - act * 0.85)
    return None


def _exo_camera_distance_m(sim: Any) -> float | None:
    tick_phys = getattr(sim, "_tick_phys_state", None)
    if not callable(tick_phys):
        return None
    try:
        st = tick_phys()
    except Exception:
        return None
    if not isinstance(st, dict):
        return None
    try:
        cx = float(st.get("com_x", 0.0))
        cy = float(st.get("com_y", 0.0))
        cz = float(st.get("com_z", 0.75))
    except (TypeError, ValueError):
        return None
    ex, ey, ez = _EXO_CAMERA_POS
    return float(math.hypot(cx - ex, cy - ey, cz - ez))


def resolve_distance_to_human(sim: Any) -> float | None:
    """
    Normalized distance to human observer: 0=contact, 1=far.
    Returns None when no sensor/scene data is available (caller uses stub).
    """
    ov = getattr(sim, "_distance_to_human_override", None)
    if ov is not None:
        try:
            return _clip_dist_norm(float(ov))
        except (TypeError, ValueError):
            pass

    for src_fn in (
        getattr(sim, "_graph_vec_cached", None),
        getattr(sim, "_env_observe_cached", None),
    ):
        if not callable(src_fn):
            continue
        try:
            obs = dict(src_fn() or {})
        except Exception:
            continue
        for key in ("distance_to_human", "phys_distance_to_human"):
            if key in obs:
                try:
                    return _clip_dist_norm(float(obs[key]))
                except (TypeError, ValueError):
                    pass

    slot_dist = _human_slot_distance(sim)
    if slot_dist is not None:
        return slot_dist

    dist_m = _exo_camera_distance_m(sim)
    if dist_m is not None:
        max_m = max(0.5, _ef("RKK_HUMAN_DIST_MAX_M", 4.0))
        return _clip_dist_norm(dist_m / max_m)

    return None


def feed_distance_to_human(sim: Any) -> float | None:
    """Resolve live distance and push into NS engine + sim cache."""
    dist = resolve_distance_to_human(sim)
    if dist is None:
        return None
    sim._distance_to_human = dist
    ns_eng = getattr(sim, "_ns_engine", None)
    if ns_eng is not None:
        ns_eng.set_distance_to_human(dist)
    return dist


def locomote_macro_active(sim: Any) -> bool:
    ic = getattr(sim, "_intention_state", None)
    if ic is not None:
        hint = str(getattr(ic, "macro_hint", "") or "").strip().upper()
        if hint in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            return True
    s2 = getattr(sim, "_system2", None)
    if s2 is not None:
        macro = str(getattr(s2, "_active_macro", "") or "").strip().upper()
        if macro in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            return True
    last = getattr(sim, "_system2_last", None) or {}
    if isinstance(last, dict):
        macro = str(last.get("macro") or "").strip().upper()
        if macro in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            return True
    return False


def enforce_sticky_locomote_priors(sim: Any) -> dict[str, float]:
    """
    Hard floor for NS locomotion priors — skills cannot pull stride/coupling below NS targets.
    """
    if not locomote_macro_active(sim):
        return {}

    stride_floor = _ef("RKK_NS_LOCOMOTE_STRIDE", 0.64)
    coupling_target = _ef("RKK_NS_LOCOMOTE_COUPLING", 0.78)
    torso_floor = _ef("RKK_NS_LOCOMOTE_TORSO", 0.58)

    agent = sim.agent
    nodes = agent.graph.nodes
    base = getattr(agent.env, "base_env", None) or agent.env
    ms = getattr(base, "_motor_state", None)
    if not isinstance(ms, dict):
        return {}

    cur_stride = float(ms.get("intent_stride", nodes.get("intent_stride", 0.5)))
    stride_val = float(np.clip(max(cur_stride, stride_floor), 0.05, 0.95))
    coupling_val = float(
        np.clip(
            float(ms.get("intent_gait_coupling", coupling_target)),
            _ef("RKK_NS_LOCOMOTE_COUPLING_MIN", 0.72),
            coupling_target,
        )
    )
    torso_val = float(
        np.clip(max(float(ms.get("intent_torso_forward", 0.5)), torso_floor), 0.05, 0.95)
    )

    sticky = {
        "intent_stride": stride_val,
        "intent_gait_coupling": coupling_val,
        "intent_torso_forward": torso_val,
    }

    for ck, tv in sticky.items():
        nodes[ck] = tv
        phys_k = f"phys_{ck}"
        if phys_k in nodes:
            nodes[phys_k] = tv
        ms[ck] = tv

    nodes["executive_macro"] = 1.0 if locomote_macro_active(sim) else 0.0
    return sticky


def collect_motor_targets(sim: Any) -> dict[str, float]:
    """Merge NS cache, intention residuals, S2 macro priors into intent_* targets."""
    targets: dict[str, float] = {}

    ns_ctx = getattr(sim, "_ns_last_ctx", None) or {}
    for k, v in (ns_ctx.get("motor_priors") or {}).items():
        ck = _canonical_intent_key(k)
        if ck.startswith("intent_"):
            targets[ck] = float(v)

    ic = getattr(sim, "_intention_state", None)
    if ic is not None:
        primary = getattr(ic, "primary", None)
        if primary is not None:
            for k, v in (getattr(primary, "intent_targets", None) or {}).items():
                ck = _canonical_intent_key(k)
                if ck.startswith("intent_"):
                    targets[ck] = float(v)
            pvar = str(getattr(primary, "var_id", ""))
            if pvar.startswith("intent_") or pvar.startswith("phys_intent_"):
                targets[_canonical_intent_key(pvar)] = float(
                    getattr(primary, "target_val", 0.5)
                )
        for k, dv in (getattr(ic, "intent_residuals", None) or {}).items():
            ck = _canonical_intent_key(k)
            if ck.startswith("intent_"):
                targets[ck] = float(np.clip(0.5 + float(dv), 0.06, 0.94))

    macro = ""
    if ic is not None:
        macro = str(getattr(ic, "macro_hint", "") or "")
    if not macro:
        s2 = getattr(sim, "_system2_last", None) or {}
        if isinstance(s2, dict):
            macro = str(s2.get("macro") or "")

    if macro.upper() == "LOCOMOTE_DELIVERY":
        targets.setdefault("intent_stride", _ef("RKK_NS_LOCOMOTE_STRIDE", 0.64))
        targets["intent_gait_coupling"] = _ef("RKK_NS_LOCOMOTE_COUPLING", 0.78)
        targets.setdefault("intent_torso_forward", _ef("RKK_NS_LOCOMOTE_TORSO", 0.58))

    return targets


def apply_motor_targets(
    sim: Any,
    targets: dict[str, float],
    *,
    graph_blend: float | None = None,
    motor_gain: float | None = None,
) -> dict[str, float]:
    """Write targets into graph.nodes and humanoid _motor_state."""
    locomote = locomote_macro_active(sim)
    if not targets:
        if locomote:
            return enforce_sticky_locomote_priors(sim)
        return {}
    gb = graph_blend if graph_blend is not None else _ef("RKK_NS_FAST_GRAPH_BLEND", 0.55)
    mg = motor_gain if motor_gain is not None else _ef("RKK_NS_FAST_MOTOR_GAIN", 0.45)

    agent = sim.agent
    nodes = agent.graph.nodes
    applied: dict[str, float] = {}
    sticky_keys = frozenset(
        {"intent_stride", "intent_gait_coupling", "intent_torso_forward"}
    )

    for k, target in targets.items():
        ck = _canonical_intent_key(k)
        if not ck.startswith("intent_"):
            continue
        tv = float(np.clip(target, 0.05, 0.95))
        if locomote and ck in sticky_keys:
            if ck == "intent_stride":
                tv = max(tv, _ef("RKK_NS_LOCOMOTE_STRIDE", 0.64))
            elif ck == "intent_gait_coupling":
                c_max = _ef("RKK_NS_LOCOMOTE_COUPLING", 0.78)
                c_min = _ef("RKK_NS_LOCOMOTE_COUPLING_MIN", 0.72)
                tv = float(np.clip(tv, c_min, c_max))
            elif ck == "intent_torso_forward":
                tv = max(tv, _ef("RKK_NS_LOCOMOTE_TORSO", 0.58))
            gb = 1.0
            mg = 1.0
        else:
            gb = graph_blend if graph_blend is not None else _ef("RKK_NS_FAST_GRAPH_BLEND", 0.55)
            mg = motor_gain if motor_gain is not None else _ef("RKK_NS_FAST_MOTOR_GAIN", 0.45)
        if ck in nodes:
            cur = float(nodes[ck])
            if locomote and ck in sticky_keys:
                nodes[ck] = float(np.clip(max(cur, tv), 0.05, 0.95))
            else:
                nodes[ck] = float(np.clip(cur + gb * (tv - cur), 0.05, 0.95))
            applied[ck] = float(nodes[ck])
        phys_k = f"phys_{ck}"
        if phys_k in nodes:
            cur = float(nodes[phys_k])
            if locomote and ck in sticky_keys:
                nodes[phys_k] = float(np.clip(max(cur, tv), 0.05, 0.95))
            else:
                nodes[phys_k] = float(np.clip(cur + gb * (tv - cur), 0.05, 0.95))

    base = getattr(agent.env, "base_env", None) or agent.env
    fn = getattr(base, "apply_motor_intent_residuals", None)
    ms = getattr(base, "_motor_state", None)
    if callable(fn) and isinstance(ms, dict):
        residuals: dict[str, float] = {}
        mg_default = motor_gain if motor_gain is not None else _ef("RKK_NS_FAST_MOTOR_GAIN", 0.45)
        for k, target in targets.items():
            ck = _canonical_intent_key(k)
            if not ck.startswith("intent_"):
                continue
            tv = float(np.clip(target, 0.05, 0.95))
            if locomote and ck in sticky_keys:
                if ck == "intent_stride":
                    tv = max(tv, _ef("RKK_NS_LOCOMOTE_STRIDE", 0.64))
                elif ck == "intent_gait_coupling":
                    tv = _ef("RKK_NS_LOCOMOTE_COUPLING", 0.78)
                elif ck == "intent_torso_forward":
                    tv = max(tv, _ef("RKK_NS_LOCOMOTE_TORSO", 0.58))
            cur = float(ms.get(ck, nodes.get(ck, 0.5)))
            if locomote and ck in sticky_keys and tv <= cur + 0.002:
                ms[ck] = float(np.clip(max(cur, tv), 0.05, 0.95))
                applied[ck] = float(ms[ck])
                continue
            delta = (tv - cur) * mg_default
            if abs(delta) >= 0.003:
                residuals[ck] = delta
        if residuals:
            try:
                fn(residuals)
            except Exception:
                pass
        for ck, nv in applied.items():
            ms[ck] = float(nv)

    if locomote:
        applied.update(enforce_sticky_locomote_priors(sim))
    return applied


def sync_ns_motor_every_tick(sim: Any) -> dict[str, float]:
    """Called before CPG each tick."""
    ns_eng = getattr(sim, "_ns_engine", None)
    dist_h = resolve_distance_to_human(sim)
    if dist_h is None:
        dist_h = 1.0
        if ns_eng is not None:
            dist_h = float(getattr(ns_eng, "_human_distance_stub", 1.0))
    agent = sim.agent
    nodes = agent.graph.nodes
    nodes["distance_to_human"] = dist_h
    nodes["phys_distance_to_human"] = dist_h
    if ns_eng is not None:
        ns_eng.set_distance_to_human(dist_h)
    bridge = getattr(sim, "_ns_bridge", None)
    if bridge is not None:
        bridge.knowledge_graph.set_runtime_fact("distance_to_human", dist_h)
        obs = dict(getattr(sim, "_graph_vec_cached", lambda: {})() or {})
        obs["distance_to_human"] = dist_h
        bridge.perceive(obs, nodes)

    w_meta = getattr(sim, "_w_meta", None) or getattr(agent, "_w_meta", None)
    if w_meta is not None:
        try:
            pe = float(w_meta.meta_prediction_error_rolling(64))
            nodes["meta_prediction_error"] = pe
        except Exception:
            pass

    if ns_eng is not None:
        veto = ns_eng.check_human_proximity({"distance_to_human": dist_h})
        if not veto.allowed:
            base = getattr(agent.env, "base_env", None) or agent.env
            ms = getattr(base, "_motor_state", None)
            if isinstance(ms, dict):
                for k in list(ms.keys()):
                    if str(k).startswith("intent_"):
                        ms[k] = 0.5
            for k in list(nodes.keys()):
                if str(k).startswith("intent_"):
                    nodes[k] = 0.5
            return {"safety_veto": 1.0, "distance_to_human": dist_h}

    targets = collect_motor_targets(sim)
    ns_ctx = getattr(sim, "_ns_last_ctx", None) or {}
    arb = getattr(sim, "_motor_arbiter", None)
    suppress = arb is not None and arb.should_suppress_substrate()
    applied: dict[str, float] = {}
    if targets and not suppress:
        applied = apply_motor_targets(sim, targets)
    elif targets and arb is not None:
        canon_targets: dict[str, float] = {}
        for k, v in targets.items():
            ck = _canonical_intent_key(k)
            if ck.startswith("intent_"):
                canon_targets[ck] = float(v)
        if canon_targets:
            arb.register_from_dict("ns_bridge", canon_targets)
    if locomote_macro_active(sim) and not suppress:
        applied.update(enforce_sticky_locomote_priors(sim))
        try:
            from engine.locomote_gait import clamp_locomote_gait_intents

            clamp_locomote_gait_intents(sim)
        except Exception:
            pass

    if arb is not None:
        priors = dict((ns_ctx or {}).get("motor_priors") or {})
        if not priors and applied:
            priors = {k: float(v) for k, v in applied.items() if str(k).startswith("intent_")}
        if priors and not suppress:
            canon = {}
            for k, v in priors.items():
                ck = _canonical_intent_key(k)
                if ck.startswith("intent_"):
                    canon[ck] = float(v)
            if canon:
                arb.register_from_dict("ns_bridge", canon, precision=0.92)

    pe_fwd = getattr(sim, "_hai_pe_fwd_ema", None)
    if pe_fwd is not None:
        try:
            nodes["hai_pe_fwd_ema"] = float(pe_fwd)
        except (TypeError, ValueError):
            pass
    ic = getattr(sim, "_intention_state", None)
    if ic is not None:
        hint = str(getattr(ic, "macro_hint", "") or "").strip().upper()
        nodes["executive_macro_hint"] = 1.0 if hint == "LOCOMOTE_DELIVERY" else 0.5 if hint == "EXPLORE" else 0.0

    return applied
