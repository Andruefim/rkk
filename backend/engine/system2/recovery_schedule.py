"""
Recovery motor schedules: LLM plans, deterministic fallback, bundle enrichment.
"""
from __future__ import annotations

import os
from typing import Any

from engine.system2.macros import macro_bundle
from engine.system2.validate import clip_intent_deltas


def recovery_fallback_enabled() -> bool:
    return os.environ.get("RKK_S2_RECOVERY_FALLBACK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def recovery_scripted_enabled() -> bool:
    return os.environ.get("RKK_S2_RECOVERY_SCRIPTED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def recovery_scripted_llm_on_entry() -> bool:
    """If false, entry uses scripted only; LLM waits for replan/stagnation."""
    return os.environ.get("RKK_S2_RECOVERY_SCRIPTED_LLM_ON_ENTRY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def default_scripted_getup_steps() -> list[dict[str, Any]]:
    """
    Fixed prone→kneel→stand-ish sequence (frame ticks per phase).
    Tuned for CPG tuck + torso/support; LLM may refine via replan only.
    """
    raw = os.environ.get("RKK_S2_RECOVERY_SCRIPTED_JSON", "").strip()
    if raw:
        try:
            import json

            data = json.loads(raw)
            if isinstance(data, list) and data:
                return [_normalize_step(s) for s in data if isinstance(s, dict)]
        except Exception:
            pass
    return [
        {
            "ticks": 50,
            "phase": "tuck",
            "intent_deltas": {
                "intent_stop_recover": 0.20,
                "intent_support_left": 0.14,
                "intent_support_right": 0.14,
                "intent_stride": -0.15,
            },
        },
        {
            "ticks": 45,
            "phase": "torso_lift",
            "intent_deltas": {
                "intent_torso_forward": 0.22,
                "intent_stop_recover": 0.12,
                "intent_arm_counterbalance": 0.10,
                "intent_stride": -0.10,
            },
        },
        {
            "ticks": 50,
            "phase": "push_up",
            "intent_deltas": {
                "intent_torso_forward": 0.18,
                "intent_lean_forward": 0.12,
                "intent_support_left": 0.16,
                "intent_support_right": 0.10,
                "intent_stop_recover": 0.10,
            },
        },
        {
            "ticks": 40,
            "phase": "kneel",
            "intent_deltas": {
                "intent_torso_forward": 0.14,
                "intent_support_left": 0.12,
                "intent_support_right": 0.12,
                "intent_stop_recover": 0.08,
                "intent_stride": -0.06,
            },
        },
        {
            "ticks": 30,
            "phase": "release_walk",
            "intent_deltas": {
                "intent_stop_recover": -0.08,
                "intent_torso_forward": 0.08,
                "intent_stride": 0.03,
            },
        },
    ]


def prepare_scripted_getup_steps() -> list[dict[str, Any]]:
    steps = enrich_recovery_steps(default_scripted_getup_steps())
    for st in steps:
        st["intent_deltas"] = sanitize_recovery_intent_deltas(st.get("intent_deltas"))
    ok, reason = validate_llm_recovery_plan(steps)
    if not ok:
        return enrich_recovery_steps(default_recovery_fallback_steps())
    return steps


_SCRIPTED_PHASE_JOINTS: dict[str, dict[str, float]] = {
    "tuck": {
        "lhip": 0.74,
        "rhip": 0.74,
        "lknee": 0.80,
        "rknee": 0.80,
        "lankle": 0.40,
        "rankle": 0.40,
        "spine_pitch": 0.56,
        "lshoulder": 0.36,
        "rshoulder": 0.36,
        "lelbow": 0.42,
        "relbow": 0.42,
    },
    "torso_lift": {
        "lhip": 0.64,
        "rhip": 0.64,
        "lknee": 0.70,
        "rknee": 0.70,
        "lankle": 0.44,
        "rankle": 0.44,
        "spine_pitch": 0.70,
        "lshoulder": 0.44,
        "rshoulder": 0.44,
        "lelbow": 0.48,
        "relbow": 0.48,
    },
    "push_up": {
        "lhip": 0.58,
        "rhip": 0.58,
        "lknee": 0.62,
        "rknee": 0.62,
        "lankle": 0.48,
        "rankle": 0.48,
        "spine_pitch": 0.76,
        "lshoulder": 0.52,
        "rshoulder": 0.52,
        "lelbow": 0.55,
        "relbow": 0.55,
    },
    "kneel": {
        "lhip": 0.52,
        "rhip": 0.52,
        "lknee": 0.56,
        "rknee": 0.56,
        "lankle": 0.50,
        "rankle": 0.50,
        "spine_pitch": 0.68,
        "lshoulder": 0.48,
        "rshoulder": 0.48,
    },
    "release_walk": {
        "lhip": 0.50,
        "rhip": 0.50,
        "lknee": 0.50,
        "rknee": 0.50,
        "lankle": 0.50,
        "rankle": 0.50,
        "spine_pitch": 0.62,
        "lshoulder": 0.50,
        "rshoulder": 0.50,
    },
}


def recovery_scripted_lock_until_exhausted() -> bool:
    return os.environ.get("RKK_S2_RECOVERY_SCRIPTED_LOCK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def scripted_getup_phase_at(
    sim_tick: int,
    anchor_tick: int,
    cumulative: list[int],
    steps: list[dict[str, Any]],
) -> tuple[int, str]:
    """Return (phase_index, phase_name) for current schedule position."""
    if not steps or not cumulative:
        return 0, "tuck"
    rel = max(0, int(sim_tick) - int(anchor_tick))
    idx = 0
    for i, bound in enumerate(cumulative):
        if rel <= bound:
            idx = i
            break
        idx = i
    st = steps[min(idx, len(steps) - 1)]
    name = str(st.get("phase") or "")
    if not name:
        names = ("tuck", "torso_lift", "push_up", "kneel", "release_walk")
        name = names[min(idx, len(names) - 1)]
    return idx, name


def scripted_getup_joint_targets(phase_name: str) -> dict[str, float]:
    return dict(_SCRIPTED_PHASE_JOINTS.get(str(phase_name), _SCRIPTED_PHASE_JOINTS["tuck"]))


def scripted_getup_episode_spec() -> dict[str, Any]:
    """Targets for tier1/tier2 relative progress from prone baseline."""
    try:
        cz = float(os.environ.get("RKK_S2_RECOVERY_SCRIPTED_TARGET_COM_Z", "0.36"))
    except ValueError:
        cz = 0.36
    try:
        ps = float(os.environ.get("RKK_S2_RECOVERY_SCRIPTED_TARGET_POSTURE", "0.38"))
    except ValueError:
        ps = 0.38
    return {
        "expected_state": {
            "com_z": cz,
            "posture_stability": ps,
            "foot_contact_l": 0.28,
            "foot_contact_r": 0.28,
        },
        "max_prediction_error": 0.55,
        "skill_id": "recovery_scripted_getup",
    }


def default_recovery_fallback_steps() -> list[dict[str, Any]]:
    """Deterministic multi-step schedule when LLM returns nothing invalid."""
    raw = os.environ.get("RKK_S2_RECOVERY_FALLBACK_JSON", "").strip()
    if raw:
        try:
            import json

            data = json.loads(raw)
            if isinstance(data, list) and data:
                return [_normalize_step(s) for s in data if isinstance(s, dict)]
        except Exception:
            pass
    bundle = macro_bundle("RECOVER_POSTURE")
    res = dict(bundle.get("residuals") or {})
    # Фазы: стоп+опора → tuck/торс (CPG) → наклон → снятие recovery-режима.
    return [
        {
            "ticks": 28,
            "intent_deltas": {
                "intent_stop_recover": max(0.12, float(res.get("intent_stop_recover", 0.06)) + 0.06),
                "intent_support_left": 0.08,
                "intent_support_right": 0.08,
                "intent_stride": -0.12,
            },
        },
        {
            "ticks": 32,
            "intent_deltas": {
                "intent_stop_recover": 0.08,
                "intent_torso_forward": 0.10,
                "intent_arm_counterbalance": res.get("intent_arm_counterbalance", 0.05),
            },
        },
        {
            "ticks": 40,
            "intent_deltas": {
                "intent_torso_forward": 0.06,
                "intent_lean_forward": res.get("intent_lean_forward", 0.05),
                "intent_stop_recover": -0.04,
            },
        },
        {
            "ticks": 24,
            "intent_deltas": {
                "intent_stop_recover": -0.06,
                "intent_stride": 0.04,
            },
        },
    ]


def _normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    try:
        ticks = int(max(1, min(80, int(step.get("ticks", 12)))))
    except (TypeError, ValueError):
        ticks = 12
    deltas = step.get("intent_deltas") or step.get("deltas") or {}
    clean: dict[str, float] = {}
    if isinstance(deltas, dict):
        for k, v in deltas.items():
            sk = str(k).strip()
            if not sk.startswith("intent_") and not sk.startswith("phys_intent_"):
                continue
            try:
                clean[sk] = float(v)
            except (TypeError, ValueError):
                continue
    out: dict[str, Any] = {"ticks": ticks, "intent_deltas": clean}
    phase = step.get("phase")
    if phase is not None:
        out["phase"] = str(phase)
    return out


def sanitize_recovery_intent_deltas(deltas: dict[str, Any] | None) -> dict[str, float]:
    """
    LLM often emits huge +intent_stop_recover deltas (misread as 'more recovery').
    Cap per-step deltas so graph + schedule do not saturate at 0.94 and stall CPG phases.
    """
    raw = clip_intent_deltas(deltas) if deltas else {}
    if not raw:
        return {}
    try:
        sr_max = float(os.environ.get("RKK_S2_RECOVER_STOP_RECOVER_DELTA_MAX", "0.22"))
    except ValueError:
        sr_max = 0.22
    try:
        stride_max = float(os.environ.get("RKK_S2_RECOVER_STRIDE_DELTA_MAX", "0.06"))
    except ValueError:
        stride_max = 0.06
    out: dict[str, float] = {}
    for k, v in raw.items():
        sk = str(k)
        if sk == "intent_stop_recover":
            out[sk] = float(max(-0.12, min(sr_max, v)))
        elif sk == "intent_stride":
            out[sk] = float(max(-0.20, min(stride_max, v)))
        else:
            out[sk] = float(v)
    return out


def enrich_recovery_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill empty intent_deltas from RECOVER bundle so schedule is executable."""
    bundle = macro_bundle("RECOVER_POSTURE")
    res = dict(bundle.get("residuals") or {})
    out: list[dict[str, Any]] = []
    for i, st in enumerate(steps):
        st = _normalize_step(st)
        d = sanitize_recovery_intent_deltas(st.get("intent_deltas") or {})
        if not d and res:
            d = dict(res)
        elif not d and i == 0:
            d = {"intent_stop_recover": 0.07, "intent_torso_forward": 0.06}
        row: dict[str, Any] = {"ticks": st["ticks"], "intent_deltas": d}
        if st.get("phase") is not None:
            row["phase"] = str(st["phase"])
        out.append(row)
    return out


def _recovery_min_step_ticks() -> int:
    try:
        return max(5, int(os.environ.get("RKK_S2_RECOVERY_MIN_STEP_TICKS", "10")))
    except ValueError:
        return 10


def _recovery_min_total_ticks() -> int:
    try:
        return max(20, int(os.environ.get("RKK_S2_RECOVERY_MIN_TOTAL_TICKS", "60")))
    except ValueError:
        return 60


def llm_ticks_look_like_step_indices(steps: list[dict[str, Any]]) -> bool:
    """
    Detect LLM treating ticks as step index (1,2,3,...) instead of frame duration.
    Does not match uniformly short steps (2,2) — those are rejected by validate.
    """
    if len(steps) < 2:
        return False
    ticks = [int(max(1, s.get("ticks", 1))) for s in steps]
    n = len(ticks)
    if max(ticks) > 15:
        return False
    if ticks == list(range(1, n + 1)):
        return True
    if n >= 3 and ticks[0] == 1 and all(
        ticks[i] == ticks[i - 1] + 1 for i in range(1, n)
    ):
        return True
    return False


def remediate_index_ticks_recovery_plan(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-map index-style ticks onto fallback phase durations; keep LLM intent_deltas."""
    fb = default_recovery_fallback_steps()
    fb_ticks = [int(s["ticks"]) for s in fb]
    out: list[dict[str, Any]] = []
    for i, st in enumerate(steps):
        norm = _normalize_step(st)
        norm["ticks"] = fb_ticks[i % len(fb_ticks)]
        out.append(norm)
    return out


def validate_llm_recovery_plan(steps: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return (ok, reject_reason)."""
    if not steps:
        return False, "empty"
    if not any(st.get("intent_deltas") for st in steps):
        return False, "no_deltas"
    min_step = _recovery_min_step_ticks()
    min_total = _recovery_min_total_ticks()
    ticks = [int(max(1, s.get("ticks", 1))) for s in steps]
    if sum(ticks) < min_total:
        return False, "total_too_short"
    if any(t < min_step for t in ticks):
        return False, "step_too_short"
    return True, ""


def prepare_llm_recovery_steps(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """
    Enrich, remediate index-style ticks, validate.
    Returns (steps_ready, was_remediated).
    """
    enriched = enrich_recovery_steps(steps)
    for st in enriched:
        st["intent_deltas"] = sanitize_recovery_intent_deltas(st.get("intent_deltas"))
    remediated = False
    if llm_ticks_look_like_step_indices(enriched):
        enriched = remediate_index_ticks_recovery_plan(enriched)
        enriched = enrich_recovery_steps(enriched)
        remediated = True
    ok, _ = validate_llm_recovery_plan(enriched)
    if not ok:
        return [], remediated
    return enriched, remediated
