"""
Wave 2: episode success = homeostatic veto + prediction error vs intentional ``expected_state``.

Прокси Active Inference: минимизация PE по заявленным сенсорам; полный Free Energy не считается.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from typing import Any

from engine.system2.schema import EpisodeSuccessSpec, filter_expected_state_raw


def _g(obs: Mapping[str, Any], key: str, default: float = 0.5) -> float:
    v = obs.get(key, obs.get(f"phys_{key}", default))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def homeostatic_veto(obs: Mapping[str, Any]) -> tuple[bool, str]:
    """
    Возвращает (allowed, reason). False = эпизод не может считаться успешным для дистилляции.
    """
    try:
        stress_max = float(os.environ.get("RKK_S2_HOMEOSTRESS_MAX", "0.92"))
    except ValueError:
        stress_max = 0.92
    try:
        energy_min = float(os.environ.get("RKK_S2_HOMEOENERGY_MIN", "0.08"))
    except ValueError:
        energy_min = 0.08
    stress_max = float(max(0.5, min(0.999, stress_max)))
    energy_min = float(max(0.001, min(0.5, energy_min)))

    stress = _g(obs, "intero_stress", 0.0)
    energy = _g(obs, "intero_energy", 1.0)
    if stress > stress_max:
        return False, "intero_stress"
    if energy < energy_min:
        return False, "intero_energy"
    return True, ""


def obs_value_for_key(obs: Mapping[str, Any], key: str) -> float | None:
    if key in obs:
        try:
            return float(obs[key])
        except (TypeError, ValueError):
            return None
    pk = f"phys_{key}" if not str(key).startswith("phys_") else str(key)
    if pk in obs:
        try:
            return float(obs[pk])
        except (TypeError, ValueError):
            return None
    if str(key).startswith("phys_"):
        bare = str(key)[5:]
        if bare in obs:
            try:
                return float(obs[bare])
            except (TypeError, ValueError):
                return None
    return None


def prediction_error_total(
    obs: Mapping[str, Any],
    expected_state: Mapping[str, float],
    *,
    missing_key_penalty: float | None = None,
) -> float:
    """Суммарный L1 PE по ключам ``expected_state``."""
    try:
        pen = float(
            os.environ.get(
                "RKK_S2_PE_MISSING_KEY_PENALTY",
                str(missing_key_penalty if missing_key_penalty is not None else 1.0),
            )
        )
    except ValueError:
        pen = 1.0
    pen = float(max(0.0, min(3.0, pen)))
    total = 0.0
    for k, expv in expected_state.items():
        av = obs_value_for_key(obs, str(k))
        if av is None:
            total += pen
        else:
            total += abs(float(av) - float(expv))
    return float(total)


def resolve_max_prediction_error(
    llm_max_pe: float | None,
    *,
    n_keys: int,
    macro: str,
    skill_id: str | None,
) -> float:
    """LLM задаёт порог; иначе адаптивный fallback от числа ключей и «сложности» макроса."""
    if llm_max_pe is not None and llm_max_pe > 0:
        return float(min(6.0, max(0.02, llm_max_pe)))
    try:
        base = float(os.environ.get("RKK_S2_MAX_PE_DEFAULT", "0.18"))
    except ValueError:
        base = 0.18
    try:
        per_key = float(os.environ.get("RKK_S2_MAX_PE_PER_KEY", "0.04"))
    except ValueError:
        per_key = 0.04
    nk = max(0, int(n_keys))
    complexity = 1.0 + 0.1 * max(0, nk - 1)
    m = str(macro or "").upper()
    if m in ("EXPLORE", "LOCOMOTE_DELIVERY"):
        complexity += 0.12
    if skill_id and len(str(skill_id)) > 24:
        complexity += 0.05
    raw = base + per_key * max(1, nk) * complexity
    return float(min(4.0, max(0.06, raw)))


def evaluate_macro_success(
    obs_end: Mapping[str, Any],
    spec: EpisodeSuccessSpec,
    *,
    macro: str,
) -> tuple[bool, dict[str, Any]]:
    """
    Успех эпизода: гомеостаз + PE ≤ порога. Пустой ``expected_state`` → (False, reason) —
    вызывающий код должен использовать wave-1 fallback.
    """
    diag: dict[str, Any] = {}
    ok_h, reason = homeostatic_veto(obs_end)
    diag["homeo_veto"] = not ok_h
    diag["veto_reason"] = reason
    if not ok_h:
        return False, diag
    if not spec.expected_state:
        diag["reason"] = "no_expected_state"
        return False, diag

    max_pe = resolve_max_prediction_error(
        spec.max_prediction_error,
        n_keys=len(spec.expected_state),
        macro=macro,
        skill_id=spec.skill_id,
    )
    pe = prediction_error_total(obs_end, spec.expected_state)
    diag["pe_total"] = round(pe, 6)
    diag["max_pe"] = round(max_pe, 6)
    diag["expected_keys"] = sorted(spec.expected_state.keys())
    ok = pe <= max_pe
    return ok, diag


def _override_foot_contact_ok(obs: Mapping[str, Any]) -> tuple[bool, float, float]:
    try:
        foot_min = float(os.environ.get("RKK_S2_OVERRIDE_MIN_FOOT_CONTACT", "0.18"))
    except ValueError:
        foot_min = 0.18
    fl = _g(obs, "foot_contact_l", 0.0)
    fr = _g(obs, "foot_contact_r", 0.0)
    return max(fl, fr) >= foot_min, fl, fr


def _override_min_com_z_abs() -> float:
    raw = os.environ.get(
        "RKK_S2_OVERRIDE_MIN_COM_Z_ABS",
        os.environ.get("RKK_S2_OVERRIDE_MIN_COM_Z", "0.38"),
    )
    try:
        return float(raw)
    except ValueError:
        return 0.38


def override_recovered_tier1_ok(
    obs: Mapping[str, Any],
    obs0: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """
    Tier1: not fallen + measurable progress vs override start (prone → crawl/kneel).
    """
    try:
        d_cz_min = float(os.environ.get("RKK_S2_RECOVER_TIER1_DCOMZ", "0.04"))
    except ValueError:
        d_cz_min = 0.04
    try:
        d_ps_min = float(os.environ.get("RKK_S2_RECOVER_TIER1_DPOSTURE", "0.06"))
    except ValueError:
        d_ps_min = 0.06
    cz0 = float(_g(obs0, "com_z", 0.5))
    cz1 = float(_g(obs, "com_z", cz0))
    ps0 = float(_g(obs0, "posture_stability", 0.5))
    ps1 = float(_g(obs, "posture_stability", ps0))
    foot_ok, fl, fr = _override_foot_contact_ok(obs)
    d_cz = cz1 - cz0
    d_ps = ps1 - ps0
    try:
        prone_cz = float(os.environ.get("RKK_S2_RECOVER_TIER1_PRONE_COM_Z", "0.22"))
    except ValueError:
        prone_cz = 0.22
    try:
        prone_ps = float(os.environ.get("RKK_S2_RECOVER_TIER1_PRONE_POSTURE", "0.18"))
    except ValueError:
        prone_ps = 0.18
    try:
        min_cz1 = float(os.environ.get("RKK_S2_RECOVER_TIER1_MIN_COM_Z", "0.10"))
    except ValueError:
        min_cz1 = 0.10
    prone = cz0 < prone_cz or ps0 < prone_ps
    if prone:
        progress = d_cz >= d_cz_min and cz1 >= min_cz1
    else:
        progress = d_cz >= d_cz_min or d_ps >= d_ps_min
    ok = foot_ok and progress
    diag = {
        "recover_tier": 1,
        "d_com_z": round(d_cz, 5),
        "d_posture": round(d_ps, 5),
        "tier1_prone_mode": prone,
        "com_z": round(cz1, 4),
        "posture_stability": round(ps1, 4),
        "foot_contact_max": round(max(fl, fr), 4),
        "tier1_d_com_z_min": d_cz_min,
        "tier1_d_posture_min": d_ps_min,
    }
    if not ok:
        if not foot_ok:
            diag["override_exit_block"] = "foot_contact_low"
        else:
            diag["override_exit_block"] = "tier1_no_progress"
    return ok, diag


def override_recovered_tier2_ok(
    obs: Mapping[str, Any],
    obs0: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Tier2: stand-ready — relative com_z lift + absolute floor + posture."""
    try:
        ps_min = float(os.environ.get("RKK_S2_OVERRIDE_MIN_POSTURE", "0.42"))
    except ValueError:
        ps_min = 0.42
    cz_abs_min = _override_min_com_z_abs()
    try:
        d_cz_min = float(os.environ.get("RKK_S2_RECOVER_TIER2_DCOMZ", "0.08"))
    except ValueError:
        d_cz_min = 0.08
    cz0 = float(_g(obs0, "com_z", 0.5))
    cz1 = float(_g(obs, "com_z", cz0))
    ps1 = float(_g(obs, "posture_stability", 0.0))
    foot_ok, fl, fr = _override_foot_contact_ok(obs)
    cz_ok = cz1 >= cz_abs_min and (cz1 - cz0) >= d_cz_min
    ok = ps1 >= ps_min and cz_ok and foot_ok
    diag = {
        "recover_tier": 2,
        "posture_stability": round(ps1, 4),
        "com_z": round(cz1, 4),
        "d_com_z": round(cz1 - cz0, 5),
        "foot_contact_max": round(max(fl, fr), 4),
        "min_posture": ps_min,
        "min_com_z_abs": cz_abs_min,
        "tier2_d_com_z_min": d_cz_min,
    }
    if not ok:
        if ps1 < ps_min:
            diag["override_exit_block"] = "posture_low"
        elif not cz_ok:
            diag["override_exit_block"] = "com_z_low"
        else:
            diag["override_exit_block"] = "foot_contact_low"
    return ok, diag


def override_recovered_posture_ok(obs: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """
    Tier2 stand gate without obs0 (legacy callers). Prefer tier2 with obs0.
    Uses a prone floor baseline for relative com_z when obs0 is unknown.
    """
    try:
        floor_cz = float(os.environ.get("RKK_S2_RECOVER_BASELINE_COM_Z", "0.10"))
    except ValueError:
        floor_cz = 0.10
    cz = float(_g(obs, "com_z", 0.5))
    obs0 = {"com_z": min(cz, floor_cz), "posture_stability": 0.0}
    return override_recovered_tier2_ok(obs, obs0)


def evaluate_override_recovery_exit(
    obs: Mapping[str, Any],
    obs0: Mapping[str, Any],
    spec: EpisodeSuccessSpec,
    *,
    macro: str = "RECOVER_POSTURE",
) -> tuple[int, bool, dict[str, Any]]:
    """Returns (tier 0|1|2, success, diag). tier 0 = no exit. Check tier2 before tier1."""
    t2_ok, t2_diag = override_recovered_tier2_ok(obs, obs0)
    if t2_ok:
        ok, pe_diag = episode_success_with_pe_fallback(obs0, obs, spec, macro=macro)
        pe_diag.update(t2_diag)
        pe_diag["override_posture_gate"] = True
        return 2, ok, pe_diag
    t1_ok, t1_diag = override_recovered_tier1_ok(obs, obs0)
    if t1_ok:
        ok, pe_diag = episode_success_with_pe_fallback(obs0, obs, spec, macro=macro)
        pe_diag.update(t1_diag)
        pe_diag["override_posture_gate"] = True
        if not ok:
            pe_diag["wave1"] = True
            ok = wave1_delta_success(obs0, obs)
        return 1, ok, pe_diag
    block = t2_diag.get("override_exit_block") or t1_diag.get("override_exit_block", "")
    return 0, False, {
        "override_posture_gate": False,
        "override_exit_block": block,
        **{k: v for k, v in t2_diag.items() if k not in t1_diag},
        **t1_diag,
    }


def wave1_delta_success(
    obs0: Mapping[str, Any],
    obs1: Mapping[str, Any],
) -> bool:
    """Эвристика волны 1 по дельтам com_z / posture_stability."""
    cz0 = float(_g(obs0, "com_z", 0.5))
    cz1 = float(_g(obs1, "com_z", cz0))
    ps0 = float(_g(obs0, "posture_stability", 0.5))
    ps1 = float(_g(obs1, "posture_stability", ps0))
    return (cz1 - cz0) > 0.018 or (ps1 - ps0) > 0.04


def episode_success_with_pe_fallback(
    obs0: Mapping[str, Any],
    obs1: Mapping[str, Any],
    spec: EpisodeSuccessSpec,
    *,
    macro: str,
) -> tuple[bool, dict[str, Any]]:
    """
    Если ``spec.expected_state`` не пуст — PE+вето; иначе wave1 дельты (без вето для совместимости
    с прежним поведением макроса; вето можно включить env ``RKK_S2_WAVE1_HOMEOSTATIC=1``).
    """
    if spec.expected_state:
        return evaluate_macro_success(obs1, spec, macro=macro)
    use_veto = os.environ.get("RKK_S2_WAVE1_HOMEOSTATIC", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if use_veto:
        ok_h, reason = homeostatic_veto(obs1)
        if not ok_h:
            return False, {"homeo_veto": True, "veto_reason": reason, "wave1": True}
    ok = wave1_delta_success(obs0, obs1)
    return ok, {"wave1": True, "homeo_veto": False}


def curriculum_stage_to_spec(stage: Any) -> EpisodeSuccessSpec:
    """Собрать ``EpisodeSuccessSpec`` из ``CurriculumStage`` (опциональные поля s2_*)."""
    raw_es = getattr(stage, "s2_expected_state", None)
    es = raw_es if isinstance(raw_es, dict) else {}
    sid = getattr(stage, "s2_skill_id", None)
    skill_id = str(sid).strip()[:120] if sid is not None and str(sid).strip() else None
    mx = getattr(stage, "s2_max_prediction_error", None)
    try:
        mx_f = float(mx) if mx is not None else None
    except (TypeError, ValueError):
        mx_f = None
    return EpisodeSuccessSpec(
        expected_state=filter_expected_state_raw(es),
        max_prediction_error=mx_f,
        skill_id=skill_id,
    )


def should_attach_curriculum_pe_spec(macro: str, gov: EpisodeSuccessSpec) -> bool:
    """
    Curriculum ``s2_expected_state`` описывает текущий *этап* (часто стойка/статика).
    Не смешивать его с макросами вроде EXPLORE/LOCOMOTE — иначе почти все эпизоды
    падают по PE и дистилляция врёт (см. лог: EXPLORE + skill_id static_stance).
    LLM/proposal ``expected_state`` по-прежнему задаётся для любого макроса отдельно.
    """
    if not gov.expected_state:
        return False
    m = str(macro or "").strip().upper()
    return m == "IDLE"


def sanitize_s2_slug(s: str, max_len: int = 48) -> str:
    x = str(s).lower().replace(".", "_").replace(" ", "_").replace("-", "_")
    x = re.sub(r"[^a-z0-9_]+", "_", x).strip("_")
    return (x or "x")[:max_len]


def build_s2_detector_id(
    macro: str,
    skill_id: str | None,
    expected_state: dict[str, float],
) -> str:
    """Стабильный detector_id: ``system2:{skill|macro}:{sorted_keys}``."""
    part = sanitize_s2_slug(skill_id or macro or "idle")
    keys = sorted(expected_state.keys())
    keys_slug = "_".join(sanitize_s2_slug(k, 40) for k in keys)[:140]
    raw = f"system2:{part}:{keys_slug}" if keys_slug else f"system2:{part}"
    if len(raw) > 200:
        h = hashlib.sha256(raw.encode()).hexdigest()[:12]
        raw = f"system2:{part}:k{len(keys)}_{h}"
    return raw


# macro_or_skill_id -> short description (logging / docs)
MACRO_SUCCESS_PREDICATE_NOTES: dict[str, str] = {
    "RECOVER_POSTURE": "override_exit: PE vs expected_state or wave1 deltas",
}
