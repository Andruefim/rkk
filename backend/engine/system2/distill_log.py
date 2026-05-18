"""
Wave A: helpers for System2 distill JSONL (schema, compression, health metrics).
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any

from engine.system2.schema import System2Proposal


def distill_log_path() -> Path:
    raw = os.environ.get("RKK_SYSTEM2_DISTILL_LOG", "logs/system2_distill.jsonl")
    return Path(raw)


def distill_enabled() -> bool:
    return os.environ.get("RKK_SYSTEM2_DISTILL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _env_float(key: str, default: str) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return float(default)


def distill_health_window() -> int:
    try:
        return max(20, int(os.environ.get("RKK_S2_DISTILL_HEALTH_WINDOW", "200")))
    except ValueError:
        return 200


def distill_min_success_rate() -> float:
    return _env_float("RKK_S2_DISTILL_MIN_SUCCESS_RATE", "0.25")


def distill_recover_min_success_rate() -> float:
    return _env_float("RKK_S2_DISTILL_RECOVER_MIN_SUCCESS_RATE", "0.25")


def blend_ready_student_conf() -> float:
    return _env_float("RKK_S2_DISTILL_BLEND_READY_STUDENT_CONF", "0.42")


def compress_recovery_steps(
    steps: list[dict[str, Any]] | None,
    *,
    max_steps: int = 24,
) -> list[dict[str, Any]]:
    """Compact recovery schedule for JSONL (ticks + intent_deltas only)."""
    out: list[dict[str, Any]] = []
    for st in list(steps or [])[:max_steps]:
        if not isinstance(st, dict):
            continue
        deltas = st.get("intent_deltas") or st.get("deltas") or {}
        clean: dict[str, float] = {}
        if isinstance(deltas, dict):
            for k, v in deltas.items():
                sk = str(k)
                if not sk.startswith("intent_"):
                    continue
                try:
                    clean[sk] = round(float(v), 4)
                except (TypeError, ValueError):
                    continue
        try:
            ticks = int(max(1, min(80, int(st.get("ticks", 1)))))
        except (TypeError, ValueError):
            ticks = 1
        out.append({"ticks": ticks, "intent_deltas": clean})
    return out


def proposal_distill_extra(
    proposal: System2Proposal | None,
    *,
    source: str,
    llm_macro: str | None = None,
) -> dict[str, Any]:
    """Fields to attach to macro-end distill when LLM/VLM shaped the episode."""
    if proposal is None:
        return {}
    out: dict[str, Any] = {}
    if source == "llm" or (proposal.intent_deltas or proposal.expected_state):
        out["plan_source"] = source
    if llm_macro:
        out["llm_macro"] = str(llm_macro)
    if proposal.intent_deltas:
        out["intent_deltas"] = {
            str(k): round(float(v), 4)
            for k, v in proposal.intent_deltas.items()
            if str(k).startswith("intent_")
        }
    if proposal.skill_id:
        out["proposal_skill_id"] = str(proposal.skill_id)
    return out


def pe_distill_extra(
    pe_diag: dict[str, Any], spec_expected_state: dict[str, float], skill_id: str | None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("pe_total", "max_pe", "homeo_veto", "veto_reason", "wave1", "reason"):
        if k in pe_diag and pe_diag[k] is not None:
            out[k] = pe_diag[k]
    if spec_expected_state:
        out["expected_state"] = {str(k): float(v) for k, v in spec_expected_state.items()}
    if skill_id:
        out["skill_id"] = skill_id
    return out


class DistillHealthTracker:
    """Rolling window over appended distill rows for diag + blend readiness hints."""

    def __init__(self) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=distill_health_window())

    def record(self, *, success: bool, macro: str, student_conf: float | None) -> None:
        self._rows.append(
            {
                "success": bool(success),
                "macro": str(macro or "IDLE").upper(),
                "student_conf": student_conf,
            }
        )

    def snapshot(self) -> dict[str, Any]:
        rows = list(self._rows)
        if not rows:
            return {}
        n = len(rows)
        overall = sum(1 for r in rows if r["success"]) / n
        recover = [r for r in rows if r["macro"] == "RECOVER_POSTURE"]
        rec_n = len(recover)
        rec_rate = (
            sum(1 for r in recover if r["success"]) / rec_n if rec_n else None
        )
        confs = [
            float(r["student_conf"])
            for r in recover
            if r.get("student_conf") is not None
        ]
        rec_conf_med = None
        if confs:
            s = sorted(confs)
            mid = len(s) // 2
            rec_conf_med = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0

        min_all = distill_min_success_rate()
        min_rec = distill_recover_min_success_rate()
        conf_th = blend_ready_student_conf()
        blend_ready = bool(
            rec_n >= max(20, min(40, n // 2))
            and rec_rate is not None
            and rec_rate >= min_rec
            and rec_conf_med is not None
            and rec_conf_med >= conf_th
        )
        warn = overall < min_all
        return {
            "distill_window": n,
            "distill_success_rate": round(overall, 4),
            "distill_recover_success_rate": (
                round(rec_rate, 4) if rec_rate is not None else None
            ),
            "distill_recover_conf_median": (
                round(rec_conf_med, 4) if rec_conf_med is not None else None
            ),
            "distill_blend_ready": blend_ready,
            "distill_quality_warn": warn,
        }


def analyze_distill_file(
    path: Path,
    *,
    window: int | None = None,
) -> dict[str, Any]:
    """Load JSONL and compute report dict (for CLI and tests)."""
    win = window or distill_health_window()
    rows: list[dict[str, Any]] = []
    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            lines = []
        for line in lines[-win:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _rate(subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return sum(1 for r in subset if r.get("success")) / len(subset)

    by_macro: dict[str, list[dict[str, Any]]] = {}
    by_source: dict[str, list[dict[str, Any]]] = {}
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        m = str(r.get("macro", "IDLE")).upper()
        by_macro.setdefault(m, []).append(r)
        s = str(r.get("source", "unknown"))
        by_source.setdefault(s, []).append(r)
        sk = r.get("skill_id") or r.get("proposal_skill_id")
        if sk:
            by_skill.setdefault(str(sk), []).append(r)

    recover = [
        r
        for r in rows
        if str(r.get("macro", "")).upper() == "RECOVER_POSTURE"
        or "fallen_override" in str(r.get("source", ""))
    ]
    d_cz = []
    d_ps = []
    pe_pass = 0
    pe_n = 0
    for r in recover:
        d = r.get("delta") or {}
        if isinstance(d, dict):
            try:
                d_cz.append(float(d.get("d_com_z", 0.0)))
            except (TypeError, ValueError):
                pass
            try:
                d_ps.append(float(d.get("d_posture", 0.0)))
            except (TypeError, ValueError):
                pass
        if r.get("success") and r.get("pe_total") is not None:
            pe_n += 1
            try:
                mx = float(r.get("max_pe", 1.0))
                if float(r["pe_total"]) <= mx:
                    pe_pass += 1
            except (TypeError, ValueError):
                pass
        elif r.get("homeo_veto") is False and r.get("success"):
            pe_pass += 1
            pe_n += 1

    llm_rows = [r for r in rows if str(r.get("source", "")) == "llm"]
    student_rows = [
        r for r in rows if "student" in str(r.get("source", ""))
    ]

    def _median(xs: list[float]) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0

    tracker = DistillHealthTracker()
    for r in rows:
        sc = r.get("student_conf")
        try:
            scf = float(sc) if sc is not None else None
        except (TypeError, ValueError):
            scf = None
        tracker.record(
            success=bool(r.get("success")),
            macro=str(r.get("macro", "IDLE")),
            student_conf=scf,
        )

    return {
        "path": str(path),
        "rows": len(rows),
        "overall_success_rate": _rate(rows),
        "by_macro": {k: _rate(v) for k, v in sorted(by_macro.items())},
        "by_source": {k: _rate(v) for k, v in sorted(by_source.items())},
        "by_skill_id": {k: _rate(v) for k, v in sorted(by_skill.items())},
        "recover": {
            "count": len(recover),
            "success_rate": _rate(recover),
            "median_d_com_z": _median(d_cz),
            "median_d_posture": _median(d_ps),
            "pe_pass_rate": (pe_pass / pe_n) if pe_n else None,
        },
        "llm_share": (len(llm_rows) / len(rows)) if rows else None,
        "student_share": (len(student_rows) / len(rows)) if rows else None,
        "health": tracker.snapshot(),
    }
