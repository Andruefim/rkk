"""
File logger for human-task progress diagnostics (understanding → plan → motion → verify → outcome).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_MAX_BYTES = 5 * 1024 * 1024
_JSONL_NAME = "task_log.jsonl"
_TXT_NAME = "task_log.txt"
_AI_ANALYSIS_NAME = "ai_task_analysis.txt"

# Runtime session logs wiped on backend start (not archival external_review/).
_SESSION_LOG_NAMES: tuple[str, ...] = (
    _JSONL_NAME,
    _TXT_NAME,
    f"{_JSONL_NAME}.1",
    f"{_TXT_NAME}.1",
    _AI_ANALYSIS_NAME,
    "live_uv_candidates.jsonl",
    "system2_distill.jsonl",
)


def task_log_enabled() -> bool:
    raw = os.environ.get("RKK_TASK_LOG", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def task_log_dir() -> Path:
    raw = os.environ.get("RKK_TASK_LOG_DIR", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent.parent / "logs"


def _jsonl_path() -> Path:
    return task_log_dir() / _JSONL_NAME


def _txt_path() -> Path:
    return task_log_dir() / _TXT_NAME


def ai_analysis_path() -> Path:
    return task_log_dir() / _AI_ANALYSIS_NAME


def clear_session_logs() -> list[str]:
    """Delete runtime log files so each project start begins with a clean slate."""
    cleared: list[str] = []
    log_dir = task_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return cleared
    with _LOCK:
        for name in _SESSION_LOG_NAMES:
            path = log_dir / name
            try:
                if path.is_file():
                    path.unlink()
                    cleared.append(name)
            except Exception:
                pass
        # Extra rotated / stray session dumps in the log root.
        try:
            for path in log_dir.glob("live_uv_candidates.jsonl*"):
                if path.is_file():
                    path.unlink()
                    cleared.append(path.name)
        except Exception:
            pass
    return cleared


def read_task_log_events(
    *,
    tick_lo: int,
    tick_hi: int,
    max_events: int = 400,
) -> list[dict[str, Any]]:
    """Load JSONL events with tick in [tick_lo, tick_hi] (inclusive)."""
    path = _jsonl_path()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        # Read without holding the write lock for the whole scan (append stays atomic).
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tick = row.get("tick")
                if tick is None:
                    continue
                try:
                    t = int(tick)
                except (TypeError, ValueError):
                    continue
                if t < int(tick_lo) or t > int(tick_hi):
                    continue
                out.append(row)
    except Exception:
        return out
    if len(out) > max_events:
        # Prefer latest samples in the window for the AI prompt.
        out = out[-max_events:]
    return out


def read_ai_analysis_text(*, max_chars: int = 12000) -> str:
    path = ai_analysis_path()
    if not path.is_file():
        return ""
    try:
        with _LOCK:
            text = path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]
    except Exception:
        return ""


def append_ai_analysis(tick_lo: int, tick_hi: int, analysis: str) -> None:
    """Append one window analysis block to ai_task_analysis.txt."""
    body = str(analysis or "").strip()
    if not body:
        body = "(empty analysis)"
    block = f"[{int(tick_lo)}-{int(tick_hi)}]: {body}\n\n"
    try:
        with _LOCK:
            log_dir = task_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            path = ai_analysis_path()
            with path.open("a", encoding="utf-8") as f:
                f.write(block)
    except Exception:
        pass


def _round_float(v: float, ndigits: int = 4) -> float:
    return round(float(v), ndigits)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _round_float(value)
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in list(value)[:32]]
    return str(value)[:200]


def summarize_expected_state(expected_state: dict[str, float] | None) -> dict[str, Any]:
    es = dict(expected_state or {})
    n_keys = len(es)
    nonzero = 0
    pairs: list[tuple[str, float]] = []
    for k, v in es.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if abs(fv) > 1e-6:
            nonzero += 1
        pairs.append((str(k), fv))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top5 = {k: _round_float(v) for k, v in pairs[:5]}
    return {
        "n_expected_keys": n_keys,
        "n_nonzero": nonzero,
        "top5": top5,
    }


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.is_file() and path.stat().st_size > _MAX_BYTES:
            rotated = Path(str(path) + ".1")
            if rotated.is_file():
                rotated.unlink()
            path.rename(rotated)
    except Exception:
        pass


def _format_human_line(record: dict[str, Any]) -> str:
    ts = str(record.get("ts", ""))
    tick = record.get("tick", "")
    event = str(record.get("event", ""))
    parts = [ts, f"tick={tick}", f"event={event}"]
    for key in sorted(record.keys()):
        if key in ("ts", "tick", "event"):
            continue
        val = record[key]
        if isinstance(val, str):
            esc = val.replace('"', "'")
            parts.append(f'{key}="{esc}"' if (" " in esc or "=" in esc) else f"{key}={esc}")
        elif isinstance(val, (dict, list)):
            try:
                parts.append(f"{key}={json.dumps(val, ensure_ascii=False, separators=(',', ':'))}")
            except Exception:
                parts.append(f"{key}={val!r}")
        else:
            parts.append(f"{key}={val}")
    return " ".join(parts)


def task_log_event(event: str, *, tick: int | None = None, **fields: Any) -> None:
    """Append one task-progress event (JSONL + human-readable). Never raises."""
    if not task_log_enabled():
        return
    try:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "tick": int(tick) if tick is not None else None,
            "event": str(event),
        }
        for k, v in fields.items():
            if v is None:
                continue
            record[str(k)] = _json_safe(v)

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        human = _format_human_line(record)

        with _LOCK:
            log_dir = task_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            jsonl = _jsonl_path()
            txt = _txt_path()
            _rotate_if_needed(jsonl)
            _rotate_if_needed(txt)
            with jsonl.open("a", encoding="utf-8") as fj:
                fj.write(line + "\n")
            with txt.open("a", encoding="utf-8") as ft:
                ft.write(human + "\n")
    except Exception:
        pass