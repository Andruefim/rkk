"""
Neural / latent / WM diagnostic logger.

Separate from task_log: explains *why* vision/OWM/WM/AI chose something,
not only what the body did. Writes:
  backend/logs/neural_log.jsonl
  backend/logs/neural_log.txt

Channels (RKK_NEURAL_LOG_CHANNELS, comma-separated, default all):
  vision | owm | wm | active_inf | nav | latent | snapshot

Env:
  RKK_NEURAL_LOG=1
  RKK_NEURAL_LOG_EVERY=5          # throttle for high-freq channels
  RKK_NEURAL_LOG_LATENT_FULL=0    # 1 = dump truncated latent vectors
  RKK_NEURAL_LOG_MAX_MB=8
"""
from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_LOCK = threading.Lock()
_JSONL_NAME = "neural_log.jsonl"
_TXT_NAME = "neural_log.txt"
_DEFAULT_CHANNELS = frozenset(
    {"vision", "owm", "wm", "active_inf", "nav", "latent", "snapshot"}
)
# High-frequency channels throttled by RKK_NEURAL_LOG_EVERY (tick-based).
_THROTTLED = frozenset({"owm", "wm", "active_inf", "nav", "snapshot"})
_last_tick_by_channel: dict[str, int] = {}


def neural_log_enabled() -> bool:
    raw = os.environ.get("RKK_NEURAL_LOG", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def neural_log_every() -> int:
    try:
        return max(1, int(os.environ.get("RKK_NEURAL_LOG_EVERY", "5")))
    except ValueError:
        return 5


def neural_log_latent_full() -> bool:
    raw = os.environ.get("RKK_NEURAL_LOG_LATENT_FULL", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def neural_log_max_bytes() -> int:
    try:
        mb = float(os.environ.get("RKK_NEURAL_LOG_MAX_MB", "8"))
    except ValueError:
        mb = 8.0
    return int(max(1.0, mb) * 1024 * 1024)


def neural_log_channels() -> frozenset[str]:
    raw = os.environ.get("RKK_NEURAL_LOG_CHANNELS", "").strip()
    if not raw:
        return _DEFAULT_CHANNELS
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return frozenset(parts) if parts else _DEFAULT_CHANNELS


def neural_log_dir() -> Path:
    raw = os.environ.get("RKK_NEURAL_LOG_DIR", "").strip()
    if raw:
        return Path(raw)
    raw_task = os.environ.get("RKK_TASK_LOG_DIR", "").strip()
    if raw_task:
        return Path(raw_task)
    return Path(__file__).resolve().parent.parent / "logs"


def neural_log_session_files() -> tuple[str, ...]:
    return (_JSONL_NAME, _TXT_NAME, f"{_JSONL_NAME}.1", f"{_TXT_NAME}.1")


def _round_float(v: float, ndigits: int = 4) -> float:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(fv):
        return 0.0
    return round(fv, ndigits)


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)[:120]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _round_float(value)
    if isinstance(value, str):
        return value[:800]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 48:
                out["…"] = f"+{len(value) - 48}"
                break
            out[str(k)[:80]] = _json_safe(v, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        seq = list(value)
        head = [_json_safe(v, depth=depth + 1) for v in seq[:24]]
        if len(seq) > 24:
            head.append(f"…+{len(seq) - 24}")
        return head
    if hasattr(value, "detach"):
        try:
            import numpy as np

            arr = value.detach().float().cpu().numpy().reshape(-1)
            return summarize_latent(arr)
        except Exception:
            return {"type": "tensor", "error": "unreadable"}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return summarize_latent(value)
    except Exception:
        pass
    return str(value)[:200]


def summarize_latent(vec: Any, *, max_dump: int = 16) -> dict[str, Any]:
    """Compact latent stats (always); optional truncated vector dump."""
    out: dict[str, Any] = {"dim": 0, "l2": 0.0, "mean": 0.0, "abs_max": 0.0}
    try:
        import numpy as np

        if hasattr(vec, "detach"):
            arr = vec.detach().float().cpu().numpy().reshape(-1)
        else:
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return out
        out["dim"] = int(arr.size)
        out["l2"] = _round_float(float(np.linalg.norm(arr)))
        out["mean"] = _round_float(float(arr.mean()))
        out["abs_max"] = _round_float(float(np.max(np.abs(arr))))
        if neural_log_latent_full():
            out["head"] = [_round_float(float(x)) for x in arr[:max_dump].tolist()]
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


def latent_pair_stats(a: Any, b: Any) -> dict[str, Any]:
    """Cosine + L2 delta between two latents (empty-safe)."""
    try:
        import numpy as np

        def _arr(x: Any) -> np.ndarray:
            if x is None:
                return np.zeros(0, dtype=np.float64)
            if hasattr(x, "detach"):
                return x.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
            return np.asarray(x, dtype=np.float64).reshape(-1)

        aa, bb = _arr(a), _arr(b)
        if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
            return {"cosine": 0.0, "l2_delta": 0.0, "aligned": False}
        na = float(np.linalg.norm(aa))
        nb = float(np.linalg.norm(bb))
        cos = float(np.dot(aa, bb) / max(na * nb, 1e-12)) if na > 0 and nb > 0 else 0.0
        return {
            "cosine": _round_float(cos),
            "l2_delta": _round_float(float(np.linalg.norm(aa - bb))),
            "aligned": True,
        }
    except Exception:
        return {"cosine": 0.0, "l2_delta": 0.0, "aligned": False}


def _rotate_if_needed(path: Path) -> None:
    try:
        if path.is_file() and path.stat().st_size > neural_log_max_bytes():
            rotated = Path(str(path) + ".1")
            if rotated.is_file():
                rotated.unlink()
            path.rename(rotated)
    except Exception:
        pass


def _format_human_line(record: dict[str, Any]) -> str:
    ts = str(record.get("ts", ""))
    tick = record.get("tick", "")
    ch = str(record.get("channel", ""))
    event = str(record.get("event", ""))
    parts = [ts, f"tick={tick}", f"ch={ch}", f"event={event}"]
    for key in sorted(record.keys()):
        if key in ("ts", "tick", "channel", "event"):
            continue
        val = record[key]
        if isinstance(val, str):
            esc = val.replace('"', "'")
            parts.append(
                f'{key}="{esc}"' if (" " in esc or "=" in esc) else f"{key}={esc}"
            )
        elif isinstance(val, (dict, list)):
            try:
                parts.append(
                    f"{key}={json.dumps(val, ensure_ascii=False, separators=(',', ':'))}"
                )
            except Exception:
                parts.append(f"{key}={val!r}")
        else:
            parts.append(f"{key}={val}")
    return " ".join(parts)


def _should_throttle(channel: str, tick: int | None) -> bool:
    if channel not in _THROTTLED:
        return False
    if tick is None:
        return False
    every = neural_log_every()
    last = _last_tick_by_channel.get(channel)
    if last is not None and int(tick) - int(last) < every:
        return True
    _last_tick_by_channel[channel] = int(tick)
    return False


def neural_log_event(
    channel: str,
    event: str,
    *,
    tick: int | None = None,
    force: bool = False,
    **fields: Any,
) -> None:
    """Append one neural diagnostic event. Never raises."""
    if not neural_log_enabled():
        return
    ch = str(channel or "snapshot").strip().lower() or "snapshot"
    if ch not in neural_log_channels():
        return
    if not force and _should_throttle(ch, tick):
        return
    try:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "tick": int(tick) if tick is not None else None,
            "channel": ch,
            "event": str(event),
        }
        for k, v in fields.items():
            if v is None:
                continue
            record[str(k)] = _json_safe(v)

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        human = _format_human_line(record)

        with _LOCK:
            log_dir = neural_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            jsonl = log_dir / _JSONL_NAME
            txt = log_dir / _TXT_NAME
            _rotate_if_needed(jsonl)
            _rotate_if_needed(txt)
            with jsonl.open("a", encoding="utf-8") as fj:
                fj.write(line + "\n")
            with txt.open("a", encoding="utf-8") as ft:
                ft.write(human + "\n")
    except Exception:
        pass


def summarize_slot_table(slots: Iterable[dict[str, Any]] | None, *, top: int = 5) -> list[dict[str, Any]]:
    """Compact view of slot candidates for resolve debugging."""
    rows = list(slots or [])
    out: list[dict[str, Any]] = []
    for s in rows[:top]:
        item = {
            "slot_id": s.get("slot_id"),
            "label": s.get("label"),
            "uv_valid": bool(s.get("uv_valid")),
            "u": _round_float(float(s.get("u") or 0.5)),
            "v": _round_float(float(s.get("v") or 0.5)),
            "activation": _round_float(float(s.get("activation") or 0.0)),
            "mask_peakiness": _round_float(float(s.get("mask_peakiness") or 0.0)),
            "match_score": _round_float(float(s.get("match_score") or s.get("score") or 0.0)),
            "match_label": _round_float(float(s.get("match_label") or 0.0)),
            "match_concept": _round_float(float(s.get("match_concept") or 0.0)),
            "match_ontology": _round_float(float(s.get("match_ontology") or 0.0)),
        }
        vec = s.get("latent", s.get("vector"))
        if vec is not None:
            item["latent"] = summarize_latent(vec)
        out.append(item)
    return out


def summarize_prediction_gaps(
    current: dict[str, float] | None,
    predicted: dict[str, float] | None,
    targets: dict[str, float] | None,
    *,
    top: int = 8,
) -> dict[str, Any]:
    """Largest |pred-target| and |pred-current| gaps for Active Inference / WM."""
    curr = dict(current or {})
    pred = dict(predicted or {})
    targ = dict(targets or {})
    keys = set(targ) | set(pred)
    gaps: list[tuple[str, float, float, float, float]] = []
    for k in keys:
        try:
            c = float(curr.get(k, 0.5))
            p = float(pred.get(k, c))
            t = float(targ.get(k, p))
        except (TypeError, ValueError):
            continue
        gaps.append((str(k), abs(p - t), abs(p - c), p, t))
    gaps.sort(key=lambda row: row[1], reverse=True)
    top_rows = [
        {
            "node": k,
            "abs_pred_target": _round_float(apt),
            "abs_pred_curr": _round_float(apc),
            "pred": _round_float(p),
            "target": _round_float(t),
        }
        for k, apt, apc, p, t in gaps[:top]
    ]
    return {"n": len(gaps), "top": top_rows}
