"""In-memory ring of vision/OWM snapshots; dump on is_usable True→False.

Continuous per-tick disk dumps are expensive. Keep the last N ticks in RAM and
write ``bind_dumps/loss_tick_N/`` plus a neural_log event only on the loss edge.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any


def _ei(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return int(default)


def vision_trace_ticks() -> int:
    return max(8, min(400, _ei("RKK_VISION_TRACE_TICKS", 90)))


def _round(v: Any, nd: int = 4) -> Any:
    try:
        if v is None:
            return None
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


class VisionLossTrace:
    """Ring buffer of compact per-tick vision/OWM dicts."""

    def __init__(self, maxlen: int | None = None) -> None:
        n = int(maxlen) if maxlen is not None else vision_trace_ticks()
        self._buf: deque[dict[str, Any]] = deque(maxlen=max(8, n))

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def push(self, snap: dict[str, Any]) -> None:
        self._buf.append(dict(snap))

    def snapshots(self) -> list[dict[str, Any]]:
        return list(self._buf)

    def dump_to_dir(self, dump_root: Path, *, extra: dict[str, Any] | None = None) -> Path:
        dump_root.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "n": len(self._buf),
            "trace": list(self._buf),
        }
        if extra:
            payload["extra"] = extra
        path = dump_root / "trace.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path


def snapshot_from_scene(
    *,
    tick: int,
    scene: Any,
    visual_env: Any | None = None,
    slots: list[dict[str, Any]] | None = None,
    encode_this_tick: bool = False,
    dtheta: float = 0.0,
) -> dict[str, Any]:
    """Build one ring record from LatentSceneMemory + optional slot table."""
    act = scene.active() if scene is not None else None
    diags = dict(getattr(act, "diagnostics", None) or {}) if act is not None else {}
    slot_rows: list[dict[str, Any]] = []
    peaks: list[float] = []
    best_id = ""
    best_peak = -1.0
    uv_valid_n = 0
    bbox = None
    for s in list(slots or []):
        peak = float(s.get("mask_peakiness") or 0.0)
        peaks.append(peak)
        sid = str(s.get("slot_id") or "")
        uv_ok = bool(s.get("uv_valid"))
        if uv_ok:
            uv_valid_n += 1
        if peak > best_peak:
            best_peak = peak
            best_id = sid
            bbox = s.get("bbox")
        slot_rows.append(
            {
                "slot_id": sid,
                "mask_peakiness": _round(peak),
                "uv_valid": uv_ok,
                "u": _round(s.get("u")),
                "v": _round(s.get("v")),
                "bbox": s.get("bbox"),
                "label": s.get("label"),
            }
        )
    mean_peak = float(sum(peaks) / len(peaks)) if peaks else 0.0
    if visual_env is not None:
        encode_this_tick = bool(
            encode_this_tick
            or getattr(visual_env, "_encode_this_intervene", False)
        )
        if dtheta == 0.0:
            dtheta = float(getattr(visual_env, "_last_turn_dtheta", 0.0) or 0.0)
    return {
        "tick": int(tick),
        "mask_peakiness_mean": _round(mean_peak),
        "mask_peakiness": slot_rows,
        "best_slot_id": best_id,
        "uv_valid": uv_valid_n > 0,
        "uv_valid_n": uv_valid_n,
        "bbox": bbox,
        "encode_this_tick": bool(encode_this_tick),
        "dtheta": _round(dtheta if dtheta else getattr(scene, "last_odom_dtheta", 0.0)),
        "owm": {
            "bearing": _round(getattr(act, "bearing", None)) if act else None,
            "range_m": _round(getattr(act, "range_m", None)) if act else None,
            "conf": _round(getattr(act, "confidence", None)) if act else None,
            "bearing_sigma": _round(getattr(act, "bearing_sigma", None)) if act else None,
            "live_conf": _round(diags.get("live_conf")),
            "bearing_live_delta": _round(diags.get("bearing_live_delta")),
            "kalman_gain": _round(diags.get("kalman_gain")),
            "hard_lock": bool(getattr(scene, "hard_lock_active", False)),
            "source": diags.get("source"),
            "usable": bool(act.is_usable(int(tick))) if act is not None else False,
        },
    }
