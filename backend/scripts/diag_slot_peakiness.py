#!/usr/bin/env python3
"""2A: SlotAttention peakiness diagnostic suite (offline).

Measures mask peakiness on synthetic attention maps and optional bind-dump
meta.json slot stats. Does NOT change the hot path — diagnosis before treatment.

Usage:
  cd backend && python scripts/diag_slot_peakiness.py
  python scripts/diag_slot_peakiness.py --dump logs/bind_dumps/tick_18678
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.vision_resolve import _mask_peakiness, mask_peakiness_min  # noqa: E402


def _synthetic_cases() -> list[tuple[str, np.ndarray]]:
    h, w = 16, 16
    flat = np.ones((h, w), dtype=np.float32)
    blob = np.zeros((h, w), dtype=np.float32)
    blob[6:10, 6:10] = 1.0
    sharp = np.zeros((h, w), dtype=np.float32)
    sharp[8, 8] = 1.0
    ring = np.zeros((h, w), dtype=np.float32)
    ring[4:12, 4] = 0.8
    ring[4:12, 11] = 0.8
    ring[4, 4:12] = 0.8
    ring[11, 4:12] = 0.8
    soft = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    soft = np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 18.0).astype(np.float32)
    return [
        ("flat_uniform", flat),
        ("soft_gaussian", soft),
        ("box_blob", blob),
        ("single_pixel", sharp),
        ("ring", ring),
    ]


def _summarize_dump(dump_dir: Path) -> dict:
    meta_path = dump_dir / "meta.json"
    if not meta_path.is_file():
        return {"error": f"missing {meta_path}"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    slots = list(meta.get("slot_peakiness") or [])
    peaks = [float(s.get("mask_peakiness") or 0.0) for s in slots]
    uv_ok = sum(1 for s in slots if s.get("uv_valid"))
    thr = float(meta.get("mask_peakiness_min") or mask_peakiness_min())
    peaked = sum(1 for p in peaks if p >= thr)
    return {
        "tick": meta.get("tick"),
        "n_slots": len(slots),
        "uv_valid_count": uv_ok,
        "peakiness_min_threshold": thr,
        "peakiness_min": min(peaks) if peaks else None,
        "peakiness_max": max(peaks) if peaks else None,
        "peakiness_mean": float(np.mean(peaks)) if peaks else None,
        "slots_above_threshold": peaked,
        "fraction_peaked": (peaked / len(peaks)) if peaks else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dump",
        type=str,
        default="",
        help="Optional bind dump dir with meta.json (e.g. logs/bind_dumps/tick_18678)",
    )
    args = ap.parse_args()

    thr = float(mask_peakiness_min())
    print(f"mask_peakiness_min threshold = {thr:.3f}")
    print("--- synthetic attention maps ---")
    rows = []
    for name, mask in _synthetic_cases():
        p = float(_mask_peakiness(mask))
        rows.append((name, p, p >= thr))
        flag = "PASS" if p >= thr else "fail"
        print(f"  {name:16s}  peakiness={p:7.3f}  vs thr {flag}")

    peaked_syn = sum(1 for _, _, ok in rows if ok)
    print(f"synthetic peaked {peaked_syn}/{len(rows)}")

    if args.dump:
        dump_dir = Path(args.dump)
        if not dump_dir.is_absolute():
            dump_dir = ROOT / dump_dir
        summary = _summarize_dump(dump_dir)
        print("--- bind dump ---")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        default = ROOT / "logs" / "bind_dumps" / "tick_18678"
        if default.is_dir():
            summary = _summarize_dump(default)
            print("--- default bind dump tick_18678 ---")
            print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
