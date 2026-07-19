"""Monitor RKK sim for N ticks: logs + camera screenshots every minute."""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "http://localhost:8000"
TICK_SPAN = 1000
INTERVAL_SEC = 60
OUT_DIR = Path(__file__).resolve().parent / "monitor_run"


def _get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _snapshot() -> dict:
    return _get(f"{API}/api/snapshot")


def _camera_frame() -> dict | None:
    try:
        return _get(f"{API}/camera/frame?view=fp", timeout=45.0)
    except Exception as exc:
        return {"error": str(exc)}


def _tail_task_log(since_tick: int) -> list[dict]:
    log_path = Path(__file__).resolve().parents[1] / "backend" / "logs" / "task_log.jsonl"
    if not log_path.is_file():
        return []
    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("tick", -1)) >= since_tick:
                rows.append(row)
    return rows


def _save_shot(out: Path, frame_b64: str | None) -> bool:
    if not frame_b64:
        return False
    try:
        out.write_bytes(base64.b64decode(frame_b64))
        return True
    except Exception:
        return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        snap0 = _snapshot()
    except urllib.error.URLError as exc:
        print(f"Backend not reachable at {API}: {exc}", file=sys.stderr)
        return 1

    start_tick = int(snap0.get("tick") or 0)
    target_tick = start_tick + TICK_SPAN
    meta = {
        "start_tick": start_tick,
        "target_tick": target_tick,
        "interval_sec": INTERVAL_SEC,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Monitor run {run_id}: tick {start_tick} -> {target_tick}")

    shot_idx = 0
    last_log_tick = start_tick
    summary_lines: list[str] = []

    while True:
        now = datetime.now(timezone.utc).isoformat()
        try:
            snap = _snapshot()
        except Exception as exc:
            snap = {"error": str(exc), "tick": None}

        tick = int(snap.get("tick") or 0)
        cam = _camera_frame()
        overlay = (cam or {}).get("overlay") if isinstance(cam, dict) else None
        active = (overlay or {}).get("active") if isinstance(overlay, dict) else None

        shot_idx += 1
        prefix = f"{shot_idx:03d}_t{tick}"
        snap_path = run_dir / f"{prefix}_snapshot.json"
        snap_path.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

        if isinstance(cam, dict):
            (run_dir / f"{prefix}_overlay.json").write_text(
                json.dumps(overlay or {}, indent=2), encoding="utf-8"
            )
            ok = _save_shot(run_dir / f"{prefix}_cam.jpg", cam.get("frame"))
            if not ok:
                (run_dir / f"{prefix}_cam_missing.txt").write_text(
                    str(cam.get("error", "no frame")), encoding="utf-8"
                )

        new_logs = _tail_task_log(last_log_tick)
        if new_logs:
            with (run_dir / "task_log_slice.jsonl").open("a", encoding="utf-8") as f:
                for row in new_logs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            last_log_tick = max(last_log_tick, max(int(r.get("tick", 0)) for r in new_logs))

        line = (
            f"{now} tick={tick} delta={tick - start_tick} "
            f"fallen={snap.get('fallen')} "
            f"hud={active.get('label') if active else None} "
            f"brg={active.get('bearing') if active else None} "
            f"rng={active.get('range_m') if active else None} "
            f"conf={active.get('conf') if active else None}"
        )
        summary_lines.append(line)
        print(line)

        # Append latest task_progress if any
        for row in reversed(new_logs):
            if row.get("event") == "task_progress":
                tp = (
                    f"  progress: oracle={row.get('oracle_dist_m')} "
                    f"vision_rng={row.get('vision_range_m')} "
                    f"heading_err={row.get('task_heading_err')} "
                    f"vision_brg={row.get('vision_bearing')}"
                )
                summary_lines.append(tp)
                print(tp)
                break

        (run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        if tick >= target_tick:
            meta["ended_at"] = now
            meta["end_tick"] = tick
            meta["shots"] = shot_idx
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"Done: {tick} >= {target_tick}")
            return 0

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
