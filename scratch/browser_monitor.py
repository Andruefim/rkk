"""Browser screenshot every 60s until +1000 ticks from start."""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

API = "http://localhost:8000/api/snapshot"
UI = "http://localhost:5173/"
INTERVAL = 60
TICK_SPAN = 1000
OUT = Path(__file__).resolve().parent / "monitor_run" / "browser_minute"


def snapshot() -> dict:
    with urllib.request.urlopen(API, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap0 = snapshot()
    start = int(snap0.get("tick") or 0)
    target = start + TICK_SPAN
    meta = {"start_tick": start, "target_tick": target, "interval_sec": INTERVAL}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(UI, wait_until="networkidle", timeout=60000)
        time.sleep(3)

        idx = 0
        while True:
            idx += 1
            snap = snapshot()
            tick = int(snap.get("tick") or 0)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = OUT / f"{idx:03d}_t{tick}_{ts}.png"
            page.screenshot(path=str(path), full_page=False)
            tt = snap.get("task_tree") or {}
            line = (
                f"{ts} tick={tick} delta={tick-start} progress={tt.get('progress')} "
                f"node={tt.get('current_node_id')} fallen={snap.get('fallen')}"
            )
            lines.append(line)
            (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(line, flush=True)
            if tick >= target:
                meta["end_tick"] = tick
                meta["shots"] = idx
                (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"DONE tick {tick} >= {target}", flush=True)
                break
            time.sleep(INTERVAL)
        browser.close()


if __name__ == "__main__":
    main()
