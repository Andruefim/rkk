"""Poll logs/rkk_run.jsonl until target tick reached."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "rkk_run.jsonl"


def last_tick() -> int:
    if not LOG.exists():
        return 0
    tick = 0
    with LOG.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "tick":
                tick = int(row["tick"])
    return tick


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    max_wait = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0

    start = time.time()
    prev = last_tick()
    print(f"wait_for_tick target={target} start={prev}", flush=True)

    while True:
        time.sleep(interval)
        cur = last_tick()
        dt = time.time() - start
        rate = (cur - prev) / interval if interval > 0 else 0.0
        eta = (target - cur) / rate if rate > 0 else float("inf")
        print(
            f"tick={cur} delta={cur-prev} rate={rate:.1f}/s eta={eta/60:.1f}min elapsed={dt/60:.1f}min",
            flush=True,
        )
        prev = cur
        if cur >= target:
            print(f"DONE tick={cur}", flush=True)
            return
        if dt >= max_wait:
            print(f"TIMEOUT tick={cur} (< {target})", flush=True)
            return


if __name__ == "__main__":
    main()
