"""Poll rkk_run.jsonl until target tick count."""
import json
import sys
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "rkk_run.jsonl"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def load_ticks() -> list[dict]:
    rows: list[dict] = []
    if not LOG.exists():
        return rows
    for line in LOG.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "tick":
            rows.append(r)
    return rows


for i in range(180):
    rows = load_ticks()
    last = rows[-1]["tick"] if rows else 0
    if i == 0 or i % 6 == 0 or last >= TARGET:
        s2 = (rows[-1].get("system2") or {}) if rows else {}
        ov = sum(1 for x in rows if (x.get("system2") or {}).get("fallen_override_active"))
        err = sum(1 for x in rows if (x.get("system2") or {}).get("error"))
        print(
            f"poll {i}: ticks={last} ov={ov} s2_err={err} macro={s2.get('macro')}",
            flush=True,
        )
    if last >= TARGET:
        print(f"done at tick {last}", flush=True)
        break
    time.sleep(10)
else:
    rows = load_ticks()
    last = rows[-1]["tick"] if rows else 0
    print(f"timeout at tick {last}", flush=True)
    sys.exit(1)
