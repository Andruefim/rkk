"""Poll /state while browser WS drives sim; detect stalls."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def fetch_state(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/health")
    p.add_argument("--delta", type=int, default=800)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--stall-sec", type=float, default=45.0)
    p.add_argument("--timeout", type=float, default=90.0)
    args = p.parse_args()

    t0 = time.time()
    start_tick: int | None = None
    last_tick = -1
    last_change = t0
    llm_hits = 0

    while True:
        try:
            d = fetch_state(args.url, args.timeout)
        except Exception as e:
            print(f"[poll] fetch error: {e}", flush=True)
            time.sleep(args.interval)
            continue

        tick = int(d.get("tick", 0))
        s2 = d.get("system2") if isinstance(d.get("system2"), dict) else {}
        sleep = d.get("sleep") if isinstance(d.get("sleep"), dict) else {}
        if not sleep and d.get("sleeping") is not None:
            sleep = {"sleeping": d.get("sleeping"), "phase": d.get("sleep_phase", "")}
        llm = bool(s2.get("llm_inflight"))
        if llm:
            llm_hits += 1

        if start_tick is None:
            start_tick = tick
            print(f"[poll] start_tick={start_tick}", flush=True)

        now = time.time()
        if tick != last_tick:
            last_change = now
            last_tick = tick

        delta = tick - start_tick
        elapsed = now - t0
        rate = delta / elapsed if elapsed > 0 else 0.0
        print(
            f"[poll] tick={tick} +{delta} rate={rate:.2f}/s "
            f"S2={s2.get('macro','')} src={s2.get('source','')} "
            f"sleep={sleep.get('phase', sleep.get('sleeping', ''))} llm={llm}",
            flush=True,
        )

        if delta >= args.delta:
            print(f"[poll] OK +{delta} in {elapsed:.1f}s llm_hits={llm_hits}", flush=True)
            sys.exit(0)

        if now - last_change > args.stall_sec:
            print(f"[poll] STUCK at tick={tick} for {now-last_change:.0f}s", flush=True)
            sys.exit(1)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
