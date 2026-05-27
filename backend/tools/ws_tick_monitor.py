"""Monitor causal-stream WS: tick progress, stalls, System2/LLM flags."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import websockets


async def run(target: int, stall_sec: float, sample_every: int) -> int:
    uri = "ws://localhost:8000/ws/causal-stream"
    t0 = time.time()
    last_tick = -1
    last_change = t0
    max_gap = 0.0
    llm_inflight_msgs = 0
    same_tick_streak = 0
    errors = 0
    milestones: list[tuple[int, float]] = []

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        print(f"[monitor] connected -> target tick {target}", flush=True)
        start_tick: int | None = None
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=stall_sec)
            except asyncio.TimeoutError:
                print(
                    f"[monitor] STALL: no WS message for {stall_sec:.0f}s "
                    f"(last tick={last_tick})",
                    flush=True,
                )
                return 2

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                errors += 1
                continue

            tick = int(data.get("tick", 0))
            if start_tick is None:
                start_tick = tick
                print(f"[monitor] start_tick={start_tick}", flush=True)
            s2 = data.get("system2") if isinstance(data.get("system2"), dict) else {}
            src = str(s2.get("source") or "")
            macro = str(s2.get("macro") or "")
            idle = bool(s2.get("idle"))
            llm = bool(s2.get("llm_inflight"))

            if llm:
                llm_inflight_msgs += 1

            now = time.time()
            if tick != last_tick:
                gap = now - last_change
                if last_tick >= 0:
                    max_gap = max(max_gap, gap)
                last_change = now
                last_tick = tick
                same_tick_streak = 0
                delta = tick - (start_tick or 0)
                if sample_every > 0 and delta > 0 and delta % sample_every == 0:
                    elapsed = now - t0
                    rate = delta / elapsed if elapsed > 0 else 0
                    milestones.append((tick, rate))
                    print(
                        f"[monitor] tick={tick} +{delta} rate={rate:.2f}/s "
                        f"S2={macro or '—'} src={src or '—'} "
                        f"idle={idle} llm={llm} d={data.get('graph', {}).get('d')}",
                        flush=True,
                    )
            else:
                same_tick_streak += 1

            delta = tick - (start_tick or 0)
            if delta >= target:
                elapsed = now - t0
                print(
                    f"[monitor] OK +{delta} ticks ({start_tick}->{tick}) in {elapsed:.1f}s "
                    f"avg={delta/elapsed:.2f}/s max_gap={max_gap:.1f}s "
                    f"llm_inflight_msgs={llm_inflight_msgs} ws_errors={errors}",
                    flush=True,
                )
                return 0

            if now - last_change > stall_sec:
                print(
                    f"[monitor] STUCK: tick frozen at {tick} for {now-last_change:.1f}s "
                    f"S2 macro={macro} src={src} idle={idle} llm={llm}",
                    flush=True,
                )
                return 1

    return 3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target",
        type=int,
        default=3000,
        help="ticks to advance from first observed tick (not absolute)",
    )
    p.add_argument("--stall-sec", type=float, default=120.0)
    p.add_argument("--sample-every", type=int, default=500)
    args = p.parse_args()
    rc = asyncio.run(run(args.target, args.stall_sec, args.sample_every))
    sys.exit(rc)


if __name__ == "__main__":
    main()
