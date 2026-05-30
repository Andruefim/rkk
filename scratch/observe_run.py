"""Rolling analysis of logs/rkk_run.jsonl for long-run observation."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "rkk_run.jsonl"


def load_ticks() -> list[dict]:
    if not LOG.exists():
        return []
    lines = [json.loads(l) for l in LOG.open(encoding="utf-8") if l.strip()]
    return [r for r in lines if r.get("type") == "tick"]


def window_stats(ticks: list[dict], n: int = 300) -> dict:
    w = ticks[-n:] if len(ticks) >= n else ticks
    if not w:
        return {}

    def mean(key, *path):
        vals = []
        for t in w:
            d = t
            for p in path:
                d = (d or {}).get(p, {})
            v = d.get(key) if not path else d.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return round(sum(vals) / len(vals), 4) if vals else None

    fallen = sum(1 for t in w if t.get("body", {}).get("fallen"))
    blocked = sum(1 for t in w if t.get("action", {}).get("blocked"))
    s2 = sum(1 for t in w if t.get("action", {}).get("from_system2"))
    macros = Counter(str(t.get("system2", {}).get("macro", "?")) for t in w)
    actions = Counter(str(t.get("action", {}).get("variable", "")) for t in w)

    edge_deltas = [t.get("hud", {}).get("edge_delta", 0) for t in w]
    net_edges = sum(int(x or 0) for x in edge_deltas)

    return {
        "n": len(w),
        "tick_lo": w[0]["tick"],
        "tick_hi": w[-1]["tick"],
        "phi": mean("phi", "hud"),
        "discovery": mean("discovery_rate", "hud"),
        "alpha": mean("alpha_mean", "hud"),
        "compression": mean("compression_gain", "hud"),
        "block_rate": mean("block_rate", "hud"),
        "edges": mean("edge_count", "hud"),
        "net_edge_delta": net_edges,
        "fallen_pct": round(100 * fallen / len(w), 1),
        "blocked_pct": round(100 * blocked / len(w), 1),
        "from_s2_pct": round(100 * s2 / len(w), 1),
        "posture": mean("posture_stability", "body"),
        "com_z": mean("com_z", "phys"),
        "loco_reward": mean("reward_ema", "locomotion"),
        "s2_ema": mean("outcome_ema", "system2"),
        "student_conf": mean("student_conf", "system2"),
        "fixed_root": w[-1].get("body", {}).get("fixed_root"),
        "scope": w[-1].get("scope"),
        "top_macro": macros.most_common(3),
        "top_actions": actions.most_common(5),
        "s2_last": {
            k: w[-1].get("system2", {}).get(k)
            for k in (
                "macro",
                "idle",
                "outcome_ema",
                "student_conf",
                "fallen_override_active",
                "recover_tier",
                "distill_success_rate",
            )
        },
    }


def print_report(label: str, ticks: list[dict]) -> None:
    if not ticks:
        print(f"[{label}] no ticks yet")
        return
    last = ticks[-1]
    w300 = window_stats(ticks, 300)
    w1000 = window_stats(ticks, 1000) if len(ticks) >= 200 else None
    print(f"\n=== {label} | total_ticks_logged={len(ticks)} sim_tick={last['tick']} ===")
    print("window_300:", json.dumps(w300, ensure_ascii=False))
    if w1000:
        print("window_1000:", json.dumps(w1000, ensure_ascii=False))
    ev = last.get("events") or []
    if ev:
        print("last_events:", [e.get("text", "")[:120] for e in ev[-3:]])


def main() -> None:
    polls = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    for i in range(polls):
        ticks = load_ticks()
        print_report(f"poll_{i+1}/{polls}", ticks)
        if i + 1 < polls:
            time.sleep(interval)


if __name__ == "__main__":
    main()
