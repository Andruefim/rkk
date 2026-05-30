import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

lines = [json.loads(l) for l in Path("logs/rkk_run.jsonl").open(encoding="utf-8") if l.strip()]
ticks = [l for l in lines if l.get("type") == "tick"]
print("=== RUN SUMMARY tick", ticks[-1]["tick"] if ticks else 0, "===")

fixed = []
curriculum = []
for t in ticks:
    b = t.get("body", {})
    if b.get("fixed_root") is not None:
        fixed.append((t["tick"], b["fixed_root"]))
    sc = t.get("scope") or {}
    if sc.get("phase") is not None:
        curriculum.append((t["tick"], sc["phase"], sc.get("mastery_quality")))

for name, arr, idx in [("fixed_root", fixed, 1), ("curriculum_phase", curriculum, 1)]:
    prev = None
    changes = []
    for row in arr:
        v = row[idx]
        if prev is None:
            prev = v
            continue
        if v != prev:
            changes.append((row[0], prev, v))
            prev = v
    print(name, "transitions", changes)

fall_streak = 0
max_fall_streak = 0
fall_episodes = 0
for t in ticks:
    if t.get("body", {}).get("fallen"):
        fall_streak += 1
        max_fall_streak = max(max_fall_streak, fall_streak)
    else:
        if fall_streak > 0:
            fall_episodes += 1
        fall_streak = 0
print("fall_episodes", fall_episodes, "max_consecutive_fallen_ticks", max_fall_streak)
print("total_fallen_ticks", sum(1 for t in ticks if t.get("body", {}).get("fallen")))

chunks = [ticks[i : i + 200] for i in range(0, len(ticks), 200)]
print("--- chunk trends (200 ticks) ---")
for ci, ch in enumerate(chunks):
    if not ch:
        continue
    dr = [t["hud"]["discovery_rate"] for t in ch]
    sc = [
        (t.get("system2") or {}).get("student_conf")
        for t in ch
        if (t.get("system2") or {}).get("student_conf") is not None
    ]
    ema = [
        (t.get("system2") or {}).get("outcome_ema")
        for t in ch
        if (t.get("system2") or {}).get("outcome_ema") is not None
    ]
    ps = [
        t.get("body", {}).get("posture_stability")
        for t in ch
        if t.get("body", {}).get("posture_stability") is not None
    ]
    lr = [
        (t.get("locomotion") or {}).get("reward_ema")
        for t in ch
        if (t.get("locomotion") or {}).get("reward_ema") is not None
    ]
    ec = [t["hud"]["edge_count"] for t in ch]
    fallen = sum(1 for t in ch if t.get("body", {}).get("fallen"))
    macros = Counter(str((t.get("system2") or {}).get("macro", "?")) for t in ch)
    lo, hi = ch[0]["tick"], ch[-1]["tick"]
    print(
        f"  c{ci+1} t{lo}-{hi} disc={statistics.mean(dr):.2f} posture={statistics.mean(ps):.3f} "
        f"loco={statistics.mean(lr):.3f} fallen={fallen} edges~{statistics.mean(ec):.0f} "
        f"s2conf={statistics.mean(sc) if sc else 0:.2f} s2ema={statistics.mean(ema) if ema else 0:.2f} "
        f"macros={dict(macros.most_common(2))}"
    )

last = ticks[-1]
wm = last.get("wm") or {}
print("wm_last_keys", list(wm.keys()))
if wm.get("ensemble"):
    print("ensemble_last", wm["ensemble"])
if wm.get("eig"):
    print("eig_last", wm["eig"])

sleep_ticks = sum(1 for t in ticks if (t.get("sleep") or {}).get("sleeping"))
print("sleep_ticks", sleep_ticks)

keywords = ["Phase", "Sleep", "Neuro", "fixed_root", "FALLEN", "fall", "Стабильно", "Open Reality", "VLM"]
found = defaultdict(int)
samples = defaultdict(list)
for t in ticks:
    for e in t.get("events") or []:
        txt = str(e.get("text", ""))
        for kw in keywords:
            if kw.lower() in txt.lower():
                found[kw] += 1
                if len(samples[kw]) < 3:
                    samples[kw].append((t["tick"], txt[:100]))
print("event_keywords", dict(found))
for kw, samp in samples.items():
    if samp:
        print(" sample", kw, samp)

print(
    "alpha first200/last200",
    statistics.mean([t["hud"]["alpha_mean"] for t in ticks[:200]]),
    statistics.mean([t["hud"]["alpha_mean"] for t in ticks[-200:]]),
)
print("edges first/last", ticks[0]["hud"]["edge_count"], ticks[-1]["hud"]["edge_count"])
print("phi last", ticks[-1]["hud"]["phi"], "hud phase", ticks[-1]["hud"].get("phase"))
