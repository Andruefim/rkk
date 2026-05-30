import json
import statistics
from collections import Counter
from pathlib import Path

lines = [json.loads(l) for l in Path("logs/rkk_run.jsonl").open(encoding="utf-8") if l.strip()]
ticks = [l for l in lines if l.get("type") == "tick"]
print("TOTAL", len(ticks), "last", ticks[-1]["tick"])

# Compare segments
for label, lo, hi in [
    ("phase1_crisis", 2600, 2800),
    ("recovery", 2800, 3500),
    ("mid_plateau", 3500, 4500),
    ("final", 4500, 5020),
]:
    seg = [t for t in ticks if lo <= t["tick"] <= hi]
    if not seg:
        continue
    fallen = sum(1 for t in seg if t.get("body", {}).get("fallen"))
    ps = [t.get("body", {}).get("posture_stability") for t in seg if t.get("body", {}).get("posture_stability") is not None]
    dr = [t["hud"]["discovery_rate"] for t in seg]
    sc = [(t.get("system2") or {}).get("student_conf") for t in seg if (t.get("system2") or {}).get("student_conf") is not None]
    ema = [(t.get("system2") or {}).get("outcome_ema") for t in seg if (t.get("system2") or {}).get("outcome_ema") is not None]
    lr = [(t.get("locomotion") or {}).get("reward_ema") for t in seg if (t.get("locomotion") or {}).get("reward_ema") is not None]
    ec = [t["hud"]["edge_count"] for t in seg]
    macros = Counter(str((t.get("system2") or {}).get("macro", "?")) for t in seg)
    fr = seg[-1].get("body", {}).get("fixed_root")
    print(
        f"\n{label} [{lo}-{hi}] n={len(seg)} fallen={fallen}({100*fallen/len(seg):.0f}%) "
        f"posture={statistics.mean(ps):.3f} disc={statistics.mean(dr):.2f} loco={statistics.mean(lr):.3f} "
        f"edges={statistics.mean(ec):.0f} s2conf={statistics.mean(sc) if sc else 0:.2f} s2ema={statistics.mean(ema) if ema else 0:.2f} "
        f"fixed_root_end={fr} macros={dict(macros.most_common(3))}"
    )

last = ticks[-1]
print("\nFINAL STATE")
print("  fallen", last.get("body", {}).get("fallen"), "fall_count", last.get("body", {}).get("fall_count"))
print("  posture", last.get("body", {}).get("posture_stability"))
print("  phi", last["hud"]["phi"], "discovery", last["hud"]["discovery_rate"], "alpha", last["hud"]["alpha_mean"])
print("  edges", last["hud"]["edge_count"], "gnn_d", last["hud"]["gnn_d"])
print("  s2", {k: (last.get("system2") or {}).get(k) for k in ["macro", "idle", "outcome_ema", "student_conf", "fallen_override_active", "distill_success_rate"]})
print("  scope", last.get("scope"))
print("  action", last.get("action", {}).get("variable"), last.get("action", {}).get("value"))

# Fall timeline after 2725
post = [t for t in ticks if t["tick"] >= 2725]
fall_runs = []
st = None
for t in post:
    f = bool(t.get("body", {}).get("fallen"))
    if f and st is None:
        st = t["tick"]
    elif not f and st is not None:
        fall_runs.append((st, t["tick"] - 1, t["tick"] - st))
        st = None
if st is not None:
    fall_runs.append((st, post[-1]["tick"], post[-1]["tick"] - st + 1))
print("\nFALL EPISODES after 2725:", fall_runs)
