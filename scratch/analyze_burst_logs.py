import json
from pathlib import Path

p = Path(r"c:\Users\Andrey\Desktop\agi\rkk\backend\logs\task_log.jsonl")
rows = []
with p.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(r.get("tick", -1)) >= 1400:
            rows.append(r)

print("events from tick>=1400:", len(rows))
progs = [r for r in rows if r.get("event") == "task_progress"]
print("task_progress count", len(progs))
if progs:
    a, b = progs[0], progs[-1]
    for label, r in (("first", a), ("last", b)):
        print(
            label,
            r.get("tick"),
            "oracle",
            r.get("oracle_dist_m"),
            "vision",
            r.get("vision_range_m"),
            "heading",
            r.get("task_heading_err"),
            "nav",
            r.get("task_nav_active"),
            "brg",
            r.get("vision_bearing"),
        )

print("--- last 12 progress ---")
for r in progs[-12:]:
    print(
        f"t={r.get('tick')} nav={r.get('task_nav_active')} he={r.get('task_heading_err')} "
        f"brg={r.get('vision_bearing')} vr={r.get('vision_range_m')} od={r.get('oracle_dist_m')} "
        f"cv={r.get('closing_vel')} hl={r.get('hard_lock')}"
    )

corr = [
    r
    for r in rows
    if r.get("event") in ("vision_range_correct", "vision_rebind", "com_teleport", "nav_hold")
]
print("--- correct/rebind/teleport ---")
for r in corr[-20:]:
    keys = (
        "tick",
        "event",
        "reason",
        "vision_range_m",
        "oracle_dist_m",
        "corrected_range_m",
        "jump_m",
        "new_range_m",
    )
    print({k: r.get(k) for k in keys if r.get(k) is not None})
