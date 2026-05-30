import json
from collections import Counter
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "rkk_run.jsonl"
MAX_TICK = 2100

rows = []
for line in LOG.open(encoding="utf-8"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("type") == "tick" and r["tick"] <= MAX_TICK:
        rows.append(r)

print(f"analyzed {len(rows)} ticks (1..{rows[-1]['tick']})")
for start in (1, 500, 1000, 1500, 2000):
    w = [x for x in rows if start <= x["tick"] < start + 500]
    if not w:
        continue
    n = len(w)
    fallen = sum(1 for x in w if (x.get("body") or {}).get("fallen"))
    s2e = sum(1 for x in w if (x.get("system2") or {}).get("error"))
    ov = sum(1 for x in w if (x.get("system2") or {}).get("fallen_override_active"))
    learned = sum(1 for x in w if (x.get("system2") or {}).get("learned_recovery_active"))
    mo = Counter(
        (x.get("system2") or {}).get("motor_owner")
        for x in w
        if (x.get("system2") or {}).get("motor_owner")
    )
    beh = [x.get("behavioral_score") or (x.get("hud") or {}).get("behavioral_score") for x in w]
    beh = [b for b in beh if b is not None]
    com = [
        (x.get("behavior") or {}).get("com_x_vel_ema")
        for x in w
        if (x.get("behavior") or {}).get("com_x_vel_ema") is not None
    ]
    ens = [
        (x.get("ensemble") or {}).get("max_weight")
        for x in w
        if (x.get("ensemble") or {}).get("max_weight") is not None
    ]
    edges = [x.get("hud", {}).get("edge_count") for x in w if x.get("hud", {}).get("edge_count")]
    print(
        f"{start}-{start+499}: fallen={100*fallen/n:.0f}% ov={100*ov/n:.0f}% "
        f"learned={100*learned/n:.0f}% s2_err={100*s2e/n:.0f}% motor={dict(mo)} "
        f"beh={round(sum(beh)/len(beh), 2) if beh else 'n/a'} "
        f"com_vel={round(sum(com)/len(com), 4) if com else 'n/a'} "
        f"ens={max(ens) if ens else 'n/a'} edges={edges[-1] if edges else 'n/a'}"
    )

for r in rows:
    s2 = r.get("system2") or {}
    if s2.get("fallen_override_active"):
        print("first override tick", r["tick"])
        print(
            " ",
            {k: s2.get(k) for k in [
                "fallen_override_active", "learned_recovery_active", "motor_owner",
                "source", "recovery_schedule_source", "s2_fallen_streak",
            ]},
        )
        break

# override exits
recovered = sum(1 for r in rows if (r.get("system2") or {}).get("override_recovered"))
max_reset = sum(1 for r in rows if (r.get("system2") or {}).get("override_max_reset"))
print(f"override_recovered ticks={recovered} override_max_reset={max_reset}")

sources = Counter(
    (r.get("system2") or {}).get("recovery_schedule_source")
    for r in rows
    if (r.get("system2") or {}).get("fallen_override_active")
)
print("recovery_schedule_source during override:", dict(sources))
