#!/usr/bin/env python3
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/api/snapshot"
N = int(__import__("os").environ.get("POLL_N", "8"))
INTERVAL = float(__import__("os").environ.get("POLL_INTERVAL", "12"))


def fetch():
    with urllib.request.urlopen(URL, timeout=12) as resp:
        return json.load(resp)


for i in range(N):
    try:
        r = fetch()
        a = (r.get("agents") or [{}])[0]
        b = r.get("behavioral") or {}
        s2 = r.get("system2") or {}
        wm = ((r.get("wm") or {}).get("ensemble") or {})
        w = wm.get("weights") or []
        wmax = max(float(x) for x in w) if w else None
        print(
            f"[{i+1}/{N}] tick={r.get('tick')} fallen={r.get('fallen')} step={r.get('curriculum_step')} "
            f"dr={a.get('discovery_rate')} edges={a.get('edge_count')} beh={b.get('behavioral_score')} "
            f"com={b.get('com_x_vel_ema')} s2ov={s2.get('fallen_override_active')} "
            f"rec={s2.get('override_recovered')} src={s2.get('recovery_schedule_source')} wmax={wmax}"
        )
    except Exception as exc:
        print(f"[{i+1}/{N}] error: {exc}")
    if i + 1 < N:
        time.sleep(INTERVAL)
