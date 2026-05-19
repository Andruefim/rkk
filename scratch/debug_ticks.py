import json

with open('logs/rkk_run.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        if d.get('type') == 'tick':
            t = d.get('tick', 0)
            if 680 <= t <= 780:
                s2 = d.get('system2', {})
                print(f"tick={t} s2={json.dumps(s2, ensure_ascii=False)}")
