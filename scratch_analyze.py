import json
with open('logs/rkk_run.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        if d.get('type') == 'tick' and d.get('tick', 0) == 1495:
            print(json.dumps(d, indent=2))
            break
