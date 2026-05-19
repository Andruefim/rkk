import json

last_state = None
with open('logs/rkk_run.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        if d.get('type') == 'tick':
            t = d.get('tick', 0)
            s2 = d.get('system2', {})
            body = d.get('body', {})
            fallen = body.get('fallen')
            s2_source = s2.get('source')
            s2_macro = s2.get('macro')
            rec_source = s2.get('recovery_schedule_source')
            current_state = (fallen, s2_source, s2_macro, rec_source)
            if current_state != last_state:
                print(f"tick={t:4d} fallen={fallen} s2_source={s2_source} s2_macro={s2_macro} rec_source={rec_source} llm_inflight={s2.get('llm_inflight')} s2_wait={s2.get('llm_wait_ticks')}")
                last_state = current_state
