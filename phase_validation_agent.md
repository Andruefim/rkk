# Phase Validation Agent — Инструкция

> **Для кого этот документ.** Ты — агент с доступом к браузеру (визуальное наблюдение за гуманоидом),
> терминалу (команды), файловой системе (логи, `.env`, JSON).
> Все todo реализованы. Твоя задача — **последовательно валидировать каждую фазу**
> через реальные прогоны и убедиться что числа в scorecard соответствуют порогам.
> Smoke-тесты (pytest) не являются подтверждением прохождения фазы.

---

## Модель тестирования (прочитай один раз)

```
pytest smoke gate     → код не падает, поля присутствуют       → НЕОБХОДИМО, но НЕ ДОСТАТОЧНО
behavioral gate       → числа из реального прогона > порогов   → ДОСТАТОЧНО для закрытия фазы
```

Behavioral gate = запуск `eval_transfer.py` → чтение `logs/autonomy_scorecard.json` → сравнение
с порогами из таблицы ниже. **Ты читаешь JSON глазами и сравниваешь числа.**

---

## Текущее состояние (перед началом работы)

Прочитай `logs/autonomy_scorecard.json`. Если файл существует — запиши значения в таблицу ниже.
Это базовая линия.

| Метрика | Текущее | Порог | Статус |
|---------|---------|-------|--------|
| `worlds.humanoid.script_override_frac_post_warmup` | ? | < 0.20 | ? |
| `worlds.humanoid.emergency_override_frac_post_warmup` | ? | < 0.15 | ? |
| `discovery_new_frac` | ? | > 0.60 | ? |
| `cross_env_success_rate_200` | ? | ≥ 0.40 | ? |
| `meta_prediction_error` | ? | < 0.15 | ? |
| `continual_forgetting_ratio` | ? | ≥ 0.50 | ? |
| `meta_recovery_ticks` | ? | ≤ 1000 | ? |

---

## Правила работы

1. **Фаза открывается только если предыдущая закрыта.** Не включай фичи следующей фазы пока
   текущая не дала зелёные числа.
2. **Одно изменение `.env` за раз.** Включаешь один master switch → запускаешь прогон → смотришь.
3. **Если числа ухудшились** — немедленно выключи последний включённый switch, запиши в лог.
4. **Визуальное наблюдение** — необходимо для фаз 0–2. Для фаз 3–6 — достаточно JSON.
5. **Не трогай код.** Только `.env`. Если нужно менять код — это не валидация, это баг.

---

## Команды которые ты будешь использовать

```bash
# Запуск прогона humanoid (основная команда)
python backend/tools/eval_transfer.py \
  --train-ticks 1500 \
  --eval-ticks 200 \
  --scorecard \
  --world humanoid

# Запуск с несколькими мирами (Phase 6b+)
python backend/tools/eval_transfer.py \
  --train-ticks 1500 \
  --scorecard \
  --worlds humanoid,grid_nav,symbolic_control

# Запуск только smoke тестов конкретной фазы
pytest backend/tests/integration/test_phase{N}_*.py -v -m "not slow"

# Посмотреть scorecard
cat logs/autonomy_scorecard.json

# Посмотреть последние строки transfer log
tail -20 logs/transfer_eval.jsonl

# Посмотреть что сейчас включено
grep -E "^RKK_.*=1" .env
```

---

## PHASE 0 — Валидация (Instrumentation)

### Статус: предположительно завершена (все todos completed)

### Шаг 0.1 — Проверить что Phase 0 features включены в .env

```bash
grep -E "RKK_POST_FR_ALPHA_DECAY|RKK_ADVANCE_EVAL_FALLEN_MAX|RKK_EVAL_MODE" .env
```

**Ожидаемый вывод:**
```
RKK_POST_FR_ALPHA_DECAY=0.40
RKK_ADVANCE_EVAL_FALLEN_MAX=0.35
RKK_ADVANCE_EVAL_QUALITY_MIN=0.30
RKK_EVAL_MODE=0
```

Если значений нет — добавь их в `.env` и продолжай.

### Шаг 0.2 — Запустить прогон

```bash
python backend/tools/eval_transfer.py --train-ticks 1500 --scorecard --world humanoid
```

### Шаг 0.3 — Визуальная проверка в браузере

Открой браузер с симуляцией. Наблюдай 200 тиков после запуска.

**Что смотреть:**
- Гуманоид стоит и балансирует без постоянного скриптового вмешательства
- После падения — поднимается сам (не через hard reset с телепортацией)
- В overlay/debug панели нет постоянного флага `s2_override=ACTIVE`

**Красные флаги (фаза не готова):**
- Гуманоид падает и не поднимается > 5 секунд
- Постоянное мигание `fallen_override` в логах
- Телепортация в исходную позицию чаще чем раз в 30 секунд

### Шаг 0.4 — Проверить scorecard

```bash
cat logs/autonomy_scorecard.json
```

**Критерии прохождения Phase 0:**

| Поле | Порог | Действие если не выполнено |
|------|-------|---------------------------|
| `worlds.humanoid.a1_pass` | `true` | Проверить `RKK_ADVANCE_EVAL_QUALITY_MIN`, уменьшить до 0.20 |
| `worlds.humanoid.a4_pass` | `true` | Проверить `RKK_POST_FR_ALPHA_DECAY`, увеличить до 0.50 |
| Файл `logs/transfer_eval.jsonl` существует | да | Проверить `RKK_TRANSFER_EVAL_LOG` в `.env` |

### ✅ Phase 0 CLOSED если: a1_pass=true, a4_pass=true, JSONL файл существует.

---

## PHASE 1 — Валидация (Same-topology transfer)

### Предусловие: Phase 0 closed.

### Шаг 1.1 — Активировать Phase 1 features

В `.env` должны быть включены:
```
RKK_ROLE_TYPE_ENABLED=1
RKK_ROLE_TYPE_STRICT=1
RKK_GENOME_RANK=8
RKK_CROSS_ENV_ALLOW_WM_TRAIN=0
```

### Шаг 1.2 — Запустить cross-env benchmark

```bash
python backend/tools/eval_transfer.py \
  --train-ticks 1500 \
  --eval-ticks 200 \
  --benchmark cross_env_same_topology \
  --scorecard \
  --world humanoid
```

### Шаг 1.3 — Проверить JSONL

```bash
tail -5 logs/transfer_eval.jsonl | python -m json.tool
```

**Ожидаемые поля в последней строке:**
```json
{
  "cross_env_success_rate_200": 0.43,
  "eval_kind": "cross_env_same_topology"
}
```

### Критерии прохождения Phase 1:

| Поле | Порог | Действие если не выполнено |
|------|-------|---------------------------|
| `cross_env_success_rate_200` | ≥ 0.40 | Увеличить `--train-ticks` до 2000, повторить |
| `ticks_to_success_0_5` присутствует | да | Smoke issue: `pytest test_phase1_*.py -v` |

### ✅ Phase 1 CLOSED если: `cross_env_success_rate_200 ≥ 0.40`.

---

## PHASE 2 — Валидация (pass_core_embodied) ← КРИТИЧЕСКАЯ ФАЗА

### Предусловие: Phase 1 closed.

> Phase 2 — единственная точка где A1/A4 для humanoid **замораживаются**.
> Числа которые ты получишь здесь будут эталоном навсегда.
> После прохождения НЕ ПЕРЕСЧИТЫВАТЬ.

### Шаг 2.1 — Активировать Phase 2 features (по одному)

**Сначала только C2 (bridge loss):**
```bash
# Добавить в .env:
RKK_WORLD_BRIDGE_ENABLED=1
RKK_BRIDGE_LOSS_WEIGHT=0.20
RKK_BRIDGE_LOSS_EVERY=1
```

Запустить прогон 500 тиков и **визуально проверить** что гуманоид не стал хуже:
```bash
python backend/tools/eval_transfer.py --train-ticks 500 --world humanoid
```

Если за 500 тиков падений не стало больше — включить C3:
```bash
RKK_STRUCTURE_LEARN_EVERY=50
RKK_VSTRUCTURE_ENSEMBLE_N=4
RKK_LOG_DISCOVERY_SPLIT=1
```

### Шаг 2.2 — Финальный прогон Phase 2

```bash
python backend/tools/eval_transfer.py \
  --train-ticks 1500 \
  --eval-ticks 200 \
  --scorecard \
  --world humanoid
```

### Шаг 2.3 — Визуальная проверка

Наблюдай в браузере после тика 800 (post-warmup window):

**Что должно быть:**
- Гуманоид стоит, ходит, балансирует
- S2 override НЕ мигает постоянно в debug overlay
- `fallen_override` НЕ активен > 15% времени

**Тест на падение:** принудительно "толкни" гуманоида (если есть такая функция в UI).
Он должен подняться самостоятельно без телепортации в течение ~50 тиков.

### Шаг 2.4 — Проверить и заморозить числа

```bash
cat logs/autonomy_scorecard.json
```

**BLOCKING критерии Phase 2:**

| Поле JSON | Порог | Результат |
|-----------|-------|-----------|
| `worlds.humanoid.script_override_frac_post_warmup` | **< 0.20** | запиши значение |
| `worlds.humanoid.emergency_override_frac_post_warmup` | **< 0.15** | запиши значение |
| `discovery_new_frac` | **> 0.60** | запиши значение |
| `worlds.humanoid.a1_pass` | `true` | — |
| `worlds.humanoid.a4_pass` | `true` | — |
| `pass_core_embodied` | `true` | — |

### Contingency paths (что делать при провале):

| Что провалилось | Что делать |
|-----------------|------------|
| A1 miss (script_override > 0.20) | Проверить `RKK_EVAL_MODE=0`; уменьшить `RKK_SKILL_CHAIN_PE_MAX` до 0.20 |
| A4 miss (emergency > 0.15) | Увеличить `RKK_POST_FR_ALPHA_DECAY` до 0.50; проверить probe подключён |
| #3 miss (discovery < 0.60) | Включить `RKK_LOG_DISCOVERY_SPLIT=1`; проверить `edge_age_at_activation` в snapshot |
| L_bridge не сходится | Выключить `RKK_WORLD_BRIDGE_ENABLED=0`, идти дальше без него |
| Crash / JSONL пустой | СТОП. Запустить `pytest test_phase2_*.py -v` найти smoke issue |

### ✅ Phase 2 CLOSED если: a1_pass=true, a4_pass=true, discovery_new_frac > 0.60.
### Запиши финальные числа — они будут сравниваться в Phase 6.

---

## PHASE 3 — Валидация (Latents + WorldAutonomyContract)

### Предусловие: Phase 2 closed + числа заморожены.

### Шаг 3.1 — Активировать C4 (только)

```bash
# Добавить в .env:
RKK_C4_ENABLED=1
RKK_LATENT_RESIDUAL_THRESH=0.30
RKK_LATENT_TTL_TICKS=500
RKK_LATENT_MIN_IG=0.05
```

Запустить 1000 тиков:
```bash
python backend/tools/eval_transfer.py --train-ticks 1000 --world humanoid
```

### Шаг 3.2 — Проверить что C4 работает (не что он помогает)

```bash
cat logs/transfer_eval.jsonl | python -c "
import sys, json
lines = [json.loads(l) for l in sys.stdin if l.strip()]
last = lines[-1] if lines else {}
print('c4_active:', last.get('c4_active', 'NOT FOUND'))
print('latent_injections:', last.get('latent_injections', 'NOT FOUND'))
print('latent_nodes_alive:', last.get('latent_nodes_alive', 'NOT FOUND'))
"
```

**Ожидаемый вывод (smoke — не benchmark):**
```
c4_active: True
latent_injections: N (любое >= 0)
latent_nodes_alive: N (может быть 0 если среда не требует латентов)
```

Если `c4_active: NOT FOUND` — smoke issue. Запусти `pytest test_phase3_*.py -v`.

### Шаг 3.3 — Активировать C5

```bash
RKK_C5_ENABLED=1
RKK_PROMOTE_MIN_WORLDS=2
```

### Шаг 3.4 — Проверить WorldAutonomyContract

```bash
python -c "
import json
with open('logs/autonomy_scorecard.json') as f:
    sc = json.load(f)
worlds = sc.get('worlds', {})
print('Registered worlds:', list(worlds.keys()))
for w in ['humanoid', 'cartpole', 'grid_nav']:
    print(f'{w}: metrics_applicable =', worlds.get(w, {}).get('metrics_applicable', 'NOT REGISTERED'))
"
```

**Ожидаемый вывод:**
```
Registered worlds: ['humanoid', 'cartpole', 'grid_nav']
humanoid: metrics_applicable = True
cartpole: metrics_applicable = True
grid_nav: metrics_applicable = True
```

### Критерии прохождения Phase 3:

| Проверка | Порог | xfail OK? |
|----------|-------|-----------|
| `c4_active` в логах | True | Нет — блокирует |
| WorldAutonomyContract зарегистрирован для ≥3 worlds | да | Нет — блокирует |
| `learned_roles_count` ≥ 0 (не ошибка) | поле присутствует | Нет |
| `mean residual ↓` после latent @500 ticks | качественно | **Да** — xfail OK |
| A1/A4 humanoid НЕ ухудшились vs Phase 2 frozen | drift < 0.02 | Нет — блокирует |

**Важно:** Сравни `script_override_frac` сейчас с замороженным значением Phase 2.
Если разница > 0.02 — C4 дестабилизирует W. Выключи `RKK_C4_ENABLED=0`.

### ✅ Phase 3 CLOSED если: C4 active, contracts registered, humanoid A1/A4 не ухудшились.

---

## PHASE 4 — Валидация (Spectral + Skeleton)

### Предусловие: Phase 3 closed.

### Шаг 4.1 — Активировать B4

```bash
RKK_SPECTRAL_TRANSFER_ENABLED=1
RKK_SPECTRAL_K=8
RKK_SPECTRAL_ALIGN_THRESH=0.55
```

### Шаг 4.2 — Запустить cross-topology benchmark

```bash
python backend/tools/eval_transfer.py \
  --benchmark cross_topology_spectral \
  --src humanoid \
  --dst cartpole \
  --eval-ticks 200
```

```bash
tail -3 logs/transfer_eval.jsonl | python -m json.tool
```

Ищи поля: `cross_topology_spectral_success_200`, baseline `random_init_success_200`.

### Шаг 4.3 — Активировать C6 + E

```bash
RKK_C6_ENABLED=1
RKK_ROLE_DISCOVERY_THRESH=0.65
RKK_SKELETON_TRANSFER_ENABLED=1
RKK_SKELETON_CMI_THRESH=0.12
```

```bash
python backend/tools/eval_transfer.py \
  --benchmark skeleton_transfer \
  --dst cartpole \
  --eval-ticks 200
```

### Критерии Phase 4:

| Метрика | Порог | xfail OK? |
|---------|-------|-----------|
| `cross_topology_spectral_success_200` | ≥ 2× random OR ≥ 40% | **Да** |
| `skeleton_transfer_success_200` | ≥ 30% OR ≥ 1.5× random | **Да** |
| JSONL поля присутствуют | да | Нет — блокирует |
| cartpole: ≥1 node с `learned_role` (C6) | да | Нет — блокирует |

> Оба benchmark — xfail OK. Phase 5 открывается даже если числа не достигли порога,
> если поля **присутствуют** в JSONL.

### ✅ Phase 4 CLOSED если: JSONL поля присутствуют + cartpole role discovery работает (smoke).

---

## PHASE 5 — Валидация (pass_agi_extended)

### Предусловие: Phase 4 closed.

### Шаг 5.1 — Активировать F (W_meta)

```bash
RKK_META_CAUSAL_ENABLED=1
RKK_META_UPDATE_EVERY=50
RKK_META_DO_SAFE=1
```

Запустить 1000 тиков и проверить:
```bash
python backend/tools/eval_transfer.py --train-ticks 1000 --world humanoid

python -c "
import json
lines = open('logs/transfer_eval.jsonl').readlines()
last = json.loads(lines[-1])
print('meta_prediction_error:', last.get('meta_prediction_error', 'NOT FOUND'))
"
```

`meta_prediction_error` должен быть числом (не `NOT FOUND`). Значение сейчас не важно — важно что поле есть.

### Шаг 5.2 — Активировать G (GoalGenerator)

```bash
RKK_GOAL_GEN_ENABLED=1
RKK_GOAL_PROPOSE_EVERY=200
RKK_GOAL_WMETA_MIN_SUCCESS=0.30
RKK_GOAL_DIVERSITY_WINDOW=10
RKK_GOAL_COOLDOWN_MAX=3
RKK_GOAL_SATURATION_FRAC=0.50
```

```bash
RKK_CURRICULUM_GRAPH_ENABLED=1
RKK_CURRICULUM_GRAPH_HUMAN_SEED=1
```

### Шаг 5.3 — Финальный прогон Phase 5

```bash
python backend/tools/eval_transfer.py \
  --train-ticks 2000 \
  --scorecard \
  --world humanoid
```

### Шаг 5.4 — Визуальная проверка в браузере

Наблюдай за debug/event overlay. Должны появляться события:
- `GoalGenerator: proposed goal [X]` — GoalGenerator предложил субцель
- `CurriculumGraph: node completed source=generated` — выполнена автогенерированная задача
- НЕТ постоянного `goal_gen_blocked` — если есть, saturation guard срабатывает слишком агрессивно

Если `goal_gen_blocked` мигает > 50% времени → увеличь `RKK_GOAL_COOLDOWN_MAX` до 5.

### Критерии Phase 5 (BLOCKING):

| Метрика JSON | Порог |
|--------------|-------|
| `meta_prediction_error` | **< 0.15** rolling 500 ticks |
| `autonomous_goals_crossworld_pass` | **≥ 3 goals**, SR ≥ 0.4, ≥ 2 worlds |
| `pass_agi_extended` | **true** |
| `pass_core_embodied` (frozen echo) | **true** (не изменился) |

### Шаг 5.5 — Проверить saturation guard

```bash
python -c "
import json
lines = open('logs/transfer_eval.jsonl').readlines()[-50:]
blocked = sum(1 for l in lines if 'goal_gen_blocked' in l)
print(f'goal_gen_blocked events in last 50 entries: {blocked}')
print('OK' if blocked < 10 else 'WARNING: saturation guard too aggressive')
"
```

### ✅ Phase 5 CLOSED если: meta_prediction_error < 0.15, autonomous_goals ≥ 3, pass_agi_extended=true.

---

## PHASE 6a — Валидация (Non-physical domains)

### Предусловие: Phase 5 closed.

### Шаг 6a.1 — Активировать H stubs

```bash
RKK_H_GRID_NAV_ENABLED=1
RKK_H_SYMBOLIC_ENABLED=1
```

### Шаг 6a.2 — Проверить что stubs загружаются

```bash
python -c "
from backend.engine.core.world import WORLDS
print('grid_nav registered:', 'grid_nav' in WORLDS)
print('symbolic_control registered:', 'symbolic_control' in WORLDS)
"
```

### Шаг 6a.3 — Запустить skeleton transfer на grid_nav

```bash
RKK_SYMBOLIC_GROUNDING_ENABLED=1

python backend/tools/eval_transfer.py \
  --benchmark skeleton_nonphys \
  --dst grid_nav \
  --eval-ticks 500
```

### Критерии Phase 6a (smoke — нет blocking benchmark):

| Проверка | Порог |
|----------|-------|
| `grid_nav` и `symbolic_control` загружаются без crash | да |
| `skeleton_nonphys_success_500` поле присутствует в JSONL | да |
| SymbolicGrounding: ≥1 правило с CMI > 0.12 в логах | да |
| A1/A4 humanoid НЕ ухудшились | drift < 0.02 |
| `#8 skeleton_nonphys_success_500 ≥ 1.5× random` | **xfail OK** |

### ✅ Phase 6a CLOSED если: stubs работают без crash, JSONL поля присутствуют.

---

## PHASE 6b — Валидация (Continual + A1/A4 non-phys) ← BLOCKING

### Предусловие: Phase 6a closed.

### Шаг 6b.1 — Активировать I1 (EWC)

```bash
RKK_EWC_ENABLED=1
RKK_EWC_LAMBDA=1000
RKK_EWC_ROLES_ONLY=1
RKK_EWC_STABLE_AGE_MIN=200
RKK_EWC_GRAPH_CHANGE_THRESH=0.20
```

### Шаг 6b.2 — Активировать I2 (CausalHealthMonitor)

```bash
RKK_HEALTH_MONITOR_ENABLED=1
RKK_HEALTH_CHECK_EVERY=100
RKK_HEALTH_DISCOVERY_MIN=0.40
RKK_HEALTH_REPAIR_DRY_RUN=1
```

### Шаг 6b.3 — 3-world continual run

```bash
python backend/tools/eval_transfer.py \
  --train-ticks 1500 \
  --worlds humanoid,humanoid_variant,grid_nav \
  --continual \
  --scorecard
```

После прогона:
```bash
python -c "
import json
sc = json.load(open('logs/autonomy_scorecard.json'))
print('continual_forgetting_ratio:', sc.get('continual_forgetting_ratio', 'NOT FOUND'))
print('ewc_stable_edge_count:', sc.get('ewc_stable_edge_count', 'NOT FOUND'))
"
```

### Шаг 6b.4 — Проверить A1/A4 на non-phys worlds

```bash
python backend/tools/eval_transfer.py \
  --train-ticks 1500 \
  --scorecard \
  --worlds grid_nav,symbolic_control,humanoid
```

```bash
python -c "
import json
sc = json.load(open('logs/autonomy_scorecard.json'))
for world in ['grid_nav', 'symbolic_control']:
    w = sc['worlds'].get(world, {})
    print(f'{world}:')
    print(f'  script_override_frac: {w.get(\"script_override_frac_post_warmup\", \"N/A\")} (need < 0.20)')
    print(f'  emergency_override_frac: {w.get(\"emergency_override_frac_post_warmup\", \"N/A\")} (need < 0.15)')
    print(f'  a1_pass: {w.get(\"a1_pass\", \"N/A\")}')
    print(f'  a4_pass: {w.get(\"a4_pass\", \"N/A\")}')
"
```

### Шаг 6b.5 — Тест намеренной деградации (I2 smoke)

Если есть CLI флаг для деградации:
```bash
python backend/tools/eval_transfer.py \
  --train-ticks 500 \
  --inject-degradation eig_off \
  --world humanoid
```

Иначе — проверить логи после 6b.3 прогона:
```bash
grep "CausalHealthMonitor" logs/transfer_eval.jsonl | tail -5
```

Должно быть ≥1 строки с `health_event` или `repair_suggested`.

### BLOCKING критерии Phase 6b:

| Метрика | Порог |
|---------|-------|
| `continual_forgetting_ratio` | **≥ 0.50** |
| `worlds.grid_nav.a1_pass` | **true** |
| `worlds.grid_nav.a4_pass` | **true** |
| `worlds.symbolic_control.a1_pass` | **true** |
| `worlds.symbolic_control.a4_pass` | **true** |
| `ewc_stable_edge_count` поле присутствует | да |

### Если A1/A4 не проходят на non-phys worlds:

1. Проверить что `WorldAutonomyContract` зарегистрирован (Phase 3).
2. Проверить что probes (`stuck_override_active`, `constraint_violation_override`) действительно
   пишутся в snapshot при реальных событиях в grid_nav/symbolic_control.
3. Запустить более длинный прогон (`--train-ticks 3000`) — non-phys мирам нужно больше тиков.

### ✅ Phase 6b CLOSED если: continual_forgetting_ratio ≥ 0.50, A1/A4 pass на grid_nav и symbolic_control.

---

## PHASE 6c — Валидация (MetaCircuitBreaker + pass_agi_full)

### Предусловие: Phase 6b closed.

### Шаг 6c.1 — Активировать I3

```bash
RKK_META_CB_PE_OPEN=0.25
RKK_META_CB_PE_CLOSE=0.12
RKK_META_CB_AGE_OPEN=2000
RKK_META_CB_RESET_AFTER=500
```

### Шаг 6c.2 — Проверить state transitions

```bash
python -c "
# Тест переходов MetaCircuitBreaker напрямую
from backend.engine.meta_causal import MetaCircuitBreaker
import os
os.environ['RKK_META_CB_PE_OPEN'] = '0.25'
os.environ['RKK_META_CB_PE_CLOSE'] = '0.12'
os.environ['RKK_META_CB_RESET_AFTER'] = '500'

cb = MetaCircuitBreaker()
print('Initial state:', cb.state)  # closed

# Trigger OPEN
for i in range(5):
    cb.observe(meta_pe=0.30, meta_age=100, tick=i)
print('After PE spike:', cb.state)  # open
print('wmeta_active:', cb.wmeta_active)  # False

# Wait for HALF_OPEN
for i in range(500):
    cb.observe(meta_pe=0.30, meta_age=100, tick=i+10)
print('After timeout:', cb.state)  # half_open

# Stabilize → CLOSED
for i in range(10):
    cb.observe(meta_pe=0.10, meta_age=100, tick=i+520)
print('After stabilize:', cb.state)  # closed
print('Recovery ticks:', cb.recovery_ticks(530))
"
```

**Ожидаемый вывод:**
```
Initial state: closed
After PE spike: open
wmeta_active: False
After timeout: half_open
After stabilize: closed
Recovery ticks: N (< 1000)
```

### Шаг 6c.3 — Финальный прогон pass_agi_full

```bash
python backend/tools/eval_transfer.py \
  --train-ticks 2000 \
  --scorecard \
  --worlds humanoid,grid_nav,symbolic_control
```

### Шаг 6c.4 — Проверить финальный scorecard

```bash
python -c "
import json
sc = json.load(open('logs/autonomy_scorecard.json'))

checks = [
    ('pass_agi_full', sc.get('pass_agi_full'), True),
    ('pass_agi_extended', sc.get('pass_agi_extended'), True),
    ('pass_core_embodied', sc.get('pass_core_embodied'), True),
    ('autonomy_integrity_nonphys', sc.get('autonomy_integrity_nonphys'), True),
    ('meta_recovery_ticks', sc.get('meta_recovery_ticks', 9999), '≤ 1000'),
    ('continual_forgetting_ratio', sc.get('continual_forgetting_ratio', 0), '≥ 0.50'),
    ('meta_prediction_error', sc.get('meta_prediction_error', 1.0), '< 0.15'),
    ('discovery_new_frac', sc.get('discovery_new_frac', 0), '> 0.60'),
]

print('=== FINAL SCORECARD ===')
all_pass = True
for name, val, expected in checks:
    if isinstance(expected, bool):
        ok = val == expected
    elif str(expected).startswith('≤'):
        ok = val is not None and val <= int(str(expected)[1:].strip())
    elif str(expected).startswith('≥'):
        ok = val is not None and val >= float(str(expected)[1:].strip())
    elif str(expected).startswith('<'):
        ok = val is not None and val < float(str(expected)[1:].strip())
    elif str(expected).startswith('>'):
        ok = val is not None and val > float(str(expected)[1:].strip())
    else:
        ok = val == expected
    status = '✅' if ok else '❌'
    if not ok:
        all_pass = False
    print(f'{status} {name}: {val} (need {expected})')

print()
print('=== RESULT:', '✅ PASS_AGI_FULL' if all_pass else '❌ NOT YET' , '===')
"
```

### BLOCKING критерии Phase 6c (финальные):

| Метрика | Порог |
|---------|-------|
| `meta_recovery_ticks` | **≤ 1000** |
| `pass_agi_full` | **true** |
| `autonomy_integrity_nonphys` | **true** |
| `pass_agi_extended` (frozen echo) | **true** |
| `pass_core_embodied` (frozen echo) | **true** |

### ✅ Phase 6c CLOSED если: pass_agi_full=true.

---

## Итоговый чеклист (быстрая сводка)

| Phase | Ключевая команда | Ключевой порог | Заморозить |
|-------|-----------------|----------------|-----------|
| 0 | `eval_transfer --scorecard --world humanoid` | a1_pass, a4_pass | нет |
| 1 | `eval_transfer --benchmark cross_env_same_topology` | cross_env_success ≥ 0.40 | нет |
| **2** | `eval_transfer --train-ticks 1500 --scorecard` | **A1<0.20, A4<0.15, disc>0.60** | **ДА — заморозить числа** |
| 3 | проверить C4 active + contracts registered | humanoid A1/A4 drift < 0.02 | нет |
| 4 | `eval_transfer --benchmark cross_topology_spectral` | поля присутствуют (xfail OK) | нет |
| 5 | `eval_transfer --train-ticks 2000 --scorecard` | meta_pe<0.15, goals≥3 | нет |
| 6a | проверить stubs load + JSONL поля | crash-free (xfail OK) | нет |
| 6b | `eval_transfer --worlds humanoid,... --continual` | **forgetting≥0.50, A1/A4 non-phys** | нет |
| **6c** | `eval_transfer --worlds all --scorecard` | **pass_agi_full=true** | — |

---

## Действия при регрессии

Если после включения фичи следующей фазы **A1 или A4 humanoid ухудшились на > 0.02** от замороженного значения:

1. Выключи последний включённый master switch.
2. Запусти `eval_transfer --train-ticks 1500 --scorecard --world humanoid`.
3. Убедись что числа вернулись к замороженным ± 0.01.
4. Запиши в лог: какой switch вызвал регрессию + значения до/после.
5. Не переходи к следующей фазе пока не разберёшься.

---

## Файлы которые ты читаешь

| Файл | Когда читать |
|------|-------------|
| `logs/autonomy_scorecard.json` | после каждого `eval_transfer --scorecard` |
| `logs/transfer_eval.jsonl` | tail -10 для проверки последних метрик |
| `logs/research_gate.json` | если xfail benchmark — проверить что записалось |
| `.env` | перед каждой фазой — убедиться что включено только нужное |
| `logs/eval_gate_result.json` | для дебага curriculum advance gate |
