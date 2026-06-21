# RKK: Честный разбор архитектуры для AGI в теле гуманоида

> Написано после полного просмотра кода. Без дипломатии — только то, что реально мешает цели.

---

## Что есть и работает (сильные стороны)

| Компонент | Файл | Статус |
|-----------|------|--------|
| Каузальный GNN (NOTEARS + message passing) | `causal_graph.py` | ✅ есть |
| Активное выведение (HAI + precision channels) | `hierarchical_active_inference.py` | ✅ есть |
| System 2 — макро-планирование + WM beam search | `system2/controller.py` + `wm_planner.py` | ✅ есть |
| Нейро-символьный мост (fuzzy predicates → STRIPS) | `neuro_symbolic/` | ✅ есть (только что добавлен) |
| Символьный движок (SafetyAxiom, veto, Lukasiewicz) | `neuro_symbolic/engine.py` | ✅ есть |
| Эпизодическая память + Sleep-фаза | `episodic_memory.py` + `sleep_consolidation.py` | ✅ есть |
| Геном (врождённые приоры, spectral transfer) | `genome/` | ✅ есть |
| CPG + моторный кортекс | `cpg_locomotion.py`, `motor_cortex.py` | ✅ есть |
| JEPA/sequence WM training | `causal_graph.py` (Phase N) | ✅ есть |
| Нейрогенез (RSI) | `rsi_structural.py`, `neurogenesis_coordinator.py` | ✅ есть |

Архитектура **очень богатая**. Гораздо богаче большинства академических систем.

---

## Критические пробелы (то, чего нет или сломано)

### ❌ 1. STRIPS-плanner не учится — это хардкод на 8 действий

**Файл**: [`neuro_symbolic/planner.py`](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/neuro_symbolic/planner.py)

`HUMANOID_ACTIONS` — это список из 8 вручную написанных действий (`RecoverPosture`, `StepForward`, `Turn`...).
Плюс `plan_to_goal` — жадный BFS глубиной 4.

**Проблема**: Это не планирование — это lookup table. AGI в теле должен **открывать** новые действия из опыта, а не работать с заранее перечисленными примитивами. Когда гуманоид встретит ситуацию вне этих 8 действий — плаунер ничего не предложит.

**Что нужно**: Learned symbolic vocabulary — предикаты и операторы должны возникать из каузального графа через регуляризацию, а не быть прошиты руками.

---

### ❌ 2. Символьный уровень — только вето, не генерация

**Файл**: [`neuro_symbolic/engine.py`](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/neuro_symbolic/engine.py)

`SymbolicCognitiveEngine` умеет только: **заблокировать** действие (hard/soft veto). Он **не умеет** предлагать альтернативы, рассуждать, объяснять, или модифицировать цели.

Нынешняя System 2 — это не мышление, это firewall. Для AGI System 2 должна:
- Формулировать гипотезы о мире
- Проводить мысленные эксперименты
- Переписывать цели из первых принципов

**Сейчас**: `veto_prediction()` → allow/block + penalty. Всё.

---

### ❌ 3. Fuzzy predicates → символы — односторонний path (только снизу вверх)

**Файл**: [`neuro_symbolic/bridge.py`](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/neuro_symbolic/bridge.py#L266-L278)

`priors_for_active_inference()` работает так: `obs → ProbabilisticState → motor_priors`.
Это "восходящий" канал: нейронное → символьное → моторное.

**Отсутствует нисходящий канал**: символьный план должен менять **приоры убеждений** (belief priors) в каузальном графе. `apply_priors_to_graph()` существует, но только 8 intent-переменных с blend=0.38 — это не настоящая нисходящая причинность. Символы не управляют вниманием, не задают, что "предсказывать", не задают precision весов.

В реальном мозге: cortex → thalamus → cortex. Символьные цели меняют то, **что** сенсорная система воспринимает. У вас этого нет.

---

### ❌ 4. Нет рабочего обучения System 2 из опыта (RL для планировщика)

**Файл**: [`system2/controller.py`](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/system2/controller.py), [`system2/learned_student.py`](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/system2/learned_student.py)

`LearnedMacroStudent` существует, но из диагностического дока (`agi_issues_diagnosis.md`) видно: **Recovery = scripted в 100% случаев**, S2 RECOVER_POSTURE не обучен. Это значит System 2 фактически работает как скриптовый контроллер, а не как обученный "медленный ум".

`MacroStudent.choose_macro_from_obs()` — выбор макроса, но сигнал обучения (gradient) для самого планировщика не замкнут. WM-beam-search дает действие, но потери планировщика не propagate назад.

---

### ❌ 5. Отсутствует семантическая рабочая память (Working Memory)

AGI требует способности удерживать **контекст** через десятки секунд — "что я делал 30 секунд назад", "какая цель ещё не достигнута", "что я обещал сделать".

**Есть**: `episodic_memory.py` — хранит эпизоды падений и успехов (структурированные записи).
**Нет**: Unbounded working memory buffer, который System 2 может **читать и писать** во время решения задачи. `EpisodicMemory` — это долгосрочная память, а не рабочая.

Аналог: у человека рабочая память = prefrontal cortex slots (~7 items). Без неё AGI не может держать цепочку рассуждений длиннее нескольких шагов.

---

### ❌ 6. Knowledge Graph — stub, не обучается

**Файл**: [`neuro_symbolic/knowledge_graph.py`](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/neuro_symbolic/knowledge_graph.py) (3583 байт = очень маленький)

`bootstrap_humanoid_ontology()` создаёт фиксированную онтологию. `KnowledgeGraph.add()` добавляет факты, но нет:
- Обучения из ошибок планирования
- Forgetting curve (устаревшие факты удаляются)
- Surprise-driven update (необычное наблюдение → новое правило)

Для AGI knowledge graph должен быть **живым** — расти из опыта, не из bootstrap.

---

### ❌ 7. Intrinsic objective — EIG исторически ≈ 0

Из диагностики: `agi_issues_diagnosis.md` — EIG был 0 из-за Bayesian ensemble update отсутствует.
Даже если это исправлено: **intrinsic objective (compression gain / mutual information)** —
это недостаточная цель для AGI. Агент стремится удивляться, но не стремится **решать задачи**.

Нет внешней оценки полезности: что именно нужно сделать в мире? Какова конечная задача?
Чистый curiosity-driven agent останется вечным исследователем, никогда не закрывая задачи.

---

### ❌ 8. Нет мета-познания / самомодели рассуждения

Агент не знает:
- Насколько он уверен в своём планировании
- Когда лучше спросить (exploit S2 deliberation vs fast S1)
- Что он не знает (unknown unknowns)

`system1.py` и `system2/controller.py` переключаются по простым условиям (fallen flag, curriculum stage).
Настоящее meta-cognition = динамическое управление "глубиной мышления" в зависимости от ситуации.
В нейронауке: prefrontal-parietal network. В ML: confidence estimation + adaptive compute.

---

### ❌ 9. Causal reasoning — только одношаговый

**Файл**: [`causal_graph.py`](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/causal_graph.py)

`forward_dynamics()` — один шаг. Beam search делает несколько шагов, но это **последовательный перебор**, не причинно-следственный вывод.

Pearl's framework уровень 2 (интервенции) частично есть (`do()` оператор). Уровень 3 (контрфактические рассуждения) — **отсутствует**. "Что было бы, если бы я не упал?" — агент ответить не может.

Контрфактический reasoning критичен для:
- Планирования из альтернатив
- Постмортем-анализа ошибок
- Социального взаимодействия ("что думает другой агент")

---

### ❌ 10. Verbal/language layer — изолированная заглушка

**Файлы**: `verbal_action.py`, `inner_voice_net.py`, `neural_causal_language.py`

Языковой слой существует, но он **не интегрирован** в loop принятия решений. `InnerVoiceNet` генерирует "мысли", но они не влияют на планирование, не сохраняются как символы, не передаются в Knowledge Graph.

Для AGI язык — не output, это **substrate мышления** (language of thought). Внутренний монолог должен быть частью рассуждения System 2.

---

## Что нужно для реального AGI (дорожная карта)

```
Уровень 0 (сейчас): Нейронная + символьная + физика
Уровень 1 (нужно): Двусторонняя нейро-символьная причинность
Уровень 2 (нужно): Обучаемый планировщик + рабочая память
Уровень 3 (нужно): Контрфактическое рассуждение + мета-познание
Уровень 4 (нужно): Язык как substrate мышления
```

### Ближайшие конкретные шаги (по приоритету)

| Приоритет | Что сделать | Где | Блокирует |
|-----------|------------|-----|-----------|
| **P0** | Замкнуть gradient loop для S2 плаунера (loss из outcome → LearnedMacroStudent) | `system2/learned_student.py` | Пункт 4 |
| **P0** | Нисходящий precision-weighting из символов → GNN (attention reweighting) | `neuro_symbolic/bridge.py` + `causal_graph.py` | Пункт 3 |
| **P1** | Working Memory buffer (~16 slots, readable/writable by S2) | новый `working_memory.py` | Пункт 5 |
| **P1** | Knowledge Graph online learning (surprise-driven rule creation) | `neuro_symbolic/knowledge_graph.py` | Пункт 6 |
| **P2** | Discovered symbolic actions из каузального графа (не хардкод 8 actions) | `neuro_symbolic/planner.py` | Пункт 1 |
| **P2** | Контрфактическое рассуждение (counterfactual rollout) | `causal_graph.py` | Пункт 9 |
| **P3** | Meta-cognition: уверенность S2 → динамический бюджет вычислений | `system2/controller.py` | Пункт 8 |
| **P3** | Language as reasoning substrate (InnerVoice → S2 input) | `inner_voice_net.py` + `system2/controller.py` | Пункт 10 |

---

## Главный структурный дефект

> Сейчас System 2 — это **надстройка над S1**, которая вмешивается по расписанию.
> Для AGI System 2 должна быть **основным циклом**, который делегирует S1 рутину.

В мозге: медленный cortex ставит задачи → быстрые подкорковые структуры их исполняют.
В RKK: быстрый S1 (GNN tick) работает всегда → S2 иногда вмешивается.

Это инверсия. И она объясняет, почему recovery scripted, почему CPG не учится ходить, и почему EIG нулевой — потому что нет агента, который **хотел бы** чего-то достичь на уровне S2.

---

## Справедливая оценка

**Уровень архитектуры**: ~70% от необходимого для "сильного ИИ в гуманоиде"  
**Уровень обученности/замкнутости loop**: ~30%  
**Главный риск**: дальнейшее добавление компонентов при незамкнутых существующих

Нужно не добавлять, а **замыкать контуры**.
