# План перехода к AGI: Удаление хардкода и Оптимизация GNN

Мы успешно внедрили открытую архитектуру (dynamic ontology, datadriven sleep, learned motor primitives). Теперь нужно (1) сжечь старые "костыли" и (2) снять 'оковы' с производительности, которая упёрлась в O(d²).

## Goal Description

**1. Удаление хардкода.** Настоящий AGI не опирается на предзаданные человеком правила. Все функции, содержащие "зашитые" знания (формульная моторика, ground truth рёбра, фиксированные триггеры), подлежат физическому удалению.
**2. Ускорение (Performance).** При $d=114$ один батч в 256 примеров раздувает тензор `h_src` в `causal_gnn.py` до 638 МБ. GNN делает $O(B \cdot d^2)$ нелинейных трансформаций для message passing. Нужно перевести систему на sparse operations.

## User Review Required

> [!WARNING]
> **Мы полностью удаляем Ground Truth рёбра (gt_edges) и "ручную" моторику.** 
> Системе придётся выучивать координацию *с нуля* (опираясь исключительно на intrinsic reward и compression gain). Агент первые 50-100к тиков может выглядеть неуклюже, пока нейронные моторные примитивы (learned primitives) не построят нужные паттерны. Это необходимая цена за настоящую автономию.

## Proposed Changes

---

### Phase 1: Архитектурное Ускорение GNN (Откуда берутся 3 секунды на тик)

Проблема кроется в файле `causal_gnn.py`.
В методе `_message_pass`:
```python
h_src = h.unsqueeze(1).expand(B, d, d, self.hidden)
msg = self.msg_fn(torch.cat([h_src, h_dst], dim=-1))
```
Для батча EIG $B=256$ и текущего $d=114$, тензор весит сотни мегабайт, и MLP `msg_fn` отрабатывает $\approx 3.3$ миллиона раз за *один шаг*.

#### [MODIFY] [causal_gnn.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/causal_gnn.py)
*   **Внедрение Sparse Message Passing:** Мы будем вычислять сообщения *только* для пар вершин, где матрица смежности $|W_{m}[j,i]| > \epsilon$ (или хотя бы $> 10^{-4}$). Учитывая, что реальный причинный граф имеет плотность 5-10%, это даст **10-кратное** ускорение и сокращение памяти.
*   Реализация (pure PyTorch): собираем индексы через `W_masked().nonzero()`, выбираем фичи из `h`, прогоняем через (гораздо меньший) батч в `msg_fn` и раскидываем их обратно через `scatter_add`.

#### [MODIFY] [agent.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/agent.py)
*   Внутри `_batch_hypothesis_eig` используется `integrate_world_model_step`. Это означает, что для проверки гипотез агент делает *решение ОДУ*. Это overkill для скоринга чувствительности!
*   **Решение:** в цикле EIG напрямую вызывать `_core.forward_dynamics` вместо дорогого `odeint`. При EIG нас интересует первая производная связи, а не точечная траектория в непрерывном времени.

---

### Phase 2: Удаление замененного хардкода

#### [MODIFY] [environment.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/features/humanoid/environment.py)
*   **Удалить** `_apply_motor_intents`. Человек написал формулы: $hip = 0.50 + 0.14 \times stride - 0.08 \times sup$. Это нужно стереть. Метод будет просто пропускать значения, отдавая управление в `LearnedMotorProgram`.
*   **Удалить** `gt_edges()`. Полностью стереть массив из 40+ хардкоженных рёбер. 

#### [MODIFY] [constants.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/features/humanoid/constants.py)
*   **Удалить** `VAR_NAMES`. Оставить только диапазоны нормализации (`_RANGES`) и списки нужные для `VariableRegistry`. 
*   **Удалить** `HUMANOID_KINEMATIC_EDGE_PRIORS`.

#### [MODIFY] [sleep_consolidation.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/sleep_consolidation.py)
*   **Отключить** periodic сон (`if (tick - self.last_sleep_tick) >= self._every_ticks...`).
*   Оставить только Data-driven (compression stagnation) и аварийный сон (при частых падениях).

#### [MODIFY] [skill_library_humanoid.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/skill_library_humanoid.py)
*   Мы можем безопасно выкинуть хардкоженные цепочки вроде `step_forward_L = [doIntent_support_right, ...]`, оставив этот модуль как "заглушку" с перенаправлением в `MotorPrimitiveLibrary`.

---

## Open Questions

> [!CAUTION]
> Когда я удалю формульную моторику (`_apply_motor_intents`), гуманоид в первые часы начнёт дёргать ногами хаотично, пока `MotorPatternDetector` не соберёт первые успешные транзиции и не обучит примитивы. **Вы готовы к длительному периоду (возможно 100k+ тиков) полностью хаотичного падения, пока он не "нащупает" моторику?** Либо мне оставить базовый CPG-локомоушен (`_maybe_apply_cpg_locomotion`) как рефлекторный низший уровень (System 0)? Рекомендуется оставить CPG как спинной мозг.

## Verification Plan

1. **Performance Check:** Я выведу `time.time()` до и после `tick_step()` в консоль симулятора, чтобы убедиться, что время тика упало с ~3 секунд до ~0.1-0.2с даже при больших размерах графа ($d \sim 150$).
2. **Graph Growth Check:** Посмотреть логи нейрогенеза и `VariableRegistry.snapshot()` чтобы понять, что AGI стабильно и с нужной скоростью расширяет своё представление мира (от 11 до 114+).
3. **Hardcode Zero Check:** Выполню `grep` по `0.14*stride` и `gt_edges`, чтобы убедиться, что в исходниках больше не осталось читерских правил AGI.
