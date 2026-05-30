# RKK: Диагноз по 5000-тиковому прогону и план исправлений

## Диагноз: 4 реальные проблемы

### Проблема 1 — Graph growth убивает физику (CRITICAL)
**Симптом**: падение на тиках 2500 и 4600 точно совпадает со всплеском рёбер (~600 → ~3000 и ~4500).  
**Причина**: `NeurogenesisEngine.scan_and_grow()` вызывает `graph.rebind_variables()` → `resize_to()` на CausalGNNCore. При ресайзе вся матрица `W` расширяется, новые механизмы добавляются без обучения, и один шаг forward_dynamics возвращает случайные числа. `HomeostaticController` получает garbage intent → CPG получает garbage → тело падает.  
**Корень**: нейрогенез происходит прямо во время шага симуляции, без заморозки тела.

### Проблема 2 — Recovery scripted, не learned (CRITICAL)
**Симптом**: `fallen_override:scripted` в 100% эпизодов падения. S2 RECOVER_POSTURE не обучен.  
**Причина**: В `intristic_objective.py` EIG = 0 (мы только что исправили), поэтому `GoalImagination` никогда не генерировала цель `intent_stop_recover`. Motor cortex не получал gradient signal на recovery позу.  
**Корень**: circular dependency — чтобы научиться вставать, нужно упасть + получить EIG-цель на вставание. Но EIG был 0.

### Проблема 3 — CPG стоит, не ходит (HIGH)
**Симптом**: Locomotion reward EMA = 0.07–0.08. 100% действий = gait_coupling. com_x не растёт.  
**Причина**: CPG генерирует ритмичные сигналы, но `LocomotionController.get_joint_targets()` отдаёт суставам delta ≈ ±0.18 от 0.5 — это маленький размах **при правильном масштабе**, но физика PyBullet требует, чтобы нога реально выдвигалась вперёд. Мы подняли joint_amp до 0.18, что дало ходьбу, но затем CPG "остывает" — intrinsic reward уменьшается по мере того как WM запоминает стояние.  
**Корень**: нет forward momentum reward. CPG оптимизируется только по intrinsic (EIG), который исчезает после нескольких тиков.

### Проблема 4 — Ensemble не обновляется (MEDIUM)
**Симптом**: ensemble weights = [0.25, 0.25, 0.25, 0.25] на протяжении всего прогона.  
**Причина**: `ensemble_log_likelihood()` вызывается в `intristic_objective.py`, но `ens.update_posterior()` — нет. Байесовское обновление написано, но не подключено к основному циклу.  
**Корень**: отсутствует вызов `graph._ensemble.update_posterior(ll)` в `IntrinsicObjective.step()`.

---

## План исправлений

### Fix 1: Изолировать нейрогенез от физики

#### [MODIFY] [rsi_structural.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/rsi_structural.py)
- Добавить флаг `_pending_growth: dict | None` — вместо немедленного `rebind_variables` ставим рост в очередь
- Фактический рост выполнять только в момент `fixed_root=True` или когда агент только что встал (posture > 0.8 в течение 20+ тиков)

#### [MODIFY] [intristic_objective.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/intristic_objective.py)  
- В `maybe_trigger_neurogenesis`: проверять `is_fallen` и `posture_stability` перед вызовом `scan_and_grow`
- Минимальный cooldown нейрогенеза: 2000 тиков (сейчас 500)

---

### Fix 2: Подключить Bayesian ensemble update

#### [MODIFY] [intristic_objective.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/intristic_objective.py)
- В `IntrinsicObjective.step()` после вычисления EIG добавить:
```python
if graph._ensemble is not None and cf_pred and cf_obs:
    ll = ensemble_log_likelihood(graph, cf_pred, cf_obs, best_intent_var, best_intent_val)
    graph._ensemble.update_posterior(ll)
```
- Это позволит ансамблю дифференцироваться — некоторые гипотезы начнут получать больший вес, EIG станет информативным.

---

### Fix 3: Forward momentum reward для CPG

#### [MODIFY] [cpg_locomotion.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/cpg_locomotion.py)
- В `train_cpg_from_intrinsic_history()`: добавить `com_x_velocity_bonus` к reward signal
- Формула: `r_total = r_intrinsic + 0.4 * clip(com_x_vel - 0.01, 0, 1)`
- Это даёт CPG постоянный стимул двигаться вперёд независимо от EIG

#### [MODIFY] [intristic_objective.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/intristic_objective.py)
- В `_apply_intrinsic_reward()`: извлекать `com_x_vel` из obs и добавлять к reward перед передачей в locomotion_ctrl

---

### Fix 4: Обучаемый recovery (GoalImagination → recovery цели)

#### [MODIFY] [intristic_objective.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/intristic_objective.py)
- В `generate_goal()`: когда `is_fallen` (posture < 0.4) — приоритизировать кандидатов `intent_stop_recover` и `intent_torso_forward`
- В `get_target_priors()`: при `posture < 0.4` всегда возвращать `{"intent_stop_recover": 0.80, "intent_torso_forward": 0.55}` независимо от GoalImagination

---

## Порядок выполнения

| # | Fix | Файл | Приоритет |
|---|-----|------|-----------|
| 1 | Нейрогенез только при posture > 0.75 + cooldown 2000 | `rsi_structural.py` | CRITICAL |
| 2 | Bayesian ensemble update в основном цикле | `intristic_objective.py` | HIGH |
| 3 | Forward momentum bonus в CPG reward | `cpg_locomotion.py` + `intristic_objective.py` | HIGH |
| 4 | Recovery goal из GoalImagination | `intristic_objective.py` | HIGH |

## Ожидаемый результат после фиксов
- Нейрогенез перестанет валить тело (graph growth в безопасный момент)
- Ensemble дифференцируется → EIG станет информативным → GoalImagination генерирует реальные цели
- CPG получает постоянный forward signal → локомоция не "остывает"
- Recovery goal появляется при падении → агент учится вставать (не scripted)

---

## Fix 5–8 (v2): AGI-critical extensions

### Fix 5 — Graded S1/S2 recovery
- `RKK_S2_LEARNED_RECOVERY=1`: Phase L (120 ticks) — S1 intrinsic, без scripted physics
- `motor_owner`: `s1_learned` | `s2_scripted` | `cpg`
- **Files**: `system2/controller.py`, `mixin_locomotion.py`

### Fix 6 — Behavioral curriculum gates
- `BehavioralTracker`, Step 3 substates `3a_learning` / `3b_locomotion_mastered`
- **Files**: `behavioral_tracker.py`, `snapshot.py`, frontend HUD

### Fix 7 — Graph growth control
- `NeurogenesisCoordinator`, edge delta limiter, sleep consolidation hook
- **Files**: `neurogenesis_coordinator.py`, `rsi_structural.py`, `sleep_consolidation.py`

### Fix 8 — Honest metrics
- `structural_discovery` vs `behavioral_score`, phi fallen penalty, extended tick log
- **Files**: `agent.py`, `tick_run_logger.py`

### Validation
```bash
RKK_TICK_RUN_LOG=1 python backend/run.py
python scratch/validate_agi_fixes.py
```
