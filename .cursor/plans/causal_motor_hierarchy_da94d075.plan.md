---
name: causal motor hierarchy
overview: "Перестроить локомоцию из разрозненных CPG/skill контуров в иерархическую causal-aware моторную систему: быстрый low-level остаётся, но попадает в общий credit-assignment, а high-level граф видит моторные команды, фазу gait, опору и перенос веса."
todos:
  - id: motor-state-layer
    content: Спроектировать общий MotorState/MotorCommandLog между L1 worker и main causal loop
    status: completed
  - id: motor-observables
    content: Определить новый набор gait/support/contact переменных и точки их инжекции в graph/L1
    status: completed
  - id: cpg-to-motor-policy
    content: Перепроектировать CPG из автономного reward learner в causal-aware MotorPolicy
    status: completed
  - id: skills-to-options
    content: Перевести skill_library с raw joint sequences на hierarchical motor options
    status: completed
  - id: intent-action-space
    content: Добавить motor-intent action space и ValueLayer/goal planning проверки
    status: completed
  - id: worker-safe-migration
    content: Сохранить single-writer и совместимость с L4 worker, persistence и RSI
    status: completed
isProject: false
---

# План Исправления Каузальной Локомоции

## Цель
Сделать моторный стек согласованным с общей AGI/RSI архитектурой: `L0` физика остаётся быстрой, `L1` становится causal-aware, а `L2/L3` начинают видеть и объяснять низкоуровневые моторные команды, gait phase, опору и перенос веса вместо только конечных наблюдений.

## Текущий Разрыв
- Нормальный causal path идёт через `agent.step() -> ValueLayer.check_action() -> graph.propagate()/record_intervention() -> train_step()`, см. [c:\Users\Andrey\Desktop\agi\rkk\backend\engine\agent.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\agent.py) и [c:\Users\Andrey\Desktop\agi\rkk\backend\engine\causal_graph.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\causal_graph.py).
- Но locomotion сейчас обходит этот путь: `CPG` пишет напрямую в физику и учится на суррогатном reward (`com_x/com_z/fallen`) в [cpg_locomotion.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\cpg_locomotion.py), а `skill_library` гоняет кадры через `env.intervene(..., count_intervention=False)` в [simulation.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\simulation.py).
- Из-за этого low-level живёт своей жизнью, а граф почти не получает credit за ходьбу. Скрин с `block 95%`, `edges 0`, `discovery 0%` этому соответствует.

## Целевая Архитектура
```mermaid
flowchart LR
  subgraph l0[L0 Physics 120Hz]
    phys[PyBulletEnv]
  end
  subgraph l1[L1 Motor Controller 60Hz]
    motorState[MotorState]
    motorPolicy[MotorPolicy]
    motorLog[MotorCommandLog]
  end
  subgraph l23[L2 L3 Cognitive Graph]
    graph[CausalGraph]
    vl[ValueLayer]
    planner[PlannerAndImagination]
  end
  graph -->|"motor intents"| motorPolicy
  motorPolicy -->|"joint targets"| phys
  phys -->|"obs contacts gait phase support"| motorState
  motorState -->|"derived motor vars"| graph
  motorLog -->|"credit assignment / record_intervention_like"| graph
  planner -->|"goal / gait mode / support bias"| graph
  vl -->|"guardrails for motor intents"| graph
```

## Реализация По Этапам

### Этап 1. Ввести каузальный моторный интерфейс без ломки текущего fast loop
- Добавить новый слой `MotorState` / `MotorCommandLog` в [simulation.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\simulation.py), который живёт в main thread и хранит:
  - `last_motor_intents`
  - `last_joint_targets`
  - `gait_phase`
  - `support_leg`
  - `foot_contact_l/r`
  - `com_shift_target`
- Оставить `L0` и decoupled `CPG` worker, но заставить их работать через этот общий интерфейс, а не напрямую мимо каузального слоя.
- Важно: `agent.graph` и реальное применение к графу остаются single-writer; high-Hz loop только читает snapshot и пишет в thread-safe лог/снимок состояния.

### Этап 2. Сделать low-level действия наблюдаемыми для графа
- Расширить humanoid observation в [environment_humanoid.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\environment_humanoid.py) новыми моторными переменными:
  - `gait_phase_l`, `gait_phase_r`
  - `foot_contact_l`, `foot_contact_r`
  - `support_bias`
  - `motor_drive_l`, `motor_drive_r`
  - `posture_stability`
- Эти переменные должны появляться в `graph._node_ids`, persistence и snapshot так же, как другие физические наблюдения.
- `HierarchicalGraph.step_l1()` в [hierarchical_graph.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\hierarchical_graph.py) начать кормить не только суставами, но и агрегатами опоры/переноса веса.

### Этап 3. Встроить CPG в каузальный credit-assignment
- Перестроить [cpg_locomotion.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\cpg_locomotion.py) так, чтобы `LocomotionController` работал не как автономный learner по `com_x`, а как `MotorPolicy`:
  - на вход получает `motor intents` из графа
  - на выход отдаёт `joint targets` и `motor latent state`
- Вместо голого reward на `com_x/com_z` добавить structured objective:
  - устойчивость корпуса (`torso_roll`, `torso_pitch`)
  - сохранение `com_z`
  - симметрия шагов
  - чередование опоры ног
  - ограничение jerk/частоты на ankles/knees
- Главное: каждая low-level motor burst должна попадать в лог как `record_intervention_like` событие, чтобы граф мог учить причинность между `motor intents` и результатом.

### Этап 4. Перевести skill library из bypass в hierarchical options
- В [skill_library.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\skill_library.py) и skill-ветке в [simulation.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\simulation.py) перестать трактовать skill как прямую последовательность raw joint `do()`.
- Новая роль skill:
  - выдавать короткий `motor option` / `gait mode`
  - менять high-level intent (`stride_length`, `swing_leg`, `stabilize_torso`, `prepare_support_shift`)
  - low-level policy уже реализует это на 60Hz.
- Так skill перестанет обходить `ValueLayer` и `record_intervention`; вместо этого он станет частью causal plan layer.

### Этап 5. Поднять L2/L3 до управления моторными намерениями, а не суставами
- В [agent.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\agent.py) и [goal_planning.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\goal_planning.py) добавить отдельный action-space для embodied control:
  - `intent_stride`
  - `intent_support_left/right`
  - `intent_torso_forward`
  - `intent_arm_counterbalance`
  - `intent_stop_recover`
- `ValueLayer` в [value_layer.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\value_layer.py) должен проверять не только raw joint range, но и motor intent safety: overspeed, destabilizing support shift, repeated unstable gait mode.
- `imagination` начать крутить поверх motor intents, а не только поверх raw joints.

### Этап 6. Сохранить worker/thread производительность
- Не переносить `agent.graph` в high-Hz worker.
- Сохранить текущий single-writer инвариант из [simulation.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\simulation.py):
  - `L0/L1` worker only reads snapshots and writes motor state/log
  - `L2/L3/L4 apply` делает main tick
- Для `L4` concept worker и persistence:
  - моторные новые переменные должны быть сериализуемыми в текущем [persistence.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\persistence.py)
  - `memory_load()` должен сбрасывать и rebind-ить новый motor state так же, как сейчас безопасно делает для `L4`
- Для decoupled `CPG` обязательно отвязать внутренний `dt=0.05` от реального loop Hz и перейти на реальный `dt` worker-а.

### Этап 7. Привязать RSI и дальнейшие фазы к новой моторной иерархии
- В [rsi_full.py](c:\Users\Andrey\Desktop\agi\rkk\backend\engine\rsi_full.py) заменить perturbation target:
  - не просто шум в CPG amplitude/frequency,
  - а perturbation в motor intent / gait mode / recovery strategy.
- Phase 2 системы (`hierarchical_graph`, `concept_store`) должны начать видеть устойчивые моторные паттерны как часть embodied concepts.
- Phase 3 (language/social) подключать только после того, как low-level становится каузально объяснимым: тогда language сможет указывать на `concept`/`motor intent`, а не на raw ankle twitch.

## Порядок Внедрения
1. Сначала ввести `MotorState` и моторные переменные без изменения поведения.
2. Затем переписать `CPG` в `MotorPolicy`, сохранив существующие loop/threads.
3. Потом перевести skill library на `motor options`.
4. Затем расширить action-space графа и `ValueLayer` под motor intents.
5. После стабилизации локомоции подвязать RSI и позже language/social.

## Основные Риски
- Если сразу убрать raw-joint control, можно сломать текущий fixed-root и Phase 2 bootstrap.
- Если разрешить high-Hz worker писать в `graph`, появятся гонки с `agent_loop`, `L3`, `L4` и persistence.
- Если сохранить reward с упором в `com_x`, система снова будет учиться “бежать вперёд и падать”, даже в новой архитектуре.

## Критерии Успеха
- `block_rate` локомоции заметно падает с текущих экстремальных значений.
- High-level граф получает устойчивые рёбра к моторным intent/state узлам.
- В gait видны чередование опоры и стабилизация корпуса, а не только разгон ног.
- RSI расширяет поведение через motor intents, а не через хаотичное усиление CPG.
- Следующие фазы начинают строиться поверх устойчивой embodied causal stack, а не поверх обходных low-level контуров.