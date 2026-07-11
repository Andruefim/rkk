# RKK — Embodied AGI Humanoid Simulation

Платформа для симуляции воплощённого (embodied) AGI-агента: гуманоид в физическом мире PyBullet, управляемый нейрокогнитивной архитектурой — каузальная GNN-модель мира, System2-планирование, CPG-локомоция, grounded language, интероцепция/аффект и иерархическое дерево задач.

Гуманоид принимает команды человека в чате (например, «подойди к предмету и дотронься до него»), строит план (imagine → execute → verify), выполняет его в симуляции, докладывает о результате и остаётся автономным между заданиями. Ход выполнения отображается на фронтенде в виде дерева задач.

## Архитектура

- **Backend** (`backend/`) — Python: FastAPI + PyBullet + PyTorch. Тик-цикл агента: восприятие → мир-модель (causal GNN) → планирование (System2 / WM planner) → моторный арбитраж (рефлексы, CPG, executive-интенты) → действие.
- **Frontend** (`src/`) — TypeScript/React + Three.js (Vite). 3D-визуализация сцены и скелета, чат с агентом, панель дерева задач, телеметрия.
- Связь: WebSocket `ws://localhost:8000/ws/causal-stream` + REST на порту 8000.

## Запуск

```bash
# Backend (порт 8000)
cd backend
pip install -r requirements.txt
python run.py

# Frontend (порт 5173)
npm install
npm run dev
```

Конфигурация — в `.env` в корне репозитория (устройство `RKK_DEVICE`, частоты циклов, флаги AGI-контура `RKK_TASK_BINDING` / `RKK_TASK_TREE` и др.). Подробности по флагам, профилированию тиков и тюнингу производительности — в `AGENTS.md`.

## Проверки

```bash
npx tsc -b        # типы фронтенда
npx eslint .      # линт фронтенда
npm run build     # прод-сборка

cd backend
$env:RKK_RUN_TESTS="1"; python -m pytest tests/ -q   # тесты бэкенда (PowerShell)
```
