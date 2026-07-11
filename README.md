# RKK — Embodied Agent Simulation

A simulation platform for an embodied agent: a humanoid in a PyBullet physics world driven by a neurocognitive architecture — a causal GNN world model, System2 planning, CPG locomotion, grounded language, interoception/affect, and a hierarchical task tree.

The agent accepts human commands in chat (e.g. "walk to the object in front of you and touch it"), builds a plan (imagine → execute → verify), carries it out in simulation, reports the result, and stays autonomous between tasks. Execution progress is shown on the frontend as a task tree.

## Architecture

- **Backend** (`backend/`) — Python: FastAPI + PyBullet + PyTorch. Agent tick loop: perception → world model (causal GNN) → planning (System2 / WM planner) → motor arbitration (reflexes, CPG, executive intents) → action.
- **Frontend** (`src/`) — TypeScript/React + Three.js (Vite). 3D visualization of the scene and skeleton, agent chat, task tree panel, telemetry.
- Communication: WebSocket `ws://localhost:8000/ws/causal-stream` + REST on port 8000.

## Getting started

```bash
# Backend (port 8000)
cd backend
pip install -r requirements.txt
python run.py

# Frontend (port 5173)
npm install
npm run dev
```

Configuration lives in `.env` at the repo root (device via `RKK_DEVICE`, loop rates, task-loop flags `RKK_TASK_BINDING` / `RKK_TASK_TREE`, etc.). See `AGENTS.md` for details on flags, tick profiling, and performance tuning.

## Checks

```bash
npx tsc -b        # frontend type check
npx eslint .      # frontend lint
npm run build     # production build

cd backend
$env:RKK_RUN_TESTS="1"; python -m pytest tests/ -q   # backend tests (PowerShell)
```
