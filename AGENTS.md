# AGENTS.md

## Cursor Cloud specific instructions

### Architecture overview

RKK is a two-part AGI simulation platform: a **Python backend** (FastAPI + PyBullet + PyTorch) and a **TypeScript/React frontend** (Vite + Three.js). They communicate via WebSocket (`ws://localhost:8000/ws/causal-stream`) and REST API on port 8000. Configuration is in `.env` at the repo root.

### Running services

- **Backend**: `cd backend && RKK_DEVICE=cpu python3 run.py`
  - All settings come from `.env` in the repo root — do NOT override them with command-line env vars
  - Only `RKK_DEVICE=cpu` is needed in the cloud VM (no GPU; `.env` has `cuda`)
  - Starts FastAPI/Uvicorn on port 8000
  - Verbal speech (optional Ollama) may fail with ConnectError if Ollama is not running — expected in cloud VM
- **Frontend**: `npm run dev` (Vite on port 5173, see `package.json` scripts)

### Lint / Type-check / Build

- `npx eslint .` — ESLint for frontend
- `npx tsc -b` — TypeScript type check
- `npm run build` — full production build (tsc + vite build)

### Tick performance profiling

- **`RKK_TICK_PROFILE=1`** (default): per-span timings for agent (`agent.*`), simulation (`sim.*`), and background loops (`bg.*`).
- **Console**: ranked report every `RKK_TICK_PROFILE_REPORT_EVERY` ticks and when a tick exceeds `RKK_TICK_PROFILE_SLOW_MS`.
- **HTTP**: `GET http://localhost:8000/api/tick_profile` or `python backend/profile_tick.py`.
- **UI payload**: `tick_profile` field in `/api/snapshot` (top spans by EMA ms, % of wall time).
- **Simulation sub-spans** (inside former `post_agent`): `sim.post_motor_cortex`, `sim.post_episodic`, `sim.post_rssm`, `sim.post_l4`, `sim.post_cognition`, `sim.post_reflex_cereb`, `sim.post_scene_vision`, `sim.post_rsi`, etc.

### Performance tuning (env)

- **`RKK_VISION_GNN_FEED_EVERY=8`**: GNN→vision PC (`integrate_world_model_step`); was hardcoded every 2 ticks.
- **`RKK_SCENE_CACHE_EVERY=6`**: PyBullet scene/skeleton refresh interval (avoid `1`).
- **`RKK_WM_TRAIN_EVERY` / `RKK_WM_TRAIN_EVERY_FALLEN`**: WM `train_step` cadence (fallen default: off).
- **Fallen fast-path** (`agent.step(fallen=True)`): stale score cache, VL horizon 0, skip CEM/goal-plan, ensemble/temporal/traj updates.
- **Bench**: `python scratch/bench_tick_hz.py` (sync inner tick; reports median Hz and tick≥650).

### Key gotchas

- `pybullet` requires `build-essential`, `cmake`, and `python3-dev` system packages to compile from source. These must be installed before `pip install -r backend/requirements.txt`.
- PyTorch CPU variant must be installed with `--index-url https://download.pytorch.org/whl/cpu` to avoid downloading the large CUDA build.
- The `.env` file is tracked in git. To override settings without modifying it, pass environment variables directly when starting the backend.
- The backend initializes the PyBullet humanoid simulation on first request (lazy `get_sim()` call), so the first API call or WebSocket connection takes a few seconds.
- Backend Python scripts (uvicorn, fastapi, etc.) are installed to `~/.local/bin` — ensure this is on `PATH`.
