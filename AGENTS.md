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
- **`RKK_SCENE_CACHE_EVERY=6`**: full PyBullet scene rebuild interval (`static_geometry`, registry, etc.; avoid `1`).
- **`RKK_SKELETON_EVERY=1`**: lightweight per-tick skeleton/ankleQuats/cubes patch on top of cached scene (decoupled from scene cache).
- **`RKK_WS_STATIC_EVERY=30`**: WebSocket omits `scene.static_geometry` between full frames; frontend retains last known static mesh.
- **`RKK_WM_TRAIN_EVERY` / `RKK_WM_TRAIN_EVERY_FALLEN`**: WM `train_step` cadence (fallen default: off).
- **Fallen fast-path** (`agent.step(fallen=True)`): stale score cache, VL horizon 0, skip CEM/goal-plan, ensemble/temporal/traj updates.
- **Bench**: `python scratch/bench_tick_hz.py` (sync inner tick; reports median Hz and tick≥650).

### AGI humanoid validation loop

The tracked `.env` enables the **vision-first** command/task-tree path (camera + depth for control; registry for sim eval only). Set binding/tree to `0` for legacy autonomous-only runs:

- **`RKK_TASK_BINDING=1`**: human chat → counterfactual WM goal → PE verify → REPORT (requires `RKK_GROUNDED_LANG=1`).
- **`RKK_TASK_TREE=1`**: hierarchical task tree on top of task binding; optional LLM stage decompose via `RKK_TASK_TREE_LLM=1` (falls back to predicate ontology if Ollama unavailable).
- **`RKK_TASK_RESOLVE=vision`**: bind/control from ego camera slots + metric depth (`VisualTarget.bearing` + `range_m`). Use `oracle` only for ablation/legacy tests (privileged registry XY).
- **`RKK_AUTO_VISUAL=1`** / **`RKK_SLOT_LABEL_ENABLED=1`**: SlotAttention continuous + bind-time scene labels for vision resolve.
- **`RKK_MANIP_CHAIR=1`**: spawn movable chair for **sim metrics / oracle ablation**; control path must not require registry when `RKK_TASK_RESOLVE=vision`.
- **`RKK_VISUAL_GROUNDING`**: slot↔body (self) grounding — separate from scene-object resolve.
- **`RKK_TASK_MOTOR_BODYSPLIT=1`** (default): during active human tasks, register only upper-body intents as `human_task`; balance-critical fields stay with reflex/gait.
- **`RKK_TASK_MOTOR_HOLD_TICKS=60`**: after fall hard-reset, skip `human_task` motor registration for N ticks.

Depth: sim uses PyBullet ego depth buffer (`get_ego_rgbd` → `DepthCamera.range_at_uv`). Same API for real-robot RGB-D/stereo later.

### Key gotchas

- `pybullet` requires `build-essential`, `cmake`, and `python3-dev` system packages to compile from source. These must be installed before `pip install -r backend/requirements.txt`. `python3-dev` is baked into the cloud snapshot; without it the pybullet wheel build fails with `fatal error: Python.h: No such file or directory`.
- Python packages install into the user site (`~/.local`). The base image ships a PEP 668 `EXTERNALLY-MANAGED` marker, so manual pip installs need `pip install --user --break-system-packages ...` (the startup update script already uses this). `torch` (CPU build, `2.13.0+cpu`) is pre-installed in the snapshot and is NOT in `requirements.txt` — do not `pip install torch` from the default index or it pulls the large CUDA build.
- `pytest` is not in `requirements.txt`; it is baked into the snapshot. Backend tests are skipped unless `RKK_RUN_TESTS=1` (see `backend/tests/conftest.py`). Run: `cd backend && RKK_RUN_TESTS=1 RKK_DEVICE=cpu python3 -m pytest tests/ -q`.
- Known pre-existing failure (unrelated to env setup): `tests/test_manipulation_pybullet_smoke.py::test_fallback_manip_chair_push` — `_FallbackHumanoid.__init__` in `engine/features/humanoid/fallback.py` uses `self._object_registry` before it is assigned. The rest of the suite passes.
- PyTorch CPU variant must be installed with `--index-url https://download.pytorch.org/whl/cpu` to avoid downloading the large CUDA build.
- The `.env` file is tracked in git. To override settings without modifying it, pass environment variables directly when starting the backend.
- The backend initializes the PyBullet humanoid simulation on first request (lazy `get_sim()` call), so the first API call or WebSocket connection takes a few seconds.
- Backend Python scripts (uvicorn, fastapi, etc.) are installed to `~/.local/bin` — ensure this is on `PATH`.
