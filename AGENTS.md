# AGENTS.md

## Cursor Cloud specific instructions

### Architecture overview

RKK is a two-part AGI simulation platform: a **Python backend** (FastAPI + PyBullet + PyTorch) and a **TypeScript/React frontend** (Vite + Three.js). They communicate via WebSocket (`ws://localhost:8000/ws/causal-stream`) and REST API on port 8000. Configuration is in `.env` at the repo root.

### Running services

- **Backend**: `cd backend && RKK_DEVICE=cpu RKK_SKIP_AUTO_HUMANOID_LLM=1 RKK_SKIP_AUTO_VISION=1 RKK_SKIP_PHASE3_LLM=1 RKK_LLM_LOOP=0 RKK_MEMORY_RESUME_ON_START=0 python3 run.py`
  - Starts FastAPI/Uvicorn on port 8000
  - `RKK_DEVICE=cpu` is required in the cloud VM (no GPU)
  - The `RKK_SKIP_*` and `RKK_LLM_LOOP=0` flags disable Ollama-dependent features (LLM bootstrap, visual cortex auto-init, phase-3 teacher, LLM-in-the-loop) since Ollama is not available in the cloud VM
  - `RKK_MEMORY_RESUME_ON_START=0` prevents loading stale autosave state
- **Frontend**: `npm run dev` (Vite on port 5173, see `package.json` scripts)

### Lint / Type-check / Build

- `npx eslint .` — ESLint for frontend
- `npx tsc -b` — TypeScript type check
- `npm run build` — full production build (tsc + vite build)

### Key gotchas

- `pybullet` requires `build-essential`, `cmake`, and `python3-dev` system packages to compile from source. These must be installed before `pip install -r backend/requirements.txt`.
- PyTorch CPU variant must be installed with `--index-url https://download.pytorch.org/whl/cpu` to avoid downloading the large CUDA build.
- The `.env` file is tracked in git. To override settings without modifying it, pass environment variables directly when starting the backend.
- The backend initializes the PyBullet humanoid simulation on first request (lazy `get_sim()` call), so the first API call or WebSocket connection takes a few seconds.
- Backend Python scripts (uvicorn, fastapi, etc.) are installed to `~/.local/bin` — ensure this is on `PATH`.
