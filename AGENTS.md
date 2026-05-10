# AGENTS.md

## Cursor Cloud specific instructions

### Architecture overview

RKK is a two-part AGI simulation platform: a **Python backend** (FastAPI + PyBullet + PyTorch) and a **TypeScript/React frontend** (Vite + Three.js). They communicate via WebSocket (`ws://localhost:8000/ws/causal-stream`) and REST API on port 8000. Configuration is in `.env` at the repo root.

### Running services

- **Backend**: `cd backend && RKK_DEVICE=cpu python3 run.py`
  - All settings come from `.env` in the repo root — do NOT override them with command-line env vars
  - Only `RKK_DEVICE=cpu` is needed in the cloud VM (no GPU; `.env` has `cuda`)
  - Starts FastAPI/Uvicorn on port 8000
  - Ollama-dependent features (LLM teacher, VLM labeling) will gracefully fail with ConnectError — this is expected in cloud VM
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
