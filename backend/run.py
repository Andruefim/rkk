"""
run.py — точка входа для Windows.

PowerShell (из каталога репозитория rkk):
  cd backend
  python run.py

Переменные RKK_* можно держать в файле .env в корне rkk (рядом с backend/) — подхватывается при старте.

После старта API (фоном): опционально enable visual (SlotAttention).
  RKK_SKIP_AUTO_VISION=1 — не включать зрение.
  RKK_AUTO_VISION_N_SLOTS=8, RKK_AUTO_VISION_MODE=hybrid

Ручной bootstrap графа: POST /bootstrap/humanoid
Ручное зрение: POST /vision/enable

Или через uvicorn напрямую:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

# Windows: ОБЯЗАТЕЛЬНО if __name__ == "__main__"
# иначе multiprocessing сломает spawn
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path

    # Windows cp1251: Unicode in print() → UnicodeEncodeError → первый WS-тик роняет поток.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    _backend_dir = Path(__file__).resolve().parent
    _repo_root = _backend_dir.parent
    try:
        from dotenv import load_dotenv

        load_dotenv(_repo_root / ".env")
    except ImportError:
        pass

    # Добавляем backend/ в путь чтобы engine.* импортировался правильно
    sys.path.insert(0, str(_backend_dir))

    import torch
    print(f"[RKK] PyTorch: {torch.__version__}")
    print(f"[RKK] GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[RKK] Device: {torch.cuda.get_device_name(0)}")
    else:
        print("[RKK] Running on CPU (для GPU: PyTorch+cuda; при наличии GPU задайте RKK_DEVICE=cuda)")

    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # True только при разработке (не работает с GPU singleton)
        workers=1,      # Windows + GPU: строго 1 worker
        log_level="info",
    )
