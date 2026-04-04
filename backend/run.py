"""
run.py — точка входа для Windows.

PowerShell:
  cd backend
  python run.py

После старта API автоматически вызывается humanoid_structured LLM-bootstrap (Ollama).
  Отключить: RKK_SKIP_AUTO_HUMANOID_LLM=1
  URL/модель: RKK_OLLAMA_URL, RKK_OLLAMA_MODEL (по умолчанию localhost:11434, gemma4:e4b)

Или через uvicorn напрямую:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

# Windows: ОБЯЗАТЕЛЬНО if __name__ == "__main__"
# иначе multiprocessing сломает spawn
if __name__ == "__main__":
    import sys
    import os

    # Добавляем backend/ в путь чтобы engine.* импортировался правильно
    sys.path.insert(0, os.path.dirname(__file__))

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
