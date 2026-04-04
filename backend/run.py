"""
run.py — точка входа для Windows.

PowerShell:
  cd backend
  python run.py

После старта API (фоном): humanoid_structured LLM-bootstrap → автоматически vision ON → один VLM.
  Отключить LLM: RKK_SKIP_AUTO_HUMANOID_LLM=1
  Отключить авто-зрение: RKK_SKIP_AUTO_VISION=1
  Зрение да, без авто-VLM: RKK_SKIP_AUTO_VLM_BOOTSTRAP=1
  Слоты/режим: RKK_AUTO_VISION_N_SLOTS=8, RKK_AUTO_VISION_MODE=hybrid
  VLM как в UI: RKK_AUTO_VLM_WEAK_EDGES=1, RKK_AUTO_VLM_TEXT_ONLY=1, RKK_AUTO_VLM_MAX_MASKS=4
  URL/модель: RKK_OLLAMA_URL, RKK_OLLAMA_MODEL

Фаза 3 (после VLM): LLM-учитель для System1 + мягкий VL-overlay (TTL).
  Отключить авто: RKK_SKIP_PHASE3_LLM=1
  Затухание бонуса: RKK_TEACHER_T_MAX (интервенций, по умолчанию 140)

Ручной VLM: POST /vision/vlm-label. Повтор учителя: POST /teacher/refresh.

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
