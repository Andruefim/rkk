"""
run.py — точка входа для Windows.

PowerShell (из каталога репозитория rkk):
  cd backend
  python run.py

Переменные RKK_* можно держать в файле .env в корне rkk (рядом с backend/) — подхватывается при старте.
Без .env, только для сессии PowerShell:
  $env:RKK_LLM_LOOP = "1"
  cd backend; python run.py

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

Этап D — LLM в петле (не только bootstrap): RKK_LLM_LOOP=1
  Уровень 2: контрфактуальная консультация при стагнации discovery (≥RKK_LLM_STAGNATION_TICKS),
  rolling block_rate>0.4, VLM «неизвестный» слот, surprise PE>3σ.
  Уровень 3 (humanoid): редко RKK_LLM_LEVEL3_INTERVAL тиков — перезапись гипотез.
  Доп.: RKK_LLM_LEVEL2_COOLDOWN (по умолч. 240 тиков), RKK_LLM_MIN_INTERVENTIONS.
  World model: RKK_WM_PASSIVE_MIX — доля пассивных переходов в train_step GNN.
  Цель (Этап E): self_goal_active / self_goal_target_dist в humanoid; RKK_GOAL_PLANNING, RKK_PLAN_DEPTH, RKK_PLAN_VALUES.
  RSI lite (Этап G): плато discovery_rate → L1/BUFFER/imagination; RKK_RSI_LITE, RKK_RSI_PLATEAU_TICKS, RKK_RSI_MIN_INTERVENTIONS.

Или через uvicorn напрямую:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

# Windows: ОБЯЗАТЕЛЬНО if __name__ == "__main__"
# иначе multiprocessing сломает spawn
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path

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
