"""
Periodic Ollama analysis of human-task logs.

Every RKK_TASK_LOG_AI_EVERY ticks, reads detailed task_log.jsonl for that window
plus prior AI analyses, and appends a Russian diagnostic summary to
ai_task_analysis.txt in the form:

  [0-100]: ...
  [101-200]: ...
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from engine.ollama_env import (
    get_ollama_generate_url,
    get_ollama_model,
    ollama_think_disabled_payload,
)
from engine.task_logger import (
    append_ai_analysis,
    read_ai_analysis_text,
    read_task_log_events,
    task_log_enabled,
)

_LOCK = threading.Lock()
_last_analyzed_hi: int = -1
_in_flight: bool = False


def task_log_ai_enabled() -> bool:
    if not task_log_enabled():
        return False
    raw = os.environ.get("RKK_TASK_LOG_AI", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def task_log_ai_every() -> int:
    try:
        return max(20, int(os.environ.get("RKK_TASK_LOG_AI_EVERY", "100")))
    except ValueError:
        return 100


def task_log_ai_model() -> str:
    m = (os.environ.get("RKK_TASK_LOG_AI_MODEL") or "").strip()
    return m or get_ollama_model()


def task_log_ai_timeout_s() -> float:
    try:
        return max(5.0, float(os.environ.get("RKK_TASK_LOG_AI_TIMEOUT_S", "45")))
    except ValueError:
        return 45.0


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _compact_events(events: list[dict[str, Any]], *, max_chars: int = 14000) -> str:
    """Serialize events for the prompt; drop bulky nested dumps if needed."""
    if not events:
        return "(no events in this tick window)"
    lines: list[str] = []
    for row in events:
        slim = {
            k: v
            for k, v in row.items()
            if k != "ts" and v is not None and v != "" and v != []
        }
        try:
            lines.append(json.dumps(slim, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            lines.append(str(slim)[:400])
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    # Keep head (command/bind) + tail (latest progress).
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return head + "\n...\n" + tail


def _build_prompt(
    *,
    tick_lo: int,
    tick_hi: int,
    events_text: str,
    prior_analysis: str,
) -> str:
    prior = prior_analysis.strip() or "(none yet — this is the first window)"
    return f"""Ты диагностический аналитик логов гуманоидного AGI-симулятора RKK.
Пиши по-русски, коротко и конкретно (8–16 предложений или маркированный список).
Цель: помочь разработчику понять, почему таск юзера выполняется или ломается.

Фокус (обязательно отметь, если видно в логах):
- какой текст команды / когда (event=command_received и т.п.)
- resolve цели (vision / sim-oracle / uncertain), hard_lock, range/bearing
- навигация: task_nav_active, heading_err, closing_vel, com_x/y, vision_range
- motor / intention macros (IDLE vs LOCOMOTE/EXPLORE), falls
- стадии task tree, PE / progress, успех или fail
- явные аномалии (nav=0 при активном таске, premature unlock, нет движения и т.д.)

Опирайся на предыдущие анализы — если проблема продолжается или изменилась, скажи явно.
Не выдумывай чисел, которых нет в логах. Не пиши общие советы вне логов.

Окно тиков: [{tick_lo}-{tick_hi}]

Предыдущие AI-анализы (хвост):
{prior}

Сырые события task_log.jsonl за это окно:
{events_text}

Ответ — только анализ для окна [{tick_lo}-{tick_hi}], без преамбулы и без JSON.
"""


def _call_ollama(prompt: str) -> str:
    url = get_ollama_generate_url().strip().rstrip("/")
    if not url.endswith("/generate"):
        url = url + "/api/generate" if "/api/" not in url else url
    payload: dict[str, Any] = {
        "model": task_log_ai_model(),
        "prompt": prompt,
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {
            "temperature": 0.2,
            "num_predict": _env_int("RKK_TASK_LOG_AI_NUM_PREDICT", 512),
        },
    }
    import httpx

    with httpx.Client(timeout=task_log_ai_timeout_s()) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            return f"(ollama HTTP {resp.status_code})"
        raw = (resp.json().get("response") or "").strip()
        return raw or "(empty ollama response)"


def analyze_tick_window(tick_lo: int, tick_hi: int) -> str:
    """Synchronous analysis for one tick window. Returns analysis text (or empty if skipped)."""
    events = read_task_log_events(
        tick_lo=tick_lo,
        tick_hi=tick_hi,
        max_events=_env_int("RKK_TASK_LOG_AI_MAX_EVENTS", 400),
    )
    if not events:
        # No task activity in this window — do not call Ollama or pollute analysis file.
        return ""
    prior = read_ai_analysis_text(
        max_chars=_env_int("RKK_TASK_LOG_AI_PRIOR_CHARS", 8000)
    )
    prompt = _build_prompt(
        tick_lo=tick_lo,
        tick_hi=tick_hi,
        events_text=_compact_events(events),
        prior_analysis=prior,
    )
    try:
        analysis = _call_ollama(prompt)
    except Exception as ex:
        analysis = f"(ollama error: {type(ex).__name__}: {ex})"
    append_ai_analysis(tick_lo, tick_hi, analysis)
    return analysis


def _run_window_async(tick_lo: int, tick_hi: int) -> None:
    global _in_flight, _last_analyzed_hi
    try:
        analyze_tick_window(tick_lo, tick_hi)
    finally:
        with _LOCK:
            _in_flight = False
            _last_analyzed_hi = max(_last_analyzed_hi, int(tick_hi))


def maybe_analyze_task_logs(tick: int) -> bool:
    """
    If tick lands on a window boundary and the window has events, kick off
    background Ollama analysis. Returns True if a job was started.
    """
    if not task_log_ai_enabled():
        return False
    every = task_log_ai_every()
    t = int(tick)
    if t <= 0 or (t % every) != 0:
        return False
    tick_hi = t
    tick_lo = 0 if t == every else (t - every + 1)

    # Skip empty windows before starting a thread / calling Ollama.
    if not read_task_log_events(tick_lo=tick_lo, tick_hi=tick_hi, max_events=1):
        with _LOCK:
            global _last_analyzed_hi
            _last_analyzed_hi = max(_last_analyzed_hi, tick_hi)
        return False

    with _LOCK:
        global _in_flight
        if _in_flight:
            return False
        if tick_hi <= _last_analyzed_hi:
            return False
        _in_flight = True

    thread = threading.Thread(
        target=_run_window_async,
        args=(tick_lo, tick_hi),
        name=f"task-log-ai-{tick_lo}-{tick_hi}",
        daemon=True,
    )
    thread.start()
    return True


def reset_analyzer_state_for_tests() -> None:
    """Test helper: clear in-memory window bookkeeping."""
    global _last_analyzed_hi, _in_flight
    with _LOCK:
        _last_analyzed_hi = -1
        _in_flight = False
