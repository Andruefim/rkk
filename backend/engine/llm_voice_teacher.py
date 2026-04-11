"""
llm_voice_teacher.py — LLM Voice Teacher: offline distillation от LLM к InnerVoiceNet.

LLM больше НЕ работает в real-time цикле агента.
Вместо этого: раз в 60-120 секунд (τ3) LLM анализирует
текущее состояние и генерирует:
  1. verbal_annotation — текстовое описание ситуации
  2. concept_labels — список активных концептов из ConceptStore
  3. guidance_intents — рекомендуемые intent_* значения

Это используется для:
  A. Дистилляции в InnerVoiceNet (push_distill_sample)
  B. Обновления ConceptStore (contrastive_step)
  C. Soft guidance в IntentPropagator (τ3 → τ2)

Режимы работы LLM:
  ANNOTATE  — анализирует состояние, возвращает concepts + verbal
  GUIDE     — предлагает конкретные intent_* изменения (было LLM L2)
  LESSON    — глубокий анализ паттернов (раз в N минут, для sleep)

Ключевые отличия от старой LLM L2:
  - НЕ блокирует основной tick-цикл (async, timeout=30s)
  - НЕ ожидает ответа (fire-and-forget с callback)
  - НЕ применяет напрямую к env (только через IntentPropagator)
  - Результат кэшируется и применяется при следующем τ3 цикле

RKK_LLM_TEACHER_ENABLED=1
RKK_LLM_TEACHER_EVERY_SEC=90      — секунд между annotation calls
RKK_LLM_TEACHER_GUIDE_EVERY=3     — annotation вызовов до guide call
RKK_LLM_TEACHER_LESSON_EVERY=10   — guide вызовов до lesson call
RKK_LLM_TEACHER_TIMEOUT=30        — HTTP timeout
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
import numpy as np

from engine.concept_store import CONCEPT_DEFS


def teacher_enabled() -> bool:
    return os.environ.get("RKK_LLM_TEACHER_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ── Available concept names for LLM ───────────────────────────────────────────
DEFINED_CONCEPTS = [name for name, _, _ in CONCEPT_DEFS if not name.startswith("LATENT_")]
_CONCEPT_LIST_STR = ", ".join(DEFINED_CONCEPTS[:80])  # first 80 for prompt brevity


# ── Teacher result dataclass ───────────────────────────────────────────────────
@dataclass
class TeacherAnnotation:
    """Result from one LLM teacher call."""
    tick: int
    timestamp: float
    mode: str  # "annotate", "guide", "lesson"

    # Verbal annotation (for distillation training)
    verbal: str = ""

    # Concept labels (from ConceptStore vocab)
    primary_concepts: list[str] = field(default_factory=list)

    # Guidance intents (for IntentPropagator τ3 → τ2)
    intent_adjustments: dict[str, float] = field(default_factory=dict)

    # Lesson content (for sleep consolidation)
    lesson_text: str = ""
    lesson_concepts: list[str] = field(default_factory=list)

    # Seeds for GNN (causal hypotheses)
    seeds: list[dict] = field(default_factory=list)

    # Quality
    confidence: float = 1.0
    error: str = ""


# ── Prompt builders ────────────────────────────────────────────────────────────
def _build_annotate_prompt(
    obs_summary: str,
    concept_str: str,
    fall_history_brief: str,
    available_concepts: str = _CONCEPT_LIST_STR,
) -> str:
    return f"""You are the internal voice of a humanoid robot learning to walk.
Observe the robot's state and describe what is happening in simple terms.

CURRENT STATE:
{obs_summary}

CURRENT INNER VOICE: {concept_str or "UNKNOWN"}

FALL HISTORY (brief):
{fall_history_brief or "No recent falls."}

AVAILABLE CONCEPT VOCABULARY (choose from these):
{available_concepts}

Respond ONLY with valid JSON:
{{
  "verbal": "<1-2 sentences describing what the robot is experiencing right now, first person>",
  "primary_concepts": ["<CONCEPT_NAME>", "<CONCEPT_NAME>"],
  "confidence": <0.0-1.0>
}}

Rules:
- verbal should sound like genuine inner speech: "I'm leaning too far back..."
- primary_concepts: 2-5 concepts from the vocabulary above, most relevant first
- Be specific to the actual numbers (e.g. "posture=0.43 means I'm unstable")
"""


def _build_guide_prompt(
    obs_summary: str,
    concept_str: str,
    curriculum_stage: str,
    fall_patterns: str,
    valid_intents: list[str],
) -> str:
    intents_str = ", ".join(valid_intents[:15])
    return f"""You are the strategic guidance system of a humanoid robot.
The robot has an internal voice: {concept_str}.
Current curriculum stage: {curriculum_stage}.

STATE:
{obs_summary}

FALL PATTERNS:
{fall_patterns or "No clear patterns yet."}

AVAILABLE INTENT VARIABLES: {intents_str}

Based on the robot's internal state and history, suggest adjustments.
Respond ONLY with valid JSON:
{{
  "verbal": "<inner speech about strategic intent>",
  "primary_concepts": ["<CONCEPT>"],
  "intent_adjustments": {{
    "<intent_var>": <float 0.0-1.0>
  }},
  "seeds": [
    {{"from_": "<var>", "to": "<var>", "weight": <0.1-0.5>}}
  ]
}}

Rules:
- intent_adjustments: 1-3 variables only, conservative changes (0.4-0.7 range)
- seeds: 0-3 causal hypotheses based on observed patterns
- If robot is falling/recovering: prioritize intent_stop_recover, reduce intent_stride
"""


def _build_lesson_prompt(
    full_context: str,
    total_ticks: int,
    total_falls: int,
) -> str:
    return f"""You are reviewing the learning session of a humanoid robot.
Total experience: {total_ticks} ticks, {total_falls} falls.

FULL SESSION CONTEXT:
{full_context}

Generate a structured lesson for what the robot learned and what to focus on next.
Respond ONLY with valid JSON:
{{
  "verbal": "<inner reflection, 2-3 sentences>",
  "lesson_text": "<structured lesson: what worked, what failed, next priority>",
  "primary_concepts": ["<CONCEPT>"],
  "lesson_concepts": ["<key concepts to reinforce>"],
  "intent_adjustments": {{}},
  "seeds": []
}}
"""


# ── Response parser ────────────────────────────────────────────────────────────
def _parse_teacher_response(
    raw: str,
    tick: int,
    mode: str,
    valid_intents: set[str],
    valid_graph_vars: set[str],
) -> TeacherAnnotation:
    from engine.llm_json_extract import parse_json_object_loose

    ann = TeacherAnnotation(tick=tick, timestamp=time.time(), mode=mode)
    obj = parse_json_object_loose(raw)
    if not obj:
        ann.error = "parse_failed"
        return ann

    ann.verbal = str(obj.get("verbal") or "").strip()[:300]
    ann.confidence = float(np.clip(float(obj.get("confidence", 1.0)), 0.0, 1.0))
    ann.lesson_text = str(obj.get("lesson_text") or "").strip()[:500]

    # Validate concepts
    valid_concept_names = {name for name, _, _ in CONCEPT_DEFS if not name.startswith("LATENT_")}
    ann.primary_concepts = [
        c for c in (obj.get("primary_concepts") or [])
        if isinstance(c, str) and c in valid_concept_names
    ][:5]
    ann.lesson_concepts = [
        c for c in (obj.get("lesson_concepts") or [])
        if isinstance(c, str) and c in valid_concept_names
    ][:8]

    # Validate intent adjustments
    for k, v in (obj.get("intent_adjustments") or {}).items():
        if str(k) in valid_intents:
            try:
                ann.intent_adjustments[str(k)] = float(np.clip(float(v), 0.05, 0.95))
            except (TypeError, ValueError):
                pass

    # Validate seeds
    for s in (obj.get("seeds") or []):
        if not isinstance(s, dict):
            continue
        fr = str(s.get("from_") or s.get("from") or "")
        to = str(s.get("to") or "")
        if fr in valid_graph_vars and to in valid_graph_vars and fr != to:
            try:
                w = float(np.clip(float(s.get("weight", 0.3)), 0.05, 0.6))
                ann.seeds.append({"from_": fr, "to": to, "weight": w, "alpha": 0.06})
            except (TypeError, ValueError):
                pass

    return ann


# ── LLM Voice Teacher ─────────────────────────────────────────────────────────
class LLMVoiceTeacher:
    """
    Offline LLM teacher for InnerVoiceNet distillation.

    НЕ работает в real-time. Запускается async раз в 60-120 секунд.
    Результаты кэшируются и применяются при следующем τ3 цикле.

    Режимы (escalating):
      ANNOTATE → GUIDE → LESSON (цикл)
    """

    def __init__(self):
        self._every_sec = _env_float("RKK_LLM_TEACHER_EVERY_SEC", 90.0)
        self._guide_every = _env_int("RKK_LLM_TEACHER_GUIDE_EVERY", 3)
        self._lesson_every = _env_int("RKK_LLM_TEACHER_LESSON_EVERY", 10)
        self._timeout = _env_float("RKK_LLM_TEACHER_TIMEOUT", 30.0)

        self._last_call_time: float = -9999.0
        self._call_count: int = 0
        self._pending: bool = False

        # Cached results
        self._annotations: deque[TeacherAnnotation] = deque(maxlen=20)
        self._latest: TeacherAnnotation | None = None

        # Callbacks
        self._on_annotation: list[Callable[[TeacherAnnotation], None]] = []

        self.total_calls: int = 0
        self.total_errors: int = 0

    def add_callback(self, fn: Callable[[TeacherAnnotation], None]) -> None:
        """Register callback for new annotations."""
        self._on_annotation.append(fn)

    def should_call(self) -> bool:
        if not teacher_enabled():
            return False
        if self._pending:
            return False
        return (time.time() - self._last_call_time) >= self._every_sec

    def get_mode(self) -> str:
        if self._call_count % self._lesson_every == self._lesson_every - 1:
            return "lesson"
        if self._call_count % self._guide_every == self._guide_every - 1:
            return "guide"
        return "annotate"

    async def call_async(
        self,
        tick: int,
        obs: dict[str, float],
        inner_voice_controller,   # InnerVoiceController
        episodic_memory=None,
        curriculum=None,
        llm_url: str = "",
        llm_model: str = "",
        valid_intents: list[str] | None = None,
        valid_graph_vars: list[str] | None = None,
        total_ticks: int = 0,
        total_falls: int = 0,
    ) -> TeacherAnnotation | None:
        """
        Async LLM call. Non-blocking — fire from asyncio.ensure_future().
        """
        if not teacher_enabled() or self._pending:
            return None

        self._pending = True
        self._last_call_time = time.time()
        mode = self.get_mode()

        try:
            ann = await self._do_call(
                tick=tick,
                mode=mode,
                obs=obs,
                inner_voice_controller=inner_voice_controller,
                episodic_memory=episodic_memory,
                curriculum=curriculum,
                llm_url=llm_url,
                llm_model=llm_model,
                valid_intents=set(valid_intents or []),
                valid_graph_vars=set(valid_graph_vars or []),
                total_ticks=total_ticks,
                total_falls=total_falls,
            )
            if ann is not None and not ann.error:
                self._annotations.append(ann)
                self._latest = ann
                self.total_calls += 1
                self._call_count += 1
                for cb in self._on_annotation:
                    try:
                        cb(ann)
                    except Exception:
                        pass
            return ann
        except Exception as e:
            self.total_errors += 1
            print(f"[LLMTeacher] call failed: {e!r}")
            return None
        finally:
            self._pending = False

    async def _do_call(
        self,
        tick: int,
        mode: str,
        obs: dict[str, float],
        inner_voice_controller,
        episodic_memory,
        curriculum,
        llm_url: str,
        llm_model: str,
        valid_intents: set[str],
        valid_graph_vars: set[str],
        total_ticks: int,
        total_falls: int,
    ) -> TeacherAnnotation | None:
        from engine.ollama_env import get_ollama_generate_url, get_ollama_model, ollama_think_disabled_payload

        url = (llm_url or get_ollama_generate_url()).strip()
        model = (llm_model or get_ollama_model()).strip()
        if not url:
            return None
        if not url.endswith("/generate"):
            url = (url.rstrip("/") + "/api/generate") if "/api/" not in url else url

        # Build obs summary
        obs_summary = self._format_obs(obs)
        concept_str = inner_voice_controller.get_concept_str() if inner_voice_controller else "UNKNOWN"
        fall_history = ""
        if episodic_memory is not None:
            fall_history = episodic_memory.get_llm_context_block(max_falls=3, max_successes=1)
        curriculum_stage = ""
        if curriculum is not None:
            curriculum_stage = curriculum.current_stage.name

        # Choose prompt by mode
        if mode == "annotate":
            prompt = _build_annotate_prompt(obs_summary, concept_str, fall_history)
        elif mode == "guide":
            fall_patterns = ""
            if episodic_memory is not None and episodic_memory._patterns:
                fall_patterns = "\n".join(p.description for p in episodic_memory._patterns[:3])
            prompt = _build_guide_prompt(
                obs_summary, concept_str, curriculum_stage,
                fall_patterns, list(valid_intents)[:15]
            )
        else:  # lesson
            full_ctx = f"{obs_summary}\n\n{fall_history}"
            prompt = _build_lesson_prompt(full_ctx, total_ticks, total_falls)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **ollama_think_disabled_payload(),
            "options": {"temperature": 0.2, "num_predict": 400},
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            raw = (resp.json().get("response") or "").strip()

        return _parse_teacher_response(raw, tick, mode, valid_intents, valid_graph_vars)

    def _format_obs(self, obs: dict[str, float]) -> str:
        keys = [
            "posture_stability", "com_z", "com_x", "torso_pitch",
            "foot_contact_l", "foot_contact_r", "gait_phase_l", "gait_phase_r",
            "support_bias", "intent_stride", "intent_torso_forward",
            "proprio_balance", "proprio_anomaly", "proprio_empowerment",
        ]
        lines = []
        for k in keys:
            v = obs.get(k, obs.get(f"phys_{k}", None))
            if v is not None:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    fv = 0.0
                lines.append(f"  {k}={fv:.3f}")
        return "\n".join(lines)

    def get_latest(self) -> TeacherAnnotation | None:
        return self._latest

    def snapshot(self) -> dict[str, Any]:
        latest = self._latest
        return {
            "enabled": teacher_enabled(),
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "pending": self._pending,
            "every_sec": self._every_sec,
            "call_count": self._call_count,
            "latest_mode": latest.mode if latest else None,
            "latest_verbal": (latest.verbal[:80] if latest and latest.verbal else None),
            "latest_concepts": (latest.primary_concepts if latest else []),
            "latest_tick": (latest.tick if latest else None),
        }
