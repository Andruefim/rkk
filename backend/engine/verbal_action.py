"""
verbal_action.py — Phase L: Verbal Action (Variant C).

Речь как действие в пространстве действий агента.
Три типа высказываний с reward shaping:

  OBSERVE  — агент комментирует происходящее (проактивно)
  ASK      — запрашивает помощь у человека (Human-in-the-Loop)
  REPORT   — сообщает о достижении / провале

Reward структура:
  ASK при curiosity > 0.7  → +0.3 если человек отвечает
  ASK при curiosity < 0.3  → -0.1 (не спрашивай когда понятно)
  OBSERVE точное совпадение с исходом → +0.15
  REPORT подтверждённое достижение   → +0.20
  Спам (>3 сообщений за 100 тиков)   → -0.05 / сообщение

Генерация речи:
  При RKK_NEURAL_LANG=1 основной путь — NeuralLanguageGrounding (engine.neural_lang_integration).
  Ниже — короткие fallback-строки для cold start / отключённой нейросети.
  LLM для ASK — при RKK_SPEECH_LLM_ASK=1.

RKK_SPEECH_ENABLED=1
RKK_SPEECH_LANG=ru            — ru | en
RKK_SPEECH_OBSERVE_EVERY=150  — min тиков между OBSERVE
RKK_SPEECH_ASK_CURIOSITY=0.70 — порог curiosity для ASK
RKK_SPEECH_LLM_ASK=1         — использовать LLM для ASK (иначе шаблон)
RKK_SPEECH_SPAM_WINDOW=100    — тиков для спам-детектора
RKK_SPEECH_SPAM_MAX=3         — max сообщений в окне
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np


def speech_enabled() -> bool:
    return os.environ.get("RKK_SPEECH_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


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


# ── Message types ──────────────────────────────────────────────────────────────
class SpeechType(Enum):
    OBSERVE = auto()
    ASK     = auto()
    REPORT  = auto()


@dataclass
class AgentMessage:
    tick: int
    speech_type: SpeechType
    text: str
    concepts: list[str]
    curiosity: float
    posture: float
    timestamp: float = field(default_factory=time.time)

    # Reward tracking
    reward_pending: bool = False
    reward_received: float = 0.0
    human_replied: bool = False
    human_reply: str = ""
    outcome_matched: bool = False  # for OBSERVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": f"{self.tick}_{self.speech_type.name}",
            "tick": self.tick,
            "type": self.speech_type.name,
            "text": self.text,
            "concepts": self.concepts,
            "curiosity": round(self.curiosity, 3),
            "posture": round(self.posture, 3),
            "timestamp": self.timestamp,
            "human_replied": self.human_replied,
            "human_reply": self.human_reply,
            "reward": round(self.reward_received, 3),
        }


# ── Speech Decoder (minimal fallback; основной путь — neural_lang_integration) ─
class SpeechDecoder:
    """
    Converts thought_embedding + concepts + obs → natural language text.
    Основной путь: NeuralLanguageGrounding (engine.neural_lang_integration).
    Ниже — короткие строки для cold start; ASK может идти через LLM при RKK_SPEECH_LLM_ASK=1.
    """

    def __init__(self):
        self._lang = _env("RKK_SPEECH_LANG", "ru")

    def decode_observe(
        self,
        concepts: list[str],
        obs: dict[str, float],
        curiosity: float,
    ) -> str:
        """Fallback OBSERVE when neural decoder недоступен."""
        def g(k: str) -> float:
            return float(obs.get(k, obs.get(f"phys_{k}", 0.5)))

        concept_names = [c for c, _ in concepts] if concepts and isinstance(concepts[0], tuple) else concepts
        posture = g("posture_stability")
        fallen = g("com_z") < 0.35
        ru = self._lang == "ru"

        if fallen or "FALLEN" in concept_names or "FALL_BACKWARD" in concept_names or "FALL_FORWARD" in concept_names:
            return "Падаю." if ru else "Falling."
        if "LOSING_BALANCE" in concept_names or "LOW_STABILITY" in concept_names or posture < 0.5:
            return "Теряю равновесие." if ru else "Losing balance."
        if "OVERSTRIDE" in concept_names:
            return "Шаг слишком широкий." if ru else "Stride too wide."
        if curiosity > 0.65:
            return "Интересно, что будет дальше." if ru else "Curious what happens next."
        if "HIGH_STABILITY" in concept_names or "STABLE_BALANCE" in concept_names:
            return "Стою устойчиво." if ru else "Standing stable."
        if "IMPROVING" in concept_names:
            return "Становится лучше." if ru else "Getting better."
        if "PLATEAU" in concept_names:
            return "Застрял, почти без прогресса." if ru else "Stuck, little progress."
        return "Стою устойчиво." if ru else "Standing stable."

    def decode_report(
        self,
        report_type: str,
        ticks: int = 0,
        count: int = 0,
    ) -> str:
        ru = self._lang == "ru"
        if "STABLE" in report_type.upper():
            return f"Стабильно {ticks} тиков." if ru else f"Stable {ticks} ticks."
        if "SLEEP" in report_type.upper():
            return "Нужен отдых." if ru else "Need rest."
        return f"Упал {count} раз." if ru else f"Fallen {count} times."

    def decode_ask_template(self, concepts: list[str]) -> str:
        """Fallback ASK without LLM."""
        concept_names = [c for c, _ in concepts] if concepts and isinstance(concepts[0], tuple) else concepts
        ru = self._lang == "ru"
        if any(c in concept_names for c in ("LOSING_BALANCE", "HIGH_FALL_RISK", "FALLEN")):
            return "Как улучшить баланс?" if ru else "How can I improve balance?"
        if "OVERSTRIDE" in concept_names:
            return "Какой шаг безопаснее?" if ru else "What stride is safer?"
        return "Что делать дальше?" if ru else "What should I do next?"

    async def decode_ask_llm(
        self,
        concepts: list[str],
        obs: dict[str, float],
        fall_history_brief: str,
        llm_url: str,
        llm_model: str,
    ) -> str:
        """Generate ASK via LLM — a genuine question from the agent's perspective."""
        from engine.ollama_env import ollama_think_disabled_payload

        lang_hint = "Russian" if self._lang == "ru" else "English"
        concept_str = ", ".join(concepts[:4]) if concepts else "UNKNOWN"

        def g(k: str) -> float:
            return float(obs.get(k, obs.get(f"phys_{k}", 0.5)))

        prompt = f"""You are a humanoid robot learning to walk. You speak in first person.
Your current internal state: {concept_str}
posture={g('posture_stability'):.2f}, com_z={g('com_z'):.2f}, stride={g('intent_stride'):.2f}

Recent history: {fall_history_brief or 'no data'}

Generate ONE short genuine question to ask your human trainer.
- Language: {lang_hint}
- First person ("почему я..." / "why do I...")
- Specific to your current state (use the numbers)
- Max 15 words
- No punctuation except ?
- Raw text only, no quotes, no JSON

Question:"""

        url = llm_url.strip().rstrip("/")
        if not url.endswith("/generate"):
            url = url + "/api/generate" if "/api/" not in url else url

        payload = {
            "model": llm_model,
            "prompt": prompt,
            "stream": False,
            **ollama_think_disabled_payload(),
            "options": {"temperature": 0.5, "num_predict": 40},
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    raw = (resp.json().get("response") or "").strip()
                    # Clean up: remove quotes, extra punctuation
                    raw = raw.strip('"\'').split("\n")[0].strip()
                    if raw and len(raw) > 3:
                        return raw
        except Exception:
            pass

        return self.decode_ask_template(concepts)


# ── Verbal Action Controller ───────────────────────────────────────────────────
class VerbalActionController:
    """
    Полный контроллер вербальных действий.

    Решает когда говорить, что говорить, и считает reward.

    Интеграция в simulation.py:
      self._verbal = VerbalActionController()

    В тик-цикле:
      msg = await self._verbal.tick(tick, obs, inner_voice, fallen, sim)
      if msg:
          self._broadcast_agent_message(msg)
          self._add_event(f"🗣 {msg.text}", "#88ffcc", "speech")

    При ответе человека:
      self._verbal.on_human_reply(reply_text)
    """

    def __init__(self):
        self.decoder = SpeechDecoder()
        self._observe_every = _env_int("RKK_SPEECH_OBSERVE_EVERY", 150)
        self._ask_curiosity_threshold = _env_float("RKK_SPEECH_ASK_CURIOSITY", 0.70)
        self._spam_window = _env_int("RKK_SPEECH_SPAM_WINDOW", 100)
        self._spam_max = _env_int("RKK_SPEECH_SPAM_MAX", 3)
        self._use_llm_ask = _env("RKK_SPEECH_LLM_ASK", "1") in ("1", "true", "yes")

        self._last_observe_tick: int = -999
        self._last_ask_tick: int = -999
        self._last_report_tick: int = -999

        self._messages: deque[AgentMessage] = deque(maxlen=200)
        self._pending_ask: AgentMessage | None = None
        self._recent_ticks: deque[int] = deque(maxlen=50)

        # State tracking for REPORT
        self._stable_walk_since: int | None = None
        self._last_fall_count_reported: int = 0

        # Reward accumulator
        self._total_verbal_reward: float = 0.0
        self.total_messages: int = 0
        self.total_asks: int = 0
        self.total_replies_received: int = 0

        # Callbacks for UI broadcast
        self._on_message: list[Callable[[AgentMessage], None]] = []

    def add_callback(self, fn: Callable[[AgentMessage], None]) -> None:
        self._on_message.append(fn)

    def _is_spam(self, tick: int) -> bool:
        """Check if too many messages in recent window."""
        cutoff = tick - self._spam_window
        recent = [t for t in self._recent_ticks if t >= cutoff]
        return len(recent) >= self._spam_max

    def _pick_speech_type(
        self,
        tick: int,
        curiosity: float,
        fallen: bool,
        total_falls: int,
        posture: float,
        stable_ticks: int,
    ) -> SpeechType | None:
        """Decide what type of speech to produce this tick, if any."""
        if self._is_spam(tick):
            return None

        # OBSERVE on fall event
        if fallen and (tick - self._last_observe_tick) > 20:
            return SpeechType.OBSERVE

        # ASK when curiosity is high
        if (
            curiosity >= self._ask_curiosity_threshold
            and (tick - self._last_ask_tick) > 300
            and not fallen
        ):
            return SpeechType.ASK

        # REPORT stable walk milestone
        if stable_ticks > 0 and stable_ticks % 200 == 0 and (tick - self._last_report_tick) > 200:
            return SpeechType.REPORT

        # REPORT fall count milestone
        falls_since = total_falls - self._last_fall_count_reported
        if falls_since >= 10 and (tick - self._last_report_tick) > 500:
            return SpeechType.REPORT

        # OBSERVE periodically
        if (tick - self._last_observe_tick) >= self._observe_every:
            return SpeechType.OBSERVE

        return None

    async def tick(
        self,
        tick: int,
        obs: dict[str, float],
        inner_voice_ctrl,
        fallen: bool,
        total_falls: int,
        llm_url: str = "",
        llm_model: str = "",
        fall_history_brief: str = "",
    ) -> AgentMessage | None:
        """
        Main tick. Returns AgentMessage if agent speaks, else None.
        """
        if not speech_enabled():
            return None

        # Get inner voice state
        concepts = inner_voice_ctrl.get_active_concepts() if inner_voice_ctrl else []
        curiosity = 0.5
        if inner_voice_ctrl:
            # Get curiosity from last reward signal if available
            concept_names = [n for n, _ in concepts]
            if "HIGH_CURIOSITY" in concept_names:
                curiosity = 0.8
            elif "LOW_CURIOSITY" in concept_names:
                curiosity = 0.2

        def g(k: str) -> float:
            return float(obs.get(k, obs.get(f"phys_{k}", 0.5)))

        posture = g("posture_stability")

        # Track stable walk
        if posture > 0.65 and not fallen:
            if self._stable_walk_since is None:
                self._stable_walk_since = tick
        else:
            self._stable_walk_since = None
        stable_ticks = (tick - self._stable_walk_since) if self._stable_walk_since else 0

        # Decide speech type
        speech_type = self._pick_speech_type(
            tick, curiosity, fallen, total_falls, posture, stable_ticks
        )
        if speech_type is None:
            return None

        # Generate text
        concept_names = [n for n, _ in concepts]
        text = await self._generate_text(
            speech_type, concept_names, concepts, obs, curiosity,
            stable_ticks, total_falls, fall_history_brief, llm_url, llm_model,
        )
        if not text:
            return None

        # Create message
        msg = AgentMessage(
            tick=tick,
            speech_type=speech_type,
            text=text,
            concepts=concept_names[:5],
            curiosity=curiosity,
            posture=posture,
        )

        # Reward pending for ASK
        if speech_type == SpeechType.ASK:
            msg.reward_pending = True
            self._pending_ask = msg
            self._last_ask_tick = tick
            self.total_asks += 1
        elif speech_type == SpeechType.OBSERVE:
            self._last_observe_tick = tick
        elif speech_type == SpeechType.REPORT:
            self._last_report_tick = tick
            if total_falls - self._last_fall_count_reported >= 10:
                self._last_fall_count_reported = total_falls

        self._messages.append(msg)
        self._recent_ticks.append(tick)
        self.total_messages += 1

        # Fire callbacks
        for cb in self._on_message:
            try:
                cb(msg)
            except Exception:
                pass

        # Spam penalty
        if self._is_spam(tick):
            self._total_verbal_reward -= 0.05

        return msg

    async def _generate_text(
        self,
        speech_type: SpeechType,
        concept_names: list[str],
        concepts: list,
        obs: dict,
        curiosity: float,
        stable_ticks: int,
        total_falls: int,
        fall_history_brief: str,
        llm_url: str,
        llm_model: str,
    ) -> str:
        if speech_type == SpeechType.OBSERVE:
            return self.decoder.decode_observe(concept_names, obs, curiosity)

        elif speech_type == SpeechType.REPORT:
            falls_since = total_falls - self._last_fall_count_reported
            if stable_ticks > 100:
                return self.decoder.decode_report("STABLE_WALK", ticks=stable_ticks)
            elif falls_since >= 10:
                return self.decoder.decode_report("FALL_COUNT", count=total_falls)
            else:
                return self.decoder.decode_report("STABLE_WALK", ticks=stable_ticks)

        elif speech_type == SpeechType.ASK:
            if self._use_llm_ask and llm_url:
                return await self.decoder.decode_ask_llm(
                    concept_names, obs, fall_history_brief, llm_url, llm_model
                )
            return self.decoder.decode_ask_template(concept_names)

        return ""

    def on_human_reply(self, reply_text: str) -> float:
        """
        Human replied to agent's message.
        Returns reward given to agent.
        """
        self.total_replies_received += 1

        # Find pending ASK
        pending = self._pending_ask
        if pending is not None and pending.reward_pending:
            pending.human_replied = True
            pending.human_reply = reply_text
            pending.reward_pending = False

            # Reward: was curiosity high when asked?
            if pending.curiosity >= self._ask_curiosity_threshold:
                reward = 0.30
            elif pending.curiosity >= 0.5:
                reward = 0.15
            else:
                reward = -0.05  # asked when didn't need to

            pending.reward_received = reward
            self._total_verbal_reward += reward
            self._pending_ask = None
            return reward

        # Reply to non-pending message — small positive
        self._total_verbal_reward += 0.05
        return 0.05

    def on_observe_outcome(self, fell_as_predicted: bool) -> float:
        """Call after OBSERVE to check if prediction matched outcome."""
        recent_observes = [m for m in self._messages if m.speech_type == SpeechType.OBSERVE]
        if not recent_observes:
            return 0.0
        last = recent_observes[-1]
        if fell_as_predicted and ("Falling" in last.text or "Падаю" in last.text):
            last.outcome_matched = True
            last.reward_received = 0.15
            self._total_verbal_reward += 0.15
            return 0.15
        return 0.0

    def get_messages_for_ui(self, last_n: int = 50) -> list[dict]:
        return [m.to_dict() for m in list(self._messages)[-last_n:]]

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": speech_enabled(),
            "total_messages": self.total_messages,
            "total_asks": self.total_asks,
            "total_replies": self.total_replies_received,
            "total_verbal_reward": round(self._total_verbal_reward, 4),
            "pending_ask": self._pending_ask is not None,
            "last_message": (
                self._messages[-1].to_dict() if self._messages else None
            ),
        }
