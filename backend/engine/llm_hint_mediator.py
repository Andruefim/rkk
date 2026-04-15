"""
llm_hint_mediator.py — LLM → CausalGraph Mediator Discovery.

Проблема (из лога):
  "[LLM L2] missing mediator that maps high-level intent to low-level
  motor drive and subsequent physical response"

LLM уже знает где пробел в графе. Но никто его не слушает.
Этот модуль парсит LLM объяснения и автоматически создаёт узлы.

Алгоритм:
  1. Каждый раз когда phase3_teacher.py / llm_curriculum.py генерирует
     explanation → парсим его на "missing X between A and B"
  2. Если LLM говорит next_probe="perturb X" → это кандидат на новый узел
  3. Если LLM перечисляет edges которых нет в графе → добавляем их
  4. NeurogenesisEngine создаёт latent node с нужными связями

RKK_LLM_MEDIATOR_ENABLED=1
RKK_LLM_MEDIATOR_CONFIDENCE=0.25   — минимальный вес ребра из LLM edges
RKK_LLM_MEDIATOR_COOLDOWN=200      — тиков между созданиями узлов
"""
from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _mediator_enabled() -> bool:
    return os.environ.get("RKK_LLM_MEDIATOR_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

def _ef(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)))
    except ValueError: return default

def _ei(key: str, default: int) -> int:
    try: return max(1, int(os.environ.get(key, str(default))))
    except ValueError: return default


# ─── Parses ───────────────────────────────────────────────────────────────────

@dataclass
class MediatorHint:
    """Всё что LLM нам сказал об одной missing связи."""
    source_var: str                    # откуда
    target_var: str                    # куда
    mediator_name: str                 # предложенное имя нового узла
    edge_weight: float                 # сила связи (0-1)
    confidence: float = 0.5           # уверенность парсера
    explanation_text: str = ""        # оригинальный текст
    next_probe_var: str | None = None # что LLM предлагает потрогать
    tick: int = 0


class LLMHintParser:
    """
    Парсит текстовые объяснения LLM в структурированные MediatorHint.

    LLM из лога говорит:
      explanation: "missing mediator ... intent_support_left ... foot_contact_l"
      next_probe: "Perturb phys_intent_support_left ..."
      edges: ['phys_intent_support_left->phys_foot_contact_l(+0.27)', ...]

    Мы извлекаем:
      source=phys_intent_support_left
      target=phys_foot_contact_l
      weight=0.27
      mediator_name=latent_intent_motor_l  (генерируем)
    """

    # Паттерны для парсинга
    EDGE_PATTERN = re.compile(
        r"([\w_]+)->([\w_]+)\(([\+\-]?\d+\.\d+)\)"
    )
    MEDIATOR_PATTERN = re.compile(
        r"missing\s+(?:the\s+)?mediator\s+(?:that\s+maps\s+)?([a-z_\s]+?)"
        r"(?:\s+to\s+|\s+and\s+)([a-z_\s]+)",
        re.IGNORECASE
    )
    PERTURB_PATTERN = re.compile(
        r"[Pp]erturb\s+([\w_]+)",
        re.IGNORECASE
    )
    STAGNATION_PATTERN = re.compile(
        r"(?:stagnation|over.?fitting|static state|not improving|no progress)",
        re.IGNORECASE
    )

    def __init__(self):
        self._min_confidence = _ef("RKK_LLM_MEDIATOR_CONFIDENCE", 0.25)
        self._parsed_hints: deque[MediatorHint] = deque(maxlen=100)
        self.total_parsed: int = 0

    def parse(
        self,
        explanation: str,
        edges_preview: list[str],
        next_probe: str | None,
        tick: int,
    ) -> list[MediatorHint]:
        """
        Основной вход. Принимает то что приходит из LLM phase3_teacher.
        Возвращает список MediatorHint для создания узлов.
        """
        if not _mediator_enabled():
            return []

        hints: list[MediatorHint] = []

        # 1. Парсим edges_preview — самый надёжный источник
        for edge_str in (edges_preview or []):
            m = self.EDGE_PATTERN.match(edge_str.strip())
            if not m:
                continue
            src, tgt, w_str = m.group(1), m.group(2), m.group(3)
            weight = abs(float(w_str))
            if weight < self._min_confidence:
                continue

            # Генерируем имя медиатора из пары переменных
            mediator_name = self._make_mediator_name(src, tgt)

            hint = MediatorHint(
                source_var=src,
                target_var=tgt,
                mediator_name=mediator_name,
                edge_weight=weight,
                confidence=weight,
                explanation_text=explanation,
                next_probe_var=self._extract_probe_var(next_probe or ""),
                tick=tick,
            )
            hints.append(hint)

        # 2. Если LLM явно говорит "missing mediator" — повышаем confidence
        mediator_match = self.MEDIATOR_PATTERN.search(explanation or "")
        if mediator_match and hints:
            for h in hints:
                h.confidence = min(1.0, h.confidence + 0.2)

        # 3. Если LLM говорит "stagnation" — значит точно нужен новый узел
        is_stagnant = bool(self.STAGNATION_PATTERN.search(explanation or ""))
        if is_stagnant and hints:
            for h in hints:
                h.confidence = min(1.0, h.confidence + 0.15)

        # Фильтруем по confidence
        hints = [h for h in hints if h.confidence >= self._min_confidence]

        for h in hints:
            self._parsed_hints.append(h)
            self.total_parsed += 1

        return hints

    def _make_mediator_name(self, src: str, tgt: str) -> str:
        """Генерирует осмысленное имя для медиирующего узла."""
        # Убираем префиксы phys_/proprio_/mc_ для краткого имени
        def strip_prefix(s: str) -> str:
            for p in ("phys_", "proprio_", "mc_", "l1_", "l0_"):
                if s.startswith(p):
                    return s[len(p):]
            return s

        s = strip_prefix(src)
        t = strip_prefix(tgt)

        # Если оба intent → motor, используем паттерн
        if "intent" in src and ("contact" in tgt or "motor" in tgt or "gait" in tgt):
            side = "l" if "_l" in src or "_left" in src else "r" if "_r" in src or "_right" in src else ""
            return f"latent_intent_motor_{side}" if side else "latent_intent_motor"
        if "cpg" in src or "gait" in src:
            return f"latent_cpg_{strip_prefix(tgt)[:12]}"

        # Общий случай
        return f"latent_{s[:8]}_{t[:8]}"

    def _extract_probe_var(self, probe_text: str) -> str | None:
        m = self.PERTURB_PATTERN.search(probe_text)
        return m.group(1) if m else None

    def get_recent(self, n: int = 5) -> list[MediatorHint]:
        return list(self._parsed_hints)[-n:]

    def snapshot(self) -> dict[str, Any]:
        recent = self.get_recent(3)
        return {
            "total_parsed": self.total_parsed,
            "recent_hints": [
                {
                    "mediator": h.mediator_name,
                    "edge": f"{h.source_var}->{h.target_var}",
                    "weight": round(h.edge_weight, 3),
                    "confidence": round(h.confidence, 3),
                }
                for h in recent
            ],
        }


# ─── MediatorNeurogenesis ─────────────────────────────────────────────────────

class MediatorNeurogenesis:
    """
    Создаёт latent узлы в каузальном графе на основе MediatorHint.

    Что происходит с новым узлом:
      1. Добавляется в graph с начальным значением 0.5
      2. Создаются рёбра src→latent и latent→tgt с весами из LLM
      3. GNN получает новый узел — начинает его обучать
      4. Агент может через него интервенировать

    Критично: узел НЕ ЗАХАРДКОЖЕН. Его имя берётся из MediatorHint.
    Через 100-200 тиков GNN сам решит нужен ли он (если не улучшает
    compression — атрофируется через neurogenesis pruning).
    """

    def __init__(self):
        self._cooldown = _ei("RKK_LLM_MEDIATOR_COOLDOWN", 200)
        self._last_creation_tick: dict[str, int] = {}
        self._created_nodes: deque[dict[str, Any]] = deque(maxlen=50)
        self.total_created: int = 0

    def maybe_create(
        self,
        hint: MediatorHint,
        graph,
        tick: int,
    ) -> dict[str, Any] | None:
        """
        Создаёт медиирующий узел если:
          - Ещё не создан
          - Cooldown истёк
          - Оба src и tgt существуют в графе

        Возвращает событие создания или None.
        """
        if not _mediator_enabled():
            return None

        name = hint.mediator_name

        # Уже существует?
        if name in graph._node_ids:
            # Но проверим ребро — может его нет
            self._ensure_edges(hint, graph)
            return None

        # Cooldown
        last = self._last_creation_tick.get(name, -9999)
        if (tick - last) < self._cooldown:
            return None

        # Проверяем что src и tgt есть
        nodes = graph._node_ids
        if hint.source_var not in nodes or hint.target_var not in nodes:
            # Пробуем частичное совпадение
            src = self._fuzzy_match(hint.source_var, nodes)
            tgt = self._fuzzy_match(hint.target_var, nodes)
            if src is None or tgt is None:
                return None
            hint.source_var = src
            hint.target_var = tgt

        # Создаём узел
        try:
            graph.add_latent_node(
                name=name,
                initial_value=0.5,
                parents=[(hint.source_var, hint.edge_weight)],
                children=[(hint.target_var, hint.edge_weight)],
            )
        except AttributeError:
            # Fallback если add_latent_node не реализован
            graph._node_ids.add(name)
            graph._nodes[name] = 0.5
            self._ensure_edges(hint, graph)

        self._last_creation_tick[name] = tick
        self.total_created += 1

        event = {
            "tick": tick,
            "node": name,
            "source": hint.source_var,
            "target": hint.target_var,
            "weight": round(hint.edge_weight, 3),
            "confidence": round(hint.confidence, 3),
            "llm_triggered": True,
        }
        self._created_nodes.append(event)
        print(f"[MediatorNeuro] Created latent node '{name}' "
              f"({hint.source_var}→{name}→{hint.target_var}, w={hint.edge_weight:.3f})")
        return event

    def _ensure_edges(self, hint: MediatorHint, graph) -> None:
        """Добавляет рёбра если не существуют."""
        try:
            graph.add_edge_if_missing(hint.source_var, hint.mediator_name, hint.edge_weight)
            graph.add_edge_if_missing(hint.mediator_name, hint.target_var, hint.edge_weight)
        except Exception:
            pass

    def _fuzzy_match(self, name: str, nodes: set[str]) -> str | None:
        """Ищет ближайший узел по подстроке."""
        # Точное совпадение
        if name in nodes:
            return name
        # Частичное
        for n in nodes:
            if name in n or n in name:
                return n
        # По суффиксу (убираем prefix)
        suffix = name.split("_", 1)[-1] if "_" in name else name
        for n in nodes:
            if n.endswith(suffix) or suffix in n:
                return n
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_created": self.total_created,
            "recent": list(self._created_nodes)[-3:],
        }


# ─── Unified Controller ────────────────────────────────────────────────────────

class LLMMediatorController:
    """
    Объединяет Parser + Neurogenesis.
    Вешается на phase3_teacher callback.
    """

    def __init__(self):
        self.parser = LLMHintParser()
        self.neurogenesis = MediatorNeurogenesis()
        self.total_nodes_from_llm: int = 0

    def on_llm_output(
        self,
        explanation: str,
        edges_preview: list[str],
        next_probe: str | None,
        graph,
        tick: int,
    ) -> list[dict[str, Any]]:
        """
        Главный вход. Вызывать из phase3_teacher.py после каждого LLM ответа.

        Возвращает список событий создания узлов.
        """
        hints = self.parser.parse(
            explanation=explanation,
            edges_preview=edges_preview,
            next_probe=next_probe,
            tick=tick,
        )

        events = []
        for hint in hints:
            ev = self.neurogenesis.maybe_create(hint, graph, tick)
            if ev is not None:
                events.append(ev)
                self.total_nodes_from_llm += 1

        return events

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_nodes_from_llm": self.total_nodes_from_llm,
            "parser": self.parser.snapshot(),
            "neurogenesis": self.neurogenesis.snapshot(),
        }


# ─── Patch ────────────────────────────────────────────────────────────────────

def apply_llm_mediator_patch(sim) -> bool:
    """
    Патч: вешается на phase3_teacher и llm_curriculum
    для автоматического создания медиирующих узлов.

    Вызов: apply_llm_mediator_patch(sim)
    """
    if not _mediator_enabled():
        print("[LLMMediator] Disabled (RKK_LLM_MEDIATOR_ENABLED=0)")
        return False

    controller = LLMMediatorController()
    sim._llm_mediator = controller

    # Патч phase3_teacher
    teacher = getattr(sim, "_phase3_teacher", None)
    if teacher is not None:
        original_process = teacher.process_llm_response

        def patched_process(response: dict, tick: int) -> dict:
            result = original_process(response, tick)

            explanation = response.get("explanation", "")
            edges = response.get("edges", [])
            if isinstance(edges, list) and edges and isinstance(edges[0], str):
                edges_preview = edges
            else:
                edges_preview = response.get("preview", [])
            next_probe = response.get("next_probe", "")

            events = controller.on_llm_output(
                explanation=explanation,
                edges_preview=edges_preview,
                next_probe=next_probe,
                graph=sim.agent.graph,
                tick=tick,
            )

            for ev in events:
                sim._add_event(
                    f"🔬 LLMMediator: +node '{ev['node']}' "
                    f"({ev['source']}→{ev['target']}, w={ev['weight']})",
                    "#ff8844", "discovery"
                )

            return result

        teacher.process_llm_response = patched_process
        print("[LLMMediator] Hooked into phase3_teacher.process_llm_response")

    # Патч llm_curriculum если есть
    curriculum = getattr(sim, "_curriculum", None)
    if curriculum is not None and hasattr(curriculum, "_on_llm_explanation"):
        original_explain = curriculum._on_llm_explanation

        def patched_explain(text: str, edges: list, tick: int):
            original_explain(text, edges, tick)
            controller.on_llm_output(
                explanation=text,
                edges_preview=edges,
                next_probe=None,
                graph=sim.agent.graph,
                tick=tick,
            )

        curriculum._on_llm_explanation = patched_explain

    return True
