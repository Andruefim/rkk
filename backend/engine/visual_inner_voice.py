"""
visual_inner_voice.py — Phase M: Visual Inner Voice.

Расширяет InnerVoiceController визуальным потоком:
  1. SlotLabeler → visual concepts → InnerVoiceNet
  2. Визуальные концепты инжектируются в GNN отдельным prefixом concept_vis_*
  3. VerbalAction получает world_description для контекстных высказываний
  4. LLM Teacher получает полный визуальный контекст для аннотаций

Ключевое изменение verbal_action.py:
  OBSERVE раньше смотрел только на bodily concepts
  OBSERVE теперь смотрит на visual + bodily → выбирает наиболее релевантный

Новые шаблоны (добавляются к существующим в verbal_action.py):
  OBSERVE_OBJECT_AHEAD   → "Впереди что-то. Нужно решить как обойти."
  OBSERVE_RAMP_DETECTED  → "Вижу наклонную поверхность. Попробую подойти."
  OBSERVE_OPEN_SCENE     → "Путь открыт. Продолжаю."
  OBSERVE_NOVEL_OBJECT   → "Что-то новое. Не видел раньше."
  ASK_OBJECT             → "Что это за объект передо мной?"
  ASK_NAVIGATE           → "Как мне лучше обойти это препятствие?"

GNN инъекция:
  concept_vis_ramp_detected    = 0.85
  concept_vis_object_blocking  = 0.72
  concept_vis_clear_path       = 0.80
  → каузальная модель теперь знает о препятствиях

RKK_VISUAL_VOICE_ENABLED=1
RKK_VISUAL_VOICE_EVERY=30      — тиков между visual concept updates
RKK_VISUAL_INJECT_TOP=5        — топ-N визуальных концептов в GNN
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np

from engine.visual_concepts import VISUAL_CONCEPT_NAMES, VISUAL_BY_DOMAIN


def visual_voice_enabled() -> bool:
    return os.environ.get("RKK_VISUAL_VOICE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


# ── Visual concept → GNN node name ────────────────────────────────────────────
def concept_to_gnn_node(concept_name: str) -> str:
    return f"concept_vis_{concept_name.lower()}"


# ── Visual verbal templates (Russian) ─────────────────────────────────────────
VISUAL_TEMPLATES_RU: dict[str, list[str]] = {
    # Object detection
    "OBSERVE_OBJECT_AHEAD": [
        "Впереди объект. Нужно обойти или перешагнуть.",
        "Вижу что-то на пути.",
        "Путь заблокирован. Думаю как пройти.",
    ],
    "OBSERVE_OBJECT_CLOSE": [
        "Что-то совсем рядом.",
        "Объект в непосредственной близости.",
        "Очень близко к чему-то.",
    ],
    "OBSERVE_OBJECT_LEFT": [
        "Что-то слева.",
        "Объект с левой стороны.",
    ],
    "OBSERVE_OBJECT_RIGHT": [
        "Что-то справа.",
        "Объект справа от меня.",
    ],
    "OBSERVE_RAMP_DETECTED": [
        "Вижу наклонную поверхность.",
        "Рампа. Интересно, смогу ли я подняться.",
        "Поверхность под углом. Осторожно.",
    ],
    "OBSERVE_STAIRS_DETECTED": [
        "Ступени. Сложно для меня пока.",
        "Лестница впереди.",
        "Вижу ступени. Нужно попробовать.",
    ],
    "OBSERVE_BEAM_DETECTED": [
        "Узкая балка. Сложный баланс.",
        "Балка. Нужно очень точно двигаться.",
    ],
    "OBSERVE_PLATFORM_DETECTED": [
        "Платформа. Можно попробовать залезть.",
        "Возвышение впереди.",
    ],
    "OBSERVE_BALL_DETECTED": [
        "Шар. Могу попробовать его толкнуть.",
        "Круглый объект. Он подвижный.",
    ],
    "OBSERVE_OPEN_SCENE": [
        "Путь открыт. Продолжаю движение.",
        "Свободное пространство впереди.",
        "Ничего не мешает.",
    ],
    "OBSERVE_CLEAR_PATH": [
        "Путь свободен.",
        "Ничего не блокирует путь.",
    ],
    "OBSERVE_CLUTTERED": [
        "Много объектов вокруг. Сложная навигация.",
        "Загромождено. Нужно быть аккуратным.",
    ],
    "OBSERVE_NOVEL_OBJECT": [
        "Что-то незнакомое. Раньше не видел.",
        "Новый объект в поле зрения. Интересно.",
        "Это я вижу впервые.",
    ],
    "OBSERVE_SENSOR_PLATE": [
        "Светящаяся плитка. Интересно что это.",
        "Сенсорная платформа. Попробую встать.",
    ],
    "OBSERVE_WALL_NEARBY": [
        "Близко стена. Нужно повернуть.",
        "Стена рядом. Осторожно.",
    ],
    "OBSERVE_MOVING_OBJECT": [
        "Что-то движется.",
        "Объект в движении. Слежу.",
    ],
    "OBSERVE_I_MOVED_IT": [
        "Я это сдвинул!",
        "Объект отреагировал на моё действие.",
        "Я могу влиять на окружение.",
    ],
    # ASK — world-aware questions
    "ASK_OBJECT_UNKNOWN": [
        "Что это за объект передо мной?",
        "Что это такое?",
        "Не понимаю что я вижу. Можешь объяснить?",
    ],
    "ASK_HOW_TO_NAVIGATE": [
        "Как мне лучше обойти это?",
        "Как пройти мимо этого объекта?",
        "Что мне делать с этим препятствием?",
    ],
    "ASK_CAN_CLIMB": [
        "Я смогу залезть на эту поверхность?",
        "Эта наклонная поверхность безопасна для меня?",
    ],
    "ASK_WHAT_IS_GOAL": [
        "Куда мне нужно идти?",
        "Есть ли цель в этом пространстве?",
    ],
}

VISUAL_TEMPLATES_EN: dict[str, list[str]] = {
    "OBSERVE_OBJECT_AHEAD": [
        "Something ahead. Need to decide: go around or step over.",
        "Object in my path.",
        "Path is blocked. Thinking.",
    ],
    "OBSERVE_OBJECT_CLOSE": [
        "Something very close.",
        "Object nearby.",
    ],
    "OBSERVE_OBJECT_LEFT": [
        "Something to my left.",
        "Object on the left side.",
    ],
    "OBSERVE_OBJECT_RIGHT": [
        "Something to my right.",
        "Object on the right.",
    ],
    "OBSERVE_RAMP_DETECTED": [
        "I see an inclined surface.",
        "Ramp ahead. Can I climb it?",
        "Angled surface. Be careful.",
    ],
    "OBSERVE_STAIRS_DETECTED": [
        "Stairs. Hard for me right now.",
        "Steps ahead.",
    ],
    "OBSERVE_BEAM_DETECTED": [
        "Narrow beam. Challenging balance.",
        "Balance beam ahead.",
    ],
    "OBSERVE_PLATFORM_DETECTED": [
        "Platform. Could try to get up there.",
        "Elevated surface ahead.",
    ],
    "OBSERVE_BALL_DETECTED": [
        "A ball. I could push it.",
        "Round object. It can move.",
    ],
    "OBSERVE_OPEN_SCENE": [
        "Path is open. Continuing.",
        "Open space ahead.",
        "Nothing blocking.",
    ],
    "OBSERVE_CLEAR_PATH": [
        "Clear path.",
        "Nothing in the way.",
    ],
    "OBSERVE_CLUTTERED": [
        "Many objects. Complex navigation.",
        "Cluttered. Need to be careful.",
    ],
    "OBSERVE_NOVEL_OBJECT": [
        "Something unfamiliar. Never seen this before.",
        "New object in view. Interesting.",
    ],
    "OBSERVE_SENSOR_PLATE": [
        "Glowing tile. What is that?",
        "Sensor plate. Maybe step on it.",
    ],
    "OBSERVE_WALL_NEARBY": [
        "Wall close. Need to turn.",
        "Approaching boundary.",
    ],
    "OBSERVE_MOVING_OBJECT": [
        "Something is moving.",
        "Object in motion. Watching.",
    ],
    "OBSERVE_I_MOVED_IT": [
        "I moved it!",
        "Object reacted to my action.",
        "I can affect the environment.",
    ],
    "ASK_OBJECT_UNKNOWN": [
        "What is that object in front of me?",
        "What am I looking at?",
    ],
    "ASK_HOW_TO_NAVIGATE": [
        "How should I get around this?",
        "Best way past this obstacle?",
    ],
    "ASK_CAN_CLIMB": [
        "Can I climb that surface?",
        "Is that ramp safe for me?",
    ],
    "ASK_WHAT_IS_GOAL": [
        "Where should I go?",
        "Is there a goal in this space?",
    ],
}


def get_visual_templates(lang: str = "ru") -> dict[str, list[str]]:
    return VISUAL_TEMPLATES_RU if lang == "ru" else VISUAL_TEMPLATES_EN


# ── Concept → Template key mapper ─────────────────────────────────────────────
def concepts_to_observe_key(concepts: list[str]) -> str | None:
    """Select most relevant OBSERVE template from active visual concepts."""
    # Priority order
    priority = [
        ("OBJECT_BLOCKING_PATH",    "OBSERVE_OBJECT_AHEAD"),
        ("OBJECT_VERY_CLOSE",       "OBSERVE_OBJECT_CLOSE"),
        ("RAMP_DETECTED",           "OBSERVE_RAMP_DETECTED"),
        ("STAIRS_DETECTED",         "OBSERVE_STAIRS_DETECTED"),
        ("BEAM_DETECTED",           "OBSERVE_BEAM_DETECTED"),
        ("PLATFORM_DETECTED",       "OBSERVE_PLATFORM_DETECTED"),
        ("BALL_DETECTED",           "OBSERVE_BALL_DETECTED"),
        ("SENSOR_PLATE_VISIBLE",    "OBSERVE_SENSOR_PLATE"),
        ("FIRST_TIME_SEEING",       "OBSERVE_NOVEL_OBJECT"),
        ("HIGH_VISUAL_SURPRISE",    "OBSERVE_NOVEL_OBJECT"),
        ("I_MOVED_OBJECT",          "OBSERVE_I_MOVED_IT"),
        ("OBJECT_MOVING",           "OBSERVE_MOVING_OBJECT"),
        ("WALL_NEARBY",             "OBSERVE_WALL_NEARBY"),
        ("CLUTTERED_SCENE",         "OBSERVE_CLUTTERED"),
        ("OBJECT_LEFT",             "OBSERVE_OBJECT_LEFT"),
        ("OBJECT_RIGHT",            "OBSERVE_OBJECT_RIGHT"),
        ("OBJECT_AHEAD",            "OBSERVE_OBJECT_AHEAD"),
        ("CLEAR_PATH_AHEAD",        "OBSERVE_CLEAR_PATH"),
        ("OPEN_SCENE",              "OBSERVE_OPEN_SCENE"),
    ]
    for concept, key in priority:
        if concept in concepts:
            return key
    return None


def concepts_to_ask_key(concepts: list[str]) -> str | None:
    """Select ASK template from visual concepts."""
    if "FIRST_TIME_SEEING" in concepts or "HIGH_VISUAL_SURPRISE" in concepts:
        return "ASK_OBJECT_UNKNOWN"
    if "RAMP_DETECTED" in concepts or "INCLINED_SURFACE" in concepts:
        return "ASK_CAN_CLIMB"
    if "OBJECT_BLOCKING_PATH" in concepts or "MUST_GO_AROUND" in concepts:
        return "ASK_HOW_TO_NAVIGATE"
    return None


# ── Visual Inner Voice Controller ──────────────────────────────────────────────
class VisualInnerVoice:
    """
    Интегрируется в InnerVoiceController и VerbalActionController.

    Два режима:
      1. Пассивный: обновляет visual concepts в InnerVoice каждые N тиков
      2. Активный: предоставляет world_description для verbal templates

    Интеграция в simulation.py:
      self._visual_voice = VisualInnerVoice(lang="ru")

    При VLM update:
      self._visual_voice.update(
          slot_labeler_result,   # от SlotLabeler.process_slots()
          graph,                 # для инъекции узлов
          inner_voice_ctrl,      # для push_distill_sample()
          tick,
      )

    В verbal_action.py._generate_text():
      # Получить visual context:
      visual_obs_key = self._visual_voice.get_observe_key()
      visual_ask_key = self._visual_voice.get_ask_key()
      world_desc     = self._visual_voice.get_world_description()
    """

    def __init__(self, lang: str = "ru"):
        self._lang = lang
        self._templates = get_visual_templates(lang)
        self._every = _env_int("RKK_VISUAL_VOICE_EVERY", 30)
        self._inject_top = _env_int("RKK_VISUAL_INJECT_TOP", 5)
        self._last_tick = -999

        self._active_visual_concepts: list[tuple[str, float]] = []
        self._concept_names: list[str] = []
        self._world_desc: str = ""
        self._injected_nodes: set[str] = set()

        self._prev_i_moved_flag = False
        self.total_updates: int = 0
        self._history: deque[list[str]] = deque(maxlen=20)

    def update(
        self,
        visual_concepts: list[tuple[str, float]],
        world_desc: str,
        graph,
        inner_voice_ctrl,
        tick: int,
    ) -> None:
        """
        Main update from SlotLabeler result.
        """
        if not visual_voice_enabled():
            return
        if (tick - self._last_tick) < self._every:
            return
        self._last_tick = tick

        self._active_visual_concepts = visual_concepts
        self._concept_names = [c for c, _ in visual_concepts]
        self._world_desc = world_desc
        self._history.append(self._concept_names[:5])
        self.total_updates += 1

        # Inject into GNN
        if graph is not None:
            self._inject_to_graph(graph, visual_concepts[:self._inject_top])

        # Push to InnerVoiceNet distillation (visual concepts as training labels)
        if inner_voice_ctrl is not None and visual_concepts:
            node_ids = list(graph._node_ids) if graph and hasattr(graph, "_node_ids") else []
            state_vec = [float(graph.nodes.get(n, 0.5)) for n in node_ids] if node_ids else []
            if state_vec:
                top_visual = [c for c, _ in visual_concepts[:3]]
                inner_voice_ctrl.push_distill_sample(state_vec, top_visual)

    def _inject_to_graph(
        self,
        graph,
        concepts: list[tuple[str, float]],
    ) -> None:
        """Inject visual concept nodes into GNN with prefix concept_vis_."""
        # Fade out old nodes
        for node_key in list(self._injected_nodes):
            if node_key in graph.nodes:
                graph.nodes[node_key] = max(0.0, graph.nodes[node_key] * 0.7)

        self._injected_nodes = set()

        for name, activation in concepts:
            node_key = concept_to_gnn_node(name)
            if node_key not in graph.nodes:
                try:
                    graph.set_node(node_key, float(activation))
                    # Add semantic edges
                    self._add_semantic_edges(graph, name, node_key)
                except Exception:
                    pass
            else:
                graph.nodes[node_key] = float(activation)
            self._injected_nodes.add(node_key)

    def _add_semantic_edges(self, graph, concept_name: str, node_key: str) -> None:
        """Add causal edges from visual concepts to relevant body/intent variables."""
        edges: list[tuple[str, str, float]] = []

        if concept_name in ("OBJECT_BLOCKING_PATH", "OBJECT_VERY_CLOSE", "COLLISION_RISK"):
            edges = [
                (node_key, "intent_stop_recover", 0.30),
                (node_key, "intent_stride", -0.20),  # slow down
            ]
        elif concept_name in ("RAMP_DETECTED", "STAIRS_DETECTED", "INCLINED_SURFACE"):
            edges = [
                (node_key, "intent_torso_forward", 0.25),
                (node_key, "intent_stride", -0.15),
            ]
        elif concept_name in ("CLEAR_PATH_AHEAD", "OPEN_SCENE"):
            edges = [
                (node_key, "intent_stride", 0.15),
            ]
        elif concept_name in ("HIGH_VISUAL_SURPRISE", "FIRST_TIME_SEEING"):
            edges = [
                (node_key, "causal_eig", 0.30),
            ]
        elif concept_name in ("WALL_NEARBY", "BOUNDARY_NEAR"):
            edges = [
                (node_key, "intent_stride", -0.25),
                (node_key, "intent_stop_recover", 0.20),
            ]

        for fr, to, w in edges:
            if fr in graph.nodes and to in graph.nodes:
                try:
                    graph.set_edge(fr, to, w, alpha=0.07)
                except Exception:
                    pass

    def get_observe_key(self) -> str | None:
        """Get best OBSERVE template key for current visual state."""
        return concepts_to_observe_key(self._concept_names)

    def get_ask_key(self) -> str | None:
        """Get ASK template key if visual curiosity warrants asking."""
        return concepts_to_ask_key(self._concept_names)

    def get_template(self, key: str) -> str | None:
        """Get random template text for given key."""
        import random
        templates = self._templates.get(key, [])
        return random.choice(templates) if templates else None

    def get_world_description(self) -> str:
        return self._world_desc

    def get_active_concepts(self) -> list[tuple[str, float]]:
        return self._active_visual_concepts

    def is_novel_scene(self) -> bool:
        return any(c in self._concept_names for c in (
            "FIRST_TIME_SEEING", "HIGH_VISUAL_SURPRISE", "UNEXPECTED_OBJECT"
        ))

    def has_blocking_object(self) -> bool:
        return any(c in self._concept_names for c in (
            "OBJECT_BLOCKING_PATH", "OBJECT_VERY_CLOSE", "COLLISION_RISK"
        ))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": visual_voice_enabled(),
            "total_updates": self.total_updates,
            "active_visual_concepts": [(c, round(v, 3)) for c, v in self._active_visual_concepts[:6]],
            "world_description": self._world_desc,
            "injected_nodes": len(self._injected_nodes),
            "observe_key": self.get_observe_key(),
        }
