"""
episodic_memory.py — Level 2-D: Episodic Fall Memory + Living Memory thread.

Агент запоминает эпизоды падений с полным контекстом и использует
эту память для улучшения LLM L2 консультаций и собственного обучения.

Living Memory (человекоподобная нить опыта):
  - timeline: компактная шкала времени (posture, fall, curriculum) каждый тик humanoid
  - semantic_narrative: короткие «буллеты» из консолидации окон и фазы сна
  - retrieve_prior_adjustments(obs): мягкие поправки к гомеостатическим приорам Active Inference
  - сохранение narrative в autosave.meta (persistent_state)

Архитектура:
  EpisodicFallMemory — хранит deque(maxlen=50) эпизодов падений:
    - obs_before: состояние тела за N тиков до падения
    - trigger_action: последнее действие перед падением
    - cause_analysis: автоматический анализ причины
    - recovery_ticks: сколько тиков потребовалось встать
    - context: intent_* значения в момент падения

  EpisodicSuccessMemory — хранит deque(maxlen=30) успешных эпизодов ходьбы:
    - obs_snapshot: состояние при устойчивой ходьбе
    - duration_ticks: как долго удерживалась ходьба
    - mean_posture: средняя стабильность

  MemoryAugmentedLLMContext — форматирует историю для LLM L2 промпта:
    - последние 5 падений с причинами
    - паттерны: какие intent_* значения коррелируют с падениями
    - последние 3 успеха: что позволило устоять

  PatternDetector — находит повторяющиеся паттерны:
    - "падает когда intent_stride > 0.65 при posture < 0.6"
    - "Recovery занимает в среднем 12 тиков при rknee > 0.7"

RKK_EPISODE_MEMORY=1           — включить (default)
RKK_EPISODE_FALL_MAXLEN=50     — размер буфера падений
RKK_EPISODE_SUCCESS_MAXLEN=30  — размер буфера успехов
RKK_EPISODE_PRETRIGGER=8       — тиков истории до падения
RKK_EPISODE_PATTERN_MIN=3      — min повторений для паттерна
RKK_REM_WM_TRAIN=0             — REM фаза: дополнительный wm.train_step по top-k surprise (Phase D)

Living Memory:
  RKK_LIVING_MEMORY_TRACE=1      — шкала времени (timeline) для humanoid
  RKK_LIVING_MEMORY_CONSOLIDATE_EVERY=400 — тиков между свёрткой в semantic bullets
"""
from __future__ import annotations

import os
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Obs / intent values may be str; defaults like '?' must never reach :.3f format."""
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def episode_memory_enabled() -> bool:
    return os.environ.get("RKK_EPISODE_MEMORY", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def living_memory_trace_enabled() -> bool:
    return os.environ.get("RKK_LIVING_MEMORY_TRACE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class TimelineEntry:
    tick: int
    posture: float
    fallen: bool
    curriculum: str | None
    com_z: float
    summary: str


@dataclass
class FallEpisode:
    tick: int
    obs_before: dict[str, float]       # состояние за ~8 тиков до
    obs_at_fall: dict[str, float]      # состояние в момент падения
    trigger_action: tuple[str, float] | None  # (var, value) последнее действие
    intents_at_fall: dict[str, float]  # intent_* в момент падения
    cause: str                         # автоматическая классификация
    recovery_ticks: int = 0            # сколько тиков восстановления
    recovered: bool = False
    physics_context: dict[str, float] = field(default_factory=dict)
    surprise_score: float = 0.0

    def to_llm_lines(self, idx: int) -> list[str]:
        cz_b = _safe_float(self.obs_before.get("com_z", self.obs_before.get("phys_com_z")))
        cz_a = _safe_float(self.obs_at_fall.get("com_z", self.obs_at_fall.get("phys_com_z")))
        tp_a = _safe_float(
            self.obs_at_fall.get("torso_pitch", self.obs_at_fall.get("phys_torso_pitch"))
        )
        ps_a = _safe_float(
            self.obs_at_fall.get(
                "posture_stability", self.obs_at_fall.get("phys_posture_stability")
            )
        )
        lines = [
            f"Fall #{idx} at tick={self.tick}:",
            f"  cause: {self.cause}",
            f"  com_z_before={cz_b:.3f} → com_z_at={cz_a:.3f}",
            f"  torso_pitch_at={tp_a:.3f}",
            f"  posture_at={ps_a:.3f}",
        ]
        if "com_z_raw_m" in self.obs_at_fall:
            lines.insert(
                3,
                f"  com_z_raw_m={_safe_float(self.obs_at_fall.get('com_z_raw_m')):.3f} "
                f"(PyBullet base Z; gate RKK_FALLEN_Z / constants)",
            )
        if self.trigger_action:
            ta_v = _safe_float(self.trigger_action[1])
            lines.append(f"  last_action: do({self.trigger_action[0]}={ta_v:.3f})")
        intent_str = ", ".join(
            f"{k.replace('intent_','')}={_safe_float(v):.2f}"
            for k, v in self.intents_at_fall.items()
            if abs(_safe_float(v) - 0.5) > 0.05
        )
        if intent_str:
            lines.append(f"  intents: {intent_str}")
        if self.recovery_ticks > 0:
            lines.append(f"  recovery: {self.recovery_ticks} ticks ({'ok' if self.recovered else 'failed'})")
        return lines


@dataclass
class SuccessEpisode:
    tick: int
    obs_snapshot: dict[str, float]
    intents: dict[str, float]
    duration_ticks: int
    mean_posture: float
    mean_com_z: float
    physics_context: dict[str, float] = field(default_factory=dict)

    def to_llm_lines(self, idx: int) -> list[str]:
        intent_str = ", ".join(
            f"{k.replace('intent_','')}={_safe_float(v):.2f}"
            for k, v in self.intents.items()
            if abs(_safe_float(v) - 0.5) > 0.05
        )
        return [
            f"Success #{idx} at tick={self.tick}: {self.duration_ticks} ticks stable walk",
            f"  posture={self.mean_posture:.3f}, com_z={self.mean_com_z:.3f}",
            f"  intents: {intent_str or 'neutral'}",
        ]


@dataclass
class PatternEntry:
    description: str
    count: int
    last_tick: int
    confidence: float  # fraction of falls matching this pattern


# ── Pattern detector ──────────────────────────────────────────────────────────
class PatternDetector:
    """
    Находит повторяющиеся паттерны в эпизодах падений.
    Паттерны вида: "падает когда stride > threshold и posture < threshold"
    """

    INTENT_VARS = [
        "intent_stride", "intent_torso_forward", "intent_support_left",
        "intent_support_right", "intent_stop_recover", "intent_gait_coupling",
    ]
    STATE_VARS = ["posture_stability", "com_z", "com_x", "torso_pitch",
                  "foot_contact_l", "foot_contact_r", "support_bias"]

    def analyze(self, falls: list[FallEpisode]) -> list[PatternEntry]:
        if len(falls) < 2:
            return []

        patterns: list[PatternEntry] = []
        n = len(falls)

        # Pattern 1: high stride + low posture
        matching = [
            f for f in falls
            if f.intents_at_fall.get("intent_stride", 0.5) > 0.62
            and f.obs_at_fall.get("posture_stability",
                f.obs_at_fall.get("phys_posture_stability", 1.0)) < 0.55
        ]
        if len(matching) >= 2:
            patterns.append(PatternEntry(
                description=f"stride>0.62 + posture<0.55 → fall ({len(matching)}/{n} falls)",
                count=len(matching),
                last_tick=matching[-1].tick,
                confidence=len(matching) / n,
            ))

        # Pattern 2: backward torso lean during stride
        matching2 = [
            f for f in falls
            if f.obs_at_fall.get("torso_pitch", f.obs_at_fall.get("phys_torso_pitch", 0.0)) < -0.10
            and f.intents_at_fall.get("intent_stride", 0.5) > 0.55
        ]
        if len(matching2) >= 2:
            patterns.append(PatternEntry(
                description=f"backward lean (torso_pitch<-0.10) during stride → fall ({len(matching2)}/{n})",
                count=len(matching2),
                last_tick=matching2[-1].tick,
                confidence=len(matching2) / n,
            ))

        # Pattern 3: uneven foot contact during stride
        matching3 = [
            f for f in falls
            if abs(
                f.obs_at_fall.get("foot_contact_l", f.obs_at_fall.get("phys_foot_contact_l", 0.5))
                - f.obs_at_fall.get("foot_contact_r", f.obs_at_fall.get("phys_foot_contact_r", 0.5))
            ) > 0.30
        ]
        if len(matching3) >= 2:
            patterns.append(PatternEntry(
                description=f"uneven foot contact (|l-r|>0.30) → fall ({len(matching3)}/{n})",
                count=len(matching3),
                last_tick=matching3[-1].tick,
                confidence=len(matching3) / n,
            ))

        # Pattern 4: recovery failure (long recovery)
        long_recovery = [f for f in falls if f.recovery_ticks > 30]
        if len(long_recovery) >= 2:
            avg_r = float(np.mean([f.recovery_ticks for f in long_recovery]))
            patterns.append(PatternEntry(
                description=f"slow recovery: avg {avg_r:.0f} ticks ({len(long_recovery)}/{n} falls)",
                count=len(long_recovery),
                last_tick=long_recovery[-1].tick,
                confidence=len(long_recovery) / n,
            ))

        # Sort by confidence
        patterns.sort(key=lambda p: -p.confidence)
        return patterns[:5]


# ── Main episodic memory ──────────────────────────────────────────────────────
class EpisodicMemory:
    """
    Центральный контроллер эпизодической памяти.

    Интегрируется в simulation.py:
      - _on_fall_detected(): записывает эпизод падения
      - _on_stable_walk(): записывает успешный эпизод ходьбы
      - get_llm_context(): форматирует историю для LLM L2 промпта
      - maybe_close_recovery(): закрывает эпизод после восстановления
    """

    def __init__(self):
        fall_max = _env_int("RKK_EPISODE_FALL_MAXLEN", 50)
        success_max = _env_int("RKK_EPISODE_SUCCESS_MAXLEN", 30)
        self._pretrigger = _env_int("RKK_EPISODE_PRETRIGGER", 8)

        self.falls: deque[FallEpisode] = deque(maxlen=fall_max)
        self.successes: deque[SuccessEpisode] = deque(maxlen=success_max)

        # Rolling obs buffer for pre-trigger history
        self._obs_history: deque[dict[str, float]] = deque(maxlen=self._pretrigger + 2)
        self._action_history: deque[tuple[str, float]] = deque(maxlen=self._pretrigger + 2)

        # State tracking
        self._last_fall_tick: int = -999
        self._in_recovery: bool = False
        self._recovery_start_tick: int = 0
        self._current_fall_ep: FallEpisode | None = None

        # Success tracking
        self._stable_since: int | None = None
        self._posture_buf: deque[float] = deque(maxlen=60)
        self._com_z_buf: deque[float] = deque(maxlen=60)

        self._pattern_detector = PatternDetector()
        self._patterns: list[PatternEntry] = []
        self._last_pattern_tick: int = -999

        self.total_falls_recorded: int = 0
        self.total_successes_recorded: int = 0
        self._last_physics_context: dict[str, float] = {}

        # Living Memory — непрерывная «жизнь» в симуляции (humanoid)
        self._timeline: deque[TimelineEntry] = deque(maxlen=_env_int("RKK_LIVING_TIMELINE_MAX", 1024))
        self._semantic_narrative: deque[str] = deque(
            maxlen=_env_int("RKK_LIVING_SEMANTIC_MAX", 48)
        )
        self._consolidate_every: int = _env_int("RKK_LIVING_MEMORY_CONSOLIDATE_EVERY", 400)
        self._last_consolidate_tick: int = -999999

    def restore_semantic_narrative(self, bullets: list[str]) -> None:
        """После загрузки autosave.meta."""
        self._semantic_narrative.clear()
        for b in bullets[: self._semantic_narrative.maxlen or 48]:
            if isinstance(b, str) and b.strip():
                self._semantic_narrative.append(b.strip()[:240])

    def semantic_narrative_list(self) -> list[str]:
        return list(self._semantic_narrative)

    def append_timeline_tick(
        self,
        tick: int,
        obs: dict[str, float],
        fallen: bool,
        posture: float,
        curriculum_name: str | None = None,
    ) -> None:
        """Вызывать каждый тик на humanoid (включая fixed_root если trace включён)."""
        if not living_memory_trace_enabled():
            return
        cz = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        cn = (curriculum_name or "").strip()[:40] or None
        summ = f"t={tick} ps={posture:.2f} cz={cz:.2f} fall={int(fallen)} cur={cn or '-'}"

        self._timeline.append(
            TimelineEntry(
                tick=tick,
                posture=float(posture),
                fallen=bool(fallen),
                curriculum=cn,
                com_z=cz,
                summary=summ,
            )
        )
        if tick - self._last_consolidate_tick >= self._consolidate_every:
            self._consolidate_timeline_to_semantic(tick)
            self._last_consolidate_tick = tick

    def _consolidate_timeline_to_semantic(self, tick: int) -> None:
        if len(self._timeline) < 40:
            return
        chunk = list(self._timeline)[-min(120, len(self._timeline)) :]
        mean_ps = float(np.mean([e.posture for e in chunk]))
        falls_n = sum(1 for e in chunk if e.fallen)
        rng = f"{chunk[0].tick}–{chunk[-1].tick}"
        bullet = (
            f"[window {rng}] mean_posture={mean_ps:.2f}, falls_window={falls_n}/{len(chunk)}"
        )
        if self._semantic_narrative and self._semantic_narrative[-1] == bullet:
            return
        # Не засорять память «пустыми» плато
        if falls_n == 0 and 0.58 <= mean_ps <= 0.74:
            return
        self._semantic_narrative.append(bullet[:240])

    def on_sleep_complete(self, tick: int, session: Any) -> None:
        """После фазы сна — закрепить эпизод консолидации в семантической памяти."""
        if not living_memory_trace_enabled() and not episode_memory_enabled():
            return
        rem = int(getattr(session, "rem_episodes_replayed", 0) or 0)
        pruned = int(getattr(session, "edges_pruned", 0) or 0)
        verbal = str(getattr(session, "lesson_verbal", "") or "")[:120].replace("\n", " ")
        self._semantic_narrative.append(
            f"sleep_consolidation@{tick}: REM_ep={rem}, pruned_edges={pruned}, lesson='{verbal}'"
        )

    def retrieve_prior_adjustments(self, obs: dict[str, float]) -> dict[str, float]:
        """
        Мягкие поправки к ключам phys_* для Active Inference (узкий диапазон).
        Основаны на выученных паттернах падений и недавней средней стабильности.
        """
        out: dict[str, float] = {}
        posture = float(
            obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
        )

        if episode_memory_enabled():
            for p in self._patterns:
                desc = p.description.lower()
                conf = float(p.confidence)
                if conf < 0.28:
                    continue
                if "overstride" in desc:
                    out["phys_com_x_vel"] = min(out.get("phys_com_x_vel", 0.40), 0.22)
                if "backward" in desc and "stride" in desc:
                    out["phys_com_z"] = max(out.get("phys_com_z", 0.80), 0.84)
                if "uneven" in desc or "support" in desc:
                    out["phys_posture_stability"] = max(
                        out.get("phys_posture_stability", 0.97), 1.0
                    )

        if len(self._timeline) >= 24:
            tail = list(self._timeline)[-24:]
            mps = float(np.mean([e.posture for e in tail]))
            if mps < 0.52 and posture < 0.58:
                out["phys_posture_stability"] = 1.0
                out["phys_com_z"] = max(out.get("phys_com_z", 0.78), 0.83)

        return out

    def tick_update(
        self,
        tick: int,
        obs: dict[str, float],
        last_action: tuple[str, float] | None,
        fallen: bool,
        posture: float,
        *,
        physics_context: dict[str, float] | None = None,
    ) -> None:
        """Call every tick to maintain rolling buffers."""
        if not episode_memory_enabled():
            return

        if physics_context:
            self._last_physics_context = dict(physics_context)
        self._obs_history.append(dict(obs))
        if last_action is not None:
            self._action_history.append(last_action)

        # Track posture / com_z for success detection
        self._posture_buf.append(posture)
        com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        self._com_z_buf.append(com_z)

        # Success tracking: >40 ticks of posture>0.68 and com_z>0.40
        if posture > 0.68 and com_z > 0.40 and not fallen:
            if self._stable_since is None:
                self._stable_since = tick
        else:
            if self._stable_since is not None:
                duration = tick - self._stable_since
                if duration >= 40:
                    self._record_success(tick, obs, duration)
            self._stable_since = None

        # Recovery tracking
        if self._in_recovery and self._current_fall_ep is not None:
            if not fallen and posture > 0.55:
                # Recovered
                ep = self._current_fall_ep
                ep.recovery_ticks = tick - self._recovery_start_tick
                ep.recovered = True
                self._in_recovery = False
                self._current_fall_ep = None
            elif (tick - self._recovery_start_tick) > 200:
                # Recovery timeout
                if self._current_fall_ep:
                    self._current_fall_ep.recovery_ticks = 200
                    self._current_fall_ep.recovered = False
                self._in_recovery = False
                self._current_fall_ep = None

        # Pattern refresh every 100 ticks
        if tick - self._last_pattern_tick > 100 and len(self.falls) >= 3:
            self._patterns = self._pattern_detector.analyze(list(self.falls))
            self._last_pattern_tick = tick

    def on_fall(
        self,
        tick: int,
        obs_at_fall: dict[str, float],
        intents: dict[str, float],
        physics_context: dict[str, float] | None = None,
    ) -> FallEpisode | None:
        """Record a fall episode. Returns the episode object."""
        if not episode_memory_enabled():
            return None
        # Debounce: don't record if too close to last fall
        if tick - self._last_fall_tick < 10:
            return None
        self._last_fall_tick = tick

        # Pre-trigger state (8 ticks before fall)
        obs_before = dict(self._obs_history[0]) if self._obs_history else dict(obs_at_fall)
        last_action = self._action_history[-1] if self._action_history else None

        # Classify cause
        cause = self._classify_fall_cause(obs_at_fall, intents, last_action)

        pc = dict(physics_context or self._last_physics_context or {})
        ps_a = _safe_float(
            obs_at_fall.get(
                "posture_stability", obs_at_fall.get("phys_posture_stability", 0.5)
            )
        )
        surprise = float(max(0.0, min(1.0, 0.82 - ps_a)))

        episode = FallEpisode(
            tick=tick,
            obs_before=obs_before,
            obs_at_fall=dict(obs_at_fall),
            trigger_action=last_action,
            intents_at_fall={k: v for k, v in intents.items()},
            cause=cause,
            physics_context=pc,
            surprise_score=surprise,
        )
        self.falls.append(episode)
        self.total_falls_recorded += 1

        # Start recovery tracking
        self._in_recovery = True
        self._recovery_start_tick = tick
        self._current_fall_ep = episode

        return episode

    def _classify_fall_cause(
        self,
        obs: dict[str, float],
        intents: dict[str, float],
        last_action: tuple[str, float] | None,
    ) -> str:
        def g(k: str, default: float = 0.5) -> float:
            return float(obs.get(k, obs.get(f"phys_{k}", default)))

        torso_pitch = g("torso_pitch", 0.0)
        posture = g("posture_stability", 0.5)
        stride = float(intents.get("intent_stride", 0.5))
        foot_l = g("foot_contact_l", 0.5)
        foot_r = g("foot_contact_r", 0.5)

        if torso_pitch < -0.12 and stride > 0.55:
            return "backward_lean_during_stride"
        if abs(foot_l - foot_r) > 0.35:
            return "uneven_support_collapse"
        if stride > 0.70 and posture < 0.50:
            return "overstride"
        if posture < 0.35:
            return "total_balance_failure"
        if last_action and "stop_recover" in last_action[0] and last_action[1] < 0.4:
            return "premature_recovery_release"
        return "unstable_gait"

    def _record_success(self, tick: int, obs: dict[str, float], duration: int) -> None:
        """Record a stable walking episode."""
        ep = SuccessEpisode(
            tick=tick,
            obs_snapshot=dict(obs),
            intents={
                k: float(obs.get(k, obs.get(f"phys_{k}", 0.5)))
                for k in [
                    "intent_stride", "intent_torso_forward",
                    "intent_support_left", "intent_support_right",
                    "intent_gait_coupling", "intent_arm_counterbalance",
                ]
            },
            duration_ticks=duration,
            mean_posture=float(np.mean(self._posture_buf)) if self._posture_buf else 0.5,
            mean_com_z=float(np.mean(self._com_z_buf)) if self._com_z_buf else 0.5,
            physics_context=dict(self._last_physics_context or {}),
        )
        self.successes.append(ep)
        self.total_successes_recorded += 1

    # ── LLM context formatting ─────────────────────────────────────────────────
    def get_llm_context_block(self, max_falls: int = 5, max_successes: int = 3) -> str:
        """
        Returns formatted memory context for LLM L2 / embodied reward prompts.
        """
        lines: list[str] = []

        if episode_memory_enabled():
            # Recent falls
            recent_falls = list(self.falls)[-max_falls:]
            if recent_falls:
                lines.append(f"FALL HISTORY (last {len(recent_falls)} of {len(self.falls)} total):")
                for i, ep in enumerate(recent_falls):
                    lines.extend(ep.to_llm_lines(i + 1))
                    lines.append("")

            # Patterns
            if self._patterns:
                lines.append("DETECTED PATTERNS:")
                for p in self._patterns[:3]:
                    lines.append(f"  [conf={p.confidence:.2f}] {p.description}")
                lines.append("")

            # Recent successes
            recent_success = list(self.successes)[-max_successes:]
            if recent_success:
                lines.append(f"STABLE WALKING EPISODES (last {len(recent_success)}):")
                for i, ep in enumerate(recent_success):
                    lines.extend(ep.to_llm_lines(i + 1))
                lines.append("")

            lines.append(f"STATISTICS: {self.total_falls_recorded} total falls, "
                         f"{self.total_successes_recorded} stable walk episodes recorded.")

        # Autobiographical semantic narrative (works even when fall buffers are sparse)
        if self._semantic_narrative:
            lines.append("")
            lines.append("AUTOBIOGRAPHY (consolidated lived experience):")
            for b in list(self._semantic_narrative)[-12:]:
                lines.append(f"  • {b}")

        if not lines:
            return ""
        return "\n".join(lines)

    def get_seeds_from_patterns(self, valid_vars: set[str]) -> list[dict]:
        """
        Паттерны падений → causal seeds для GNN.
        "backward lean + stride → fall" стает ребром spine_pitch → com_x.
        """
        seeds: list[dict] = []
        for p in self._patterns:
            if "backward_lean" in p.description and "spine_pitch" in valid_vars and "com_x" in valid_vars:
                w = float(np.clip(0.2 + p.confidence * 0.3, 0.2, 0.55))
                seeds.append({"from_": "spine_pitch", "to": "com_x", "weight": w, "alpha": 0.05})
            if "uneven_support" in p.description:
                for pair in [("foot_contact_l", "support_bias"), ("foot_contact_r", "support_bias")]:
                    if pair[0] in valid_vars and pair[1] in valid_vars:
                        seeds.append({"from_": pair[0], "to": pair[1], "weight": 0.25, "alpha": 0.05})
            if "overstride" in p.description and "posture_stability" in valid_vars and "intent_stride" in valid_vars:
                seeds.append({"from_": "posture_stability", "to": "intent_stride", "weight": 0.30, "alpha": 0.05})
        return seeds

    def get_top_k_by_surprise(self, k: int) -> list[FallEpisode]:
        """Episodes ranked by ``surprise_score`` (proxy PE from posture instability)."""
        if not self.falls:
            return []
        k = max(1, int(k))
        ranked = sorted(self.falls, key=lambda e: float(e.surprise_score), reverse=True)
        return ranked[:k]

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": episode_memory_enabled(),
            "living_trace": living_memory_trace_enabled(),
            "total_falls": self.total_falls_recorded,
            "total_successes": self.total_successes_recorded,
            "buffered_falls": len(self.falls),
            "buffered_successes": len(self.successes),
            "in_recovery": self._in_recovery,
            "n_patterns": len(self._patterns),
            "patterns": [
                {"desc": p.description[:80], "conf": round(p.confidence, 3), "count": p.count}
                for p in self._patterns
            ],
            "recent_fall_causes": [ep.cause for ep in list(self.falls)[-5:]],
            "timeline_len": len(self._timeline),
            "semantic_narrative_preview": list(self._semantic_narrative)[-6:],
        }
