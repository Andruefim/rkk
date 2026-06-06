"""
sleep_consolidation.py — Phase K: Sleep Consolidation.

Аналог сна для ИИ: офлайн-обучение на накопленном опыте.

Три фазы сна:
  PHASE_REM:    Replay эпизодов из EpisodicMemory → offline RL (lr×10)
  PHASE_LESSON: врождённые priors / seeds (без Ollama)
  PHASE_PRUNE:  Synaptic pruning — обрезка слабых GNN edges

Триггеры (любой из):
  - Каждые RKK_SLEEP_EVERY_TICKS тиков с последнего пробуждения (periodic)
  - После RKK_SLEEP_FALL_THRESHOLD падений с последнего сна
  - По команде через API endpoint /sleep

Во время сна:
  - fixed_root=True (тело зафиксировано, не падает)
  - Снижается learning rate основного цикла (агент «отдыхает»)
  - Replay прогоняется offline через GNN + Motor Cortex
  - InnerVoiceNet обучается на lesson concepts
  - ConceptStore получает reinforcement

После сна:
  - fixed_root=False
  - Curriculum advance check (возможно переход к следующему навыку)
  - PhysicalCurriculum.inject_into_scheduler() если нужны новые навыки
  - RKK_POST_SLEEP_SCORE_RELAX_TICKS (default 480): N тиков после пробуждения реже
    полный пересчёт score_interventions (мин. интервал = max(RKK_SCORE_CACHE_EVERY,
    RKK_POST_SLEEP_SCORE_CACHE_EVERY_FLOOR)); 0 — выключить.
  - RKK_POST_SLEEP_SCORE_CACHE_EVERY_FLOOR (default 32): нижняя граница интервала в этом окне.

RKK_SLEEP_ENABLED=1
RKK_SLEEP_EVERY_TICKS=10000
RKK_SLEEP_FALL_THRESHOLD=50
RKK_SLEEP_DURATION_TICKS=200   — суммарных engine-тиков на сон; фазы REM:LESSON:PRUNE ≈ 30:80:20 от этого числа (раньше были захардкожены 30+80+20=130).
RKK_SLEEP_REM_LR_MULT=10.0     — множитель lr во время REM
RKK_SLEEP_PRUNE_THRESHOLD=0.05 — обрезать edges с |w| < threshold

MoCap inverse (inject_mocap_dreams):
  RKK_SLEEP_MOCAP_DREAMS — 1 (default): проигрывать MoCap-клипы в REM и учить WM на них; 0/off — пропустить (сон и replay эпизодов без изменений).
  RKK_SLEEP_QUALITY_THRESH — если MSE(forward_seq, X|A=0) > порога, инверсию не делаем (negative transfer).
  RKK_SLEEP_INVERSE_INNER — Adam-шагов на каждый одношаговый переход (дефолт 72).
  RKK_SLEEP_INVERSE_LR — lr для пошагового Adam.
  forward_dynamics_seq = teacher-forcing stack одношаговых f(X_t,A_t); инверсия по t согласована с этим (не авторегрессия).

Диагностика памяти:
  RKK_MEMORY_DIAG=1 — RSS + размеры GNN/мостов (см. engine.memory_diag)
  RKK_MEMORY_TRACE=1 — tracemalloc diff между этапами сна
"""
from __future__ import annotations

import asyncio
import gc
import os
import pathlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import torch


@dataclass
class SleepLessonAnnotation:
    """Innate sleep lesson payload (no Ollama)."""

    tick: int
    timestamp: float
    mode: str
    verbal: str = ""
    primary_concepts: list[str] = field(default_factory=list)
    lesson_text: str = ""
    lesson_concepts: list[str] = field(default_factory=list)
    seeds: list[dict] = field(default_factory=list)
    confidence: float = 1.0
    error: str = ""
    intent_adjustments: dict[str, float] = field(default_factory=dict)


def sleep_enabled() -> bool:
    return os.environ.get("RKK_SLEEP_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _sleep_min_tick() -> int:
    try:
        return max(0, int(os.environ.get("RKK_SLEEP_MIN_TICKS", "2000")))
    except ValueError:
        return 2000


def _compression_sleep_cooldown() -> int:
    try:
        return max(500, int(os.environ.get("RKK_SLEEP_COMPRESSION_COOLDOWN", "2000")))
    except ValueError:
        return 2000


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


def _memory_diag_log(sim: Any, tag: str) -> None:
    try:
        from engine.memory_diag import log_sim_memory, trace_snapshot

        log_sim_memory(sim, tag)
        trace_snapshot(tag)
    except Exception:
        pass


# ── Sleep phases ───────────────────────────────────────────────────────────────
class SleepPhase(Enum):
    AWAKE      = auto()
    REM        = auto()   # Episodic replay
    LESSON     = auto()   # LLM teacher lesson
    PRUNE      = auto()   # Synaptic pruning


@dataclass
class SleepSession:
    """One complete sleep cycle."""
    trigger_tick: int
    trigger_reason: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # REM replay stats
    rem_episodes_replayed: int = 0
    rem_loss_before: float = 0.0
    rem_loss_after: float = 0.0

    # Lesson stats
    lesson_verbal: str = ""
    lesson_concepts: list[str] = field(default_factory=list)
    lesson_seeds_injected: int = 0

    # Prune stats
    edges_pruned: int = 0
    edges_before: int = 0
    edges_after: int = 0

    # Overall
    completed: bool = False
    ticks_slept: int = 0

    # Grounded world ↔ semantic (world_state_bridge)
    grounded_samples: int = 0
    grounded_loss_last: float = 0.0

    def duration_sec(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    def summary(self) -> str:
        return (
            f"Sleep @ tick={self.trigger_tick} ({self.trigger_reason}): "
            f"REM={self.rem_episodes_replayed} eps, "
            f"grounded={self.grounded_samples} (loss={self.grounded_loss_last:.4f}), "
            f"pruned={self.edges_pruned} edges, "
            f"lesson={self.lesson_concepts[:3]}"
        )


# ── Synaptic pruner ────────────────────────────────────────────────────────────
class SynapticPruner:
    """
    Обрезает слабые каузальные связи в GNN.

    Аналог synaptic homeostasis: слабые синапсы ослабевают → обрезаются.
    Сильные синапсы укрепляются (normalization after pruning).

    Правила:
    - Не трогаем узлы с prefix "concept_", "proprio_", "mc_" (функциональные)
    - Не трогаем edges с alpha > 0.3 (приоритетные seeds)
    - Обрезаем только |W_ij| < threshold для ненужных edges
    """

    PROTECTED_PREFIXES = ("concept_", "proprio_", "mc_", "intent_", "phys_")

    def prune(
        self,
        graph,
        threshold: float | None = None,
    ) -> tuple[int, int]:
        """
        Обнуляет слабые веса через CausalGraph.prune_weak_W (frozen / forbidden / префиксы узлов).
        Возвращает (edge_count до, edge_count после) по текущему graph.EDGE_THRESH.
        """
        thr = threshold if threshold is not None else _env_float("RKK_SLEEP_PRUNE_THRESHOLD", 0.05)
        core = getattr(graph, "_core", None)
        if core is None:
            return 0, 0
        try:
            before = int(graph.edge_count)
        except Exception:
            before = 0
        try:
            graph.prune_weak_W(thr)
        except Exception:
            pass
        try:
            after = int(graph.edge_count)
        except Exception:
            after = before
        return before, after


# ── REM Replay ─────────────────────────────────────────────────────────────────
class REMReplay:
    """
    Offline replay of episodic memory during REM phase.

    Прогоняет fall/success эпизоды через GNN + Motor Cortex
    с повышенным learning rate, чтобы «закрепить» уроки.
    """

    def replay_falls(
        self,
        episodic_memory,
        graph,
        motor_cortex,
        lr_mult: float = 10.0,
    ) -> tuple[int, float, float]:
        """
        Replay fall episodes through GNN offline.
        Returns: (n_replayed, loss_before, loss_after)
        """
        if episodic_memory is None or not episodic_memory.falls:
            return 0, 0.0, 0.0

        try:
            kk = max(1, int(os.environ.get("RKK_SLEEP_REM_TOP_K", "32")))
        except ValueError:
            kk = 32
        sort_surprise = os.environ.get(
            "RKK_SLEEP_REM_SURPRISE_SORT", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        if sort_surprise and hasattr(episodic_memory, "get_top_k_by_surprise"):
            episodes = episodic_memory.get_top_k_by_surprise(kk)
        else:
            episodes = list(episodic_memory.falls)[-min(20, len(episodic_memory.falls)) :]

        if not episodes:
            return 0, 0.0, 0.0

        core = getattr(graph, "_core", None)
        if core is None:
            return 0, 0.0, 0.0

        node_ids = list(graph._node_ids)
        d = len(node_ids)
        try:
            dev = next(core.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

        # Temporarily boost LR
        optim = getattr(graph, "_optim", None)
        original_lrs = []
        if optim is not None:
            for pg in optim.param_groups:
                original_lrs.append(pg["lr"])
                pg["lr"] = pg["lr"] * lr_mult

        losses_before = []
        losses_after = []

        n_replayed = 0
        for ep in episodes:
            # X_before: state before fall
            obs_before = ep.obs_before
            obs_fall = ep.obs_at_fall

            X_t = torch.tensor(
                [float(obs_before.get(n, obs_before.get(f"phys_{n}", 0.5))) for n in node_ids],
                dtype=torch.float32,
                device=dev,
            )
            X_fall = torch.tensor(
                [float(obs_fall.get(n, obs_fall.get(f"phys_{n}", 0.5))) for n in node_ids],
                dtype=torch.float32,
                device=dev,
            )

            # Action: what was done before falling
            action = ep.trigger_action
            a = torch.zeros(d, dtype=torch.float32, device=dev)
            if action and action[0] in node_ids:
                a[node_ids.index(action[0])] = float(action[1])

            X_t = X_t.unsqueeze(0)
            X_fall = X_fall.unsqueeze(0)
            a = a.unsqueeze(0)

            try:
                from engine.wm_neural_ode import integrate_world_model_step
                import torch.nn.functional as F

                # Metric-only: no autograd. With RKK_WM_NEURAL_ODE=1, odeint otherwise
                # materializes a huge graph; we already run a full train forward below.
                with torch.inference_mode():
                    X_pred_metric = integrate_world_model_step(graph, X_t, a)
                    loss_before = float(F.mse_loss(X_pred_metric, X_fall).item())
                losses_before.append(loss_before)

                # Train (single graph per episode)
                if optim is not None:
                    optim.zero_grad()
                    X_pred_train = integrate_world_model_step(graph, X_t, a)
                    loss = F.mse_loss(X_pred_train, X_fall)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(core.parameters(), 0.5)
                    optim.step()
                    losses_after.append(float(loss.item()))
                    del loss, X_pred_train

                n_replayed += 1
            except Exception:
                continue

        # Restore LR
        if optim is not None:
            for i, pg in enumerate(optim.param_groups):
                if i < len(original_lrs):
                    pg["lr"] = original_lrs[i]

        l_before = float(np.mean(losses_before)) if losses_before else 0.0
        l_after = float(np.mean(losses_after)) if losses_after else 0.0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return n_replayed, l_before, l_after

    def replay_one_episode(
        self,
        ep,
        graph,
        *,
        lr_mult: float = 10.0,
    ) -> tuple[bool, float, float]:
        """Train GNN on a single fall episode. Returns (ok, loss_before, loss_after)."""
        core = getattr(graph, "_core", None)
        if core is None:
            return False, 0.0, 0.0

        node_ids = list(graph._node_ids)
        d = len(node_ids)
        try:
            dev = next(core.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

        optim = getattr(graph, "_optim", None)
        original_lrs: list[float] = []
        if optim is not None:
            for pg in optim.param_groups:
                original_lrs.append(pg["lr"])
                pg["lr"] = pg["lr"] * lr_mult

        obs_before = ep.obs_before
        obs_fall = ep.obs_at_fall
        X_t = torch.tensor(
            [float(obs_before.get(n, obs_before.get(f"phys_{n}", 0.5))) for n in node_ids],
            dtype=torch.float32,
            device=dev,
        )
        X_fall = torch.tensor(
            [float(obs_fall.get(n, obs_fall.get(f"phys_{n}", 0.5))) for n in node_ids],
            dtype=torch.float32,
            device=dev,
        )
        action = ep.trigger_action
        a = torch.zeros(d, dtype=torch.float32, device=dev)
        if action and action[0] in node_ids:
            a[node_ids.index(action[0])] = float(action[1])

        X_t = X_t.unsqueeze(0)
        X_fall = X_fall.unsqueeze(0)
        a = a.unsqueeze(0)

        ok = False
        l_before = 0.0
        l_after = 0.0
        try:
            from engine.wm_neural_ode import integrate_world_model_step
            import torch.nn.functional as F

            with torch.inference_mode():
                X_pred_metric = integrate_world_model_step(graph, X_t, a)
                l_before = float(F.mse_loss(X_pred_metric, X_fall).item())

            if optim is not None:
                optim.zero_grad()
                X_pred_train = integrate_world_model_step(graph, X_t, a)
                loss = F.mse_loss(X_pred_train, X_fall)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(core.parameters(), 0.5)
                optim.step()
                l_after = float(loss.item())
                del loss, X_pred_train
            ok = True
        except Exception:
            ok = False

        if optim is not None:
            for i, pg in enumerate(optim.param_groups):
                if i < len(original_lrs):
                    pg["lr"] = original_lrs[i]

        return ok, l_before, l_after

    def replay_top_surprise_wm_train(
        self,
        episodic_memory,
        graph,
        k: int = 8,
    ) -> int:
        """
        REM → WM: вызывает ``CausalGraph.train_step()`` после записи пар наблюдений
        из top-k эпизодов по surprise (Phase D). Это и есть WM training entry point
        в RKK (LeWM / GNN world model), не отдельный ``temporal_world_model.train_step``.
        Включение: ``RKK_REM_WM_TRAIN=1``.
        """
        if episodic_memory is None:
            return 0
        if os.environ.get("RKK_REM_WM_TRAIN", "0").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return 0
        core = getattr(graph, "_core", None)
        optim = getattr(graph, "_optim", None)
        if core is None or optim is None:
            return 0
        try:
            k = max(1, int(os.environ.get("RKK_REM_WM_TOP_K", str(k))))
        except ValueError:
            k = 8
        eps = episodic_memory.get_top_k_by_surprise(k)
        if not eps:
            return 0
        node_ids = list(graph._node_ids)
        n_ok = 0

        def _row(obs: dict[str, float]) -> list[float]:
            return [float(obs.get(n, graph.nodes.get(n, 0.5))) for n in node_ids]

        for ep in eps:
            try:
                v0 = _row(ep.obs_before)
                v1 = _row(ep.obs_at_fall)
                graph.record_observation({node_ids[i]: v0[i] for i in range(len(node_ids))})
                graph.record_observation({node_ids[i]: v1[i] for i in range(len(node_ids))})
                tr = graph.train_step()
                if tr is not None:
                    n_ok += 1
            except Exception:
                continue
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return n_ok


# ── Sleep Controller ───────────────────────────────────────────────────────────
class SleepController:
    """
    Полный контроллер сна.

    Состояние машина: AWAKE → REM → LESSON → PRUNE → AWAKE

    Интеграция в simulation.py:
      self._sleep_ctrl = SleepController()

    В тик-цикле:
      trigger = self._sleep_ctrl.check_trigger(tick, total_falls, force=False)
      if trigger or self._sleep_ctrl.is_sleeping:
          result = self._sleep_ctrl.tick(tick, sim)
    """

    @staticmethod
    def _split_sleep_phase_lengths(total_ticks: int) -> tuple[int, int, int]:
        """
        Делит RKK_SLEEP_DURATION_TICKS на три фазы с тем же соотношением,
        что раньше было захардкожено (REM:LESSON:PRUNE = 30:80:20 при сумме 130).
        Остаток от округления добавляется к LESSON (ожидание LLM).
        """
        total = max(3, int(total_ticks))
        rem = max(1, round(total * 30 / 130))
        lesson = max(1, round(total * 80 / 130))
        prune = max(1, round(total * 20 / 130))
        lesson += total - (rem + lesson + prune)
        return rem, lesson, prune

    def __init__(self):
        self._phase = SleepPhase.AWAKE
        self._session: SleepSession | None = None
        self._phase_start_tick: int = 0
        self._rem_replayer = REMReplay()
        self._pruner = SynapticPruner()

        # Triggers
        self._every_ticks = _env_int("RKK_SLEEP_EVERY_TICKS", 10000)
        self._fall_threshold = _env_int("RKK_SLEEP_FALL_THRESHOLD", 50)
        self._sleep_duration = _env_int("RKK_SLEEP_DURATION_TICKS", 200)
        self._rem_ticks, self._lesson_ticks, self._prune_ticks = self._split_sleep_phase_lengths(
            self._sleep_duration
        )

        # State (0 = never woken; periodic uses tick - last_sleep_tick >= RKK_SLEEP_EVERY_TICKS)
        self.last_sleep_tick: int = 0
        self._falls_since_sleep: int = 0
        self.sleep_count: int = 0
        self.total_sleep_ticks: int = 0

        self._sessions: deque[SleepSession] = deque(maxlen=20)
        self._lesson_scheduled: bool = False
        self._lesson_result: Any = None
        self._rem_state: dict[str, Any] | None = None

    @property
    def is_sleeping(self) -> bool:
        return self._phase != SleepPhase.AWAKE

    @property
    def current_phase(self) -> SleepPhase:
        return self._phase

    def notify_fall(self) -> None:
        """Call when agent falls."""
        self._falls_since_sleep += 1

    def should_sleep(self, intrinsic_objective=None) -> str | None:
        """
        Data-driven sleep trigger: проверяет compression_is_stagnant()
        через IntrinsicObjective.causal_surprise.

        Возвращает причину ("compression_stagnant") или None.
        Это ГЛАВНЫЙ триггер сна — мозг засыпает когда перестаёт учиться.
        """
        if intrinsic_objective is None:
            return None
        cs = getattr(intrinsic_objective, "causal_surprise", None)
        if cs is None:
            return None
        # Стагнация: compression не растёт >= 50 тиков
        if cs.compression_is_stagnant(window=50):
            # Дополнительная проверка: не спать слишком часто
            # (минимум 500 тиков между compression-driven сессиями)
            if cs.total_computations > 100:
                return "compression_stagnant"
        return None

    def check_trigger(
        self,
        tick: int,
        total_falls: int,
        force: bool = False,
        intrinsic_objective=None,
        sim: Any | None = None,
    ) -> str | None:
        """
        Check if sleep should be triggered. Returns reason or None.

        Приоритет триггеров:
          1. manual (force=True)
          2. compression_stagnant (data-driven, главный)
          3. edge_spike / behavioral_drop (Fix 7)
          4. fall_threshold (аварийный, слишком много падений)
          5. periodic (fallback, если ничего не сработало)
        """
        if not sleep_enabled():
            return None
        if self.is_sleeping:
            return None
        if tick < _sleep_min_tick():
            return None

        if force:
            return "manual"

        compression_reason = self.should_sleep(intrinsic_objective)
        if compression_reason is not None:
            cooldown = _compression_sleep_cooldown()
            if (tick - self.last_sleep_tick) >= cooldown:
                return compression_reason

        if sim is not None:
            hist = getattr(sim, "_edge_delta_hist", None)
            if hist:
                try:
                    thr = int(os.environ.get("RKK_SLEEP_EDGE_DELTA_THRESHOLD", "500"))
                    window = int(os.environ.get("RKK_EDGE_DELTA_WINDOW", "100"))
                except ValueError:
                    thr, window = 500, 100
                recent = list(hist)[-window:]
                if len(recent) >= window and sum(int(x) for x in recent) > thr:
                    if (tick - self.last_sleep_tick) >= _compression_sleep_cooldown():
                        return "edge_spike"
            bt = getattr(sim, "behavioral_tracker", None)
            if bt is not None and len(getattr(bt, "_posture", [])) >= 40:
                snap = bt.snapshot()
                score = float(snap.get("behavioral_score", 0.5))
                prev = float(getattr(sim, "_prev_behavioral_score", score))
                sim._prev_behavioral_score = score
                try:
                    drop_thr = float(os.environ.get("RKK_SLEEP_BEHAVIORAL_DROP", "0.2"))
                except ValueError:
                    drop_thr = 0.2
                if prev - score > drop_thr and (tick - self.last_sleep_tick) >= _compression_sleep_cooldown():
                    return "behavioral_drop"

        if self._falls_since_sleep >= self._fall_threshold:
            return "fall_threshold"

        if self._every_ticks > 0 and tick > 0:
            if (tick - int(self.last_sleep_tick)) >= int(self._every_ticks):
                return "periodic"

        return None

    def begin_sleep(self, tick: int, reason: str, sim: Any | None = None) -> None:
        """Start a sleep cycle."""
        print(
            f"[Sleep] 😴 Beginning sleep at tick={tick} reason={reason} "
            f"(phase ticks REM={self._rem_ticks} LESSON={self._lesson_ticks} PRUNE={self._prune_ticks}; "
            f"total≈{self._sleep_duration})"
        )
        if sim is not None:
            _memory_diag_log(sim, f"sleep_begin tick={tick} reason={reason}")
        self._phase = SleepPhase.REM
        self._phase_start_tick = tick
        self._session = SleepSession(trigger_tick=tick, trigger_reason=reason)
        self._lesson_scheduled = False
        self._lesson_result = None
        self._rem_state = None

    def _init_rem_state(self, sim) -> None:
        episodes: list = []
        em = getattr(sim, "_episodic_memory", None)
        if em is not None and em.falls:
            try:
                kk = max(1, int(os.environ.get("RKK_SLEEP_REM_TOP_K", "32")))
            except ValueError:
                kk = 32
            sort_surprise = os.environ.get(
                "RKK_SLEEP_REM_SURPRISE_SORT", "1"
            ).strip().lower() in ("1", "true", "yes", "on")
            if sort_surprise and hasattr(em, "get_top_k_by_surprise"):
                episodes = list(em.get_top_k_by_surprise(kk))
            else:
                episodes = list(em.falls)[-min(20, len(em.falls)) :]

        self._rem_state = {
            "episodes": episodes,
            "idx": 0,
            "losses_before": [],
            "losses_after": [],
            "replayed": 0,
            "wm_done": False,
            "finalized": False,
        }

    def _rem_budget_ms(self) -> float:
        try:
            return max(10.0, float(os.environ.get("RKK_SLEEP_REM_BUDGET_MS", "80")))
        except ValueError:
            return 80.0

    def _rem_tick_slice(self, tick: int, sim) -> None:
        """Spread REM replay across engine ticks (avoid 10–20s stalls on tick 0)."""
        if self._rem_state is None:
            self._init_rem_state(sim)
        st = self._rem_state
        if st.get("finalized"):
            return

        session = self._session
        budget = self._rem_budget_ms() / 1000.0
        t0 = time.perf_counter()
        graph = sim.agent.graph
        lr_mult = _env_float("RKK_SLEEP_REM_LR_MULT", 10.0)

        episodes = st.get("episodes") or []
        idx = int(st.get("idx", 0))
        while idx < len(episodes) and (time.perf_counter() - t0) < budget:
            ok, lb, la = self._rem_replayer.replay_one_episode(
                episodes[idx], graph, lr_mult=lr_mult
            )
            if ok:
                st["losses_before"].append(lb)
                st["losses_after"].append(la)
                st["replayed"] = int(st.get("replayed", 0)) + 1
            idx += 1
        st["idx"] = idx

        if idx >= len(episodes) and not st.get("wm_done"):
            if os.environ.get("RKK_REM_WM_TRAIN", "1").strip().lower() not in (
                "0", "false", "no", "off",
            ):
                n_wm = self._rem_replayer.replay_top_surprise_wm_train(
                    getattr(sim, "_episodic_memory", None),
                    graph,
                )
                if n_wm > 0:
                    print(f"[Sleep] REM: WM train_step on top-surprise batches (ok={n_wm})")
            st["wm_done"] = True

        if idx >= len(episodes) and st.get("wm_done") and not st.get("finalized"):
            self._rem_finalize_rem(tick, sim, st, session)
            st["finalized"] = True

    def _rem_finalize_rem(self, tick: int, sim, st: dict, session: SleepSession | None) -> None:
        lb = st.get("losses_before") or []
        la = st.get("losses_after") or []
        l_before = float(np.mean(lb)) if lb else 0.0
        l_after = float(np.mean(la)) if la else 0.0
        n = int(st.get("replayed", 0))
        if session is not None:
            session.rem_episodes_replayed = n
            session.rem_loss_before = l_before
            session.rem_loss_after = l_after
        print(f"[Sleep] REM: replayed {n} episodes, loss {l_before:.4f}→{l_after:.4f}")

        core = getattr(sim.agent.graph, "_core", None)
        if core is not None:
            with torch.no_grad():
                w_max = core.W.abs().max()
                if w_max > 1.5:
                    core.W.data.div_(w_max / 1.5)
                    print(f"[Sleep] REM: Normalized W max {w_max:.2f} → 1.5 to prevent densification")

        _memory_diag_log(sim, "sleep_after_REM_replay")

        self._schedule_lesson(tick, sim)

    def tick(self, tick: int, sim) -> dict[str, Any]:
        """
        Drive sleep state machine. Call every tick while sleeping.
        Returns status dict.
        """
        if not self.is_sleeping:
            return {"sleeping": False}

        session = self._session
        ticks_in_phase = tick - self._phase_start_tick
        session.ticks_in_phase = ticks_in_phase
        self.total_sleep_ticks += 1
        if session:
            session.ticks_slept += 1

        # ── REM phase ──────────────────────────────────────────────────────────
        if self._phase == SleepPhase.REM:
            self._rem_tick_slice(tick, sim)
            rem_done = bool(self._rem_state and self._rem_state.get("finalized"))
            if rem_done or ticks_in_phase >= self._rem_ticks:
                self._phase = SleepPhase.LESSON
                self._phase_start_tick = tick

        # ── LESSON phase ────────────────────────────────────────────────────────
        elif self._phase == SleepPhase.LESSON:
            if ticks_in_phase >= self._lesson_ticks:
                # Apply lesson result if arrived
                if self._lesson_result is not None:
                    self._apply_lesson(tick, sim, self._lesson_result)
                    _memory_diag_log(sim, "sleep_after_lesson_applied")
                self._phase = SleepPhase.PRUNE
                self._phase_start_tick = tick

        # ── PRUNE phase ─────────────────────────────────────────────────────────
        elif self._phase == SleepPhase.PRUNE:
            if ticks_in_phase == 0:
                before, after = self._pruner.prune(sim.agent.graph)
                session.edges_pruned = before - after
                session.edges_before = before
                session.edges_after = after
                print(f"[Sleep] Prune: {before}→{after} edges ({before-after} pruned)")
                _memory_diag_log(sim, "sleep_after_prune")
                coord = getattr(sim, "neuro_coordinator", None)
                if coord is not None:
                    try:
                        applied = coord.apply_after_sleep(sim, tick=tick)
                        if applied is not None:
                            print(f"[Sleep] Applied pending neurogenesis: {applied.get('new_node')}")
                    except Exception as e:
                        print(f"[Sleep] neuro apply failed: {e}")

            if ticks_in_phase >= self._prune_ticks:
                self._end_sleep(tick, sim)

        return {
            "sleeping": True,
            "phase": self._phase.name,
            "ticks_in_phase": ticks_in_phase,
            "session": {
                "trigger": session.trigger_reason if session else "",
                "rem_replayed": session.rem_episodes_replayed if session else 0,
            },
        }

    def _schedule_lesson(self, tick: int, sim) -> None:
        """Innate genome/text priors during sleep LESSON phase."""
        from engine.environment_humanoid import humanoid_hardcoded_seeds

        seeds = list(humanoid_hardcoded_seeds())
        concepts: list[str] = []
        for s in seeds:
            c = str(s.get("concept") or "").strip()
            if c and c not in concepts:
                concepts.append(c)
        self._lesson_result = SleepLessonAnnotation(
            tick=tick,
            timestamp=time.time(),
            mode="lesson",
            verbal="Innate sleep lesson",
            primary_concepts=concepts[:8] or ["balance", "locomotion"],
            lesson_text="fixed_root curriculum consolidation",
            lesson_concepts=concepts[:8],
            seeds=seeds[:16],
            confidence=0.85,
        )
        self._lesson_scheduled = True
        print(
            f"[Sleep] Lesson scheduled: {len(seeds[:16])} seeds, "
            f"{len(self._lesson_result.primary_concepts)} concepts"
        )

    def _apply_lesson(self, tick: int, sim, ann) -> None:
        """Apply sleep lesson to InnerVoiceNet + GNN."""
        session = self._session
        if session:
            session.lesson_verbal = ann.verbal
            session.lesson_concepts = ann.primary_concepts

        # Distill into InnerVoiceNet (multiple times = stronger signal)
        inner_voice = getattr(sim, "_inner_voice", None)
        if inner_voice and ann.primary_concepts:
            node_ids = list(sim.agent.graph._node_ids)
            state_vec = [float(sim.agent.graph.nodes.get(n, 0.5)) for n in node_ids]
            if state_vec:
                # Multiple pushes during sleep = stronger consolidation
                for _ in range(5):
                    inner_voice.push_distill_sample(state_vec, ann.primary_concepts)
                for _ in range(3):
                    inner_voice.train_step()
            if session:
                session.lesson_concepts = ann.primary_concepts

        # Inject seeds into GNN
        if ann.seeds:
            n_seeds = 0
            try:
                result = sim.agent.inject_text_priors(ann.seeds)
                n_seeds = int(result.get("injected", 0))
            except Exception:
                pass
            if session:
                session.lesson_seeds_injected = n_seeds

        # Apply curriculum hints from lesson
        if ann.intent_adjustments and hasattr(sim, "_timescale") and sim._timescale:
            for var, val in ann.intent_adjustments.items():
                sim._timescale.set_intent(3, var, val)

        print(f"[Sleep] Lesson applied: {ann.primary_concepts[:3]} verbal='{ann.verbal[:60]}'")

    def _end_sleep(self, tick: int, sim) -> None:
        """Finalize sleep, wake up."""
        _memory_diag_log(sim, f"sleep_wake tick={tick}")
        session = self._session
        if session:
            session.end_time = time.time()
            session.completed = True
            self._sessions.append(session)
            print(f"[Sleep] ✅ {session.summary()}")

        em = getattr(sim, "_episodic_memory", None)
        if em is not None and session:
            try:
                em.on_sleep_complete(tick, session)
            except Exception:
                pass

        self._phase = SleepPhase.AWAKE
        self.last_sleep_tick = tick
        self._falls_since_sleep = 0
        self.sleep_count += 1

        try:
            g = sim.agent.graph
            g.deduplicate_edges_keep_strongest()
            if os.environ.get("RKK_POST_SLEEP_W_PRUNE", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            ):
                nz = g.prune_weak_W()
                ec = int(g.edge_count)
                print(
                    f"[Sleep] post-wake W prune: zeroed_weak_slots={nz} edge_count≥thresh={ec}",
                    flush=True,
                )
        except Exception:
            pass

        # После REM граф плотнее — score_interventions дороже; окно с более редким sync refresh.
        try:
            ag = getattr(sim, "agent", None)
            if ag is not None:
                try:
                    rel = int(os.environ.get("RKK_POST_SLEEP_SCORE_RELAX_TICKS", "480"))
                except ValueError:
                    rel = 480
                rel = max(0, rel)
                if rel > 0:
                    setattr(ag, "_post_sleep_score_cache_relax_until", int(tick) + rel)
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        sessions_summary = [
            s.summary() for s in list(self._sessions)[-3:]
        ]
        return {
            "enabled": sleep_enabled(),
            "is_sleeping": self.is_sleeping,
            "current_phase": self._phase.name,
            "sleep_count": self.sleep_count,
            "last_sleep_tick": self.last_sleep_tick,
            "total_sleep_ticks": self.total_sleep_ticks,
            "falls_since_sleep": self._falls_since_sleep,
            "fall_threshold": self._fall_threshold,
            "every_ticks": self._every_ticks,
            "recent_sessions": sessions_summary,
        }
