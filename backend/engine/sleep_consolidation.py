"""
sleep_consolidation.py — Phase K: Sleep Consolidation.

Аналог сна для ИИ: офлайн-обучение на накопленном опыте.

Три фазы сна:
  PHASE_REM:    Replay эпизодов из EpisodicMemory → offline RL (lr×10)
  PHASE_LESSON: LLM teacher генерирует структурированный урок
  PHASE_PRUNE:  Synaptic pruning — обрезка слабых GNN edges

Триггеры (любой из):
  - Каждые RKK_SLEEP_EVERY_TICKS тиков (default: 10000)
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

RKK_SLEEP_ENABLED=1
RKK_SLEEP_EVERY_TICKS=10000
RKK_SLEEP_FALL_THRESHOLD=50
RKK_SLEEP_DURATION_TICKS=200   — тиков на сон (с fixed_root)
RKK_SLEEP_REM_LR_MULT=10.0     — множитель lr во время REM
RKK_SLEEP_PRUNE_THRESHOLD=0.05 — обрезать edges с |w| < threshold
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import torch


def sleep_enabled() -> bool:
    return os.environ.get("RKK_SLEEP_ENABLED", "1").strip().lower() not in (
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

    def duration_sec(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    def summary(self) -> str:
        return (
            f"Sleep @ tick={self.trigger_tick} ({self.trigger_reason}): "
            f"REM={self.rem_episodes_replayed} eps, "
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
        Prune weak edges from GNN weight matrix W.
        Returns (edges_before, edges_after).
        """
        threshold = threshold or _env_float("RKK_SLEEP_PRUNE_THRESHOLD", 0.05)
        core = getattr(graph, "_core", None)
        if core is None:
            return 0, 0

        W = getattr(core, "W", None)
        if W is None:
            return 0, 0

        node_ids = list(graph._node_ids)
        n = W.shape[0]

        with torch.no_grad():
            W_abs = W.data.abs()
            # Count non-zero before
            before = int((W_abs > 0.005).sum().item())

            # Build mask: don't prune protected nodes
            prune_mask = torch.ones(n, n, dtype=torch.bool, device=W.device)
            for i, name in enumerate(node_ids):
                if any(name.startswith(p) for p in self.PROTECTED_PREFIXES):
                    prune_mask[i, :] = False
                    prune_mask[:, i] = False

            # Zero out small weights on non-protected edges
            weak = (W_abs < threshold) & prune_mask
            W.data[weak] = 0.0

            after = int((W.data.abs() > 0.005).sum().item())
            pruned = before - after

        if pruned > 0:
            graph._invalidate_cache()

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

        episodes = list(episodic_memory.falls)
        if not episodes:
            return 0, 0.0, 0.0

        core = getattr(graph, "_core", None)
        if core is None:
            return 0, 0.0, 0.0

        node_ids = list(graph._node_ids)
        d = len(node_ids)

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
        for ep in episodes[-20:]:  # last 20 fall episodes
            # X_before: state before fall
            obs_before = ep.obs_before
            obs_fall = ep.obs_at_fall

            X_t = torch.tensor(
                [float(obs_before.get(n, obs_before.get(f"phys_{n}", 0.5))) for n in node_ids],
                dtype=torch.float32,
            )
            X_fall = torch.tensor(
                [float(obs_fall.get(n, obs_fall.get(f"phys_{n}", 0.5))) for n in node_ids],
                dtype=torch.float32,
            )

            # Action: what was done before falling
            action = ep.trigger_action
            a = torch.zeros(d)
            if action and action[0] in node_ids:
                a[node_ids.index(action[0])] = float(action[1])

            X_t = X_t.unsqueeze(0)
            X_fall = X_fall.unsqueeze(0)
            a = a.unsqueeze(0)

            try:
                from engine.wm_neural_ode import integrate_world_model_step
                import torch.nn.functional as F

                # Forward pass
                X_pred = integrate_world_model_step(core, X_t, a)

                # Loss: prediction should have matched X_fall
                loss_before = float(F.mse_loss(X_pred.detach(), X_fall).item())
                losses_before.append(loss_before)

                # Train
                if optim is not None:
                    optim.zero_grad()
                    X_pred_train = integrate_world_model_step(core, X_t.detach(), a.detach())
                    loss = F.mse_loss(X_pred_train, X_fall.detach())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(core.parameters(), 0.5)
                    optim.step()
                    losses_after.append(float(loss.item()))

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
        return n_replayed, l_before, l_after


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

        # State
        self.last_sleep_tick: int = -self._every_ticks  # allow first sleep
        self._falls_since_sleep: int = 0
        self.sleep_count: int = 0
        self.total_sleep_ticks: int = 0

        self._sessions: deque[SleepSession] = deque(maxlen=20)
        self._lesson_scheduled: bool = False
        self._lesson_result: Any = None

    @property
    def is_sleeping(self) -> bool:
        return self._phase != SleepPhase.AWAKE

    @property
    def current_phase(self) -> SleepPhase:
        return self._phase

    def notify_fall(self) -> None:
        """Call when agent falls."""
        self._falls_since_sleep += 1

    def check_trigger(
        self,
        tick: int,
        total_falls: int,
        force: bool = False,
    ) -> str | None:
        """
        Check if sleep should be triggered. Returns reason or None.
        """
        if not sleep_enabled():
            return None
        if self.is_sleeping:
            return None

        if force:
            return "manual"
        if (tick - self.last_sleep_tick) >= self._every_ticks:
            return "periodic"
        if self._falls_since_sleep >= self._fall_threshold:
            return "fall_threshold"
        return None

    def begin_sleep(self, tick: int, reason: str) -> None:
        """Start a sleep cycle."""
        print(f"[Sleep] 😴 Beginning sleep at tick={tick} reason={reason}")
        self._phase = SleepPhase.REM
        self._phase_start_tick = tick
        self._session = SleepSession(trigger_tick=tick, trigger_reason=reason)
        self._lesson_scheduled = False
        self._lesson_result = None

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
            if ticks_in_phase == 0:
                # Execute REM replay (once, synchronously)
                n, l_before, l_after = self._rem_replayer.replay_falls(
                    episodic_memory=getattr(sim, "_episodic_memory", None),
                    graph=sim.agent.graph,
                    motor_cortex=getattr(sim, "_motor_cortex", None),
                    lr_mult=_env_float("RKK_SLEEP_REM_LR_MULT", 10.0),
                )
                session.rem_episodes_replayed = n
                session.rem_loss_before = l_before
                session.rem_loss_after = l_after
                print(f"[Sleep] REM: replayed {n} episodes, loss {l_before:.4f}→{l_after:.4f}")

                # Schedule LLM lesson (async, non-blocking)
                self._schedule_lesson(tick, sim)

            if ticks_in_phase >= 30:
                self._phase = SleepPhase.LESSON
                self._phase_start_tick = tick

        # ── LESSON phase ────────────────────────────────────────────────────────
        elif self._phase == SleepPhase.LESSON:
            if ticks_in_phase >= 80:
                # Apply lesson result if arrived
                if self._lesson_result is not None:
                    self._apply_lesson(tick, sim, self._lesson_result)
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

            if ticks_in_phase >= 20:
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
        """Fire async LLM lesson (non-blocking)."""
        teacher = getattr(sim, "_llm_teacher", None)
        if teacher is None:
            return

        obs = {}
        try:
            obs = dict(sim.agent.env.observe())
        except Exception:
            pass

        total_falls = getattr(sim._episodic_memory, "total_falls_recorded", 0) if sim._episodic_memory else 0
        valid_intents = [k for k in sim.agent.graph.nodes if k.startswith("intent_")]
        valid_vars = list(sim.agent.graph.nodes.keys())

        from engine.ollama_env import get_ollama_generate_url, get_ollama_model

        self._lesson_scheduled = True

        def _on_lesson(ann):
            self._lesson_result = ann

        # Temporarily override teacher mode to "lesson"
        async def _lesson_call():
            old_count = teacher._call_count
            teacher._call_count = (
                teacher._lesson_every - 1
            )  # force lesson mode
            ann = await teacher.call_async(
                tick=tick,
                obs=obs,
                inner_voice_controller=getattr(sim, "_inner_voice", None),
                episodic_memory=getattr(sim, "_episodic_memory", None),
                curriculum=getattr(sim, "_curriculum", None),
                llm_url=get_ollama_generate_url(),
                llm_model=get_ollama_model(),
                valid_intents=valid_intents,
                valid_graph_vars=valid_vars,
                total_ticks=tick,
                total_falls=total_falls,
            )
            teacher._call_count = old_count
            if ann:
                _on_lesson(ann)

        # Agent tick runs in rkk-agent-loop thread — no asyncio loop there.
        def _run_lesson_in_thread() -> None:
            try:
                asyncio.run(_lesson_call())
            except Exception as e:
                print(f"[Sleep] Lesson async error: {e}")

        try:
            threading.Thread(
                target=_run_lesson_in_thread,
                daemon=True,
                name="rkk-sleep-lesson",
            ).start()
        except Exception as e:
            print(f"[Sleep] Lesson schedule error: {e}")

    def _apply_lesson(self, tick: int, sim, ann) -> None:
        """Apply LLM lesson annotation to InnerVoiceNet + GNN."""
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
        session = self._session
        if session:
            session.end_time = time.time()
            session.completed = True
            self._sessions.append(session)
            print(f"[Sleep] ✅ {session.summary()}")

        self._phase = SleepPhase.AWAKE
        self.last_sleep_tick = tick
        self._falls_since_sleep = 0
        self.sleep_count += 1

        # Post-sleep: advance curriculum if possible
        self._post_sleep_curriculum(tick, sim)

        # Post-sleep: inject physical curriculum skills if scheduler running low
        phys = getattr(sim, "_physical_curriculum", None)
        sched = getattr(sim, "_curriculum", None)
        if phys is not None and sched is not None:
            added = phys.inject_into_scheduler(sched)
            if added > 0:
                next_name = sched._stages[-1].name
                sim._add_event(
                    f"🏃 Physical skill unlocked: {next_name}",
                    "#aaffaa", "curriculum"
                )

    def _post_sleep_curriculum(self, tick: int, sim) -> None:
        """Check curriculum mastery after sleep."""
        sched = getattr(sim, "_curriculum", None)
        phys = getattr(sim, "_physical_curriculum", None)
        if sched is None or phys is None:
            return

        cur_stage = sched.current_stage
        # If we've been in this stage for a long time and still failing → mark failed
        if cur_stage.ticks_in_stage > cur_stage.min_ticks * 3:
            if phys._active_skill_id:
                phys.mark_failed(phys._active_skill_id)
                print(f"[Sleep] Marked {phys._active_skill_id} as failed (too long)")
                # Try next skill
                added = phys.inject_into_scheduler(sched)
                if added:
                    sched._advance_stage(tick)

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
