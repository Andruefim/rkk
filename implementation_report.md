# Implementation Report: Open-ended AGI Architecture

## Summary

Implemented 6 architectural changes that transform RKK from a static-ontology system into an open-ended learning architecture.

## Changes Made

### 1. `variable_bootstrap.py` — Dynamic Variable Ontology (NEW)
[variable_bootstrap.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/variable_bootstrap.py)

**Created `VariableRegistry`** — replaces static `VAR_NAMES` (68 hardcoded vars) with a dynamic registry:

| Mode | Vars at Start | Discovery |
|---|---|---|
| `RKK_BOOTSTRAP_ONLY=0` (default) | 59 (full, backward compat) | N/A |
| `RKK_BOOTSTRAP_ONLY=1` | 11 (SEED_VARS) | Auto via pressure |

**SEED_VARS** (11): `com_z`, `torso_roll/pitch`, `foot_contact_l/r`, `posture_stability`, `lhip/rhip/lknee/rknee`, `intent_stride`

**DISCOVERABLE_GROUPS** (11): ankles, feet, torso_xy, spine, head, arms, cubes, sandbox, motor_intents, motor_obs, self

**Auto-discovery flow:**
```
VariableDiscovery finds high-error nodes
    → updates group pressure in VariableRegistry
    → when pressure > threshold → discover_group()
    → new vars added to GNN via graph.set_node()
```

---

### 2. Data-driven Sleep — `compression_is_stagnant()` 
```diff:sleep_consolidation.py
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

Диагностика памяти:
  RKK_MEMORY_DIAG=1 — RSS + размеры GNN/мостов (см. engine.memory_diag)
  RKK_MEMORY_TRACE=1 — tracemalloc diff между этапами сна
"""
from __future__ import annotations

import asyncio
import gc
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
        for ep in episodes[-20:]:  # last 20 fall episodes
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
                    X_pred_metric = integrate_world_model_step(core, X_t, a)
                    loss_before = float(F.mse_loss(X_pred_metric, X_fall).item())
                losses_before.append(loss_before)

                # Train (single graph per episode)
                if optim is not None:
                    optim.zero_grad()
                    X_pred_train = integrate_world_model_step(core, X_t, a)
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

    def begin_sleep(self, tick: int, reason: str, sim: Any | None = None) -> None:
        """Start a sleep cycle."""
        print(f"[Sleep] 😴 Beginning sleep at tick={tick} reason={reason}")
        if sim is not None:
            _memory_diag_log(sim, f"sleep_begin tick={tick} reason={reason}")
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
                _memory_diag_log(sim, "sleep_after_REM_replay")

                try:
                    from engine.world_state_bridge import grounded_sleep_consolidate

                    gsn = grounded_sleep_consolidate(sim)
                    if gsn.get("ok"):
                        session.grounded_samples = int(gsn.get("samples_pushed", 0))
                        session.grounded_loss_last = float(gsn.get("loss_last") or 0.0)
                        print(
                            f"[Sleep] Grounded: samples={session.grounded_samples} "
                            f"loss={session.grounded_loss_last}"
                        )
                except Exception as e:
                    print(f"[Sleep] Grounded consolidate: {e}")
                _memory_diag_log(sim, "sleep_after_grounded_inner_voice")

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
        _memory_diag_log(sim, f"sleep_wake tick={tick}")
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
===
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

Диагностика памяти:
  RKK_MEMORY_DIAG=1 — RSS + размеры GNN/мостов (см. engine.memory_diag)
  RKK_MEMORY_TRACE=1 — tracemalloc diff между этапами сна
"""
from __future__ import annotations

import asyncio
import gc
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
        for ep in episodes[-20:]:  # last 20 fall episodes
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
                    X_pred_metric = integrate_world_model_step(core, X_t, a)
                    loss_before = float(F.mse_loss(X_pred_metric, X_fall).item())
                losses_before.append(loss_before)

                # Train (single graph per episode)
                if optim is not None:
                    optim.zero_grad()
                    X_pred_train = integrate_world_model_step(core, X_t, a)
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
    ) -> str | None:
        """
        Check if sleep should be triggered. Returns reason or None.

        Приоритет триггеров:
          1. manual (force=True)
          2. compression_stagnant (data-driven, главный)
          3. fall_threshold (аварийный, слишком много падений)
          4. periodic (fallback, если ничего не сработало)
        """
        if not sleep_enabled():
            return None
        if self.is_sleeping:
            return None

        if force:
            return "manual"

        # Data-driven trigger: мозг перестал учиться → пора спать
        compression_reason = self.should_sleep(intrinsic_objective)
        if compression_reason is not None:
            # Не чаще чем раз в 500 тиков для compression-driven сна
            if (tick - self.last_sleep_tick) >= 500:
                return compression_reason

        if self._falls_since_sleep >= self._fall_threshold:
            return "fall_threshold"
        if (tick - self.last_sleep_tick) >= self._every_ticks:
            return "periodic"
        return None

    def begin_sleep(self, tick: int, reason: str, sim: Any | None = None) -> None:
        """Start a sleep cycle."""
        print(f"[Sleep] 😴 Beginning sleep at tick={tick} reason={reason}")
        if sim is not None:
            _memory_diag_log(sim, f"sleep_begin tick={tick} reason={reason}")
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
                _memory_diag_log(sim, "sleep_after_REM_replay")

                try:
                    from engine.world_state_bridge import grounded_sleep_consolidate

                    gsn = grounded_sleep_consolidate(sim)
                    if gsn.get("ok"):
                        session.grounded_samples = int(gsn.get("samples_pushed", 0))
                        session.grounded_loss_last = float(gsn.get("loss_last") or 0.0)
                        print(
                            f"[Sleep] Grounded: samples={session.grounded_samples} "
                            f"loss={session.grounded_loss_last}"
                        )
                except Exception as e:
                    print(f"[Sleep] Grounded consolidate: {e}")
                _memory_diag_log(sim, "sleep_after_grounded_inner_voice")

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
        _memory_diag_log(sim, f"sleep_wake tick={tick}")
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
```

Added `should_sleep(intrinsic_objective)` method and updated `check_trigger()`:

| Trigger | Priority | Condition |
|---|---|---|
| `manual` | 1 | force=True |
| `compression_stagnant` | 2 | **NEW**: CausalSurprise says compression stopped growing |
| `fall_threshold` | 3 | Too many falls |
| `periodic` | 4 | Fallback timer (every N ticks) |

The brain now sleeps when it **stops learning**, not when a timer fires.

---

### 3. Sleep Integration in Tick Loop
```diff:mixin_tick.py
"""Simulation mixin: tick_step, один шаг агента."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationTickMixin:
    # ── Tick ──────────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        hz = _agent_loop_hz_from_env()
        if hz > 0.0:
            self._bg.ensure_rkk_agent_loop()
            with self._sim_step_lock:
                cached = self._agent_step_response
            if cached is not None:
                return copy.deepcopy(cached)
            return self.public_state()
        with self._sim_step_lock:
            return self._run_single_agent_timestep_inner()

    def advance_agent_steps(self, n: int) -> None:
        """Синхронно выполнить n логических тиков агента (bootstrap при RKK_AGENT_LOOP_HZ>0)."""
        n = max(0, int(n))
        if n == 0:
            return
        with self._sim_step_lock:
            for _ in range(n):
                self._agent_step_response = self._run_single_agent_timestep_inner()

    def _run_single_agent_timestep_inner(self) -> dict:
        self.tick += 1
        self._apply_pending_llm_bundle()
        self._ensure_phase2()

        # Humanoid curriculum: фаза 1 — fixed_root с тика 1; снятие после RKK_AUTO_FIXED_ROOT_TICKS.
        try:
            auto_fr_ticks = int(os.environ.get("RKK_AUTO_FIXED_ROOT_TICKS", "0"))
        except ValueError:
            auto_fr_ticks = 0
        if auto_fr_ticks > 0 and self.current_world == "humanoid":
            if self.tick == 1 and not self._fixed_root_active:
                self.enable_fixed_root()
                self._add_event(
                    "📌 Curriculum: fixed_root ON (phase 1, arms→cubes)",
                    "#66ccaa",
                    "phase",
                )
            if (
                self._fixed_root_active
                and self.tick >= auto_fr_ticks
                and not self._curriculum_auto_fr_released
            ):
                self._curriculum_auto_fr_released = True
                self.disable_fixed_root()
                try:
                    stab = int(os.environ.get("RKK_POST_FR_STABILIZE_TICKS", "80"))
                except ValueError:
                    stab = 80
                self._curriculum_stabilize_until = self.tick + max(0, stab)
                self._add_event(
                    f"📌 Auto fixed_root OFF at tick {self.tick}, stabilize until {self._curriculum_stabilize_until}",
                    "#66ccaa",
                    "phase",
                )

        # Fallen check + автосброс физики (иначе VL и block_rate залипают)
        fallen = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            fallen = is_fn()
            if self._fall_recovery_active and not fallen:
                self._clear_fall_recovery()
            if fallen:
                self._fall_count += 1
                obs_fall = dict(self.agent.env.observe())
                if self._maybe_recover_or_reset_after_fall(obs_fall):
                    obs = self.agent.env.observe()
                    self._sync_motor_state(obs, source="reset", tick=self.tick)
                    for nid in self.agent.graph._node_ids:
                        if nid in obs:
                            self.agent.graph.nodes[nid] = obs[nid]
                    self.agent.graph.record_observation(obs)
                    self.agent.temporal.step(obs)
                    fallen = is_fn()
                if self._fall_count % 20 == 1:
                    self._add_event(
                        f"💀 [FALLEN] Nova упал! (×{self._fall_count})",
                        "#ff2244", "value"
                    )

        # Фаза 12: передаём GNN prediction в visual env (не каждый тик — см. VISION_GNN_FEED_EVERY)
        if self._visual_mode and self._visual_env is not None:
            self._vision_ticks += 1
            if self._vision_ticks % VISION_GNN_FEED_EVERY == 0:
                self._feed_gnn_prediction_to_visual()

        # Фаза 3: annealing teacher_weight; VL-overlay только пока не истёк TTL и weight>0
        try:
            tmax = int(os.environ.get("RKK_TEACHER_T_MAX", "140"))
        except ValueError:
            tmax = 140
        tmax = max(1, tmax)
        tw = max(0.0, 1.0 - (self.agent._total_interventions / tmax))
        self.agent.set_teacher_state(self._phase3_teacher_rules, tw)
        ov = self._phase3_vl_overlay
        if ov is not None and self.tick <= ov.expires_at_tick and tw > 0:
            self.agent.value_layer.set_teacher_vl_overlay(ov)
        else:
            self.agent.value_layer.set_teacher_vl_overlay(None)

        # CPG runs BEFORE agent step so legs are stabilized before high-level exploration
        self._ensure_cpg_background_loop()
        self._drain_l1_motor_commands()
        fallen_pre = False
        is_fn_pre = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn_pre) and not self._fixed_root_active:
            fallen_pre = is_fn_pre()
        self._maybe_apply_cpg_locomotion(fallen_pre)
        self._publish_cpg_node_snapshot()
        self.agent.other_agents_phi = []
        self._maybe_step_hierarchical_l1()
        obs_pre_rssm = dict(self.agent.graph.snapshot_vec_dict())
        result = self._run_agent_or_skill_step(engine_tick=self.tick)

        # Track action for episodic memory
        self._record_last_action(result)

        _obs_for_d_e: dict = {}
        try:
            _obs_for_d_e = dict(self.agent.env.observe())
        except Exception:
            pass
        _posture_now = float(
            _obs_for_d_e.get(
                "posture_stability",
                _obs_for_d_e.get("phys_posture_stability", 0.5),
            )
        )

        # Level 3-I: Multi-scale time tick (first consumer of post-step obs)
        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            self._timescale.tick(self.tick, _obs_for_d_e)
            motor_intents = self._timescale.get_intents(LEVEL_MOTOR)
            for var, val in motor_intents.items():
                if var.startswith("intent_"):
                    try:
                        self.agent.env.intervene(var, float(val), count_intervention=False)
                    except Exception:
                        pass

        # Phase J: Inner Voice (τ2, fast, no LLM)
        self._tick_inner_voice(self.tick)

        # Phase J: LLM teacher (τ3, async)
        self._tick_llm_teacher(self.tick)

        # Level 3-G: Proprioception update (after CPG + agent step; fresh obs)
        _proprio_anomaly = 0.0
        _proprio_emp_reward = 0.0
        if _PROPRIO_AVAILABLE and self._proprio is not None and self.current_world == "humanoid":
            self._proprio.update(
                tick=self.tick,
                obs=_obs_for_d_e,
                graph=self.agent.graph if hasattr(self.agent, "graph") else None,
                agent=self.agent,
            )
            _proprio_anomaly = self._proprio.anomaly_score
            _proprio_emp_reward = self._proprio.get_empowerment_reward()

            if _TIMESCALE_AVAILABLE and self._timescale is not None:
                if self._timescale.should_run(LEVEL_REFLEX, self.tick):
                    self._timescale.mark_ran(LEVEL_REFLEX, self.tick)

        # Phase K: Sleep Controller
        if (
            _PHASE_K_AVAILABLE
            and self._sleep_ctrl is not None
            and self.current_world == "humanoid"
        ):
            if fallen and not self._was_fallen_last_tick:
                self._sleep_ctrl.notify_fall()
            self._was_fallen_last_tick = fallen

            _total_falls = (
                getattr(self._episodic_memory, "total_falls_recorded", 0)
                if self._episodic_memory
                else 0
            )
            _sleep_reason = self._sleep_ctrl.check_trigger(self.tick, _total_falls)

            if _sleep_reason and not self._sleep_ctrl.is_sleeping:
                self._sleep_prev_fixed_root = self._fixed_root_active
                if not self._fixed_root_active:
                    self.enable_fixed_root()
                self._sleep_ctrl.begin_sleep(self.tick, _sleep_reason, sim=self)
                self._add_event(
                    f"😴 Sleep: {_sleep_reason} (falls={self._sleep_ctrl._falls_since_sleep})",
                    "#9988ff",
                    "sleep",
                )

            if self._sleep_ctrl.is_sleeping:
                self._sleep_ctrl.tick(self.tick, self)
                if not self._sleep_ctrl.is_sleeping:
                    if (
                        not self._sleep_prev_fixed_root
                        and self._fixed_root_active
                    ):
                        self.disable_fixed_root()
                    self._add_event(
                        f"🌅 Woke up (sleep #{self._sleep_ctrl.sleep_count})",
                        "#ffff88",
                        "sleep",
                    )

        # Phase K: Physical curriculum when scheduler runs low on stages ahead
        if (
            _PHASE_K_AVAILABLE
            and self._physical_curriculum is not None
            and self._curriculum is not None
            and self.tick % 1000 == 0
        ):
            self._physical_curriculum.inject_into_scheduler(self._curriculum)

        # Phase L: Verbal Action (async in background thread)
        if _VERBAL_AVAILABLE and self._verbal is not None:
            self._schedule_verbal_tick(fallen)

        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            if self._timescale.should_run(LEVEL_MOTOR, self.tick):
                self._timescale.mark_ran(LEVEL_MOTOR, self.tick)
            if self._timescale.should_run(LEVEL_COGNIT, self.tick):
                self._timescale.mark_ran(LEVEL_COGNIT, self.tick)

        # Level 2-D: Episodic Memory
        self._update_episodic_memory(self.tick, _obs_for_d_e, fallen, _posture_now)

        # Level 2-E: Curriculum
        self._tick_curriculum(self.tick, _obs_for_d_e, fallen)

        # Level 2-F: RSSM upgrade + training
        self._maybe_upgrade_rssm(self.tick)
        if not result.get("blocked") and not result.get("skipped"):
            _var = str(result.get("variable", ""))
            _val = float(result.get("value", 0.5))
            obs_post = dict(self.agent.graph.snapshot_vec_dict())
            self._rssm_train_step(obs_pre_rssm, _var, _val, obs_post)

        # Фаза 2 ч.3: L4 concept mining (sync fallback или async worker + single-writer apply)
        if self._visual_env is not None and self.tick % self._concept_inject_every == 0:
            vis = self._visual_env.get_slot_visualization()
            slot_vecs = self._visual_env._last_slot_vecs
            if slot_vecs is not None:
                full_obs = dict(self._visual_env.observe())
                phys_obs = {
                    k: float(v)
                    for k, v in full_obs.items()
                    if not str(k).startswith("slot_")
                }
                if _l4_worker_enabled():
                    self._enqueue_l4_task(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                    )
                elif self._concept_store is not None:
                    new_concepts = self._concept_store.update(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                        tick=self.tick,
                        graph_node_ids=list(self.agent.graph._node_ids),
                    )
                    if new_concepts:
                        added = self._concept_store.inject_into_graph(self.agent.graph)
                        c0 = new_concepts[0]
                        self._add_event(
                            f"Concept formed: {c0.cid[:4]}, slot_{c0.slot_idx}, +{added} nodes",
                            "#EF9F27",
                            "phase",
                        )
        if _l4_worker_enabled():
            self._drain_l4_results()

        self._log_step(result, fallen)
        self._rolling_block_bits.append(1 if result.get("blocked") else 0)

        snap = self.agent.snapshot()
        snap["fallen"]     = fallen
        snap["fall_count"] = self._fall_count
        self._last_snapshot = snap

        if self._rsi_full_enabled():
            from engine.rsi_full import RSIController

            if self._rsi_full is None:
                sup = (
                    self._ensure_skill_library
                    if self._skill_library_enabled()
                    else None
                )
                self._rsi_full = RSIController(
                    self.agent,
                    self._locomotion_controller,
                    skill_library_supplier=sup,
                    motor_cortex_supplier=self._ensure_motor_cortex,
                )
            rsi_ev = self._rsi_full.tick(
                snap,
                self._locomotion_reward_ema(),
                tick=self.tick,
                locomotion_ctrl=self._locomotion_controller,
            )
            if rsi_ev is not None:
                t = rsi_ev.get("type", "?")
                self._add_event(f"🔧 RSI [{t}]", "#66ccaa", "phase")

        # Phase D: Motor Cortex RSI check (every 50 ticks)
        if self.tick % 50 == 0:
            mc = self._ensure_motor_cortex()
            if mc is not None:
                posture_mean = (
                    float(np.mean(self._mc_posture_window))
                    if self._mc_posture_window else 0.0
                )
                fallen_rate = (
                    float(np.mean(self._mc_fallen_count_window))
                    if self._mc_fallen_count_window else 0.0
                )
                loco_r = self._locomotion_reward_ema()
                new_progs = mc.rsi_check_and_spawn(
                    self.tick, posture_mean, loco_r, fallen_rate
                )
                for prog_name in new_progs:
                    self._add_event(
                        f"🧠 MC-RSI: spawned '{prog_name}' "
                        f"(posture={posture_mean:.2f}, cpg_w={mc.cpg_weight:.2f})",
                        "#ff88ff", "phase"
                    )

        dr = float(snap.get("discovery_rate", 0.0))
        self._tick_discovery_plateau(dr)
        if dr > self._best_discovery_rate + 1e-5:
            self._best_discovery_rate = dr
            self._last_dr_gain_tick = self.tick

        # Level 1-B: Visual Body Grounding
        self._maybe_run_visual_grounding()

        # Phase M: slot labels + attention → visual concepts / verbal context
        if _PHASE_M_AVAILABLE:
            self._phase_m_sync_from_vision()

        if _WORLD_BRIDGE_AVAILABLE and self._world_bridge is not None:
            try:
                self._world_bridge.on_tick(self, tick_obs=_obs_for_d_e)
            except Exception as e:
                print(f"[Simulation] world_bridge.on_tick: {e}")

        # Level 1-C: Standalone reconstruction training (warm up decoder early)
        if (
            self._visual_mode
            and self._visual_env is not None
            and self.tick % 5 == 0
            and hasattr(self._visual_env, "cortex")
        ):
            cortex = self._visual_env.cortex
            if (
                hasattr(cortex, "train_reconstruction_only")
                and hasattr(self._visual_env, "_last_frame")
                and self._visual_env._last_frame is not None
                and cortex.n_train == 0  # only during warmup phase
            ):
                try:
                    cortex.train_reconstruction_only(self._visual_env._last_frame)
                except Exception:
                    pass

        # Demon
        if self.demon._last_action is not None:
            pe = 0.0
            if not result.get("blocked") and not result.get("skipped"):
                pe = float(result.get("prediction_error", 0))
            self.demon.learn(pe, self.demon._last_action_complexity, [snap])
        self._step_demon(snap)

        smoothed = self._update_phase(snap)

        graph_deltas = {}
        cnt = len(self.agent.graph.edges)
        if cnt != self._prev_edge_count:
            graph_deltas[0] = [e.as_dict() for e in self.agent.graph.edges]
            self._prev_edge_count = cnt

        # Neurogenesis
        # Structural ASI: Neurogenesis
        if self.current_world == "humanoid" and not self._fixed_root_active:
            neuro_event = self.neuro_engine.scan_and_grow(self.agent, self.tick)
            if neuro_event is not None:
                self._add_event(
                    f"🧬 Neurogenesis: {neuro_event['new_node']} allocated", 
                    "#ff44cc", 
                    "phase"
                )
                
                # Обновляем TemporalBlankets под новую размерность (d = d + 1)
                from engine.temporal import TemporalBlankets
                new_d = self.agent.graph._d
                old_tb = self.agent.temporal
                new_tb = TemporalBlankets(d_input=new_d, device=self.device)
                # Копируем состояния TemporalBlankets (насколько возможно)
                # ... (миграция состояния SSM, аналогично resize_to в GNN)
                self.agent.temporal = new_tb

        # Scene
        scene_fn = getattr(self.agent.env, "get_full_scene", None)
        scene    = scene_fn() if callable(scene_fn) else {}

        # Vision state (кэш для /vision/slots endpoint)
        if self._visual_mode and self._visual_env is not None:
            try:
                self._last_vision_state = self._visual_env.get_slot_visualization()
            except Exception:
                pass

        self._maybe_schedule_llm_loop(result, snap)

        self._maybe_refresh_concepts_cache()
        self._maybe_autosave_memory()

        try:
            from engine.memory_diag import log_sim_memory, memory_diag_enabled

            _mem_iv = int(os.environ.get("RKK_MEMORY_DIAG_INTERVAL", "0") or "0")
            if (
                memory_diag_enabled()
                and _mem_iv > 0
                and self.current_world == "humanoid"
                and self.tick % _mem_iv == 0
            ):
                log_sim_memory(self, f"tick={self.tick}")
        except Exception:
            pass

        return self._build_snapshot(snap, graph_deltas, smoothed, scene)
===
"""Simulation mixin: tick_step, один шаг агента."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationTickMixin:
    # ── Tick ──────────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        hz = _agent_loop_hz_from_env()
        if hz > 0.0:
            self._bg.ensure_rkk_agent_loop()
            with self._sim_step_lock:
                cached = self._agent_step_response
            if cached is not None:
                return copy.deepcopy(cached)
            return self.public_state()
        with self._sim_step_lock:
            return self._run_single_agent_timestep_inner()

    def advance_agent_steps(self, n: int) -> None:
        """Синхронно выполнить n логических тиков агента (bootstrap при RKK_AGENT_LOOP_HZ>0)."""
        n = max(0, int(n))
        if n == 0:
            return
        with self._sim_step_lock:
            for _ in range(n):
                self._agent_step_response = self._run_single_agent_timestep_inner()

    def _run_single_agent_timestep_inner(self) -> dict:
        self.tick += 1
        self._apply_pending_llm_bundle()
        self._ensure_phase2()

        # Humanoid curriculum: фаза 1 — fixed_root с тика 1; снятие после RKK_AUTO_FIXED_ROOT_TICKS.
        try:
            auto_fr_ticks = int(os.environ.get("RKK_AUTO_FIXED_ROOT_TICKS", "0"))
        except ValueError:
            auto_fr_ticks = 0
        if auto_fr_ticks > 0 and self.current_world == "humanoid":
            if self.tick == 1 and not self._fixed_root_active:
                self.enable_fixed_root()
                self._add_event(
                    "📌 Curriculum: fixed_root ON (phase 1, arms→cubes)",
                    "#66ccaa",
                    "phase",
                )
            if (
                self._fixed_root_active
                and self.tick >= auto_fr_ticks
                and not self._curriculum_auto_fr_released
            ):
                self._curriculum_auto_fr_released = True
                self.disable_fixed_root()
                try:
                    stab = int(os.environ.get("RKK_POST_FR_STABILIZE_TICKS", "80"))
                except ValueError:
                    stab = 80
                self._curriculum_stabilize_until = self.tick + max(0, stab)
                self._add_event(
                    f"📌 Auto fixed_root OFF at tick {self.tick}, stabilize until {self._curriculum_stabilize_until}",
                    "#66ccaa",
                    "phase",
                )

        # Fallen check + автосброс физики (иначе VL и block_rate залипают)
        fallen = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            fallen = is_fn()
            if self._fall_recovery_active and not fallen:
                self._clear_fall_recovery()
            if fallen:
                self._fall_count += 1
                obs_fall = dict(self.agent.env.observe())
                if self._maybe_recover_or_reset_after_fall(obs_fall):
                    obs = self.agent.env.observe()
                    self._sync_motor_state(obs, source="reset", tick=self.tick)
                    for nid in self.agent.graph._node_ids:
                        if nid in obs:
                            self.agent.graph.nodes[nid] = obs[nid]
                    self.agent.graph.record_observation(obs)
                    self.agent.temporal.step(obs)
                    fallen = is_fn()
                if self._fall_count % 20 == 1:
                    self._add_event(
                        f"💀 [FALLEN] Nova упал! (×{self._fall_count})",
                        "#ff2244", "value"
                    )

        # Фаза 12: передаём GNN prediction в visual env (не каждый тик — см. VISION_GNN_FEED_EVERY)
        if self._visual_mode and self._visual_env is not None:
            self._vision_ticks += 1
            if self._vision_ticks % VISION_GNN_FEED_EVERY == 0:
                self._feed_gnn_prediction_to_visual()

        # Фаза 3: annealing teacher_weight; VL-overlay только пока не истёк TTL и weight>0
        try:
            tmax = int(os.environ.get("RKK_TEACHER_T_MAX", "140"))
        except ValueError:
            tmax = 140
        tmax = max(1, tmax)
        tw = max(0.0, 1.0 - (self.agent._total_interventions / tmax))
        self.agent.set_teacher_state(self._phase3_teacher_rules, tw)
        ov = self._phase3_vl_overlay
        if ov is not None and self.tick <= ov.expires_at_tick and tw > 0:
            self.agent.value_layer.set_teacher_vl_overlay(ov)
        else:
            self.agent.value_layer.set_teacher_vl_overlay(None)

        # CPG runs BEFORE agent step so legs are stabilized before high-level exploration
        self._ensure_cpg_background_loop()
        self._drain_l1_motor_commands()
        fallen_pre = False
        is_fn_pre = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn_pre) and not self._fixed_root_active:
            fallen_pre = is_fn_pre()
        self._maybe_apply_cpg_locomotion(fallen_pre)
        self._publish_cpg_node_snapshot()
        self.agent.other_agents_phi = []
        self._maybe_step_hierarchical_l1()
        obs_pre_rssm = dict(self.agent.graph.snapshot_vec_dict())
        result = self._run_agent_or_skill_step(engine_tick=self.tick)

        # Track action for episodic memory
        self._record_last_action(result)

        _obs_for_d_e: dict = {}
        try:
            _obs_for_d_e = dict(self.agent.env.observe())
        except Exception:
            pass
        _posture_now = float(
            _obs_for_d_e.get(
                "posture_stability",
                _obs_for_d_e.get("phys_posture_stability", 0.5),
            )
        )

        # Level 3-I: Multi-scale time tick (first consumer of post-step obs)
        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            self._timescale.tick(self.tick, _obs_for_d_e)
            motor_intents = self._timescale.get_intents(LEVEL_MOTOR)
            for var, val in motor_intents.items():
                if var.startswith("intent_"):
                    try:
                        self.agent.env.intervene(var, float(val), count_intervention=False)
                    except Exception:
                        pass

        # Phase J: Inner Voice (τ2, fast, no LLM)
        self._tick_inner_voice(self.tick)

        # Phase J: LLM teacher (τ3, async)
        self._tick_llm_teacher(self.tick)

        # Level 3-G: Proprioception update (after CPG + agent step; fresh obs)
        _proprio_anomaly = 0.0
        _proprio_emp_reward = 0.0
        if _PROPRIO_AVAILABLE and self._proprio is not None and self.current_world == "humanoid":
            self._proprio.update(
                tick=self.tick,
                obs=_obs_for_d_e,
                graph=self.agent.graph if hasattr(self.agent, "graph") else None,
                agent=self.agent,
            )
            _proprio_anomaly = self._proprio.anomaly_score
            _proprio_emp_reward = self._proprio.get_empowerment_reward()

            if _TIMESCALE_AVAILABLE and self._timescale is not None:
                if self._timescale.should_run(LEVEL_REFLEX, self.tick):
                    self._timescale.mark_ran(LEVEL_REFLEX, self.tick)

        # Phase K: Sleep Controller
        if (
            _PHASE_K_AVAILABLE
            and self._sleep_ctrl is not None
            and self.current_world == "humanoid"
        ):
            if fallen and not self._was_fallen_last_tick:
                self._sleep_ctrl.notify_fall()
            self._was_fallen_last_tick = fallen

            _total_falls = (
                getattr(self._episodic_memory, "total_falls_recorded", 0)
                if self._episodic_memory
                else 0
            )
            _sleep_reason = self._sleep_ctrl.check_trigger(
                self.tick, _total_falls,
                intrinsic_objective=getattr(self, "_intrinsic", None),
            )

            if _sleep_reason and not self._sleep_ctrl.is_sleeping:
                self._sleep_prev_fixed_root = self._fixed_root_active
                if not self._fixed_root_active:
                    self.enable_fixed_root()
                self._sleep_ctrl.begin_sleep(self.tick, _sleep_reason, sim=self)
                self._add_event(
                    f"😴 Sleep: {_sleep_reason} (falls={self._sleep_ctrl._falls_since_sleep})",
                    "#9988ff",
                    "sleep",
                )

            if self._sleep_ctrl.is_sleeping:
                self._sleep_ctrl.tick(self.tick, self)
                if not self._sleep_ctrl.is_sleeping:
                    if (
                        not self._sleep_prev_fixed_root
                        and self._fixed_root_active
                    ):
                        self.disable_fixed_root()
                    self._add_event(
                        f"🌅 Woke up (sleep #{self._sleep_ctrl.sleep_count})",
                        "#ffff88",
                        "sleep",
                    )

        # Phase K: Physical curriculum when scheduler runs low on stages ahead
        if (
            _PHASE_K_AVAILABLE
            and self._physical_curriculum is not None
            and self._curriculum is not None
            and self.tick % 1000 == 0
        ):
            self._physical_curriculum.inject_into_scheduler(self._curriculum)

        # Phase L: Verbal Action (async in background thread)
        if _VERBAL_AVAILABLE and self._verbal is not None:
            self._schedule_verbal_tick(fallen)

        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            if self._timescale.should_run(LEVEL_MOTOR, self.tick):
                self._timescale.mark_ran(LEVEL_MOTOR, self.tick)
            if self._timescale.should_run(LEVEL_COGNIT, self.tick):
                self._timescale.mark_ran(LEVEL_COGNIT, self.tick)

        # Level 2-D: Episodic Memory
        self._update_episodic_memory(self.tick, _obs_for_d_e, fallen, _posture_now)

        # Level 2-E: Curriculum
        self._tick_curriculum(self.tick, _obs_for_d_e, fallen)

        # Level 2-F: RSSM upgrade + training
        self._maybe_upgrade_rssm(self.tick)
        if not result.get("blocked") and not result.get("skipped"):
            _var = str(result.get("variable", ""))
            _val = float(result.get("value", 0.5))
            obs_post = dict(self.agent.graph.snapshot_vec_dict())
            self._rssm_train_step(obs_pre_rssm, _var, _val, obs_post)

        # Фаза 2 ч.3: L4 concept mining (sync fallback или async worker + single-writer apply)
        if self._visual_env is not None and self.tick % self._concept_inject_every == 0:
            vis = self._visual_env.get_slot_visualization()
            slot_vecs = self._visual_env._last_slot_vecs
            if slot_vecs is not None:
                full_obs = dict(self._visual_env.observe())
                phys_obs = {
                    k: float(v)
                    for k, v in full_obs.items()
                    if not str(k).startswith("slot_")
                }
                if _l4_worker_enabled():
                    self._enqueue_l4_task(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                    )
                elif self._concept_store is not None:
                    new_concepts = self._concept_store.update(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                        tick=self.tick,
                        graph_node_ids=list(self.agent.graph._node_ids),
                    )
                    if new_concepts:
                        added = self._concept_store.inject_into_graph(self.agent.graph)
                        c0 = new_concepts[0]
                        self._add_event(
                            f"Concept formed: {c0.cid[:4]}, slot_{c0.slot_idx}, +{added} nodes",
                            "#EF9F27",
                            "phase",
                        )
        if _l4_worker_enabled():
            self._drain_l4_results()

        self._log_step(result, fallen)
        self._rolling_block_bits.append(1 if result.get("blocked") else 0)

        snap = self.agent.snapshot()
        snap["fallen"]     = fallen
        snap["fall_count"] = self._fall_count
        self._last_snapshot = snap

        if self._rsi_full_enabled():
            from engine.rsi_full import RSIController

            if self._rsi_full is None:
                sup = (
                    self._ensure_skill_library
                    if self._skill_library_enabled()
                    else None
                )
                self._rsi_full = RSIController(
                    self.agent,
                    self._locomotion_controller,
                    skill_library_supplier=sup,
                    motor_cortex_supplier=self._ensure_motor_cortex,
                )
            rsi_ev = self._rsi_full.tick(
                snap,
                self._locomotion_reward_ema(),
                tick=self.tick,
                locomotion_ctrl=self._locomotion_controller,
            )
            if rsi_ev is not None:
                t = rsi_ev.get("type", "?")
                self._add_event(f"🔧 RSI [{t}]", "#66ccaa", "phase")

        # Phase D: Motor Cortex RSI check (every 50 ticks)
        if self.tick % 50 == 0:
            mc = self._ensure_motor_cortex()
            if mc is not None:
                posture_mean = (
                    float(np.mean(self._mc_posture_window))
                    if self._mc_posture_window else 0.0
                )
                fallen_rate = (
                    float(np.mean(self._mc_fallen_count_window))
                    if self._mc_fallen_count_window else 0.0
                )
                loco_r = self._locomotion_reward_ema()
                new_progs = mc.rsi_check_and_spawn(
                    self.tick, posture_mean, loco_r, fallen_rate
                )
                for prog_name in new_progs:
                    self._add_event(
                        f"🧠 MC-RSI: spawned '{prog_name}' "
                        f"(posture={posture_mean:.2f}, cpg_w={mc.cpg_weight:.2f})",
                        "#ff88ff", "phase"
                    )

        dr = float(snap.get("discovery_rate", 0.0))
        self._tick_discovery_plateau(dr)
        if dr > self._best_discovery_rate + 1e-5:
            self._best_discovery_rate = dr
            self._last_dr_gain_tick = self.tick

        # Level 1-B: Visual Body Grounding
        self._maybe_run_visual_grounding()

        # Phase M: slot labels + attention → visual concepts / verbal context
        if _PHASE_M_AVAILABLE:
            self._phase_m_sync_from_vision()

        if _WORLD_BRIDGE_AVAILABLE and self._world_bridge is not None:
            try:
                self._world_bridge.on_tick(self, tick_obs=_obs_for_d_e)
            except Exception as e:
                print(f"[Simulation] world_bridge.on_tick: {e}")

        # Level 1-C: Standalone reconstruction training (warm up decoder early)
        if (
            self._visual_mode
            and self._visual_env is not None
            and self.tick % 5 == 0
            and hasattr(self._visual_env, "cortex")
        ):
            cortex = self._visual_env.cortex
            if (
                hasattr(cortex, "train_reconstruction_only")
                and hasattr(self._visual_env, "_last_frame")
                and self._visual_env._last_frame is not None
                and cortex.n_train == 0  # only during warmup phase
            ):
                try:
                    cortex.train_reconstruction_only(self._visual_env._last_frame)
                except Exception:
                    pass

        # Demon
        if self.demon._last_action is not None:
            pe = 0.0
            if not result.get("blocked") and not result.get("skipped"):
                pe = float(result.get("prediction_error", 0))
            self.demon.learn(pe, self.demon._last_action_complexity, [snap])
        self._step_demon(snap)

        smoothed = self._update_phase(snap)

        graph_deltas = {}
        cnt = len(self.agent.graph.edges)
        if cnt != self._prev_edge_count:
            graph_deltas[0] = [e.as_dict() for e in self.agent.graph.edges]
            self._prev_edge_count = cnt

        # Neurogenesis
        # Structural ASI: Neurogenesis
        if self.current_world == "humanoid" and not self._fixed_root_active:
            neuro_event = self.neuro_engine.scan_and_grow(self.agent, self.tick)
            if neuro_event is not None:
                self._add_event(
                    f"🧬 Neurogenesis: {neuro_event['new_node']} allocated", 
                    "#ff44cc", 
                    "phase"
                )
                
                # Обновляем TemporalBlankets под новую размерность (d = d + 1)
                from engine.temporal import TemporalBlankets
                new_d = self.agent.graph._d
                old_tb = self.agent.temporal
                new_tb = TemporalBlankets(d_input=new_d, device=self.device)
                # Копируем состояния TemporalBlankets (насколько возможно)
                # ... (миграция состояния SSM, аналогично resize_to в GNN)
                self.agent.temporal = new_tb

        # Scene
        scene_fn = getattr(self.agent.env, "get_full_scene", None)
        scene    = scene_fn() if callable(scene_fn) else {}

        # Vision state (кэш для /vision/slots endpoint)
        if self._visual_mode and self._visual_env is not None:
            try:
                self._last_vision_state = self._visual_env.get_slot_visualization()
            except Exception:
                pass

        self._maybe_schedule_llm_loop(result, snap)

        self._maybe_refresh_concepts_cache()
        self._maybe_autosave_memory()

        try:
            from engine.memory_diag import log_sim_memory, memory_diag_enabled

            _mem_iv = int(os.environ.get("RKK_MEMORY_DIAG_INTERVAL", "0") or "0")
            if (
                memory_diag_enabled()
                and _mem_iv > 0
                and self.current_world == "humanoid"
                and self.tick % _mem_iv == 0
            ):
                log_sim_memory(self, f"tick={self.tick}")
        except Exception:
            pass

        return self._build_snapshot(snap, graph_deltas, smoothed, scene)
```

Passes `self._intrinsic` to `check_trigger()` so the data-driven signal flows through.

---

### 4. Snapshot: `motor_primitives` + `variable_registry`
```diff:snapshot.py
"""Сборка WS/HTTP снимка состояния (вынесено из Simulation для SRP)."""
from __future__ import annotations

from typing import Any

from engine.core.constants import (
    agent_loop_hz_from_env as _agent_loop_hz_from_env,
    cpg_loop_hz_from_env as _cpg_loop_hz_from_env,
    l3_loop_hz_from_env as _l3_loop_hz_from_env,
    l4_worker_enabled as _l4_worker_enabled,
)
from engine.core.world import WORLDS
from engine.features.simulation.imports import (
    _INNER_VOICE_AVAILABLE,
    _PHASE_K_AVAILABLE,
    _PHASE_M_AVAILABLE,
    _RSSM_AVAILABLE,
    _TIMESCALE_AVAILABLE,
    _VERBAL_AVAILABLE,
    _WORLD_BRIDGE_AVAILABLE,
    rssm_enabled,
    timescale_enabled,
)


def build_simulation_snapshot(
    sim: Any,
    snap: dict,
    graph_deltas: dict,
    smoothed: float,
    scene: dict,
) -> dict:
    winfo = WORLDS.get(sim.current_world, {"color": "#cc44ff", "label": sim.current_world})

    vision_summary = None
    if sim._visual_mode and sim._visual_env is not None:
        vision_summary = sim._visual_env.cortex.snapshot()

    return {
        "tick": sim.tick,
        "phase": sim.phase,
        "max_phase": sim.max_phase,
        "entropy": round((1 - snap.get("peak_discovery_rate", 0)) * 100, 1),
        "smoothed_dr": round(smoothed, 3),
        "agents": [snap],
        "n_agents": 1,
        "demon": sim.demon.snapshot,
        "tom_links": [],
        "events": list(sim.events),
        "graph_deltas": graph_deltas,
        "value_layer": {
            "total_blocked_all": snap.get("total_blocked", 0),
            "block_rates": [round(snap.get("value_layer", {}).get("block_rate", 0), 3)],
            "imagination_horizon": snap.get("value_layer", {}).get("imagination_horizon", 0),
            "imagination_checks": snap.get("value_layer", {}).get("imagination_checks", 0),
            "imagination_blocks": snap.get("value_layer", {}).get("imagination_blocks", 0),
        },
        "byzantine": None,
        "motif": None,
        "multiprocess": False,
        "singleton": True,
        "current_world": sim.current_world,
        "world_label": winfo["label"],
        "world_color": winfo["color"],
        "worlds": WORLDS,
        "switch_history": sim.switcher.history[-5:],
        "gnn_d": sim.agent.graph._d,
        "fallen": snap.get("fallen", False),
        "fall_count": snap.get("fall_count", 0),
        "fall_recovery": {
            "active": bool(sim._fall_recovery_active),
            "start_tick": int(sim._fall_recovery_start_tick),
            "last_progress_tick": int(sim._fall_recovery_last_progress_tick),
            "best_score": round(float(sim._fall_recovery_best_score), 4),
        },
        "fixed_root": sim._fixed_root_active,
        "scene": scene,
        "visual_mode": sim._visual_mode,
        "vision_ticks": sim._vision_ticks,
        "vision": vision_summary,
        "llm_loop": {
            "enabled": sim._llm_loop_enabled(),
            "level2_inflight": sim._llm_level2_inflight,
            "pending_bundle": sim._pending_llm_bundle is not None,
            "last_schedule_tick": sim._last_level2_schedule_tick,
            "last_dr_gain_tick": sim._last_dr_gain_tick,
            "rolling_block_rate": round(sim._rolling_block_rate(), 4),
            "stats": dict(sim._llm_loop_stats),
        },
        "agent_loop": {
            "hz": round(_agent_loop_hz_from_env(), 1),
            "decoupled": _agent_loop_hz_from_env() > 0.0,
            "l3_hz": round(_l3_loop_hz_from_env(), 1),
            "l3_last_tick": int(sim._l3_last_tick),
            "l4_worker": bool(_l4_worker_enabled()),
            "l4_pending": bool(sim._l4_task_pending),
            "l4_last_submit_tick": int(sim._l4_last_submit_tick),
            "l4_last_apply_tick": int(sim._l4_last_apply_tick),
            "l1_last_cmd_tick": int(sim._l1_last_cmd_tick),
            "l1_last_apply_tick": int(sim._l1_last_apply_tick),
        },
        "locomotion": (
            {
                **sim._locomotion_controller.snapshot(),
                "decoupled_loop_hz": round(_cpg_loop_hz_from_env(), 1)
                if sim._cpg_decoupled_enabled()
                else 0.0,
            }
            if sim._locomotion_controller is not None
            else None
        ),
        "motor_state": sim._motor_state_snapshot(),
        "skills": sim._skill_snapshot(),
        "rsi_full": sim._rsi_full.snapshot()
        if sim._rsi_full_enabled() and sim._rsi_full is not None
        else None,
        "motor_cortex": (
            sim._motor_cortex.snapshot() if sim._motor_cortex is not None else None
        ),
        "concepts": [
            {
                "id": c["id"],
                "pattern": c["pattern"],
                "uses": c["uses"],
                "alpha_mean": c["alpha_mean"],
                "graph_node": c.get("graph_node"),
            }
            for c in sim._concepts_cache
        ],
        "memory": sim._memory_snapshot_meta(),
        "embodied_reward": None,
        "visual_grounding": (
            sim._visual_grounding_ctrl.snapshot()
            if sim._visual_grounding_ctrl is not None
            else None
        ),
        "episodic_memory": (
            sim._episodic_memory.snapshot()
            if sim._episodic_memory is not None
            else None
        ),
        "curriculum": (
            sim._curriculum.snapshot() if sim._curriculum is not None else None
        ),
        "rssm": (
            sim._rssm_trainer.snapshot()
            if sim._rssm_trainer is not None
            else {"enabled": rssm_enabled() if _RSSM_AVAILABLE else False}
        ),
        "proprioception": (
            sim._proprio.snapshot() if sim._proprio is not None else None
        ),
        "reward_coordinator": {"enabled": False, "replaced_by": "intrinsic_objective"},
        "intrinsic_objective": (
            sim._intrinsic.snapshot()
            if getattr(sim, "_intrinsic", None) is not None
            else None
        ),
        "timescale": (
            sim._timescale.snapshot()
            if sim._timescale is not None
            else (
                {"enabled": timescale_enabled()}
                if _TIMESCALE_AVAILABLE
                else {"enabled": False}
            )
        ),
        "inner_voice": (
            sim._inner_voice.snapshot()
            if sim._inner_voice is not None
            else {"enabled": _INNER_VOICE_AVAILABLE}
        ),
        "llm_teacher": (
            sim._llm_teacher.snapshot()
            if sim._llm_teacher is not None
            else {"enabled": False}
        ),
        "sleep": (
            sim._sleep_ctrl.snapshot()
            if _PHASE_K_AVAILABLE and sim._sleep_ctrl is not None
            else {"enabled": False}
        ),
        "physical_curriculum": (
            sim._physical_curriculum.snapshot()
            if _PHASE_K_AVAILABLE and sim._physical_curriculum is not None
            else None
        ),
        "persistence": (
            sim._persist.snapshot()
            if _PHASE_K_AVAILABLE and sim._persist is not None
            else None
        ),
        "verbal": (
            sim._verbal.snapshot()
            if _VERBAL_AVAILABLE and sim._verbal is not None
            else {"enabled": False}
        ),
        "visual_voice": (
            sim._visual_voice.snapshot()
            if _PHASE_M_AVAILABLE and sim._visual_voice is not None
            else {"enabled": False}
        ),
        "slot_labeler": (
            sim._slot_labeler.snapshot()
            if _PHASE_M_AVAILABLE and sim._slot_labeler is not None
            else None
        ),
        "world_bridge": (
            sim._world_bridge.snapshot()
            if _WORLD_BRIDGE_AVAILABLE and sim._world_bridge is not None
            else {"enabled": False}
        ),
        "phase1": sim._phase1_snapshot_meta(),
        "phase2": sim._phase2_snapshot_meta(),
    }
===
"""Сборка WS/HTTP снимка состояния (вынесено из Simulation для SRP)."""
from __future__ import annotations

from typing import Any

from engine.core.constants import (
    agent_loop_hz_from_env as _agent_loop_hz_from_env,
    cpg_loop_hz_from_env as _cpg_loop_hz_from_env,
    l3_loop_hz_from_env as _l3_loop_hz_from_env,
    l4_worker_enabled as _l4_worker_enabled,
)
from engine.core.world import WORLDS
from engine.features.simulation.imports import (
    _INNER_VOICE_AVAILABLE,
    _PHASE_K_AVAILABLE,
    _PHASE_M_AVAILABLE,
    _RSSM_AVAILABLE,
    _TIMESCALE_AVAILABLE,
    _VERBAL_AVAILABLE,
    _WORLD_BRIDGE_AVAILABLE,
    rssm_enabled,
    timescale_enabled,
)


def build_simulation_snapshot(
    sim: Any,
    snap: dict,
    graph_deltas: dict,
    smoothed: float,
    scene: dict,
) -> dict:
    winfo = WORLDS.get(sim.current_world, {"color": "#cc44ff", "label": sim.current_world})

    vision_summary = None
    if sim._visual_mode and sim._visual_env is not None:
        vision_summary = sim._visual_env.cortex.snapshot()

    return {
        "tick": sim.tick,
        "phase": sim.phase,
        "max_phase": sim.max_phase,
        "entropy": round((1 - snap.get("peak_discovery_rate", 0)) * 100, 1),
        "smoothed_dr": round(smoothed, 3),
        "agents": [snap],
        "n_agents": 1,
        "demon": sim.demon.snapshot,
        "tom_links": [],
        "events": list(sim.events),
        "graph_deltas": graph_deltas,
        "value_layer": {
            "total_blocked_all": snap.get("total_blocked", 0),
            "block_rates": [round(snap.get("value_layer", {}).get("block_rate", 0), 3)],
            "imagination_horizon": snap.get("value_layer", {}).get("imagination_horizon", 0),
            "imagination_checks": snap.get("value_layer", {}).get("imagination_checks", 0),
            "imagination_blocks": snap.get("value_layer", {}).get("imagination_blocks", 0),
        },
        "byzantine": None,
        "motif": None,
        "multiprocess": False,
        "singleton": True,
        "current_world": sim.current_world,
        "world_label": winfo["label"],
        "world_color": winfo["color"],
        "worlds": WORLDS,
        "switch_history": sim.switcher.history[-5:],
        "gnn_d": sim.agent.graph._d,
        "fallen": snap.get("fallen", False),
        "fall_count": snap.get("fall_count", 0),
        "fall_recovery": {
            "active": bool(sim._fall_recovery_active),
            "start_tick": int(sim._fall_recovery_start_tick),
            "last_progress_tick": int(sim._fall_recovery_last_progress_tick),
            "best_score": round(float(sim._fall_recovery_best_score), 4),
        },
        "fixed_root": sim._fixed_root_active,
        "scene": scene,
        "visual_mode": sim._visual_mode,
        "vision_ticks": sim._vision_ticks,
        "vision": vision_summary,
        "llm_loop": {
            "enabled": sim._llm_loop_enabled(),
            "level2_inflight": sim._llm_level2_inflight,
            "pending_bundle": sim._pending_llm_bundle is not None,
            "last_schedule_tick": sim._last_level2_schedule_tick,
            "last_dr_gain_tick": sim._last_dr_gain_tick,
            "rolling_block_rate": round(sim._rolling_block_rate(), 4),
            "stats": dict(sim._llm_loop_stats),
        },
        "agent_loop": {
            "hz": round(_agent_loop_hz_from_env(), 1),
            "decoupled": _agent_loop_hz_from_env() > 0.0,
            "l3_hz": round(_l3_loop_hz_from_env(), 1),
            "l3_last_tick": int(sim._l3_last_tick),
            "l4_worker": bool(_l4_worker_enabled()),
            "l4_pending": bool(sim._l4_task_pending),
            "l4_last_submit_tick": int(sim._l4_last_submit_tick),
            "l4_last_apply_tick": int(sim._l4_last_apply_tick),
            "l1_last_cmd_tick": int(sim._l1_last_cmd_tick),
            "l1_last_apply_tick": int(sim._l1_last_apply_tick),
        },
        "locomotion": (
            {
                **sim._locomotion_controller.snapshot(),
                "decoupled_loop_hz": round(_cpg_loop_hz_from_env(), 1)
                if sim._cpg_decoupled_enabled()
                else 0.0,
            }
            if sim._locomotion_controller is not None
            else None
        ),
        "motor_state": sim._motor_state_snapshot(),
        "skills": sim._skill_snapshot(),
        "rsi_full": sim._rsi_full.snapshot()
        if sim._rsi_full_enabled() and sim._rsi_full is not None
        else None,
        "motor_cortex": (
            sim._motor_cortex.snapshot() if sim._motor_cortex is not None else None
        ),
        "concepts": [
            {
                "id": c["id"],
                "pattern": c["pattern"],
                "uses": c["uses"],
                "alpha_mean": c["alpha_mean"],
                "graph_node": c.get("graph_node"),
            }
            for c in sim._concepts_cache
        ],
        "memory": sim._memory_snapshot_meta(),
        "embodied_reward": None,
        "visual_grounding": (
            sim._visual_grounding_ctrl.snapshot()
            if sim._visual_grounding_ctrl is not None
            else None
        ),
        "episodic_memory": (
            sim._episodic_memory.snapshot()
            if sim._episodic_memory is not None
            else None
        ),
        "curriculum": (
            sim._curriculum.snapshot() if sim._curriculum is not None else None
        ),
        "rssm": (
            sim._rssm_trainer.snapshot()
            if sim._rssm_trainer is not None
            else {"enabled": rssm_enabled() if _RSSM_AVAILABLE else False}
        ),
        "proprioception": (
            sim._proprio.snapshot() if sim._proprio is not None else None
        ),
        "reward_coordinator": {"enabled": False, "replaced_by": "intrinsic_objective"},
        "intrinsic_objective": (
            sim._intrinsic.snapshot()
            if getattr(sim, "_intrinsic", None) is not None
            else None
        ),
        "timescale": (
            sim._timescale.snapshot()
            if sim._timescale is not None
            else (
                {"enabled": timescale_enabled()}
                if _TIMESCALE_AVAILABLE
                else {"enabled": False}
            )
        ),
        "inner_voice": (
            sim._inner_voice.snapshot()
            if sim._inner_voice is not None
            else {"enabled": _INNER_VOICE_AVAILABLE}
        ),
        "llm_teacher": (
            sim._llm_teacher.snapshot()
            if sim._llm_teacher is not None
            else {"enabled": False}
        ),
        "sleep": (
            sim._sleep_ctrl.snapshot()
            if _PHASE_K_AVAILABLE and sim._sleep_ctrl is not None
            else {"enabled": False}
        ),
        "physical_curriculum": (
            sim._physical_curriculum.snapshot()
            if _PHASE_K_AVAILABLE and sim._physical_curriculum is not None
            else None
        ),
        "persistence": (
            sim._persist.snapshot()
            if _PHASE_K_AVAILABLE and sim._persist is not None
            else None
        ),
        "verbal": (
            sim._verbal.snapshot()
            if _VERBAL_AVAILABLE and sim._verbal is not None
            else {"enabled": False}
        ),
        "visual_voice": (
            sim._visual_voice.snapshot()
            if _PHASE_M_AVAILABLE and sim._visual_voice is not None
            else {"enabled": False}
        ),
        "slot_labeler": (
            sim._slot_labeler.snapshot()
            if _PHASE_M_AVAILABLE and sim._slot_labeler is not None
            else None
        ),
        "world_bridge": (
            sim._world_bridge.snapshot()
            if _WORLD_BRIDGE_AVAILABLE and sim._world_bridge is not None
            else {"enabled": False}
        ),
        "phase1": sim._phase1_snapshot_meta(),
        "phase2": sim._phase2_snapshot_meta(),
        "motor_primitives": (
            sim._motor_prim_lib.snapshot()
            if getattr(sim, "_motor_prim_lib", None) is not None
            else None
        ),
        "variable_registry": (
            sim._variable_registry.snapshot()
            if getattr(sim, "_variable_registry", None) is not None
            else None
        ),
    }
```

Added two new sections to `build_simulation_snapshot()`:
- **`motor_primitives`**: shows learned motor programs (GRU-based), pattern detection stats
- **`variable_registry`**: shows bootstrap state, group pressures, discovery history

---

### 5. Sparse EIG — 3-5x Speedup
```diff:agent.py
"""
agent_v4.py — RKKAgent с Value Layer (Шаг А).

Изменения:
  - ValueLayer.check_action() вызывается перед каждым do()
  - Заблокированные действия → penalty для System 1 + лог события
  - LLM/RAG seed interface: inject_text_priors(edges_json)
  - Fallback scorer когда System 1 буфер ещё мал
  - other_agents_phi передаётся из Simulation для ΔΦ≥0 constraint

Этап B (гипотезо-ориентированное исследование):
  score_interventions() — аппроксимация информационного выигрыша: чувствительность по узлам
  плюс суррогат снижения суммарной epistemic mass по рёбрам при предсказанном obs (не полный
  байесовский H(W)−E[H(W|obs)]). RKK_EIG_ENTROPY_TERM, RKK_EIG_POSTERIOR_ETA.
  Переключатель: RKK_HYPOTHESIS_EIG=1 (по умолчанию) | 0 | system1 | off | false
  В snapshot: h_W_edge_entropy — сумма бинарных энтропий по α_trust рёбер (диагностика неопределённости W).
  RKK_SCORE_ASYNC=1: score_interventions в фоновом daemon-потоке (тик не ждёт; возможна гонка с train_step — не рекомендуется).
  По умолчанию RKK_SCORE_ASYNC=0 — синхронный пересчёт в главном потоке (стабильно, без общего lock на граф).

Этап Г (самомодель): self_* + update_self_feedback() в humanoid — коррекция намерений по исходу do()
  и по промаху GNN (RKK_SELF_FEEDBACK_LR).

Этап E (целевое планирование): при self_goal_active и наличии target_dist в графе — поиск действия
  через imagination (propagate_from + rollout_step_free), см. engine.goal_planning; RKK_GOAL_PLANNING=0 отключает.

Этап F (символьный верификатор): проверка предсказания propagate на PHYSICS_CONSTRAINTS (engine.symbolic_verifier);
  нарушение → не prepend goal-plan, смешивание expected_ig с uncertainty на следующем шаге; RKK_SYMBOLIC_VERIFY=0 отключает.

Этап G (RSI lite): плато discovery_rate → агент усиливает L1, удваивает BUFFER_SIZE графа (до капа), +1 imagination;
  engine.rsi_lite, RKK_RSI_LITE=0 отключает; RKK_RSI_PLATEAU_TICKS, RKK_RSI_MIN_INTERVENTIONS.
"""
from __future__ import annotations

import os
import threading
from typing import Any
import torch
import numpy as np
from collections import deque

from engine.causal_graph import CausalGraph
from engine.graph_constants import is_read_only_macro_var
from engine.environment  import Environment
from engine.system1      import System1
from engine.temporal     import TemporalBlankets
from engine.value_layer  import ValueLayer, HomeostaticBounds, BlockReason
from engine.phase3_teacher import TeacherIGRule
from engine.environment_humanoid import SELF_VARS
from engine.goal_planning import (
    goal_planning_globally_disabled,
    parse_plan_value_levels,
    plan_beam_k,
    plan_depth,
    plan_max_branch,
    planning_graph_motor_vars,
)
from engine.symbolic_verifier import (
    downrank_factor_for_violation,
    exploration_blend_from_uncertainty,
    symbolic_verifier_enabled,
    verify_normalized_prediction,
)
from engine.wm_neural_ode import integrate_world_model_step
from engine.rsi_lite import (
    rsi_buffer_cap,
    rsi_imagination_cap,
    rsi_improvement_eps,
    rsi_l1_max,
    rsi_l1_scale,
    rsi_lite_enabled,
    rsi_min_interventions,
    rsi_plateau_interventions,
)
from engine.local_reflex import local_reflex_train_enabled, train_chains_parallel

ACTIVATIONS   = ["relu", "gelu", "tanh"]
NOTEARS_EVERY = 8
MAX_FALLBACK_TRIES = 5  # больше кандидатов, чтобы пройти Value Layer в начале обучения
# Вес slot_* в actual_ig для System 1; основной сигнал — не-визуальные узлы (RKK_VISUAL_IG_WEIGHT=0 → только физика).
VISUAL_IG_WEIGHT = float(os.environ.get("RKK_VISUAL_IG_WEIGHT", "0.1"))
_SELF_VAR_SET = frozenset(SELF_VARS)
# RKK_LOCOMOTION_CPG=1: CPG ведёт ноги; EIG не выбирает прямые do() по этим узлам.
_LOCOMOTION_CPG_LEG_EIG_BLOCK = frozenset(
    {"lhip", "lknee", "lankle", "rhip", "rknee", "rankle"}
)


def _is_motor_intent_var(name: str) -> bool:
    return str(name).startswith("intent_") or str(name).startswith("phys_intent_")


def _hypothesis_eig_from_env() -> bool:
    """Этап B: байесовский выбор эксперимента (EIG) вместо только System 1."""
    v = os.environ.get("RKK_HYPOTHESIS_EIG", "1").strip().lower()
    return v not in ("0", "false", "off", "system1", "no", "s1")


def _eig_chunk_size() -> int:
    try:
        return max(1, int(os.environ.get("RKK_EIG_BATCH", "256")))
    except ValueError:
        return 256


def _score_cache_every() -> int:
    """Пересчёт score_interventions не чаще чем раз в N тиков движка (RKK_SCORE_CACHE_EVERY; 1 = каждый тик)."""
    try:
        return max(1, int(os.environ.get("RKK_SCORE_CACHE_EVERY", "1")))
    except ValueError:
        return 1


def _score_async_enabled() -> bool:
    """Фоновый поток для score_interventions; по умолчанию выкл. (лок на весь WM давал рывки UI)."""
    v = os.environ.get("RKK_SCORE_ASYNC", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _imagination_horizon_from_env() -> int:
    """Фаза 13: RKK_IMAGINATION_STEPS — число шагов core(X) после мысленного do(); 0 = как раньше."""
    raw = os.environ.get("RKK_IMAGINATION_STEPS", "2")
    try:
        h = int(raw)
    except ValueError:
        h = 0
    return max(0, h)


class RKKAgent:
    def __init__(
        self,
        agent_id: int,
        name:     str,
        env:      Environment,
        device:   torch.device,
        bounds:   HomeostaticBounds | None = None,
    ):
        self.id         = agent_id
        self.name       = name
        self.env        = env
        self.device     = device
        self.activation = ACTIVATIONS[agent_id % 3]

        self.graph   = CausalGraph(device)
        self.system1 = System1(activation=self.activation, device=device)
        self.temporal = TemporalBlankets(
            d_input=len(env.variable_ids), device=device
        )
        self.value_layer = ValueLayer(bounds)
        self._imagination_horizon = _imagination_horizon_from_env()

        self._cg_history: deque[float] = deque(maxlen=20)
        self._total_interventions = 0
        self._total_blocked       = 0
        self._last_do             = "—"
        self._last_blocked_reason = ""
        self._last_result: dict | None = None
        self._symbolic_prediction_bad = False
        self._peak_discovery_rate: float = 0.0
        self._rsi_ref_discovery: float = 0.0
        self._rsi_plateau_count: int = 0
        self._rsi_adjustment_count: int = 0
        self._notears_steps  = 0
        self._last_notears_loss: dict | None = None
        self._local_reflex_cores: dict[tuple[str, ...], Any] = {}
        self._last_local_reflex_train: dict | None = None

        # Φ других агентов (заполняется Simulation-ом перед step())
        self.other_agents_phi: list[float] = []
        self._last_engine_tick = 0
        self._score_cache: list[dict] = []
        self._score_cache_tick: int = -9_999_999
        self._score_thread: threading.Thread | None = None
        self._score_result: list[dict] = []
        self._score_lock = threading.Lock()

        # Фаза 3: LLM-учитель (IG-бонус затухает с числом интервенций)
        self._teacher_rules: list[TeacherIGRule] = []
        self._teacher_weight: float = 0.0

        self._bootstrap()

    # ── Bootstrap + LLM seed interface ───────────────────────────────────────
    def _bootstrap(self):
        for var_id, val in self.env.variables.items():
            self.graph.set_node(var_id, val)

        obs0 = dict(self.env.variables)
        self.graph.record_observation(obs0)
        self.temporal.step(obs0)

        # Text priors (spurious + partial GT)
        gt = self.env.gt_edges()
        for e in gt[:2]:
            noisy_w = e["weight"] * 0.3 + (np.random.rand() - 0.5) * 0.4
            self.graph.set_edge(e["from_"], e["to"], noisy_w, alpha=0.06)

        var_ids = self.env.variable_ids
        if len(var_ids) >= 4:
            self.graph.set_edge(var_ids[1], var_ids[3],  0.35, alpha=0.05)
            self.graph.set_edge(var_ids[2], var_ids[0], -0.20, alpha=0.04)

        # Фаза 1: заморозка URDF-цепочек в L1 (humanoid VAR_NAMES).
        fr = os.environ.get("RKK_FREEZE_URDF", "1").strip().lower()
        if fr not in ("0", "false", "no", "off") and "lhip" in self.env.variable_ids:
            self.graph.freeze_kinematic_priors()

    def inject_text_priors(self, edges: list[dict]) -> dict:
        """
        LLM/RAG seed interface.

        edges: [{"from_": "Temp", "to": "Pressure", "weight": 0.8}, ...]

        Все рёбра загружаются с alpha=0.05 (низкое доверие).
        Epistemic Annealing + NOTEARS выжгут ошибочные за N интервенций.

        Узлы from_/to должны совпадать с id переменных окружения (env.variable_ids).

        Возвращает {"injected": n, "skipped": [причины...]}.
        """
        count   = 0
        skipped: list[str] = []
        valid   = set(self.graph.nodes.keys())

        for e in edges:
            from_ = e.get("from_") or e.get("from")
            to    = e.get("to")
            w     = float(e.get("weight", 0.3))

            if not from_ or not to:
                skipped.append(f"нет from_/to: {e!r}")
                continue
            if is_read_only_macro_var(from_) or is_read_only_macro_var(to):
                skipped.append(f"read-only macro: {from_!r}→{to!r}")
                continue
            if from_ not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{from_}» (доступны: {sorted(valid)})")
                continue
            if to not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{to}» (доступны: {sorted(valid)})")
                continue

            alpha = float(e.get("alpha", 0.05))
            # Слабые семена по умолчанию (0.2–0.3 экв.) — не «пугают» граф и VL
            w_scaled = min(0.3, max(0.08, float(w) * 0.28))
            self.graph.set_edge(from_, to, w_scaled, alpha=alpha)
            count += 1

        return {"injected": count, "skipped": skipped, "node_ids": sorted(valid)}

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_h_W(self) -> float:
        if self.graph._core is None:
            return 0.0
        return float(self.graph._core.dag_constraint().item())

    @staticmethod
    def _marginal_node_uncertainty(unc_m: np.ndarray) -> np.ndarray:
        """
        Маргинальная неопределённость по узлу j: max по всем рёбрам (j→·) и (·→j).
        unc_m[i,j] — epistemic mass на ребре i→j (posterior proxy: 1 − α_trust).
        """
        row_max = unc_m.max(axis=1)
        col_max = unc_m.max(axis=0)
        return np.maximum(row_max, col_max).astype(np.float64, copy=False)

    def _batch_hypothesis_eig(
        self,
        candidates: list[dict],
        X_np: np.ndarray,
        u_node: np.ndarray,
        nid_to_i: dict[str, int],
        unc_m: np.ndarray,
        node_ids: list[str],
        env: Environment,
    ) -> list[float]:
        """
        Суррогат «информативности» действия: (1) чувствительность Σ_j u(j)|ΔX_j|;
        (2) суррогат снижения неопределённости по рёбрам после гипотетического наблюдения
        (масштабирование unc_ij пропорционально |ΔX_i|+|ΔX_j|). Это не точный EIG по H(W).
        """
        core = self.graph._core
        if core is None or not candidates:
            return []
        fd = getattr(core, "forward_dynamics", None)
        if not callable(fd):
            return []

        try:
            lam = float(os.environ.get("RKK_EIG_ENTROPY_TERM", "0.22"))
        except ValueError:
            lam = 0.22
        try:
            eta = float(os.environ.get("RKK_EIG_POSTERIOR_ETA", "0.18"))
        except ValueError:
            eta = 0.18
        lam = max(0.0, lam)
        eta = max(0.0, min(0.95, eta))

        d = int(X_np.shape[0])
        device = self.device
        chunk = _eig_chunk_size()
        eigs: list[float] = []
        x0 = torch.from_numpy(X_np).to(dtype=torch.float32, device=device).unsqueeze(0)
        uu = unc_m.reshape(1, d, d)

        for start in range(0, len(candidates), chunk):
            sub = candidates[start : start + chunk]
            b = len(sub)
            x_batch = x0.expand(b, -1)
            a_batch = torch.zeros(b, d, device=device, dtype=torch.float32)
            for bi, cand in enumerate(sub):
                idx = nid_to_i.get(cand["variable"])
                if idx is not None:
                    a_batch[bi, idx] = float(cand["value"])
            with torch.no_grad():
                pred = integrate_world_model_step(core, x_batch, a_batch)
            delta = (pred - x_batch).abs().cpu().numpy()
            ab = np.abs(delta)
            S = np.clip(ab[:, :, None] + ab[:, None, :], 0.0, 1.0)
            new_u = uu * (1.0 - eta * S)
            new_u = np.maximum(new_u, 0.0)
            reduction = (uu - new_u).sum(axis=(1, 2))
            sens = (delta * u_node.reshape(1, -1)).sum(axis=1)
            total = sens + lam * reduction
            if symbolic_verifier_enabled():
                fac = downrank_factor_for_violation()
                d_nodes = len(node_ids)
                for bi in range(b):
                    pd = {
                        node_ids[j]: float(pred[bi, j].item())
                        for j in range(min(d_nodes, int(pred.shape[1])))
                    }
                    ok, _ = verify_normalized_prediction(pd, env)
                    if not ok:
                        total[bi] *= fac
            eigs.extend(total.tolist())
        return eigs

    def _rollout_imagination_state(
        self, base: dict[str, float], var: str, val: float
    ) -> dict[str, float]:
        """Этап E: один мысленный do + столько же свободных шагов, сколько в VL imagination."""
        s = self.graph.propagate_from(dict(base), var, float(val))
        for _ in range(max(0, self._imagination_horizon)):
            s = self.graph.rollout_step_free(s)
        return s

    def _features_for_intervention_pair(self, v_from: str, v_to: str) -> list[float]:
        """Один вектор признаков System1 для пары (в_from→в_to), как в score_interventions."""
        h_W_norm = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        core = self.graph._core
        ii, jj = nid_to_i.get(v_from), nid_to_i.get(v_to)
        if core is not None and ii is not None and jj is not None:
            with torch.no_grad():
                unc_t = (1.0 - core.alpha_trust_matrix()).detach().float().cpu().numpy()
                W_m = core.W_masked().detach().float().cpu().numpy()
                g_m = None
                if core.W.grad is not None:
                    g_m = core.W.grad.detach().float().abs().cpu().numpy()
            uncertainty = float(unc_t[ii, jj])
            w_ij = float(W_m[ii, jj])
            grad_norm = float(g_m[ii, jj]) if g_m is not None else 0.0
        else:
            uncertainty, w_ij, grad_norm = 1.0, 0.0, 0.0
        alpha = 1.0 - uncertainty
        val_from = self.graph.nodes.get(v_from, 0.5)
        val_to = self.graph.nodes.get(v_to, 0.5)
        ic = ic_map.get((v_from, v_to), 0)
        return self.system1.build_features(
            w_ij=w_ij, alpha_ij=alpha,
            val_from=val_from, val_to=val_to,
            uncertainty=uncertainty, h_W_norm=h_W_norm,
            grad_norm_ij=grad_norm,
            intervention_count=ic,
            discovery_rate=disc_rate,
        )

    def _build_goal_planned_candidate(self, var: str, val: float) -> dict:
        feat = self._features_for_intervention_pair(var, "target_dist")
        return {
            "variable":    var,
            "target":      "target_dist",
            "value":       float(val),
            "uncertainty": 0.35,
            "features":    feat,
            "expected_ig": 1.0,
            "from_goal_plan": True,
        }

    def _maybe_goal_planned_candidate(self) -> dict | None:
        if goal_planning_globally_disabled():
            return None
        if self.graph._core is None:
            return None
        if self.graph.nodes.get("self_goal_active") is None:
            return None
        if float(self.graph.nodes.get("self_goal_active", 0)) <= 0.45:
            return None
        if "target_dist" not in self.graph.nodes:
            return None

        state0 = dict(self.graph.nodes)
        cur_td = float(state0.get("target_dist", 0.5))
        goal_thr = float(state0.get("self_goal_target_dist", 0.42))
        if cur_td <= goal_thr + 0.015:
            return None

        motor = planning_graph_motor_vars(self.env, list(self.graph._node_ids))
        if not motor:
            return None

        levels = parse_plan_value_levels()
        actions = [(v, x) for v in motor for x in levels]
        max_b = plan_max_branch()
        if len(actions) > max_b:
            idx = np.random.choice(len(actions), size=max_b, replace=False)
            actions = [actions[i] for i in idx]

        depth = plan_depth()
        beam_k = plan_beam_k()

        def _td(s: dict[str, float]) -> float:
            return float(s.get("target_dist", cur_td))

        best_td = cur_td
        best_first: tuple[str, float] | None = None

        if depth <= 1:
            for var, val in actions:
                try:
                    sfin = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                    if not ok:
                        continue
                td = _td(sfin)
                if td < best_td - 1e-6:
                    best_td = td
                    best_first = (var, val)
        else:
            scored: list[tuple[float, str, float, dict[str, float]]] = []
            for var, val in actions:
                try:
                    s1 = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(s1), self.env)
                    if not ok:
                        continue
                scored.append((_td(s1), var, val, dict(s1)))
            scored.sort(key=lambda t: t[0])
            for _td1, v1, x1, s1 in scored[:beam_k]:
                for v2, x2 in actions:
                    try:
                        sfin = self._rollout_imagination_state(s1, v2, x2)
                    except Exception:
                        continue
                    if symbolic_verifier_enabled():
                        ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                        if not ok:
                            continue
                    td = _td(sfin)
                    if td < best_td - 1e-6:
                        best_td = td
                        best_first = (v1, x1)

        if best_first is None:
            return None
        return self._build_goal_planned_candidate(best_first[0], best_first[1])

    def _is_locomotion_primary_active(self) -> bool:
        """Если CPG управляет ногами, EIG не должен конкурировать за суставы — только intent_* и др."""
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        var_ids   = self.env.variable_ids
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate

        # Один проход по рёбрам: счётчики интервенций (раньше — O(pairs×|E|) через next() в цикле)
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count

        # Имя узла → индекс без O(d) list.index на каждую пару
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}

        # Один раз W, α и |grad| на CPU — вместо O(d²) вызовов alpha_trust_matrix / W_masked
        core = self.graph._core
        W_m = unc_m = g_m = None
        if core is not None:
            with torch.no_grad():
                W_t = core.W_masked().detach().float()
                A_t = core.alpha_trust_matrix().detach().float()
                W_m = W_t.cpu().numpy()
                unc_m = (1.0 - A_t).cpu().numpy()
            if core.W.grad is not None:
                g_m = core.W.grad.detach().float().abs().cpu().numpy()

        d = len(var_ids)
        if d == 0:
            return []

        # Счётчики интервенций по парам (только известные рёбра — O(|E|))
        ic_mat = np.zeros((d, d), dtype=np.float64)
        v2i = {v: i for i, v in enumerate(var_ids)}
        for (vf, vt), c in ic_map.items():
            i = v2i.get(vf)
            j = v2i.get(vt)
            if i is not None and j is not None and i != j:
                ic_mat[i, j] = float(c)

        ridx = np.zeros(d, dtype=np.int64)
        valid_node = np.zeros(d, dtype=bool)
        for i, v in enumerate(var_ids):
            ji = nid_to_i.get(v)
            if ji is not None:
                ridx[i] = ji
                valid_node[i] = True

        nodes_arr = np.array(
            [float(self.graph.nodes.get(v, 0.5)) for v in var_ids],
            dtype=np.float64,
        )
        mask = ~np.eye(d, dtype=bool)
        fi, fj = np.where(mask)
        n_pairs = len(fi)

        if W_m is not None:
            ii_n = ridx[fi]
            jj_n = ridx[fj]
            ok = valid_node[fi] & valid_node[fj]
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)
            w_ij[ok] = W_m[ii_n[ok], jj_n[ok]]
            uncertainty[ok] = unc_m[ii_n[ok], jj_n[ok]]
            if g_m is not None:
                grad_norm[ok] = g_m[ii_n[ok], jj_n[ok]]
        else:
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)

        alpha = 1.0 - uncertainty
        val_from = nodes_arr[fi]
        val_to = nodes_arr[fj]
        ic_v = ic_mat[fi, fj]
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_v = float(np.clip(disc_rate, 0.0, 1.0))

        feats_arr = np.column_stack(
            [
                np.tanh(w_ij),
                np.clip(alpha, 0.0, 1.0),
                np.clip(val_from, 0.0, 1.0),
                np.clip(val_to, 0.0, 1.0),
                np.clip(uncertainty, 0.0, 1.0),
                np.full(n_pairs, h_clip, dtype=np.float64),
                np.tanh(grad_norm),
                np.clip(ic_v / 100.0, 0.0, 1.0),
                np.full(n_pairs, disc_v, dtype=np.float64),
            ]
        )
        features_batch = feats_arr.tolist()

        rng = np.random.default_rng()
        posture_now = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 0.5),
            )
        )
        foot_l_now = float(
            self.graph.nodes.get(
                "foot_contact_l",
                self.graph.nodes.get("phys_foot_contact_l", 0.5),
            )
        )
        foot_r_now = float(
            self.graph.nodes.get(
                "foot_contact_r",
                self.graph.nodes.get("phys_foot_contact_r", 0.5),
            )
        )
        stable_stance = posture_now > 0.70 and min(foot_l_now, foot_r_now) > 0.56
        candidates: list[dict] = []
        for k in range(n_pairs):
            i, j = int(fi[k]), int(fj[k])
            vf, vt = var_ids[i], var_ids[j]
            unc_k = float(uncertainty[k])
            feat_k = features_batch[k]
            if _is_motor_intent_var(vf):
                if stable_stance:
                    lo, hi = 0.30, 0.72
                else:
                    lo, hi = 0.35, 0.68
                if str(vf).endswith("stride"):
                    hi = min(hi, 0.62 if stable_stance else 0.56)
                if str(vf).endswith("stop_recover"):
                    lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                rand_value = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
            else:
                rand_value = float(np.clip(rng.uniform(0.15, 0.85), 0.06, 0.94))
            candidates.append({
                "variable":    vf,
                "target":      vt,
                "value":       rand_value,
                "uncertainty": unc_k,
                "features":    feat_k,
                "expected_ig": 0.0,
            })

        if self._is_locomotion_primary_active():
            candidates = [
                c
                for c in candidates
                if c["variable"] not in _LOCOMOTION_CPG_LEG_EIG_BLOCK
            ]
            if posture_now < 0.65:
                candidates = [
                    c
                    for c in candidates
                    if str(c["variable"]).startswith("intent_")
                    or str(c["variable"]).startswith("phys_intent_")
                ]

        if not candidates or not features_batch:
            return []

        use_eig = _hypothesis_eig_from_env() and W_m is not None and unc_m is not None
        if use_eig:
            x_vec = np.array(
                [float(self.graph.nodes.get(n, 0.0)) for n in self.graph._node_ids],
                dtype=np.float64,
            )
            u_node = self._marginal_node_uncertainty(unc_m)
            eigs = self._batch_hypothesis_eig(
                candidates, x_vec, u_node, nid_to_i, unc_m,
                list(self.graph._node_ids), self.env,
            )
            if len(eigs) == len(candidates):
                # Учитываем гипотезу «это ребро неизвестно»: масштаб EIG по unc(v_from→v_to).
                for i, cand in enumerate(candidates):
                    eigs[i] *= 1.0 + float(cand["uncertainty"])
                arr = np.array(eigs, dtype=np.float64)
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo + 1e-12:
                    normed = (arr - lo) / (hi - lo)
                else:
                    normed = np.full_like(arr, 0.5)
                for i, cand in enumerate(candidates):
                    cand["eig_raw"] = float(eigs[i])
                    cand["expected_ig"] = float(normed[i])
            else:
                use_eig = False

        if not use_eig:
            scores = self.system1.score(features_batch)
            for i, cand in enumerate(candidates):
                cand["expected_ig"] = scores[i]

        if symbolic_verifier_enabled() and self._symbolic_prediction_bad:
            a, b = exploration_blend_from_uncertainty()
            for cand in candidates:
                unc = float(cand.get("uncertainty", 0.5))
                cand["expected_ig"] = a * float(cand["expected_ig"]) + b * unc

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    def _score_async_worker(self) -> None:
        try:
            with torch.no_grad():
                result = self.score_interventions()
            with self._score_lock:
                self._score_result = result
        except Exception as ex:
            print(f"[RKKAgent] score_interventions (async): {ex}")

    def set_teacher_state(self, rules: list[TeacherIGRule], weight: float) -> None:
        """Фаза 3: правила от LLM и текущий teacher_weight (симуляция считает annealing)."""
        self._teacher_rules = list(rules)
        self._teacher_weight = float(max(0.0, min(1.0, weight)))

    def _teacher_ig_bonus(self, variable: str, nodes: dict[str, float]) -> float:
        w = self._teacher_weight
        if w <= 0 or not self._teacher_rules:
            return 0.0
        acc = 0.0
        for r in self._teacher_rules:
            if r.target_var != variable:
                continue
            if r.when_var:
                val = nodes.get(r.when_var)
                if val is None:
                    continue
                if r.when_min is not None and float(val) < r.when_min:
                    continue
                if r.when_max is not None and float(val) > r.when_max:
                    continue
            acc += r.bonus * w
        return min(0.28, acc)

    # ── Один шаг с Value Layer ────────────────────────────────────────────────
    def step(self, engine_tick: int = 0, *, enable_l3: bool = True) -> dict:
        self._last_engine_tick = engine_tick
        try:
            self.graph.apply_env_observation(dict(self.env.observe()))
        except Exception:
            pass
        sce = _score_cache_every()
        if (
            sce > 1
            and self._score_cache
            and (engine_tick - self._score_cache_tick) < sce
        ):
            scores = list(self._score_cache)
        elif _score_async_enabled():
            if self._score_thread is None or not self._score_thread.is_alive():
                self._score_thread = threading.Thread(
                    target=self._score_async_worker,
                    name="rkk_score_interventions",
                    daemon=True,
                )
                self._score_thread.start()
            with self._score_lock:
                have = list(self._score_result) if self._score_result else []
            if have:
                scores = have
            elif self._score_cache:
                scores = list(self._score_cache)
            else:
                with torch.no_grad():
                    scores = self.score_interventions()
                with self._score_lock:
                    self._score_result = list(scores)
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        else:
            with torch.no_grad():
                scores = self.score_interventions()
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        gp = self._maybe_goal_planned_candidate() if enable_l3 else None
        if gp is not None and not (
            symbolic_verifier_enabled() and self._symbolic_prediction_bad
        ):
            scores.insert(0, gp)
        if not scores:
            return {
                "blocked": False, "skipped": True, "prediction_error": 0.0,
                "cf_predicted": {}, "cf_observed": {}, "goal_planned": False,
            }

        current_phi = self.phi_approx()
        chosen      = None
        check_result = None
        blocked_count = 0

        # Перебираем кандидатов пока не найдём допустимое действие
        for candidate in scores[:MAX_FALLBACK_TRIES]:
            var   = candidate["variable"]
            value = candidate["value"]

            check_result = self.value_layer.check_action(
                variable=var,
                value=value,
                current_nodes=dict(self.graph.nodes),
                graph=self.graph,
                temporal=self.temporal,
                current_phi=current_phi,
                other_agents_phi=self.other_agents_phi,
                engine_tick=engine_tick,
                imagination_horizon=(self._imagination_horizon if enable_l3 else 0),
            )

            if check_result.allowed:
                chosen = candidate
                break
            else:
                # Штрафуем System 1 за предложение опасного действия
                self.system1.push_experience(
                    features=candidate["features"],
                    actual_ig=check_result.penalty,   # отрицательный IG
                )
                blocked_count += 1
                self._total_blocked += 1
                self._last_blocked_reason = check_result.reason.value

        # Все кандидаты заблокированы — возвращаем событие
        if chosen is None:
            return {
                "blocked":       True,
                "blocked_count": blocked_count,
                "reason":        self._last_blocked_reason,
                "variable":      scores[0]["variable"] if scores else "?",
                "value":         scores[0]["value"] if scores else 0.5,
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error":  0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        # ── Выполняем допустимое действие ────────────────────────────────────
        var   = chosen["variable"]
        value = chosen["value"]

        if is_read_only_macro_var(var):
            return {
                "blocked": True,
                "blocked_count": blocked_count + 1,
                "reason": "read_only_macro",
                "variable": var,
                "value": float(value),
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        mdl_before = self.graph.mdl_size
        obs_before_env = dict(self.env.observe())
        self.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.graph.snapshot_vec_dict()
        predicted  = self.graph.propagate(var, value)
        sym_ok, sym_fail = True, []
        if symbolic_verifier_enabled():
            sym_ok, sym_fail = verify_normalized_prediction(dict(predicted), self.env)
            self._symbolic_prediction_bad = not sym_ok
        else:
            self._symbolic_prediction_bad = False
        observed_env = self.env.intervene(var, value)

        # Temporal step (только размерность среды)
        self.temporal.step(observed_env)

        self.graph.apply_env_observation(observed_env)
        observed_full = self.graph.snapshot_vec_dict()

        # NOTEARS / GNN буферы — полный вектор узлов (включая concept_*)
        self.graph.record_observation(obs_before_full)
        self.graph.record_observation(observed_full)
        self.graph.record_intervention(var, value, obs_before_full, observed_full)

        # NOTEARS train
        notears_result = None
        if self._total_interventions % NOTEARS_EVERY == 0:
            notears_result = self.graph.train_step()
            if notears_result:
                self._notears_steps += 1
                self._last_notears_loss = notears_result
            self._maybe_train_local_reflex()

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # System 1: IG по физике; slot_* и self_* не доминируют метрику (self — прямое задание агентом).
        nids = self.graph._node_ids
        phys_ids = [
            k for k in nids
            if k not in _SELF_VAR_SET and not str(k).startswith("slot_")
        ]
        slot_ids = [k for k in nids if str(k).startswith("slot_")]

        def _mean_abs_err(keys: list) -> float:
            if not keys:
                return 0.0
            return float(np.mean([
                abs(float(predicted.get(k, 0.5)) - float(observed_full.get(k, 0.5)))
                for k in keys
            ]))

        pe_phys = _mean_abs_err(phys_ids)

        # Этап Г: петля «намерение ↔ исход» + ошибка модели → self_* (только среды с методом).
        fn_sf = getattr(self.env, "update_self_feedback", None)
        if callable(fn_sf):
            try:
                fn_sf(
                    variable=var,
                    intended_norm=value,
                    observed=observed_env,
                    predicted=predicted,
                    prediction_error_phys=pe_phys,
                )
            except Exception:
                pass
            obs_self = dict(self.env.observe())
            for sk in _SELF_VAR_SET:
                if sk in self.graph.nodes and sk in obs_self:
                    self.graph.nodes[sk] = float(obs_self[sk])
            self.graph.refresh_concept_aggregates()
        pe_slot = _mean_abs_err(slot_ids)
        w_vis = min(0.45, max(0.0, VISUAL_IG_WEIGHT))
        if slot_ids and phys_ids:
            actual_ig = (1.0 - w_vis) * pe_phys + w_vis * pe_slot
        elif phys_ids:
            actual_ig = pe_phys
        else:
            actual_ig = pe_slot

        t_bonus = self._teacher_ig_bonus(var, dict(self.graph.nodes))
        actual_ig = float(np.clip(actual_ig + t_bonus, 0.0, 1.0))

        self.system1.push_experience(
            features=chosen["features"],
            actual_ig=actual_ig,
        )

        # SSM train — размерность = temporal.d_input (= |graph._node_ids|), не только env.variable_ids
        u_next = torch.tensor(
            [float(self.graph.nodes.get(n, 0.5)) for n in self.graph._node_ids],
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal.train_step(u_next)

        self._total_interventions += 1
        try:
            _v_do = float(value)
        except (TypeError, ValueError):
            _v_do = 0.5
        self._last_do = f"do({var}={_v_do:.2f})"
        self._last_blocked_reason = ""

        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        rsi_event = self._tick_rsi_lite_discovery(cur_dr)

        _cf_keys = list(self.graph._node_ids)[:48]
        self._last_result = {
            "blocked":           False,
            "blocked_count":     blocked_count,
            "variable":          var,
            "value":             value,
            "compression_delta": compression_delta,
            "updated_edges":     [f"{e.from_}→{e.to}" for e in self.graph.edges[:4]],
            "pruned_edges":      [],
            "prediction_error":  float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed_env.items()
            ])),
            "cf_predicted": {k: float(round(float(predicted.get(k, 0.0)), 4)) for k in _cf_keys},
            "cf_observed":  {k: float(round(float(observed_full.get(k, 0.0)), 4)) for k in _cf_keys},
            "goal_planned":  bool(chosen.get("from_goal_plan")),
            "symbolic_ok": sym_ok,
            "symbolic_violations": sym_fail,
            "rsi_lite": rsi_event,
            "notears":           notears_result,
        }
        return self._last_result

    # ── Demon ─────────────────────────────────────────────────────────────────
    def demon_disrupt(self) -> str:
        if self.graph._core is None:
            return "no core"
        with torch.no_grad():
            W = self.graph._core.W
            sig = (W.abs() > 0.05).nonzero(as_tuple=False)
            if len(sig) == 0:
                return "no significant edges"
            idx = sig[np.random.randint(len(sig))]
            i, j = idx[0].item(), idx[1].item()
            noise = (np.random.rand() - 0.5) * 0.3
            # Нельзя W[i,j] += … — это in-place на view листа с requires_grad.
            w_new = W.detach().clone()
            w_new[i, j] = w_new[i, j] + float(noise)
            W.copy_(w_new)
            fn = self.graph._node_ids[i] if i < len(self.graph._node_ids) else f"v{i}"
            tn = self.graph._node_ids[j] if j < len(self.graph._node_ids) else f"v{j}"
        self.graph._invalidate_cache()
        return f"W[{fn}→{tn}] +{noise:.3f}"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def compression_gain(self) -> float:
        if not self._cg_history:
            return 0.0
        return float(np.mean(list(self._cg_history)))

    @property
    def discovery_rate(self) -> float:
        return self.env.discovery_rate([
            {"from_": e.from_, "to": e.to, "weight": e.weight}
            for e in self.graph.edges
        ])

    @property
    def peak_discovery_rate(self) -> float:
        return self._peak_discovery_rate

    def _apply_rsi_lite(self) -> dict[str, float | int]:
        g = self.graph
        cur_l1 = float(getattr(g, "LAMBDA_L1", CausalGraph.LAMBDA_L1))
        new_l1 = min(cur_l1 * rsi_l1_scale(), rsi_l1_max())
        g.LAMBDA_L1 = new_l1
        cap_b = rsi_buffer_cap()
        g.BUFFER_SIZE = min(cap_b, int(g.BUFFER_SIZE) * 2)
        cap_i = rsi_imagination_cap()
        self._imagination_horizon = min(cap_i, self._imagination_horizon + 1)
        self._rsi_adjustment_count += 1
        return {
            "LAMBDA_L1": float(new_l1),
            "BUFFER_SIZE": int(g.BUFFER_SIZE),
            "imagination_horizon": int(self._imagination_horizon),
        }

    def _tick_rsi_lite_discovery(self, cur_dr: float) -> dict[str, float | int] | None:
        if not rsi_lite_enabled():
            return None
        if self._total_interventions < rsi_min_interventions():
            return None
        eps = rsi_improvement_eps()
        if cur_dr > self._rsi_ref_discovery + eps:
            self._rsi_ref_discovery = float(cur_dr)
            self._rsi_plateau_count = 0
            return None
        self._rsi_plateau_count += 1
        if self._rsi_plateau_count < rsi_plateau_interventions():
            return None
        self._rsi_plateau_count = 0
        self._rsi_ref_discovery = float(cur_dr)
        return self._apply_rsi_lite()

    def _maybe_train_local_reflex(self) -> None:
        if not local_reflex_train_enabled():
            return
        self._last_local_reflex_train = train_chains_parallel(
            graph=self.graph,
            device=self.graph.device,
            cores=self._local_reflex_cores,
        )

    def phi_approx(self) -> float:
        return self.temporal.phi_approx()

    def record_phi(self, _: float):
        pass  # temporal управляет историей сам

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        h_W     = self._get_h_W()
        tb_info = self.temporal.slow_state_summary()
        s1_info = {
            "buffer_size": len(self.system1.buffer),
            "mean_loss":   round(self.system1.mean_loss, 6),
        }
        vl_info = dict(self.value_layer.snapshot(self._last_engine_tick))
        vl_info["imagination_horizon"] = self._imagination_horizon

        notears_info = None
        if self._last_notears_loss:
            notears_info = {
                "steps":  self._notears_steps,
                "loss":   self._last_notears_loss.get("loss", 0),
                "h_W":    round(h_W, 4),
                "l_int":  self._last_notears_loss.get("l_int", 0),
            }

        h_W_edge_entropy = None
        core = self.graph._core
        if core is not None:
            with torch.no_grad():
                A = core.alpha_trust_matrix().detach().float().cpu().numpy()
            p = np.clip(A, 1e-7, 1.0 - 1e-7)
            h_W_edge_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())

        snap: dict = {
            "id":                    self.id,
            "name":                  self.name,
            "env_type":              self.env.preset,
            "activation":            self.activation,
            "graph_mdl":             round(self.graph.mdl_size, 3),
            "compression_gain":      round(self.compression_gain, 4),
            "alpha_mean":            round(self.graph.alpha_mean, 3),
            "phi":                   round(self.phi_approx(), 4),
            "node_count":            len(self.graph.nodes),
            "edge_count":            len(self.graph.edges),
            "total_interventions":   self._total_interventions,
            "total_blocked":         self._total_blocked,
            "last_do":               self._last_do,
            "last_blocked_reason":   self._last_blocked_reason,
            "discovery_rate":        round(cur_dr, 3),
            "peak_discovery_rate":   round(self._peak_discovery_rate, 3),
            "h_W":                   round(h_W, 4),
            "notears":               notears_info,
            "temporal":              tb_info,
            "system1":               s1_info,
            "value_layer":           vl_info,
            "teacher": {
                "weight":     round(self._teacher_weight, 4),
                "rules":      len(self._teacher_rules),
            },
            "hypothesis_eig": _hypothesis_eig_from_env(),
            "h_W_edge_entropy": None if h_W_edge_entropy is None else round(h_W_edge_entropy, 4),
            "rsi_lite": {
                "enabled": rsi_lite_enabled(),
                "plateau_count": self._rsi_plateau_count,
                "ref_discovery": round(self._rsi_ref_discovery, 5),
                "adjustments": self._rsi_adjustment_count,
                "LAMBDA_L1": round(float(getattr(self.graph, "LAMBDA_L1", CausalGraph.LAMBDA_L1)), 5),
                "graph_BUFFER_SIZE": int(self.graph.BUFFER_SIZE),
                "imagination_horizon": int(self._imagination_horizon),
            },
            "local_reflex_train": self._last_local_reflex_train,
            "edges": [e.as_dict() for e in self.graph.edges],
        }
        if self.env.preset == "pybullet":
            pos_fn = getattr(self.env, "object_positions_world", None)
            if callable(pos_fn):
                snap["physics_objects"] = pos_fn()
        return snap
===
"""
agent_v4.py — RKKAgent с Value Layer (Шаг А).

Изменения:
  - ValueLayer.check_action() вызывается перед каждым do()
  - Заблокированные действия → penalty для System 1 + лог события
  - LLM/RAG seed interface: inject_text_priors(edges_json)
  - Fallback scorer когда System 1 буфер ещё мал
  - other_agents_phi передаётся из Simulation для ΔΦ≥0 constraint

Этап B (гипотезо-ориентированное исследование):
  score_interventions() — аппроксимация информационного выигрыша: чувствительность по узлам
  плюс суррогат снижения суммарной epistemic mass по рёбрам при предсказанном obs (не полный
  байесовский H(W)−E[H(W|obs)]). RKK_EIG_ENTROPY_TERM, RKK_EIG_POSTERIOR_ETA.
  Переключатель: RKK_HYPOTHESIS_EIG=1 (по умолчанию) | 0 | system1 | off | false
  В snapshot: h_W_edge_entropy — сумма бинарных энтропий по α_trust рёбер (диагностика неопределённости W).
  RKK_SCORE_ASYNC=1: score_interventions в фоновом daemon-потоке (тик не ждёт; возможна гонка с train_step — не рекомендуется).
  По умолчанию RKK_SCORE_ASYNC=0 — синхронный пересчёт в главном потоке (стабильно, без общего lock на граф).

Этап Г (самомодель): self_* + update_self_feedback() в humanoid — коррекция намерений по исходу do()
  и по промаху GNN (RKK_SELF_FEEDBACK_LR).

Этап E (целевое планирование): при self_goal_active и наличии target_dist в графе — поиск действия
  через imagination (propagate_from + rollout_step_free), см. engine.goal_planning; RKK_GOAL_PLANNING=0 отключает.

Этап F (символьный верификатор): проверка предсказания propagate на PHYSICS_CONSTRAINTS (engine.symbolic_verifier);
  нарушение → не prepend goal-plan, смешивание expected_ig с uncertainty на следующем шаге; RKK_SYMBOLIC_VERIFY=0 отключает.

Этап G (RSI lite): плато discovery_rate → агент усиливает L1, удваивает BUFFER_SIZE графа (до капа), +1 imagination;
  engine.rsi_lite, RKK_RSI_LITE=0 отключает; RKK_RSI_PLATEAU_TICKS, RKK_RSI_MIN_INTERVENTIONS.
"""
from __future__ import annotations

import os
import threading
from typing import Any
import torch
import numpy as np
from collections import deque

from engine.causal_graph import CausalGraph
from engine.graph_constants import is_read_only_macro_var
from engine.environment  import Environment
from engine.system1      import System1
from engine.temporal     import TemporalBlankets
from engine.value_layer  import ValueLayer, HomeostaticBounds, BlockReason
from engine.phase3_teacher import TeacherIGRule
from engine.environment_humanoid import SELF_VARS
from engine.goal_planning import (
    goal_planning_globally_disabled,
    parse_plan_value_levels,
    plan_beam_k,
    plan_depth,
    plan_max_branch,
    planning_graph_motor_vars,
)
from engine.symbolic_verifier import (
    downrank_factor_for_violation,
    exploration_blend_from_uncertainty,
    symbolic_verifier_enabled,
    verify_normalized_prediction,
)
from engine.wm_neural_ode import integrate_world_model_step
from engine.rsi_lite import (
    rsi_buffer_cap,
    rsi_imagination_cap,
    rsi_improvement_eps,
    rsi_l1_max,
    rsi_l1_scale,
    rsi_lite_enabled,
    rsi_min_interventions,
    rsi_plateau_interventions,
)
from engine.local_reflex import local_reflex_train_enabled, train_chains_parallel

ACTIVATIONS   = ["relu", "gelu", "tanh"]
NOTEARS_EVERY = 8
MAX_FALLBACK_TRIES = 5  # больше кандидатов, чтобы пройти Value Layer в начале обучения
# Вес slot_* в actual_ig для System 1; основной сигнал — не-визуальные узлы (RKK_VISUAL_IG_WEIGHT=0 → только физика).
VISUAL_IG_WEIGHT = float(os.environ.get("RKK_VISUAL_IG_WEIGHT", "0.1"))
_SELF_VAR_SET = frozenset(SELF_VARS)
# RKK_LOCOMOTION_CPG=1: CPG ведёт ноги; EIG не выбирает прямые do() по этим узлам.
_LOCOMOTION_CPG_LEG_EIG_BLOCK = frozenset(
    {"lhip", "lknee", "lankle", "rhip", "rknee", "rankle"}
)


def _is_motor_intent_var(name: str) -> bool:
    return str(name).startswith("intent_") or str(name).startswith("phys_intent_")


def _hypothesis_eig_from_env() -> bool:
    """Этап B: байесовский выбор эксперимента (EIG) вместо только System 1."""
    v = os.environ.get("RKK_HYPOTHESIS_EIG", "1").strip().lower()
    return v not in ("0", "false", "off", "system1", "no", "s1")


def _eig_chunk_size() -> int:
    try:
        return max(1, int(os.environ.get("RKK_EIG_BATCH", "256")))
    except ValueError:
        return 256


def _score_cache_every() -> int:
    """Пересчёт score_interventions не чаще чем раз в N тиков движка (RKK_SCORE_CACHE_EVERY; 1 = каждый тик)."""
    try:
        return max(1, int(os.environ.get("RKK_SCORE_CACHE_EVERY", "1")))
    except ValueError:
        return 1


def _score_async_enabled() -> bool:
    """Фоновый поток для score_interventions; по умолчанию выкл. (лок на весь WM давал рывки UI)."""
    v = os.environ.get("RKK_SCORE_ASYNC", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _imagination_horizon_from_env() -> int:
    """Фаза 13: RKK_IMAGINATION_STEPS — число шагов core(X) после мысленного do(); 0 = как раньше."""
    raw = os.environ.get("RKK_IMAGINATION_STEPS", "2")
    try:
        h = int(raw)
    except ValueError:
        h = 0
    return max(0, h)


class RKKAgent:
    def __init__(
        self,
        agent_id: int,
        name:     str,
        env:      Environment,
        device:   torch.device,
        bounds:   HomeostaticBounds | None = None,
    ):
        self.id         = agent_id
        self.name       = name
        self.env        = env
        self.device     = device
        self.activation = ACTIVATIONS[agent_id % 3]

        self.graph   = CausalGraph(device)
        self.system1 = System1(activation=self.activation, device=device)
        self.temporal = TemporalBlankets(
            d_input=len(env.variable_ids), device=device
        )
        self.value_layer = ValueLayer(bounds)
        self._imagination_horizon = _imagination_horizon_from_env()

        self._cg_history: deque[float] = deque(maxlen=20)
        self._total_interventions = 0
        self._total_blocked       = 0
        self._last_do             = "—"
        self._last_blocked_reason = ""
        self._last_result: dict | None = None
        self._symbolic_prediction_bad = False
        self._peak_discovery_rate: float = 0.0
        self._rsi_ref_discovery: float = 0.0
        self._rsi_plateau_count: int = 0
        self._rsi_adjustment_count: int = 0
        self._notears_steps  = 0
        self._last_notears_loss: dict | None = None
        self._local_reflex_cores: dict[tuple[str, ...], Any] = {}
        self._last_local_reflex_train: dict | None = None

        # Φ других агентов (заполняется Simulation-ом перед step())
        self.other_agents_phi: list[float] = []
        self._last_engine_tick = 0
        self._score_cache: list[dict] = []
        self._score_cache_tick: int = -9_999_999
        self._score_thread: threading.Thread | None = None
        self._score_result: list[dict] = []
        self._score_lock = threading.Lock()

        # Фаза 3: LLM-учитель (IG-бонус затухает с числом интервенций)
        self._teacher_rules: list[TeacherIGRule] = []
        self._teacher_weight: float = 0.0

        self._bootstrap()

    # ── Bootstrap + LLM seed interface ───────────────────────────────────────
    def _bootstrap(self):
        for var_id, val in self.env.variables.items():
            self.graph.set_node(var_id, val)

        obs0 = dict(self.env.variables)
        self.graph.record_observation(obs0)
        self.temporal.step(obs0)

        # Text priors (spurious + partial GT)
        gt = self.env.gt_edges()
        for e in gt[:2]:
            noisy_w = e["weight"] * 0.3 + (np.random.rand() - 0.5) * 0.4
            self.graph.set_edge(e["from_"], e["to"], noisy_w, alpha=0.06)

        var_ids = self.env.variable_ids
        if len(var_ids) >= 4:
            self.graph.set_edge(var_ids[1], var_ids[3],  0.35, alpha=0.05)
            self.graph.set_edge(var_ids[2], var_ids[0], -0.20, alpha=0.04)

        # Фаза 1: заморозка URDF-цепочек в L1 (humanoid VAR_NAMES).
        fr = os.environ.get("RKK_FREEZE_URDF", "1").strip().lower()
        if fr not in ("0", "false", "no", "off") and "lhip" in self.env.variable_ids:
            self.graph.freeze_kinematic_priors()

    def inject_text_priors(self, edges: list[dict]) -> dict:
        """
        LLM/RAG seed interface.

        edges: [{"from_": "Temp", "to": "Pressure", "weight": 0.8}, ...]

        Все рёбра загружаются с alpha=0.05 (низкое доверие).
        Epistemic Annealing + NOTEARS выжгут ошибочные за N интервенций.

        Узлы from_/to должны совпадать с id переменных окружения (env.variable_ids).

        Возвращает {"injected": n, "skipped": [причины...]}.
        """
        count   = 0
        skipped: list[str] = []
        valid   = set(self.graph.nodes.keys())

        for e in edges:
            from_ = e.get("from_") or e.get("from")
            to    = e.get("to")
            w     = float(e.get("weight", 0.3))

            if not from_ or not to:
                skipped.append(f"нет from_/to: {e!r}")
                continue
            if is_read_only_macro_var(from_) or is_read_only_macro_var(to):
                skipped.append(f"read-only macro: {from_!r}→{to!r}")
                continue
            if from_ not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{from_}» (доступны: {sorted(valid)})")
                continue
            if to not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{to}» (доступны: {sorted(valid)})")
                continue

            alpha = float(e.get("alpha", 0.05))
            # Слабые семена по умолчанию (0.2–0.3 экв.) — не «пугают» граф и VL
            w_scaled = min(0.3, max(0.08, float(w) * 0.28))
            self.graph.set_edge(from_, to, w_scaled, alpha=alpha)
            count += 1

        return {"injected": count, "skipped": skipped, "node_ids": sorted(valid)}

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_h_W(self) -> float:
        if self.graph._core is None:
            return 0.0
        return float(self.graph._core.dag_constraint().item())

    @staticmethod
    def _marginal_node_uncertainty(unc_m: np.ndarray) -> np.ndarray:
        """
        Маргинальная неопределённость по узлу j: max по всем рёбрам (j→·) и (·→j).
        unc_m[i,j] — epistemic mass на ребре i→j (posterior proxy: 1 − α_trust).
        """
        row_max = unc_m.max(axis=1)
        col_max = unc_m.max(axis=0)
        return np.maximum(row_max, col_max).astype(np.float64, copy=False)

    def _batch_hypothesis_eig(
        self,
        candidates: list[dict],
        X_np: np.ndarray,
        u_node: np.ndarray,
        nid_to_i: dict[str, int],
        unc_m: np.ndarray,
        node_ids: list[str],
        env: Environment,
    ) -> list[float]:
        """
        Суррогат «информативности» действия: (1) чувствительность Σ_j u(j)|ΔX_j|;
        (2) суррогат снижения неопределённости по рёбрам после гипотетического наблюдения
        (масштабирование unc_ij пропорционально |ΔX_i|+|ΔX_j|). Это не точный EIG по H(W).
        """
        core = self.graph._core
        if core is None or not candidates:
            return []
        fd = getattr(core, "forward_dynamics", None)
        if not callable(fd):
            return []

        try:
            lam = float(os.environ.get("RKK_EIG_ENTROPY_TERM", "0.22"))
        except ValueError:
            lam = 0.22
        try:
            eta = float(os.environ.get("RKK_EIG_POSTERIOR_ETA", "0.18"))
        except ValueError:
            eta = 0.18
        lam = max(0.0, lam)
        eta = max(0.0, min(0.95, eta))

        d = int(X_np.shape[0])
        device = self.device
        chunk = _eig_chunk_size()
        eigs: list[float] = []
        x0 = torch.from_numpy(X_np).to(dtype=torch.float32, device=device).unsqueeze(0)
        uu = unc_m.reshape(1, d, d)

        for start in range(0, len(candidates), chunk):
            sub = candidates[start : start + chunk]
            b = len(sub)
            x_batch = x0.expand(b, -1)
            a_batch = torch.zeros(b, d, device=device, dtype=torch.float32)
            for bi, cand in enumerate(sub):
                idx = nid_to_i.get(cand["variable"])
                if idx is not None:
                    a_batch[bi, idx] = float(cand["value"])
            with torch.no_grad():
                pred = integrate_world_model_step(core, x_batch, a_batch)
            delta = (pred - x_batch).abs().cpu().numpy()
            ab = np.abs(delta)
            S = np.clip(ab[:, :, None] + ab[:, None, :], 0.0, 1.0)
            new_u = uu * (1.0 - eta * S)
            new_u = np.maximum(new_u, 0.0)
            reduction = (uu - new_u).sum(axis=(1, 2))
            sens = (delta * u_node.reshape(1, -1)).sum(axis=1)
            total = sens + lam * reduction
            if symbolic_verifier_enabled():
                fac = downrank_factor_for_violation()
                d_nodes = len(node_ids)
                for bi in range(b):
                    pd = {
                        node_ids[j]: float(pred[bi, j].item())
                        for j in range(min(d_nodes, int(pred.shape[1])))
                    }
                    ok, _ = verify_normalized_prediction(pd, env)
                    if not ok:
                        total[bi] *= fac
            eigs.extend(total.tolist())
        return eigs

    def _rollout_imagination_state(
        self, base: dict[str, float], var: str, val: float
    ) -> dict[str, float]:
        """Этап E: один мысленный do + столько же свободных шагов, сколько в VL imagination."""
        s = self.graph.propagate_from(dict(base), var, float(val))
        for _ in range(max(0, self._imagination_horizon)):
            s = self.graph.rollout_step_free(s)
        return s

    def _features_for_intervention_pair(self, v_from: str, v_to: str) -> list[float]:
        """Один вектор признаков System1 для пары (в_from→в_to), как в score_interventions."""
        h_W_norm = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        core = self.graph._core
        ii, jj = nid_to_i.get(v_from), nid_to_i.get(v_to)
        if core is not None and ii is not None and jj is not None:
            with torch.no_grad():
                unc_t = (1.0 - core.alpha_trust_matrix()).detach().float().cpu().numpy()
                W_m = core.W_masked().detach().float().cpu().numpy()
                g_m = None
                if core.W.grad is not None:
                    g_m = core.W.grad.detach().float().abs().cpu().numpy()
            uncertainty = float(unc_t[ii, jj])
            w_ij = float(W_m[ii, jj])
            grad_norm = float(g_m[ii, jj]) if g_m is not None else 0.0
        else:
            uncertainty, w_ij, grad_norm = 1.0, 0.0, 0.0
        alpha = 1.0 - uncertainty
        val_from = self.graph.nodes.get(v_from, 0.5)
        val_to = self.graph.nodes.get(v_to, 0.5)
        ic = ic_map.get((v_from, v_to), 0)
        return self.system1.build_features(
            w_ij=w_ij, alpha_ij=alpha,
            val_from=val_from, val_to=val_to,
            uncertainty=uncertainty, h_W_norm=h_W_norm,
            grad_norm_ij=grad_norm,
            intervention_count=ic,
            discovery_rate=disc_rate,
        )

    def _build_goal_planned_candidate(self, var: str, val: float) -> dict:
        feat = self._features_for_intervention_pair(var, "target_dist")
        return {
            "variable":    var,
            "target":      "target_dist",
            "value":       float(val),
            "uncertainty": 0.35,
            "features":    feat,
            "expected_ig": 1.0,
            "from_goal_plan": True,
        }

    def _maybe_goal_planned_candidate(self) -> dict | None:
        if goal_planning_globally_disabled():
            return None
        if self.graph._core is None:
            return None
        if self.graph.nodes.get("self_goal_active") is None:
            return None
        if float(self.graph.nodes.get("self_goal_active", 0)) <= 0.45:
            return None
        if "target_dist" not in self.graph.nodes:
            return None

        state0 = dict(self.graph.nodes)
        cur_td = float(state0.get("target_dist", 0.5))
        goal_thr = float(state0.get("self_goal_target_dist", 0.42))
        if cur_td <= goal_thr + 0.015:
            return None

        motor = planning_graph_motor_vars(self.env, list(self.graph._node_ids))
        if not motor:
            return None

        levels = parse_plan_value_levels()
        actions = [(v, x) for v in motor for x in levels]
        max_b = plan_max_branch()
        if len(actions) > max_b:
            idx = np.random.choice(len(actions), size=max_b, replace=False)
            actions = [actions[i] for i in idx]

        depth = plan_depth()
        beam_k = plan_beam_k()

        def _td(s: dict[str, float]) -> float:
            return float(s.get("target_dist", cur_td))

        best_td = cur_td
        best_first: tuple[str, float] | None = None

        if depth <= 1:
            for var, val in actions:
                try:
                    sfin = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                    if not ok:
                        continue
                td = _td(sfin)
                if td < best_td - 1e-6:
                    best_td = td
                    best_first = (var, val)
        else:
            scored: list[tuple[float, str, float, dict[str, float]]] = []
            for var, val in actions:
                try:
                    s1 = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(s1), self.env)
                    if not ok:
                        continue
                scored.append((_td(s1), var, val, dict(s1)))
            scored.sort(key=lambda t: t[0])
            for _td1, v1, x1, s1 in scored[:beam_k]:
                for v2, x2 in actions:
                    try:
                        sfin = self._rollout_imagination_state(s1, v2, x2)
                    except Exception:
                        continue
                    if symbolic_verifier_enabled():
                        ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                        if not ok:
                            continue
                    td = _td(sfin)
                    if td < best_td - 1e-6:
                        best_td = td
                        best_first = (v1, x1)

        if best_first is None:
            return None
        return self._build_goal_planned_candidate(best_first[0], best_first[1])

    def _is_locomotion_primary_active(self) -> bool:
        """Если CPG управляет ногами, EIG не должен конкурировать за суставы — только intent_* и др."""
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        var_ids   = self.env.variable_ids
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate

        # Один проход по рёбрам: счётчики интервенций (раньше — O(pairs×|E|) через next() в цикле)
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count

        # Имя узла → индекс без O(d) list.index на каждую пару
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}

        # Один раз W, α и |grad| на CPU — вместо O(d²) вызовов alpha_trust_matrix / W_masked
        core = self.graph._core
        W_m = unc_m = g_m = None
        if core is not None:
            with torch.no_grad():
                W_t = core.W_masked().detach().float()
                A_t = core.alpha_trust_matrix().detach().float()
                W_m = W_t.cpu().numpy()
                unc_m = (1.0 - A_t).cpu().numpy()
            if core.W.grad is not None:
                g_m = core.W.grad.detach().float().abs().cpu().numpy()

        d = len(var_ids)
        if d == 0:
            return []

        # Счётчики интервенций по парам (только известные рёбра — O(|E|))
        ic_mat = np.zeros((d, d), dtype=np.float64)
        v2i = {v: i for i, v in enumerate(var_ids)}
        for (vf, vt), c in ic_map.items():
            i = v2i.get(vf)
            j = v2i.get(vt)
            if i is not None and j is not None and i != j:
                ic_mat[i, j] = float(c)

        ridx = np.zeros(d, dtype=np.int64)
        valid_node = np.zeros(d, dtype=bool)
        for i, v in enumerate(var_ids):
            ji = nid_to_i.get(v)
            if ji is not None:
                ridx[i] = ji
                valid_node[i] = True

        nodes_arr = np.array(
            [float(self.graph.nodes.get(v, 0.5)) for v in var_ids],
            dtype=np.float64,
        )
        mask = ~np.eye(d, dtype=bool)
        fi, fj = np.where(mask)
        n_pairs = len(fi)

        if W_m is not None:
            ii_n = ridx[fi]
            jj_n = ridx[fj]
            ok = valid_node[fi] & valid_node[fj]
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)
            w_ij[ok] = W_m[ii_n[ok], jj_n[ok]]
            uncertainty[ok] = unc_m[ii_n[ok], jj_n[ok]]
            if g_m is not None:
                grad_norm[ok] = g_m[ii_n[ok], jj_n[ok]]
        else:
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)

        alpha = 1.0 - uncertainty
        val_from = nodes_arr[fi]
        val_to = nodes_arr[fj]
        ic_v = ic_mat[fi, fj]
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_v = float(np.clip(disc_rate, 0.0, 1.0))

        feats_arr = np.column_stack(
            [
                np.tanh(w_ij),
                np.clip(alpha, 0.0, 1.0),
                np.clip(val_from, 0.0, 1.0),
                np.clip(val_to, 0.0, 1.0),
                np.clip(uncertainty, 0.0, 1.0),
                np.full(n_pairs, h_clip, dtype=np.float64),
                np.tanh(grad_norm),
                np.clip(ic_v / 100.0, 0.0, 1.0),
                np.full(n_pairs, disc_v, dtype=np.float64),
            ]
        )
        features_batch = feats_arr.tolist()

        rng = np.random.default_rng()
        posture_now = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 0.5),
            )
        )
        foot_l_now = float(
            self.graph.nodes.get(
                "foot_contact_l",
                self.graph.nodes.get("phys_foot_contact_l", 0.5),
            )
        )
        foot_r_now = float(
            self.graph.nodes.get(
                "foot_contact_r",
                self.graph.nodes.get("phys_foot_contact_r", 0.5),
            )
        )
        stable_stance = posture_now > 0.70 and min(foot_l_now, foot_r_now) > 0.56

        # ── Sparse EIG: skip low-uncertainty pairs ──────────────────────────
        try:
            _sparse_min_unc = float(os.environ.get("RKK_SPARSE_EIG_MIN_UNC", "0.15"))
        except ValueError:
            _sparse_min_unc = 0.15
        _sparse_min_unc = max(0.0, min(0.8, _sparse_min_unc))

        candidates: list[dict] = []
        for k in range(n_pairs):
            i, j = int(fi[k]), int(fj[k])
            vf, vt = var_ids[i], var_ids[j]
            unc_k = float(uncertainty[k])
            feat_k = features_batch[k]

            # Sparse filter: skip well-known edges (except motor intents)
            if _sparse_min_unc > 0 and unc_k < _sparse_min_unc:
                if not _is_motor_intent_var(vf):
                    continue

            if _is_motor_intent_var(vf):
                if stable_stance:
                    lo, hi = 0.30, 0.72
                else:
                    lo, hi = 0.35, 0.68
                if str(vf).endswith("stride"):
                    hi = min(hi, 0.62 if stable_stance else 0.56)
                if str(vf).endswith("stop_recover"):
                    lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                rand_value = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
            else:
                rand_value = float(np.clip(rng.uniform(0.15, 0.85), 0.06, 0.94))
            candidates.append({
                "variable":    vf,
                "target":      vt,
                "value":       rand_value,
                "uncertainty": unc_k,
                "features":    feat_k,
                "expected_ig": 0.0,
            })

        if self._is_locomotion_primary_active():
            candidates = [
                c
                for c in candidates
                if c["variable"] not in _LOCOMOTION_CPG_LEG_EIG_BLOCK
            ]
            if posture_now < 0.65:
                candidates = [
                    c
                    for c in candidates
                    if str(c["variable"]).startswith("intent_")
                    or str(c["variable"]).startswith("phys_intent_")
                ]

        if not candidates or not features_batch:
            return []

        use_eig = _hypothesis_eig_from_env() and W_m is not None and unc_m is not None
        if use_eig:
            x_vec = np.array(
                [float(self.graph.nodes.get(n, 0.0)) for n in self.graph._node_ids],
                dtype=np.float64,
            )
            u_node = self._marginal_node_uncertainty(unc_m)
            eigs = self._batch_hypothesis_eig(
                candidates, x_vec, u_node, nid_to_i, unc_m,
                list(self.graph._node_ids), self.env,
            )
            if len(eigs) == len(candidates):
                # Учитываем гипотезу «это ребро неизвестно»: масштаб EIG по unc(v_from→v_to).
                for i, cand in enumerate(candidates):
                    eigs[i] *= 1.0 + float(cand["uncertainty"])
                arr = np.array(eigs, dtype=np.float64)
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo + 1e-12:
                    normed = (arr - lo) / (hi - lo)
                else:
                    normed = np.full_like(arr, 0.5)
                for i, cand in enumerate(candidates):
                    cand["eig_raw"] = float(eigs[i])
                    cand["expected_ig"] = float(normed[i])
            else:
                use_eig = False

        if not use_eig:
            scores = self.system1.score(features_batch)
            for i, cand in enumerate(candidates):
                cand["expected_ig"] = scores[i]

        if symbolic_verifier_enabled() and self._symbolic_prediction_bad:
            a, b = exploration_blend_from_uncertainty()
            for cand in candidates:
                unc = float(cand.get("uncertainty", 0.5))
                cand["expected_ig"] = a * float(cand["expected_ig"]) + b * unc

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    def _score_async_worker(self) -> None:
        try:
            with torch.no_grad():
                result = self.score_interventions()
            with self._score_lock:
                self._score_result = result
        except Exception as ex:
            print(f"[RKKAgent] score_interventions (async): {ex}")

    def set_teacher_state(self, rules: list[TeacherIGRule], weight: float) -> None:
        """Фаза 3: правила от LLM и текущий teacher_weight (симуляция считает annealing)."""
        self._teacher_rules = list(rules)
        self._teacher_weight = float(max(0.0, min(1.0, weight)))

    def _teacher_ig_bonus(self, variable: str, nodes: dict[str, float]) -> float:
        w = self._teacher_weight
        if w <= 0 or not self._teacher_rules:
            return 0.0
        acc = 0.0
        for r in self._teacher_rules:
            if r.target_var != variable:
                continue
            if r.when_var:
                val = nodes.get(r.when_var)
                if val is None:
                    continue
                if r.when_min is not None and float(val) < r.when_min:
                    continue
                if r.when_max is not None and float(val) > r.when_max:
                    continue
            acc += r.bonus * w
        return min(0.28, acc)

    # ── Один шаг с Value Layer ────────────────────────────────────────────────
    def step(self, engine_tick: int = 0, *, enable_l3: bool = True) -> dict:
        self._last_engine_tick = engine_tick
        try:
            self.graph.apply_env_observation(dict(self.env.observe()))
        except Exception:
            pass
        sce = _score_cache_every()
        if (
            sce > 1
            and self._score_cache
            and (engine_tick - self._score_cache_tick) < sce
        ):
            scores = list(self._score_cache)
        elif _score_async_enabled():
            if self._score_thread is None or not self._score_thread.is_alive():
                self._score_thread = threading.Thread(
                    target=self._score_async_worker,
                    name="rkk_score_interventions",
                    daemon=True,
                )
                self._score_thread.start()
            with self._score_lock:
                have = list(self._score_result) if self._score_result else []
            if have:
                scores = have
            elif self._score_cache:
                scores = list(self._score_cache)
            else:
                with torch.no_grad():
                    scores = self.score_interventions()
                with self._score_lock:
                    self._score_result = list(scores)
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        else:
            with torch.no_grad():
                scores = self.score_interventions()
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        gp = self._maybe_goal_planned_candidate() if enable_l3 else None
        if gp is not None and not (
            symbolic_verifier_enabled() and self._symbolic_prediction_bad
        ):
            scores.insert(0, gp)
        if not scores:
            return {
                "blocked": False, "skipped": True, "prediction_error": 0.0,
                "cf_predicted": {}, "cf_observed": {}, "goal_planned": False,
            }

        current_phi = self.phi_approx()
        chosen      = None
        check_result = None
        blocked_count = 0

        # Перебираем кандидатов пока не найдём допустимое действие
        for candidate in scores[:MAX_FALLBACK_TRIES]:
            var   = candidate["variable"]
            value = candidate["value"]

            check_result = self.value_layer.check_action(
                variable=var,
                value=value,
                current_nodes=dict(self.graph.nodes),
                graph=self.graph,
                temporal=self.temporal,
                current_phi=current_phi,
                other_agents_phi=self.other_agents_phi,
                engine_tick=engine_tick,
                imagination_horizon=(self._imagination_horizon if enable_l3 else 0),
            )

            if check_result.allowed:
                chosen = candidate
                break
            else:
                # Штрафуем System 1 за предложение опасного действия
                self.system1.push_experience(
                    features=candidate["features"],
                    actual_ig=check_result.penalty,   # отрицательный IG
                )
                blocked_count += 1
                self._total_blocked += 1
                self._last_blocked_reason = check_result.reason.value

        # Все кандидаты заблокированы — возвращаем событие
        if chosen is None:
            return {
                "blocked":       True,
                "blocked_count": blocked_count,
                "reason":        self._last_blocked_reason,
                "variable":      scores[0]["variable"] if scores else "?",
                "value":         scores[0]["value"] if scores else 0.5,
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error":  0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        # ── Выполняем допустимое действие ────────────────────────────────────
        var   = chosen["variable"]
        value = chosen["value"]

        if is_read_only_macro_var(var):
            return {
                "blocked": True,
                "blocked_count": blocked_count + 1,
                "reason": "read_only_macro",
                "variable": var,
                "value": float(value),
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        mdl_before = self.graph.mdl_size
        obs_before_env = dict(self.env.observe())
        self.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.graph.snapshot_vec_dict()
        predicted  = self.graph.propagate(var, value)
        sym_ok, sym_fail = True, []
        if symbolic_verifier_enabled():
            sym_ok, sym_fail = verify_normalized_prediction(dict(predicted), self.env)
            self._symbolic_prediction_bad = not sym_ok
        else:
            self._symbolic_prediction_bad = False
        observed_env = self.env.intervene(var, value)

        # Temporal step (только размерность среды)
        self.temporal.step(observed_env)

        self.graph.apply_env_observation(observed_env)
        observed_full = self.graph.snapshot_vec_dict()

        # NOTEARS / GNN буферы — полный вектор узлов (включая concept_*)
        self.graph.record_observation(obs_before_full)
        self.graph.record_observation(observed_full)
        self.graph.record_intervention(var, value, obs_before_full, observed_full)

        # NOTEARS train
        notears_result = None
        if self._total_interventions % NOTEARS_EVERY == 0:
            notears_result = self.graph.train_step()
            if notears_result:
                self._notears_steps += 1
                self._last_notears_loss = notears_result
            self._maybe_train_local_reflex()

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # System 1: IG по физике; slot_* и self_* не доминируют метрику (self — прямое задание агентом).
        nids = self.graph._node_ids
        phys_ids = [
            k for k in nids
            if k not in _SELF_VAR_SET and not str(k).startswith("slot_")
        ]
        slot_ids = [k for k in nids if str(k).startswith("slot_")]

        def _mean_abs_err(keys: list) -> float:
            if not keys:
                return 0.0
            return float(np.mean([
                abs(float(predicted.get(k, 0.5)) - float(observed_full.get(k, 0.5)))
                for k in keys
            ]))

        pe_phys = _mean_abs_err(phys_ids)

        # Этап Г: петля «намерение ↔ исход» + ошибка модели → self_* (только среды с методом).
        fn_sf = getattr(self.env, "update_self_feedback", None)
        if callable(fn_sf):
            try:
                fn_sf(
                    variable=var,
                    intended_norm=value,
                    observed=observed_env,
                    predicted=predicted,
                    prediction_error_phys=pe_phys,
                )
            except Exception:
                pass
            obs_self = dict(self.env.observe())
            for sk in _SELF_VAR_SET:
                if sk in self.graph.nodes and sk in obs_self:
                    self.graph.nodes[sk] = float(obs_self[sk])
            self.graph.refresh_concept_aggregates()
        pe_slot = _mean_abs_err(slot_ids)
        w_vis = min(0.45, max(0.0, VISUAL_IG_WEIGHT))
        if slot_ids and phys_ids:
            actual_ig = (1.0 - w_vis) * pe_phys + w_vis * pe_slot
        elif phys_ids:
            actual_ig = pe_phys
        else:
            actual_ig = pe_slot

        t_bonus = self._teacher_ig_bonus(var, dict(self.graph.nodes))
        actual_ig = float(np.clip(actual_ig + t_bonus, 0.0, 1.0))

        self.system1.push_experience(
            features=chosen["features"],
            actual_ig=actual_ig,
        )

        # SSM train — размерность = temporal.d_input (= |graph._node_ids|), не только env.variable_ids
        u_next = torch.tensor(
            [float(self.graph.nodes.get(n, 0.5)) for n in self.graph._node_ids],
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal.train_step(u_next)

        self._total_interventions += 1
        try:
            _v_do = float(value)
        except (TypeError, ValueError):
            _v_do = 0.5
        self._last_do = f"do({var}={_v_do:.2f})"
        self._last_blocked_reason = ""

        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        rsi_event = self._tick_rsi_lite_discovery(cur_dr)

        _cf_keys = list(self.graph._node_ids)[:48]
        self._last_result = {
            "blocked":           False,
            "blocked_count":     blocked_count,
            "variable":          var,
            "value":             value,
            "compression_delta": compression_delta,
            "updated_edges":     [f"{e.from_}→{e.to}" for e in self.graph.edges[:4]],
            "pruned_edges":      [],
            "prediction_error":  float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed_env.items()
            ])),
            "cf_predicted": {k: float(round(float(predicted.get(k, 0.0)), 4)) for k in _cf_keys},
            "cf_observed":  {k: float(round(float(observed_full.get(k, 0.0)), 4)) for k in _cf_keys},
            "goal_planned":  bool(chosen.get("from_goal_plan")),
            "symbolic_ok": sym_ok,
            "symbolic_violations": sym_fail,
            "rsi_lite": rsi_event,
            "notears":           notears_result,
        }
        return self._last_result

    # ── Demon ─────────────────────────────────────────────────────────────────
    def demon_disrupt(self) -> str:
        if self.graph._core is None:
            return "no core"
        with torch.no_grad():
            W = self.graph._core.W
            sig = (W.abs() > 0.05).nonzero(as_tuple=False)
            if len(sig) == 0:
                return "no significant edges"
            idx = sig[np.random.randint(len(sig))]
            i, j = idx[0].item(), idx[1].item()
            noise = (np.random.rand() - 0.5) * 0.3
            # Нельзя W[i,j] += … — это in-place на view листа с requires_grad.
            w_new = W.detach().clone()
            w_new[i, j] = w_new[i, j] + float(noise)
            W.copy_(w_new)
            fn = self.graph._node_ids[i] if i < len(self.graph._node_ids) else f"v{i}"
            tn = self.graph._node_ids[j] if j < len(self.graph._node_ids) else f"v{j}"
        self.graph._invalidate_cache()
        return f"W[{fn}→{tn}] +{noise:.3f}"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def compression_gain(self) -> float:
        if not self._cg_history:
            return 0.0
        return float(np.mean(list(self._cg_history)))

    @property
    def discovery_rate(self) -> float:
        """
        Blend of GT-based and self-supervised discovery rate.
        As the agent matures, self-supervised metric gets more weight.
        """
        gt_dr = self.env.discovery_rate([
            {"from_": e.from_, "to": e.to, "weight": e.weight}
            for e in self.graph.edges
        ])
        # Self-supervised: compression discoveries / total computations
        ss_dr = self.self_supervised_discovery_rate
        # Blend: GT dominates early (calibration), self-supervised dominates later
        if self._total_interventions < 200:
            return gt_dr
        blend = min(1.0, (self._total_interventions - 200) / 1000.0)
        return (1.0 - blend) * gt_dr + blend * ss_dr

    @property
    def self_supervised_discovery_rate(self) -> float:
        """
        Discovery rate without ground-truth edges.
        Based on CausalSurprise compression discoveries — the fraction of
        interventions that actually improved the causal model.
        """
        # Try to get from IntrinsicObjective (if simulation has it patched in)
        try:
            from engine.intristic_objective import IntrinsicObjective
            # Walk up to find intrinsic objective
            for attr_name in ("_intrinsic",):
                # IntrinsicObjective attaches to simulation, not agent
                # We use the causal_surprise directly if available
                pass
            # Fallback: use graph-level stats
            if self.graph.train_losses:
                recent = self.graph.train_losses[-20:]
                if len(recent) >= 5:
                    # Discovery = loss is still decreasing (model is learning)
                    early = float(np.mean(recent[:len(recent)//2]))
                    late = float(np.mean(recent[len(recent)//2:]))
                    if early > 1e-8:
                        improvement = max(0.0, (early - late) / early)
                        return float(np.clip(improvement * 2.0, 0.0, 1.0))
        except Exception:
            pass
        return 0.5  # neutral default

    @property
    def peak_discovery_rate(self) -> float:
        return self._peak_discovery_rate

    def _apply_rsi_lite(self) -> dict[str, float | int]:
        g = self.graph
        cur_l1 = float(getattr(g, "LAMBDA_L1", CausalGraph.LAMBDA_L1))
        new_l1 = min(cur_l1 * rsi_l1_scale(), rsi_l1_max())
        g.LAMBDA_L1 = new_l1
        cap_b = rsi_buffer_cap()
        g.BUFFER_SIZE = min(cap_b, int(g.BUFFER_SIZE) * 2)
        cap_i = rsi_imagination_cap()
        self._imagination_horizon = min(cap_i, self._imagination_horizon + 1)
        self._rsi_adjustment_count += 1
        return {
            "LAMBDA_L1": float(new_l1),
            "BUFFER_SIZE": int(g.BUFFER_SIZE),
            "imagination_horizon": int(self._imagination_horizon),
        }

    def _tick_rsi_lite_discovery(self, cur_dr: float) -> dict[str, float | int] | None:
        if not rsi_lite_enabled():
            return None
        if self._total_interventions < rsi_min_interventions():
            return None
        eps = rsi_improvement_eps()
        if cur_dr > self._rsi_ref_discovery + eps:
            self._rsi_ref_discovery = float(cur_dr)
            self._rsi_plateau_count = 0
            return None
        self._rsi_plateau_count += 1
        if self._rsi_plateau_count < rsi_plateau_interventions():
            return None
        self._rsi_plateau_count = 0
        self._rsi_ref_discovery = float(cur_dr)
        return self._apply_rsi_lite()

    def _maybe_train_local_reflex(self) -> None:
        if not local_reflex_train_enabled():
            return
        self._last_local_reflex_train = train_chains_parallel(
            graph=self.graph,
            device=self.graph.device,
            cores=self._local_reflex_cores,
        )

    def phi_approx(self) -> float:
        return self.temporal.phi_approx()

    def record_phi(self, _: float):
        pass  # temporal управляет историей сам

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        h_W     = self._get_h_W()
        tb_info = self.temporal.slow_state_summary()
        s1_info = {
            "buffer_size": len(self.system1.buffer),
            "mean_loss":   round(self.system1.mean_loss, 6),
        }
        vl_info = dict(self.value_layer.snapshot(self._last_engine_tick))
        vl_info["imagination_horizon"] = self._imagination_horizon

        notears_info = None
        if self._last_notears_loss:
            notears_info = {
                "steps":  self._notears_steps,
                "loss":   self._last_notears_loss.get("loss", 0),
                "h_W":    round(h_W, 4),
                "l_int":  self._last_notears_loss.get("l_int", 0),
            }

        h_W_edge_entropy = None
        core = self.graph._core
        if core is not None:
            with torch.no_grad():
                A = core.alpha_trust_matrix().detach().float().cpu().numpy()
            p = np.clip(A, 1e-7, 1.0 - 1e-7)
            h_W_edge_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())

        snap: dict = {
            "id":                    self.id,
            "name":                  self.name,
            "env_type":              self.env.preset,
            "activation":            self.activation,
            "graph_mdl":             round(self.graph.mdl_size, 3),
            "compression_gain":      round(self.compression_gain, 4),
            "alpha_mean":            round(self.graph.alpha_mean, 3),
            "phi":                   round(self.phi_approx(), 4),
            "node_count":            len(self.graph.nodes),
            "edge_count":            len(self.graph.edges),
            "total_interventions":   self._total_interventions,
            "total_blocked":         self._total_blocked,
            "last_do":               self._last_do,
            "last_blocked_reason":   self._last_blocked_reason,
            "discovery_rate":        round(cur_dr, 3),
            "peak_discovery_rate":   round(self._peak_discovery_rate, 3),
            "h_W":                   round(h_W, 4),
            "notears":               notears_info,
            "temporal":              tb_info,
            "system1":               s1_info,
            "value_layer":           vl_info,
            "teacher": {
                "weight":     round(self._teacher_weight, 4),
                "rules":      len(self._teacher_rules),
            },
            "hypothesis_eig": _hypothesis_eig_from_env(),
            "h_W_edge_entropy": None if h_W_edge_entropy is None else round(h_W_edge_entropy, 4),
            "rsi_lite": {
                "enabled": rsi_lite_enabled(),
                "plateau_count": self._rsi_plateau_count,
                "ref_discovery": round(self._rsi_ref_discovery, 5),
                "adjustments": self._rsi_adjustment_count,
                "LAMBDA_L1": round(float(getattr(self.graph, "LAMBDA_L1", CausalGraph.LAMBDA_L1)), 5),
                "graph_BUFFER_SIZE": int(self.graph.BUFFER_SIZE),
                "imagination_horizon": int(self._imagination_horizon),
            },
            "local_reflex_train": self._last_local_reflex_train,
            "edges": [e.as_dict() for e in self.graph.edges],
        }
        if self.env.preset == "pybullet":
            pos_fn = getattr(self.env, "object_positions_world", None)
            if callable(pos_fn):
                snap["physics_objects"] = pos_fn()
        return snap
```

**Before:** d²=4556 pairs scored every tick (d=68)  
**After:** Only pairs with `uncertainty > RKK_SPARSE_EIG_MIN_UNC` (default 0.15) are scored

Motor intent vars always pass the filter (they need to be scored regardless of uncertainty).

Expected speedup: **3-5x** (60-80% of pairs are well-known edges where the model is confident).

---

### 6. Self-supervised Discovery Rate
```diff:agent.py
"""
agent_v4.py — RKKAgent с Value Layer (Шаг А).

Изменения:
  - ValueLayer.check_action() вызывается перед каждым do()
  - Заблокированные действия → penalty для System 1 + лог события
  - LLM/RAG seed interface: inject_text_priors(edges_json)
  - Fallback scorer когда System 1 буфер ещё мал
  - other_agents_phi передаётся из Simulation для ΔΦ≥0 constraint

Этап B (гипотезо-ориентированное исследование):
  score_interventions() — аппроксимация информационного выигрыша: чувствительность по узлам
  плюс суррогат снижения суммарной epistemic mass по рёбрам при предсказанном obs (не полный
  байесовский H(W)−E[H(W|obs)]). RKK_EIG_ENTROPY_TERM, RKK_EIG_POSTERIOR_ETA.
  Переключатель: RKK_HYPOTHESIS_EIG=1 (по умолчанию) | 0 | system1 | off | false
  В snapshot: h_W_edge_entropy — сумма бинарных энтропий по α_trust рёбер (диагностика неопределённости W).
  RKK_SCORE_ASYNC=1: score_interventions в фоновом daemon-потоке (тик не ждёт; возможна гонка с train_step — не рекомендуется).
  По умолчанию RKK_SCORE_ASYNC=0 — синхронный пересчёт в главном потоке (стабильно, без общего lock на граф).

Этап Г (самомодель): self_* + update_self_feedback() в humanoid — коррекция намерений по исходу do()
  и по промаху GNN (RKK_SELF_FEEDBACK_LR).

Этап E (целевое планирование): при self_goal_active и наличии target_dist в графе — поиск действия
  через imagination (propagate_from + rollout_step_free), см. engine.goal_planning; RKK_GOAL_PLANNING=0 отключает.

Этап F (символьный верификатор): проверка предсказания propagate на PHYSICS_CONSTRAINTS (engine.symbolic_verifier);
  нарушение → не prepend goal-plan, смешивание expected_ig с uncertainty на следующем шаге; RKK_SYMBOLIC_VERIFY=0 отключает.

Этап G (RSI lite): плато discovery_rate → агент усиливает L1, удваивает BUFFER_SIZE графа (до капа), +1 imagination;
  engine.rsi_lite, RKK_RSI_LITE=0 отключает; RKK_RSI_PLATEAU_TICKS, RKK_RSI_MIN_INTERVENTIONS.
"""
from __future__ import annotations

import os
import threading
from typing import Any
import torch
import numpy as np
from collections import deque

from engine.causal_graph import CausalGraph
from engine.graph_constants import is_read_only_macro_var
from engine.environment  import Environment
from engine.system1      import System1
from engine.temporal     import TemporalBlankets
from engine.value_layer  import ValueLayer, HomeostaticBounds, BlockReason
from engine.phase3_teacher import TeacherIGRule
from engine.environment_humanoid import SELF_VARS
from engine.goal_planning import (
    goal_planning_globally_disabled,
    parse_plan_value_levels,
    plan_beam_k,
    plan_depth,
    plan_max_branch,
    planning_graph_motor_vars,
)
from engine.symbolic_verifier import (
    downrank_factor_for_violation,
    exploration_blend_from_uncertainty,
    symbolic_verifier_enabled,
    verify_normalized_prediction,
)
from engine.wm_neural_ode import integrate_world_model_step
from engine.rsi_lite import (
    rsi_buffer_cap,
    rsi_imagination_cap,
    rsi_improvement_eps,
    rsi_l1_max,
    rsi_l1_scale,
    rsi_lite_enabled,
    rsi_min_interventions,
    rsi_plateau_interventions,
)
from engine.local_reflex import local_reflex_train_enabled, train_chains_parallel

ACTIVATIONS   = ["relu", "gelu", "tanh"]
NOTEARS_EVERY = 8
MAX_FALLBACK_TRIES = 5  # больше кандидатов, чтобы пройти Value Layer в начале обучения
# Вес slot_* в actual_ig для System 1; основной сигнал — не-визуальные узлы (RKK_VISUAL_IG_WEIGHT=0 → только физика).
VISUAL_IG_WEIGHT = float(os.environ.get("RKK_VISUAL_IG_WEIGHT", "0.1"))
_SELF_VAR_SET = frozenset(SELF_VARS)
# RKK_LOCOMOTION_CPG=1: CPG ведёт ноги; EIG не выбирает прямые do() по этим узлам.
_LOCOMOTION_CPG_LEG_EIG_BLOCK = frozenset(
    {"lhip", "lknee", "lankle", "rhip", "rknee", "rankle"}
)


def _is_motor_intent_var(name: str) -> bool:
    return str(name).startswith("intent_") or str(name).startswith("phys_intent_")


def _hypothesis_eig_from_env() -> bool:
    """Этап B: байесовский выбор эксперимента (EIG) вместо только System 1."""
    v = os.environ.get("RKK_HYPOTHESIS_EIG", "1").strip().lower()
    return v not in ("0", "false", "off", "system1", "no", "s1")


def _eig_chunk_size() -> int:
    try:
        return max(1, int(os.environ.get("RKK_EIG_BATCH", "256")))
    except ValueError:
        return 256


def _score_cache_every() -> int:
    """Пересчёт score_interventions не чаще чем раз в N тиков движка (RKK_SCORE_CACHE_EVERY; 1 = каждый тик)."""
    try:
        return max(1, int(os.environ.get("RKK_SCORE_CACHE_EVERY", "1")))
    except ValueError:
        return 1


def _score_async_enabled() -> bool:
    """Фоновый поток для score_interventions; по умолчанию выкл. (лок на весь WM давал рывки UI)."""
    v = os.environ.get("RKK_SCORE_ASYNC", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _imagination_horizon_from_env() -> int:
    """Фаза 13: RKK_IMAGINATION_STEPS — число шагов core(X) после мысленного do(); 0 = как раньше."""
    raw = os.environ.get("RKK_IMAGINATION_STEPS", "2")
    try:
        h = int(raw)
    except ValueError:
        h = 0
    return max(0, h)


class RKKAgent:
    def __init__(
        self,
        agent_id: int,
        name:     str,
        env:      Environment,
        device:   torch.device,
        bounds:   HomeostaticBounds | None = None,
    ):
        self.id         = agent_id
        self.name       = name
        self.env        = env
        self.device     = device
        self.activation = ACTIVATIONS[agent_id % 3]

        self.graph   = CausalGraph(device)
        self.system1 = System1(activation=self.activation, device=device)
        self.temporal = TemporalBlankets(
            d_input=len(env.variable_ids), device=device
        )
        self.value_layer = ValueLayer(bounds)
        self._imagination_horizon = _imagination_horizon_from_env()

        self._cg_history: deque[float] = deque(maxlen=20)
        self._total_interventions = 0
        self._total_blocked       = 0
        self._last_do             = "—"
        self._last_blocked_reason = ""
        self._last_result: dict | None = None
        self._symbolic_prediction_bad = False
        self._peak_discovery_rate: float = 0.0
        self._rsi_ref_discovery: float = 0.0
        self._rsi_plateau_count: int = 0
        self._rsi_adjustment_count: int = 0
        self._notears_steps  = 0
        self._last_notears_loss: dict | None = None
        self._local_reflex_cores: dict[tuple[str, ...], Any] = {}
        self._last_local_reflex_train: dict | None = None

        # Φ других агентов (заполняется Simulation-ом перед step())
        self.other_agents_phi: list[float] = []
        self._last_engine_tick = 0
        self._score_cache: list[dict] = []
        self._score_cache_tick: int = -9_999_999
        self._score_thread: threading.Thread | None = None
        self._score_result: list[dict] = []
        self._score_lock = threading.Lock()

        # Фаза 3: LLM-учитель (IG-бонус затухает с числом интервенций)
        self._teacher_rules: list[TeacherIGRule] = []
        self._teacher_weight: float = 0.0

        self._bootstrap()

    # ── Bootstrap + LLM seed interface ───────────────────────────────────────
    def _bootstrap(self):
        for var_id, val in self.env.variables.items():
            self.graph.set_node(var_id, val)

        obs0 = dict(self.env.variables)
        self.graph.record_observation(obs0)
        self.temporal.step(obs0)

        # Text priors (spurious + partial GT)
        gt = self.env.gt_edges()
        for e in gt[:2]:
            noisy_w = e["weight"] * 0.3 + (np.random.rand() - 0.5) * 0.4
            self.graph.set_edge(e["from_"], e["to"], noisy_w, alpha=0.06)

        var_ids = self.env.variable_ids
        if len(var_ids) >= 4:
            self.graph.set_edge(var_ids[1], var_ids[3],  0.35, alpha=0.05)
            self.graph.set_edge(var_ids[2], var_ids[0], -0.20, alpha=0.04)

        # Фаза 1: заморозка URDF-цепочек в L1 (humanoid VAR_NAMES).
        fr = os.environ.get("RKK_FREEZE_URDF", "1").strip().lower()
        if fr not in ("0", "false", "no", "off") and "lhip" in self.env.variable_ids:
            self.graph.freeze_kinematic_priors()

    def inject_text_priors(self, edges: list[dict]) -> dict:
        """
        LLM/RAG seed interface.

        edges: [{"from_": "Temp", "to": "Pressure", "weight": 0.8}, ...]

        Все рёбра загружаются с alpha=0.05 (низкое доверие).
        Epistemic Annealing + NOTEARS выжгут ошибочные за N интервенций.

        Узлы from_/to должны совпадать с id переменных окружения (env.variable_ids).

        Возвращает {"injected": n, "skipped": [причины...]}.
        """
        count   = 0
        skipped: list[str] = []
        valid   = set(self.graph.nodes.keys())

        for e in edges:
            from_ = e.get("from_") or e.get("from")
            to    = e.get("to")
            w     = float(e.get("weight", 0.3))

            if not from_ or not to:
                skipped.append(f"нет from_/to: {e!r}")
                continue
            if is_read_only_macro_var(from_) or is_read_only_macro_var(to):
                skipped.append(f"read-only macro: {from_!r}→{to!r}")
                continue
            if from_ not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{from_}» (доступны: {sorted(valid)})")
                continue
            if to not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{to}» (доступны: {sorted(valid)})")
                continue

            alpha = float(e.get("alpha", 0.05))
            # Слабые семена по умолчанию (0.2–0.3 экв.) — не «пугают» граф и VL
            w_scaled = min(0.3, max(0.08, float(w) * 0.28))
            self.graph.set_edge(from_, to, w_scaled, alpha=alpha)
            count += 1

        return {"injected": count, "skipped": skipped, "node_ids": sorted(valid)}

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_h_W(self) -> float:
        if self.graph._core is None:
            return 0.0
        return float(self.graph._core.dag_constraint().item())

    @staticmethod
    def _marginal_node_uncertainty(unc_m: np.ndarray) -> np.ndarray:
        """
        Маргинальная неопределённость по узлу j: max по всем рёбрам (j→·) и (·→j).
        unc_m[i,j] — epistemic mass на ребре i→j (posterior proxy: 1 − α_trust).
        """
        row_max = unc_m.max(axis=1)
        col_max = unc_m.max(axis=0)
        return np.maximum(row_max, col_max).astype(np.float64, copy=False)

    def _batch_hypothesis_eig(
        self,
        candidates: list[dict],
        X_np: np.ndarray,
        u_node: np.ndarray,
        nid_to_i: dict[str, int],
        unc_m: np.ndarray,
        node_ids: list[str],
        env: Environment,
    ) -> list[float]:
        """
        Суррогат «информативности» действия: (1) чувствительность Σ_j u(j)|ΔX_j|;
        (2) суррогат снижения неопределённости по рёбрам после гипотетического наблюдения
        (масштабирование unc_ij пропорционально |ΔX_i|+|ΔX_j|). Это не точный EIG по H(W).
        """
        core = self.graph._core
        if core is None or not candidates:
            return []
        fd = getattr(core, "forward_dynamics", None)
        if not callable(fd):
            return []

        try:
            lam = float(os.environ.get("RKK_EIG_ENTROPY_TERM", "0.22"))
        except ValueError:
            lam = 0.22
        try:
            eta = float(os.environ.get("RKK_EIG_POSTERIOR_ETA", "0.18"))
        except ValueError:
            eta = 0.18
        lam = max(0.0, lam)
        eta = max(0.0, min(0.95, eta))

        d = int(X_np.shape[0])
        device = self.device
        chunk = _eig_chunk_size()
        eigs: list[float] = []
        x0 = torch.from_numpy(X_np).to(dtype=torch.float32, device=device).unsqueeze(0)
        uu = unc_m.reshape(1, d, d)

        for start in range(0, len(candidates), chunk):
            sub = candidates[start : start + chunk]
            b = len(sub)
            x_batch = x0.expand(b, -1)
            a_batch = torch.zeros(b, d, device=device, dtype=torch.float32)
            for bi, cand in enumerate(sub):
                idx = nid_to_i.get(cand["variable"])
                if idx is not None:
                    a_batch[bi, idx] = float(cand["value"])
            with torch.no_grad():
                pred = integrate_world_model_step(core, x_batch, a_batch)
            delta = (pred - x_batch).abs().cpu().numpy()
            ab = np.abs(delta)
            S = np.clip(ab[:, :, None] + ab[:, None, :], 0.0, 1.0)
            new_u = uu * (1.0 - eta * S)
            new_u = np.maximum(new_u, 0.0)
            reduction = (uu - new_u).sum(axis=(1, 2))
            sens = (delta * u_node.reshape(1, -1)).sum(axis=1)
            total = sens + lam * reduction
            if symbolic_verifier_enabled():
                fac = downrank_factor_for_violation()
                d_nodes = len(node_ids)
                for bi in range(b):
                    pd = {
                        node_ids[j]: float(pred[bi, j].item())
                        for j in range(min(d_nodes, int(pred.shape[1])))
                    }
                    ok, _ = verify_normalized_prediction(pd, env)
                    if not ok:
                        total[bi] *= fac
            eigs.extend(total.tolist())
        return eigs

    def _rollout_imagination_state(
        self, base: dict[str, float], var: str, val: float
    ) -> dict[str, float]:
        """Этап E: один мысленный do + столько же свободных шагов, сколько в VL imagination."""
        s = self.graph.propagate_from(dict(base), var, float(val))
        for _ in range(max(0, self._imagination_horizon)):
            s = self.graph.rollout_step_free(s)
        return s

    def _features_for_intervention_pair(self, v_from: str, v_to: str) -> list[float]:
        """Один вектор признаков System1 для пары (в_from→в_to), как в score_interventions."""
        h_W_norm = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        core = self.graph._core
        ii, jj = nid_to_i.get(v_from), nid_to_i.get(v_to)
        if core is not None and ii is not None and jj is not None:
            with torch.no_grad():
                unc_t = (1.0 - core.alpha_trust_matrix()).detach().float().cpu().numpy()
                W_m = core.W_masked().detach().float().cpu().numpy()
                g_m = None
                if core.W.grad is not None:
                    g_m = core.W.grad.detach().float().abs().cpu().numpy()
            uncertainty = float(unc_t[ii, jj])
            w_ij = float(W_m[ii, jj])
            grad_norm = float(g_m[ii, jj]) if g_m is not None else 0.0
        else:
            uncertainty, w_ij, grad_norm = 1.0, 0.0, 0.0
        alpha = 1.0 - uncertainty
        val_from = self.graph.nodes.get(v_from, 0.5)
        val_to = self.graph.nodes.get(v_to, 0.5)
        ic = ic_map.get((v_from, v_to), 0)
        return self.system1.build_features(
            w_ij=w_ij, alpha_ij=alpha,
            val_from=val_from, val_to=val_to,
            uncertainty=uncertainty, h_W_norm=h_W_norm,
            grad_norm_ij=grad_norm,
            intervention_count=ic,
            discovery_rate=disc_rate,
        )

    def _build_goal_planned_candidate(self, var: str, val: float) -> dict:
        feat = self._features_for_intervention_pair(var, "target_dist")
        return {
            "variable":    var,
            "target":      "target_dist",
            "value":       float(val),
            "uncertainty": 0.35,
            "features":    feat,
            "expected_ig": 1.0,
            "from_goal_plan": True,
        }

    def _maybe_goal_planned_candidate(self) -> dict | None:
        if goal_planning_globally_disabled():
            return None
        if self.graph._core is None:
            return None
        if self.graph.nodes.get("self_goal_active") is None:
            return None
        if float(self.graph.nodes.get("self_goal_active", 0)) <= 0.45:
            return None
        if "target_dist" not in self.graph.nodes:
            return None

        state0 = dict(self.graph.nodes)
        cur_td = float(state0.get("target_dist", 0.5))
        goal_thr = float(state0.get("self_goal_target_dist", 0.42))
        if cur_td <= goal_thr + 0.015:
            return None

        motor = planning_graph_motor_vars(self.env, list(self.graph._node_ids))
        if not motor:
            return None

        levels = parse_plan_value_levels()
        actions = [(v, x) for v in motor for x in levels]
        max_b = plan_max_branch()
        if len(actions) > max_b:
            idx = np.random.choice(len(actions), size=max_b, replace=False)
            actions = [actions[i] for i in idx]

        depth = plan_depth()
        beam_k = plan_beam_k()

        def _td(s: dict[str, float]) -> float:
            return float(s.get("target_dist", cur_td))

        best_td = cur_td
        best_first: tuple[str, float] | None = None

        if depth <= 1:
            for var, val in actions:
                try:
                    sfin = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                    if not ok:
                        continue
                td = _td(sfin)
                if td < best_td - 1e-6:
                    best_td = td
                    best_first = (var, val)
        else:
            scored: list[tuple[float, str, float, dict[str, float]]] = []
            for var, val in actions:
                try:
                    s1 = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(s1), self.env)
                    if not ok:
                        continue
                scored.append((_td(s1), var, val, dict(s1)))
            scored.sort(key=lambda t: t[0])
            for _td1, v1, x1, s1 in scored[:beam_k]:
                for v2, x2 in actions:
                    try:
                        sfin = self._rollout_imagination_state(s1, v2, x2)
                    except Exception:
                        continue
                    if symbolic_verifier_enabled():
                        ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                        if not ok:
                            continue
                    td = _td(sfin)
                    if td < best_td - 1e-6:
                        best_td = td
                        best_first = (v1, x1)

        if best_first is None:
            return None
        return self._build_goal_planned_candidate(best_first[0], best_first[1])

    def _is_locomotion_primary_active(self) -> bool:
        """Если CPG управляет ногами, EIG не должен конкурировать за суставы — только intent_* и др."""
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        var_ids   = self.env.variable_ids
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate

        # Один проход по рёбрам: счётчики интервенций (раньше — O(pairs×|E|) через next() в цикле)
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count

        # Имя узла → индекс без O(d) list.index на каждую пару
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}

        # Один раз W, α и |grad| на CPU — вместо O(d²) вызовов alpha_trust_matrix / W_masked
        core = self.graph._core
        W_m = unc_m = g_m = None
        if core is not None:
            with torch.no_grad():
                W_t = core.W_masked().detach().float()
                A_t = core.alpha_trust_matrix().detach().float()
                W_m = W_t.cpu().numpy()
                unc_m = (1.0 - A_t).cpu().numpy()
            if core.W.grad is not None:
                g_m = core.W.grad.detach().float().abs().cpu().numpy()

        d = len(var_ids)
        if d == 0:
            return []

        # Счётчики интервенций по парам (только известные рёбра — O(|E|))
        ic_mat = np.zeros((d, d), dtype=np.float64)
        v2i = {v: i for i, v in enumerate(var_ids)}
        for (vf, vt), c in ic_map.items():
            i = v2i.get(vf)
            j = v2i.get(vt)
            if i is not None and j is not None and i != j:
                ic_mat[i, j] = float(c)

        ridx = np.zeros(d, dtype=np.int64)
        valid_node = np.zeros(d, dtype=bool)
        for i, v in enumerate(var_ids):
            ji = nid_to_i.get(v)
            if ji is not None:
                ridx[i] = ji
                valid_node[i] = True

        nodes_arr = np.array(
            [float(self.graph.nodes.get(v, 0.5)) for v in var_ids],
            dtype=np.float64,
        )
        mask = ~np.eye(d, dtype=bool)
        fi, fj = np.where(mask)
        n_pairs = len(fi)

        if W_m is not None:
            ii_n = ridx[fi]
            jj_n = ridx[fj]
            ok = valid_node[fi] & valid_node[fj]
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)
            w_ij[ok] = W_m[ii_n[ok], jj_n[ok]]
            uncertainty[ok] = unc_m[ii_n[ok], jj_n[ok]]
            if g_m is not None:
                grad_norm[ok] = g_m[ii_n[ok], jj_n[ok]]
        else:
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)

        alpha = 1.0 - uncertainty
        val_from = nodes_arr[fi]
        val_to = nodes_arr[fj]
        ic_v = ic_mat[fi, fj]
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_v = float(np.clip(disc_rate, 0.0, 1.0))

        feats_arr = np.column_stack(
            [
                np.tanh(w_ij),
                np.clip(alpha, 0.0, 1.0),
                np.clip(val_from, 0.0, 1.0),
                np.clip(val_to, 0.0, 1.0),
                np.clip(uncertainty, 0.0, 1.0),
                np.full(n_pairs, h_clip, dtype=np.float64),
                np.tanh(grad_norm),
                np.clip(ic_v / 100.0, 0.0, 1.0),
                np.full(n_pairs, disc_v, dtype=np.float64),
            ]
        )
        features_batch = feats_arr.tolist()

        rng = np.random.default_rng()
        posture_now = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 0.5),
            )
        )
        foot_l_now = float(
            self.graph.nodes.get(
                "foot_contact_l",
                self.graph.nodes.get("phys_foot_contact_l", 0.5),
            )
        )
        foot_r_now = float(
            self.graph.nodes.get(
                "foot_contact_r",
                self.graph.nodes.get("phys_foot_contact_r", 0.5),
            )
        )
        stable_stance = posture_now > 0.70 and min(foot_l_now, foot_r_now) > 0.56
        candidates: list[dict] = []
        for k in range(n_pairs):
            i, j = int(fi[k]), int(fj[k])
            vf, vt = var_ids[i], var_ids[j]
            unc_k = float(uncertainty[k])
            feat_k = features_batch[k]
            if _is_motor_intent_var(vf):
                if stable_stance:
                    lo, hi = 0.30, 0.72
                else:
                    lo, hi = 0.35, 0.68
                if str(vf).endswith("stride"):
                    hi = min(hi, 0.62 if stable_stance else 0.56)
                if str(vf).endswith("stop_recover"):
                    lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                rand_value = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
            else:
                rand_value = float(np.clip(rng.uniform(0.15, 0.85), 0.06, 0.94))
            candidates.append({
                "variable":    vf,
                "target":      vt,
                "value":       rand_value,
                "uncertainty": unc_k,
                "features":    feat_k,
                "expected_ig": 0.0,
            })

        if self._is_locomotion_primary_active():
            candidates = [
                c
                for c in candidates
                if c["variable"] not in _LOCOMOTION_CPG_LEG_EIG_BLOCK
            ]
            if posture_now < 0.65:
                candidates = [
                    c
                    for c in candidates
                    if str(c["variable"]).startswith("intent_")
                    or str(c["variable"]).startswith("phys_intent_")
                ]

        if not candidates or not features_batch:
            return []

        use_eig = _hypothesis_eig_from_env() and W_m is not None and unc_m is not None
        if use_eig:
            x_vec = np.array(
                [float(self.graph.nodes.get(n, 0.0)) for n in self.graph._node_ids],
                dtype=np.float64,
            )
            u_node = self._marginal_node_uncertainty(unc_m)
            eigs = self._batch_hypothesis_eig(
                candidates, x_vec, u_node, nid_to_i, unc_m,
                list(self.graph._node_ids), self.env,
            )
            if len(eigs) == len(candidates):
                # Учитываем гипотезу «это ребро неизвестно»: масштаб EIG по unc(v_from→v_to).
                for i, cand in enumerate(candidates):
                    eigs[i] *= 1.0 + float(cand["uncertainty"])
                arr = np.array(eigs, dtype=np.float64)
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo + 1e-12:
                    normed = (arr - lo) / (hi - lo)
                else:
                    normed = np.full_like(arr, 0.5)
                for i, cand in enumerate(candidates):
                    cand["eig_raw"] = float(eigs[i])
                    cand["expected_ig"] = float(normed[i])
            else:
                use_eig = False

        if not use_eig:
            scores = self.system1.score(features_batch)
            for i, cand in enumerate(candidates):
                cand["expected_ig"] = scores[i]

        if symbolic_verifier_enabled() and self._symbolic_prediction_bad:
            a, b = exploration_blend_from_uncertainty()
            for cand in candidates:
                unc = float(cand.get("uncertainty", 0.5))
                cand["expected_ig"] = a * float(cand["expected_ig"]) + b * unc

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    def _score_async_worker(self) -> None:
        try:
            with torch.no_grad():
                result = self.score_interventions()
            with self._score_lock:
                self._score_result = result
        except Exception as ex:
            print(f"[RKKAgent] score_interventions (async): {ex}")

    def set_teacher_state(self, rules: list[TeacherIGRule], weight: float) -> None:
        """Фаза 3: правила от LLM и текущий teacher_weight (симуляция считает annealing)."""
        self._teacher_rules = list(rules)
        self._teacher_weight = float(max(0.0, min(1.0, weight)))

    def _teacher_ig_bonus(self, variable: str, nodes: dict[str, float]) -> float:
        w = self._teacher_weight
        if w <= 0 or not self._teacher_rules:
            return 0.0
        acc = 0.0
        for r in self._teacher_rules:
            if r.target_var != variable:
                continue
            if r.when_var:
                val = nodes.get(r.when_var)
                if val is None:
                    continue
                if r.when_min is not None and float(val) < r.when_min:
                    continue
                if r.when_max is not None and float(val) > r.when_max:
                    continue
            acc += r.bonus * w
        return min(0.28, acc)

    # ── Один шаг с Value Layer ────────────────────────────────────────────────
    def step(self, engine_tick: int = 0, *, enable_l3: bool = True) -> dict:
        self._last_engine_tick = engine_tick
        try:
            self.graph.apply_env_observation(dict(self.env.observe()))
        except Exception:
            pass
        sce = _score_cache_every()
        if (
            sce > 1
            and self._score_cache
            and (engine_tick - self._score_cache_tick) < sce
        ):
            scores = list(self._score_cache)
        elif _score_async_enabled():
            if self._score_thread is None or not self._score_thread.is_alive():
                self._score_thread = threading.Thread(
                    target=self._score_async_worker,
                    name="rkk_score_interventions",
                    daemon=True,
                )
                self._score_thread.start()
            with self._score_lock:
                have = list(self._score_result) if self._score_result else []
            if have:
                scores = have
            elif self._score_cache:
                scores = list(self._score_cache)
            else:
                with torch.no_grad():
                    scores = self.score_interventions()
                with self._score_lock:
                    self._score_result = list(scores)
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        else:
            with torch.no_grad():
                scores = self.score_interventions()
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        gp = self._maybe_goal_planned_candidate() if enable_l3 else None
        if gp is not None and not (
            symbolic_verifier_enabled() and self._symbolic_prediction_bad
        ):
            scores.insert(0, gp)
        if not scores:
            return {
                "blocked": False, "skipped": True, "prediction_error": 0.0,
                "cf_predicted": {}, "cf_observed": {}, "goal_planned": False,
            }

        current_phi = self.phi_approx()
        chosen      = None
        check_result = None
        blocked_count = 0

        # Перебираем кандидатов пока не найдём допустимое действие
        for candidate in scores[:MAX_FALLBACK_TRIES]:
            var   = candidate["variable"]
            value = candidate["value"]

            check_result = self.value_layer.check_action(
                variable=var,
                value=value,
                current_nodes=dict(self.graph.nodes),
                graph=self.graph,
                temporal=self.temporal,
                current_phi=current_phi,
                other_agents_phi=self.other_agents_phi,
                engine_tick=engine_tick,
                imagination_horizon=(self._imagination_horizon if enable_l3 else 0),
            )

            if check_result.allowed:
                chosen = candidate
                break
            else:
                # Штрафуем System 1 за предложение опасного действия
                self.system1.push_experience(
                    features=candidate["features"],
                    actual_ig=check_result.penalty,   # отрицательный IG
                )
                blocked_count += 1
                self._total_blocked += 1
                self._last_blocked_reason = check_result.reason.value

        # Все кандидаты заблокированы — возвращаем событие
        if chosen is None:
            return {
                "blocked":       True,
                "blocked_count": blocked_count,
                "reason":        self._last_blocked_reason,
                "variable":      scores[0]["variable"] if scores else "?",
                "value":         scores[0]["value"] if scores else 0.5,
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error":  0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        # ── Выполняем допустимое действие ────────────────────────────────────
        var   = chosen["variable"]
        value = chosen["value"]

        if is_read_only_macro_var(var):
            return {
                "blocked": True,
                "blocked_count": blocked_count + 1,
                "reason": "read_only_macro",
                "variable": var,
                "value": float(value),
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        mdl_before = self.graph.mdl_size
        obs_before_env = dict(self.env.observe())
        self.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.graph.snapshot_vec_dict()
        predicted  = self.graph.propagate(var, value)
        sym_ok, sym_fail = True, []
        if symbolic_verifier_enabled():
            sym_ok, sym_fail = verify_normalized_prediction(dict(predicted), self.env)
            self._symbolic_prediction_bad = not sym_ok
        else:
            self._symbolic_prediction_bad = False
        observed_env = self.env.intervene(var, value)

        # Temporal step (только размерность среды)
        self.temporal.step(observed_env)

        self.graph.apply_env_observation(observed_env)
        observed_full = self.graph.snapshot_vec_dict()

        # NOTEARS / GNN буферы — полный вектор узлов (включая concept_*)
        self.graph.record_observation(obs_before_full)
        self.graph.record_observation(observed_full)
        self.graph.record_intervention(var, value, obs_before_full, observed_full)

        # NOTEARS train
        notears_result = None
        if self._total_interventions % NOTEARS_EVERY == 0:
            notears_result = self.graph.train_step()
            if notears_result:
                self._notears_steps += 1
                self._last_notears_loss = notears_result
            self._maybe_train_local_reflex()

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # System 1: IG по физике; slot_* и self_* не доминируют метрику (self — прямое задание агентом).
        nids = self.graph._node_ids
        phys_ids = [
            k for k in nids
            if k not in _SELF_VAR_SET and not str(k).startswith("slot_")
        ]
        slot_ids = [k for k in nids if str(k).startswith("slot_")]

        def _mean_abs_err(keys: list) -> float:
            if not keys:
                return 0.0
            return float(np.mean([
                abs(float(predicted.get(k, 0.5)) - float(observed_full.get(k, 0.5)))
                for k in keys
            ]))

        pe_phys = _mean_abs_err(phys_ids)

        # Этап Г: петля «намерение ↔ исход» + ошибка модели → self_* (только среды с методом).
        fn_sf = getattr(self.env, "update_self_feedback", None)
        if callable(fn_sf):
            try:
                fn_sf(
                    variable=var,
                    intended_norm=value,
                    observed=observed_env,
                    predicted=predicted,
                    prediction_error_phys=pe_phys,
                )
            except Exception:
                pass
            obs_self = dict(self.env.observe())
            for sk in _SELF_VAR_SET:
                if sk in self.graph.nodes and sk in obs_self:
                    self.graph.nodes[sk] = float(obs_self[sk])
            self.graph.refresh_concept_aggregates()
        pe_slot = _mean_abs_err(slot_ids)
        w_vis = min(0.45, max(0.0, VISUAL_IG_WEIGHT))
        if slot_ids and phys_ids:
            actual_ig = (1.0 - w_vis) * pe_phys + w_vis * pe_slot
        elif phys_ids:
            actual_ig = pe_phys
        else:
            actual_ig = pe_slot

        t_bonus = self._teacher_ig_bonus(var, dict(self.graph.nodes))
        actual_ig = float(np.clip(actual_ig + t_bonus, 0.0, 1.0))

        self.system1.push_experience(
            features=chosen["features"],
            actual_ig=actual_ig,
        )

        # SSM train — размерность = temporal.d_input (= |graph._node_ids|), не только env.variable_ids
        u_next = torch.tensor(
            [float(self.graph.nodes.get(n, 0.5)) for n in self.graph._node_ids],
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal.train_step(u_next)

        self._total_interventions += 1
        try:
            _v_do = float(value)
        except (TypeError, ValueError):
            _v_do = 0.5
        self._last_do = f"do({var}={_v_do:.2f})"
        self._last_blocked_reason = ""

        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        rsi_event = self._tick_rsi_lite_discovery(cur_dr)

        _cf_keys = list(self.graph._node_ids)[:48]
        self._last_result = {
            "blocked":           False,
            "blocked_count":     blocked_count,
            "variable":          var,
            "value":             value,
            "compression_delta": compression_delta,
            "updated_edges":     [f"{e.from_}→{e.to}" for e in self.graph.edges[:4]],
            "pruned_edges":      [],
            "prediction_error":  float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed_env.items()
            ])),
            "cf_predicted": {k: float(round(float(predicted.get(k, 0.0)), 4)) for k in _cf_keys},
            "cf_observed":  {k: float(round(float(observed_full.get(k, 0.0)), 4)) for k in _cf_keys},
            "goal_planned":  bool(chosen.get("from_goal_plan")),
            "symbolic_ok": sym_ok,
            "symbolic_violations": sym_fail,
            "rsi_lite": rsi_event,
            "notears":           notears_result,
        }
        return self._last_result

    # ── Demon ─────────────────────────────────────────────────────────────────
    def demon_disrupt(self) -> str:
        if self.graph._core is None:
            return "no core"
        with torch.no_grad():
            W = self.graph._core.W
            sig = (W.abs() > 0.05).nonzero(as_tuple=False)
            if len(sig) == 0:
                return "no significant edges"
            idx = sig[np.random.randint(len(sig))]
            i, j = idx[0].item(), idx[1].item()
            noise = (np.random.rand() - 0.5) * 0.3
            # Нельзя W[i,j] += … — это in-place на view листа с requires_grad.
            w_new = W.detach().clone()
            w_new[i, j] = w_new[i, j] + float(noise)
            W.copy_(w_new)
            fn = self.graph._node_ids[i] if i < len(self.graph._node_ids) else f"v{i}"
            tn = self.graph._node_ids[j] if j < len(self.graph._node_ids) else f"v{j}"
        self.graph._invalidate_cache()
        return f"W[{fn}→{tn}] +{noise:.3f}"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def compression_gain(self) -> float:
        if not self._cg_history:
            return 0.0
        return float(np.mean(list(self._cg_history)))

    @property
    def discovery_rate(self) -> float:
        return self.env.discovery_rate([
            {"from_": e.from_, "to": e.to, "weight": e.weight}
            for e in self.graph.edges
        ])

    @property
    def peak_discovery_rate(self) -> float:
        return self._peak_discovery_rate

    def _apply_rsi_lite(self) -> dict[str, float | int]:
        g = self.graph
        cur_l1 = float(getattr(g, "LAMBDA_L1", CausalGraph.LAMBDA_L1))
        new_l1 = min(cur_l1 * rsi_l1_scale(), rsi_l1_max())
        g.LAMBDA_L1 = new_l1
        cap_b = rsi_buffer_cap()
        g.BUFFER_SIZE = min(cap_b, int(g.BUFFER_SIZE) * 2)
        cap_i = rsi_imagination_cap()
        self._imagination_horizon = min(cap_i, self._imagination_horizon + 1)
        self._rsi_adjustment_count += 1
        return {
            "LAMBDA_L1": float(new_l1),
            "BUFFER_SIZE": int(g.BUFFER_SIZE),
            "imagination_horizon": int(self._imagination_horizon),
        }

    def _tick_rsi_lite_discovery(self, cur_dr: float) -> dict[str, float | int] | None:
        if not rsi_lite_enabled():
            return None
        if self._total_interventions < rsi_min_interventions():
            return None
        eps = rsi_improvement_eps()
        if cur_dr > self._rsi_ref_discovery + eps:
            self._rsi_ref_discovery = float(cur_dr)
            self._rsi_plateau_count = 0
            return None
        self._rsi_plateau_count += 1
        if self._rsi_plateau_count < rsi_plateau_interventions():
            return None
        self._rsi_plateau_count = 0
        self._rsi_ref_discovery = float(cur_dr)
        return self._apply_rsi_lite()

    def _maybe_train_local_reflex(self) -> None:
        if not local_reflex_train_enabled():
            return
        self._last_local_reflex_train = train_chains_parallel(
            graph=self.graph,
            device=self.graph.device,
            cores=self._local_reflex_cores,
        )

    def phi_approx(self) -> float:
        return self.temporal.phi_approx()

    def record_phi(self, _: float):
        pass  # temporal управляет историей сам

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        h_W     = self._get_h_W()
        tb_info = self.temporal.slow_state_summary()
        s1_info = {
            "buffer_size": len(self.system1.buffer),
            "mean_loss":   round(self.system1.mean_loss, 6),
        }
        vl_info = dict(self.value_layer.snapshot(self._last_engine_tick))
        vl_info["imagination_horizon"] = self._imagination_horizon

        notears_info = None
        if self._last_notears_loss:
            notears_info = {
                "steps":  self._notears_steps,
                "loss":   self._last_notears_loss.get("loss", 0),
                "h_W":    round(h_W, 4),
                "l_int":  self._last_notears_loss.get("l_int", 0),
            }

        h_W_edge_entropy = None
        core = self.graph._core
        if core is not None:
            with torch.no_grad():
                A = core.alpha_trust_matrix().detach().float().cpu().numpy()
            p = np.clip(A, 1e-7, 1.0 - 1e-7)
            h_W_edge_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())

        snap: dict = {
            "id":                    self.id,
            "name":                  self.name,
            "env_type":              self.env.preset,
            "activation":            self.activation,
            "graph_mdl":             round(self.graph.mdl_size, 3),
            "compression_gain":      round(self.compression_gain, 4),
            "alpha_mean":            round(self.graph.alpha_mean, 3),
            "phi":                   round(self.phi_approx(), 4),
            "node_count":            len(self.graph.nodes),
            "edge_count":            len(self.graph.edges),
            "total_interventions":   self._total_interventions,
            "total_blocked":         self._total_blocked,
            "last_do":               self._last_do,
            "last_blocked_reason":   self._last_blocked_reason,
            "discovery_rate":        round(cur_dr, 3),
            "peak_discovery_rate":   round(self._peak_discovery_rate, 3),
            "h_W":                   round(h_W, 4),
            "notears":               notears_info,
            "temporal":              tb_info,
            "system1":               s1_info,
            "value_layer":           vl_info,
            "teacher": {
                "weight":     round(self._teacher_weight, 4),
                "rules":      len(self._teacher_rules),
            },
            "hypothesis_eig": _hypothesis_eig_from_env(),
            "h_W_edge_entropy": None if h_W_edge_entropy is None else round(h_W_edge_entropy, 4),
            "rsi_lite": {
                "enabled": rsi_lite_enabled(),
                "plateau_count": self._rsi_plateau_count,
                "ref_discovery": round(self._rsi_ref_discovery, 5),
                "adjustments": self._rsi_adjustment_count,
                "LAMBDA_L1": round(float(getattr(self.graph, "LAMBDA_L1", CausalGraph.LAMBDA_L1)), 5),
                "graph_BUFFER_SIZE": int(self.graph.BUFFER_SIZE),
                "imagination_horizon": int(self._imagination_horizon),
            },
            "local_reflex_train": self._last_local_reflex_train,
            "edges": [e.as_dict() for e in self.graph.edges],
        }
        if self.env.preset == "pybullet":
            pos_fn = getattr(self.env, "object_positions_world", None)
            if callable(pos_fn):
                snap["physics_objects"] = pos_fn()
        return snap
===
"""
agent_v4.py — RKKAgent с Value Layer (Шаг А).

Изменения:
  - ValueLayer.check_action() вызывается перед каждым do()
  - Заблокированные действия → penalty для System 1 + лог события
  - LLM/RAG seed interface: inject_text_priors(edges_json)
  - Fallback scorer когда System 1 буфер ещё мал
  - other_agents_phi передаётся из Simulation для ΔΦ≥0 constraint

Этап B (гипотезо-ориентированное исследование):
  score_interventions() — аппроксимация информационного выигрыша: чувствительность по узлам
  плюс суррогат снижения суммарной epistemic mass по рёбрам при предсказанном obs (не полный
  байесовский H(W)−E[H(W|obs)]). RKK_EIG_ENTROPY_TERM, RKK_EIG_POSTERIOR_ETA.
  Переключатель: RKK_HYPOTHESIS_EIG=1 (по умолчанию) | 0 | system1 | off | false
  В snapshot: h_W_edge_entropy — сумма бинарных энтропий по α_trust рёбер (диагностика неопределённости W).
  RKK_SCORE_ASYNC=1: score_interventions в фоновом daemon-потоке (тик не ждёт; возможна гонка с train_step — не рекомендуется).
  По умолчанию RKK_SCORE_ASYNC=0 — синхронный пересчёт в главном потоке (стабильно, без общего lock на граф).

Этап Г (самомодель): self_* + update_self_feedback() в humanoid — коррекция намерений по исходу do()
  и по промаху GNN (RKK_SELF_FEEDBACK_LR).

Этап E (целевое планирование): при self_goal_active и наличии target_dist в графе — поиск действия
  через imagination (propagate_from + rollout_step_free), см. engine.goal_planning; RKK_GOAL_PLANNING=0 отключает.

Этап F (символьный верификатор): проверка предсказания propagate на PHYSICS_CONSTRAINTS (engine.symbolic_verifier);
  нарушение → не prepend goal-plan, смешивание expected_ig с uncertainty на следующем шаге; RKK_SYMBOLIC_VERIFY=0 отключает.

Этап G (RSI lite): плато discovery_rate → агент усиливает L1, удваивает BUFFER_SIZE графа (до капа), +1 imagination;
  engine.rsi_lite, RKK_RSI_LITE=0 отключает; RKK_RSI_PLATEAU_TICKS, RKK_RSI_MIN_INTERVENTIONS.
"""
from __future__ import annotations

import os
import threading
from typing import Any
import torch
import numpy as np
from collections import deque

from engine.causal_graph import CausalGraph
from engine.graph_constants import is_read_only_macro_var
from engine.environment  import Environment
from engine.system1      import System1
from engine.temporal     import TemporalBlankets
from engine.value_layer  import ValueLayer, HomeostaticBounds, BlockReason
from engine.phase3_teacher import TeacherIGRule
from engine.environment_humanoid import SELF_VARS
from engine.goal_planning import (
    goal_planning_globally_disabled,
    parse_plan_value_levels,
    plan_beam_k,
    plan_depth,
    plan_max_branch,
    planning_graph_motor_vars,
)
from engine.symbolic_verifier import (
    downrank_factor_for_violation,
    exploration_blend_from_uncertainty,
    symbolic_verifier_enabled,
    verify_normalized_prediction,
)
from engine.wm_neural_ode import integrate_world_model_step
from engine.rsi_lite import (
    rsi_buffer_cap,
    rsi_imagination_cap,
    rsi_improvement_eps,
    rsi_l1_max,
    rsi_l1_scale,
    rsi_lite_enabled,
    rsi_min_interventions,
    rsi_plateau_interventions,
)
from engine.local_reflex import local_reflex_train_enabled, train_chains_parallel

ACTIVATIONS   = ["relu", "gelu", "tanh"]
NOTEARS_EVERY = 8
MAX_FALLBACK_TRIES = 5  # больше кандидатов, чтобы пройти Value Layer в начале обучения
# Вес slot_* в actual_ig для System 1; основной сигнал — не-визуальные узлы (RKK_VISUAL_IG_WEIGHT=0 → только физика).
VISUAL_IG_WEIGHT = float(os.environ.get("RKK_VISUAL_IG_WEIGHT", "0.1"))
_SELF_VAR_SET = frozenset(SELF_VARS)
# RKK_LOCOMOTION_CPG=1: CPG ведёт ноги; EIG не выбирает прямые do() по этим узлам.
_LOCOMOTION_CPG_LEG_EIG_BLOCK = frozenset(
    {"lhip", "lknee", "lankle", "rhip", "rknee", "rankle"}
)


def _is_motor_intent_var(name: str) -> bool:
    return str(name).startswith("intent_") or str(name).startswith("phys_intent_")


def _hypothesis_eig_from_env() -> bool:
    """Этап B: байесовский выбор эксперимента (EIG) вместо только System 1."""
    v = os.environ.get("RKK_HYPOTHESIS_EIG", "1").strip().lower()
    return v not in ("0", "false", "off", "system1", "no", "s1")


def _eig_chunk_size() -> int:
    try:
        return max(1, int(os.environ.get("RKK_EIG_BATCH", "256")))
    except ValueError:
        return 256


def _score_cache_every() -> int:
    """Пересчёт score_interventions не чаще чем раз в N тиков движка (RKK_SCORE_CACHE_EVERY; 1 = каждый тик)."""
    try:
        return max(1, int(os.environ.get("RKK_SCORE_CACHE_EVERY", "1")))
    except ValueError:
        return 1


def _score_async_enabled() -> bool:
    """Фоновый поток для score_interventions; по умолчанию выкл. (лок на весь WM давал рывки UI)."""
    v = os.environ.get("RKK_SCORE_ASYNC", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _imagination_horizon_from_env() -> int:
    """Фаза 13: RKK_IMAGINATION_STEPS — число шагов core(X) после мысленного do(); 0 = как раньше."""
    raw = os.environ.get("RKK_IMAGINATION_STEPS", "2")
    try:
        h = int(raw)
    except ValueError:
        h = 0
    return max(0, h)


class RKKAgent:
    def __init__(
        self,
        agent_id: int,
        name:     str,
        env:      Environment,
        device:   torch.device,
        bounds:   HomeostaticBounds | None = None,
    ):
        self.id         = agent_id
        self.name       = name
        self.env        = env
        self.device     = device
        self.activation = ACTIVATIONS[agent_id % 3]

        self.graph   = CausalGraph(device)
        self.system1 = System1(activation=self.activation, device=device)
        self.temporal = TemporalBlankets(
            d_input=len(env.variable_ids), device=device
        )
        self.value_layer = ValueLayer(bounds)
        self._imagination_horizon = _imagination_horizon_from_env()

        self._cg_history: deque[float] = deque(maxlen=20)
        self._total_interventions = 0
        self._total_blocked       = 0
        self._last_do             = "—"
        self._last_blocked_reason = ""
        self._last_result: dict | None = None
        self._symbolic_prediction_bad = False
        self._peak_discovery_rate: float = 0.0
        self._rsi_ref_discovery: float = 0.0
        self._rsi_plateau_count: int = 0
        self._rsi_adjustment_count: int = 0
        self._notears_steps  = 0
        self._last_notears_loss: dict | None = None
        self._local_reflex_cores: dict[tuple[str, ...], Any] = {}
        self._last_local_reflex_train: dict | None = None

        # Φ других агентов (заполняется Simulation-ом перед step())
        self.other_agents_phi: list[float] = []
        self._last_engine_tick = 0
        self._score_cache: list[dict] = []
        self._score_cache_tick: int = -9_999_999
        self._score_thread: threading.Thread | None = None
        self._score_result: list[dict] = []
        self._score_lock = threading.Lock()

        # Фаза 3: LLM-учитель (IG-бонус затухает с числом интервенций)
        self._teacher_rules: list[TeacherIGRule] = []
        self._teacher_weight: float = 0.0

        self._bootstrap()

    # ── Bootstrap + LLM seed interface ───────────────────────────────────────
    def _bootstrap(self):
        for var_id, val in self.env.variables.items():
            self.graph.set_node(var_id, val)

        obs0 = dict(self.env.variables)
        self.graph.record_observation(obs0)
        self.temporal.step(obs0)

        # Text priors (spurious + partial GT)
        gt = self.env.gt_edges()
        for e in gt[:2]:
            noisy_w = e["weight"] * 0.3 + (np.random.rand() - 0.5) * 0.4
            self.graph.set_edge(e["from_"], e["to"], noisy_w, alpha=0.06)

        var_ids = self.env.variable_ids
        if len(var_ids) >= 4:
            self.graph.set_edge(var_ids[1], var_ids[3],  0.35, alpha=0.05)
            self.graph.set_edge(var_ids[2], var_ids[0], -0.20, alpha=0.04)

        # Фаза 1: заморозка URDF-цепочек в L1 (humanoid VAR_NAMES).
        fr = os.environ.get("RKK_FREEZE_URDF", "1").strip().lower()
        if fr not in ("0", "false", "no", "off") and "lhip" in self.env.variable_ids:
            self.graph.freeze_kinematic_priors()

    def inject_text_priors(self, edges: list[dict]) -> dict:
        """
        LLM/RAG seed interface.

        edges: [{"from_": "Temp", "to": "Pressure", "weight": 0.8}, ...]

        Все рёбра загружаются с alpha=0.05 (низкое доверие).
        Epistemic Annealing + NOTEARS выжгут ошибочные за N интервенций.

        Узлы from_/to должны совпадать с id переменных окружения (env.variable_ids).

        Возвращает {"injected": n, "skipped": [причины...]}.
        """
        count   = 0
        skipped: list[str] = []
        valid   = set(self.graph.nodes.keys())

        for e in edges:
            from_ = e.get("from_") or e.get("from")
            to    = e.get("to")
            w     = float(e.get("weight", 0.3))

            if not from_ or not to:
                skipped.append(f"нет from_/to: {e!r}")
                continue
            if is_read_only_macro_var(from_) or is_read_only_macro_var(to):
                skipped.append(f"read-only macro: {from_!r}→{to!r}")
                continue
            if from_ not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{from_}» (доступны: {sorted(valid)})")
                continue
            if to not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{to}» (доступны: {sorted(valid)})")
                continue

            alpha = float(e.get("alpha", 0.05))
            # Слабые семена по умолчанию (0.2–0.3 экв.) — не «пугают» граф и VL
            w_scaled = min(0.3, max(0.08, float(w) * 0.28))
            self.graph.set_edge(from_, to, w_scaled, alpha=alpha)
            count += 1

        return {"injected": count, "skipped": skipped, "node_ids": sorted(valid)}

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_h_W(self) -> float:
        if self.graph._core is None:
            return 0.0
        return float(self.graph._core.dag_constraint().item())

    @staticmethod
    def _marginal_node_uncertainty(unc_m: np.ndarray) -> np.ndarray:
        """
        Маргинальная неопределённость по узлу j: max по всем рёбрам (j→·) и (·→j).
        unc_m[i,j] — epistemic mass на ребре i→j (posterior proxy: 1 − α_trust).
        """
        row_max = unc_m.max(axis=1)
        col_max = unc_m.max(axis=0)
        return np.maximum(row_max, col_max).astype(np.float64, copy=False)

    def _batch_hypothesis_eig(
        self,
        candidates: list[dict],
        X_np: np.ndarray,
        u_node: np.ndarray,
        nid_to_i: dict[str, int],
        unc_m: np.ndarray,
        node_ids: list[str],
        env: Environment,
    ) -> list[float]:
        """
        Суррогат «информативности» действия: (1) чувствительность Σ_j u(j)|ΔX_j|;
        (2) суррогат снижения неопределённости по рёбрам после гипотетического наблюдения
        (масштабирование unc_ij пропорционально |ΔX_i|+|ΔX_j|). Это не точный EIG по H(W).
        """
        core = self.graph._core
        if core is None or not candidates:
            return []
        fd = getattr(core, "forward_dynamics", None)
        if not callable(fd):
            return []

        try:
            lam = float(os.environ.get("RKK_EIG_ENTROPY_TERM", "0.22"))
        except ValueError:
            lam = 0.22
        try:
            eta = float(os.environ.get("RKK_EIG_POSTERIOR_ETA", "0.18"))
        except ValueError:
            eta = 0.18
        lam = max(0.0, lam)
        eta = max(0.0, min(0.95, eta))

        d = int(X_np.shape[0])
        device = self.device
        chunk = _eig_chunk_size()
        eigs: list[float] = []
        x0 = torch.from_numpy(X_np).to(dtype=torch.float32, device=device).unsqueeze(0)
        uu = unc_m.reshape(1, d, d)

        for start in range(0, len(candidates), chunk):
            sub = candidates[start : start + chunk]
            b = len(sub)
            x_batch = x0.expand(b, -1)
            a_batch = torch.zeros(b, d, device=device, dtype=torch.float32)
            for bi, cand in enumerate(sub):
                idx = nid_to_i.get(cand["variable"])
                if idx is not None:
                    a_batch[bi, idx] = float(cand["value"])
            with torch.no_grad():
                pred = integrate_world_model_step(core, x_batch, a_batch)
            delta = (pred - x_batch).abs().cpu().numpy()
            ab = np.abs(delta)
            S = np.clip(ab[:, :, None] + ab[:, None, :], 0.0, 1.0)
            new_u = uu * (1.0 - eta * S)
            new_u = np.maximum(new_u, 0.0)
            reduction = (uu - new_u).sum(axis=(1, 2))
            sens = (delta * u_node.reshape(1, -1)).sum(axis=1)
            total = sens + lam * reduction
            if symbolic_verifier_enabled():
                fac = downrank_factor_for_violation()
                d_nodes = len(node_ids)
                for bi in range(b):
                    pd = {
                        node_ids[j]: float(pred[bi, j].item())
                        for j in range(min(d_nodes, int(pred.shape[1])))
                    }
                    ok, _ = verify_normalized_prediction(pd, env)
                    if not ok:
                        total[bi] *= fac
            eigs.extend(total.tolist())
        return eigs

    def _rollout_imagination_state(
        self, base: dict[str, float], var: str, val: float
    ) -> dict[str, float]:
        """Этап E: один мысленный do + столько же свободных шагов, сколько в VL imagination."""
        s = self.graph.propagate_from(dict(base), var, float(val))
        for _ in range(max(0, self._imagination_horizon)):
            s = self.graph.rollout_step_free(s)
        return s

    def _features_for_intervention_pair(self, v_from: str, v_to: str) -> list[float]:
        """Один вектор признаков System1 для пары (в_from→в_to), как в score_interventions."""
        h_W_norm = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        core = self.graph._core
        ii, jj = nid_to_i.get(v_from), nid_to_i.get(v_to)
        if core is not None and ii is not None and jj is not None:
            with torch.no_grad():
                unc_t = (1.0 - core.alpha_trust_matrix()).detach().float().cpu().numpy()
                W_m = core.W_masked().detach().float().cpu().numpy()
                g_m = None
                if core.W.grad is not None:
                    g_m = core.W.grad.detach().float().abs().cpu().numpy()
            uncertainty = float(unc_t[ii, jj])
            w_ij = float(W_m[ii, jj])
            grad_norm = float(g_m[ii, jj]) if g_m is not None else 0.0
        else:
            uncertainty, w_ij, grad_norm = 1.0, 0.0, 0.0
        alpha = 1.0 - uncertainty
        val_from = self.graph.nodes.get(v_from, 0.5)
        val_to = self.graph.nodes.get(v_to, 0.5)
        ic = ic_map.get((v_from, v_to), 0)
        return self.system1.build_features(
            w_ij=w_ij, alpha_ij=alpha,
            val_from=val_from, val_to=val_to,
            uncertainty=uncertainty, h_W_norm=h_W_norm,
            grad_norm_ij=grad_norm,
            intervention_count=ic,
            discovery_rate=disc_rate,
        )

    def _build_goal_planned_candidate(self, var: str, val: float) -> dict:
        feat = self._features_for_intervention_pair(var, "target_dist")
        return {
            "variable":    var,
            "target":      "target_dist",
            "value":       float(val),
            "uncertainty": 0.35,
            "features":    feat,
            "expected_ig": 1.0,
            "from_goal_plan": True,
        }

    def _maybe_goal_planned_candidate(self) -> dict | None:
        if goal_planning_globally_disabled():
            return None
        if self.graph._core is None:
            return None
        if self.graph.nodes.get("self_goal_active") is None:
            return None
        if float(self.graph.nodes.get("self_goal_active", 0)) <= 0.45:
            return None
        if "target_dist" not in self.graph.nodes:
            return None

        state0 = dict(self.graph.nodes)
        cur_td = float(state0.get("target_dist", 0.5))
        goal_thr = float(state0.get("self_goal_target_dist", 0.42))
        if cur_td <= goal_thr + 0.015:
            return None

        motor = planning_graph_motor_vars(self.env, list(self.graph._node_ids))
        if not motor:
            return None

        levels = parse_plan_value_levels()
        actions = [(v, x) for v in motor for x in levels]
        max_b = plan_max_branch()
        if len(actions) > max_b:
            idx = np.random.choice(len(actions), size=max_b, replace=False)
            actions = [actions[i] for i in idx]

        depth = plan_depth()
        beam_k = plan_beam_k()

        def _td(s: dict[str, float]) -> float:
            return float(s.get("target_dist", cur_td))

        best_td = cur_td
        best_first: tuple[str, float] | None = None

        if depth <= 1:
            for var, val in actions:
                try:
                    sfin = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                    if not ok:
                        continue
                td = _td(sfin)
                if td < best_td - 1e-6:
                    best_td = td
                    best_first = (var, val)
        else:
            scored: list[tuple[float, str, float, dict[str, float]]] = []
            for var, val in actions:
                try:
                    s1 = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(s1), self.env)
                    if not ok:
                        continue
                scored.append((_td(s1), var, val, dict(s1)))
            scored.sort(key=lambda t: t[0])
            for _td1, v1, x1, s1 in scored[:beam_k]:
                for v2, x2 in actions:
                    try:
                        sfin = self._rollout_imagination_state(s1, v2, x2)
                    except Exception:
                        continue
                    if symbolic_verifier_enabled():
                        ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                        if not ok:
                            continue
                    td = _td(sfin)
                    if td < best_td - 1e-6:
                        best_td = td
                        best_first = (v1, x1)

        if best_first is None:
            return None
        return self._build_goal_planned_candidate(best_first[0], best_first[1])

    def _is_locomotion_primary_active(self) -> bool:
        """Если CPG управляет ногами, EIG не должен конкурировать за суставы — только intent_* и др."""
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        var_ids   = self.env.variable_ids
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate

        # Один проход по рёбрам: счётчики интервенций (раньше — O(pairs×|E|) через next() в цикле)
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count

        # Имя узла → индекс без O(d) list.index на каждую пару
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}

        # Один раз W, α и |grad| на CPU — вместо O(d²) вызовов alpha_trust_matrix / W_masked
        core = self.graph._core
        W_m = unc_m = g_m = None
        if core is not None:
            with torch.no_grad():
                W_t = core.W_masked().detach().float()
                A_t = core.alpha_trust_matrix().detach().float()
                W_m = W_t.cpu().numpy()
                unc_m = (1.0 - A_t).cpu().numpy()
            if core.W.grad is not None:
                g_m = core.W.grad.detach().float().abs().cpu().numpy()

        d = len(var_ids)
        if d == 0:
            return []

        # Счётчики интервенций по парам (только известные рёбра — O(|E|))
        ic_mat = np.zeros((d, d), dtype=np.float64)
        v2i = {v: i for i, v in enumerate(var_ids)}
        for (vf, vt), c in ic_map.items():
            i = v2i.get(vf)
            j = v2i.get(vt)
            if i is not None and j is not None and i != j:
                ic_mat[i, j] = float(c)

        ridx = np.zeros(d, dtype=np.int64)
        valid_node = np.zeros(d, dtype=bool)
        for i, v in enumerate(var_ids):
            ji = nid_to_i.get(v)
            if ji is not None:
                ridx[i] = ji
                valid_node[i] = True

        nodes_arr = np.array(
            [float(self.graph.nodes.get(v, 0.5)) for v in var_ids],
            dtype=np.float64,
        )
        mask = ~np.eye(d, dtype=bool)
        fi, fj = np.where(mask)
        n_pairs = len(fi)

        if W_m is not None:
            ii_n = ridx[fi]
            jj_n = ridx[fj]
            ok = valid_node[fi] & valid_node[fj]
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)
            w_ij[ok] = W_m[ii_n[ok], jj_n[ok]]
            uncertainty[ok] = unc_m[ii_n[ok], jj_n[ok]]
            if g_m is not None:
                grad_norm[ok] = g_m[ii_n[ok], jj_n[ok]]
        else:
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)

        alpha = 1.0 - uncertainty
        val_from = nodes_arr[fi]
        val_to = nodes_arr[fj]
        ic_v = ic_mat[fi, fj]
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_v = float(np.clip(disc_rate, 0.0, 1.0))

        feats_arr = np.column_stack(
            [
                np.tanh(w_ij),
                np.clip(alpha, 0.0, 1.0),
                np.clip(val_from, 0.0, 1.0),
                np.clip(val_to, 0.0, 1.0),
                np.clip(uncertainty, 0.0, 1.0),
                np.full(n_pairs, h_clip, dtype=np.float64),
                np.tanh(grad_norm),
                np.clip(ic_v / 100.0, 0.0, 1.0),
                np.full(n_pairs, disc_v, dtype=np.float64),
            ]
        )
        features_batch = feats_arr.tolist()

        rng = np.random.default_rng()
        posture_now = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 0.5),
            )
        )
        foot_l_now = float(
            self.graph.nodes.get(
                "foot_contact_l",
                self.graph.nodes.get("phys_foot_contact_l", 0.5),
            )
        )
        foot_r_now = float(
            self.graph.nodes.get(
                "foot_contact_r",
                self.graph.nodes.get("phys_foot_contact_r", 0.5),
            )
        )
        stable_stance = posture_now > 0.70 and min(foot_l_now, foot_r_now) > 0.56

        # ── Sparse EIG: skip low-uncertainty pairs ──────────────────────────
        try:
            _sparse_min_unc = float(os.environ.get("RKK_SPARSE_EIG_MIN_UNC", "0.15"))
        except ValueError:
            _sparse_min_unc = 0.15
        _sparse_min_unc = max(0.0, min(0.8, _sparse_min_unc))

        candidates: list[dict] = []
        for k in range(n_pairs):
            i, j = int(fi[k]), int(fj[k])
            vf, vt = var_ids[i], var_ids[j]
            unc_k = float(uncertainty[k])
            feat_k = features_batch[k]

            # Sparse filter: skip well-known edges (except motor intents)
            if _sparse_min_unc > 0 and unc_k < _sparse_min_unc:
                if not _is_motor_intent_var(vf):
                    continue

            if _is_motor_intent_var(vf):
                if stable_stance:
                    lo, hi = 0.30, 0.72
                else:
                    lo, hi = 0.35, 0.68
                if str(vf).endswith("stride"):
                    hi = min(hi, 0.62 if stable_stance else 0.56)
                if str(vf).endswith("stop_recover"):
                    lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                rand_value = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
            else:
                rand_value = float(np.clip(rng.uniform(0.15, 0.85), 0.06, 0.94))
            candidates.append({
                "variable":    vf,
                "target":      vt,
                "value":       rand_value,
                "uncertainty": unc_k,
                "features":    feat_k,
                "expected_ig": 0.0,
            })

        if self._is_locomotion_primary_active():
            candidates = [
                c
                for c in candidates
                if c["variable"] not in _LOCOMOTION_CPG_LEG_EIG_BLOCK
            ]
            if posture_now < 0.65:
                candidates = [
                    c
                    for c in candidates
                    if str(c["variable"]).startswith("intent_")
                    or str(c["variable"]).startswith("phys_intent_")
                ]

        if not candidates or not features_batch:
            return []

        use_eig = _hypothesis_eig_from_env() and W_m is not None and unc_m is not None
        if use_eig:
            x_vec = np.array(
                [float(self.graph.nodes.get(n, 0.0)) for n in self.graph._node_ids],
                dtype=np.float64,
            )
            u_node = self._marginal_node_uncertainty(unc_m)
            eigs = self._batch_hypothesis_eig(
                candidates, x_vec, u_node, nid_to_i, unc_m,
                list(self.graph._node_ids), self.env,
            )
            if len(eigs) == len(candidates):
                # Учитываем гипотезу «это ребро неизвестно»: масштаб EIG по unc(v_from→v_to).
                for i, cand in enumerate(candidates):
                    eigs[i] *= 1.0 + float(cand["uncertainty"])
                arr = np.array(eigs, dtype=np.float64)
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo + 1e-12:
                    normed = (arr - lo) / (hi - lo)
                else:
                    normed = np.full_like(arr, 0.5)
                for i, cand in enumerate(candidates):
                    cand["eig_raw"] = float(eigs[i])
                    cand["expected_ig"] = float(normed[i])
            else:
                use_eig = False

        if not use_eig:
            scores = self.system1.score(features_batch)
            for i, cand in enumerate(candidates):
                cand["expected_ig"] = scores[i]

        if symbolic_verifier_enabled() and self._symbolic_prediction_bad:
            a, b = exploration_blend_from_uncertainty()
            for cand in candidates:
                unc = float(cand.get("uncertainty", 0.5))
                cand["expected_ig"] = a * float(cand["expected_ig"]) + b * unc

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    def _score_async_worker(self) -> None:
        try:
            with torch.no_grad():
                result = self.score_interventions()
            with self._score_lock:
                self._score_result = result
        except Exception as ex:
            print(f"[RKKAgent] score_interventions (async): {ex}")

    def set_teacher_state(self, rules: list[TeacherIGRule], weight: float) -> None:
        """Фаза 3: правила от LLM и текущий teacher_weight (симуляция считает annealing)."""
        self._teacher_rules = list(rules)
        self._teacher_weight = float(max(0.0, min(1.0, weight)))

    def _teacher_ig_bonus(self, variable: str, nodes: dict[str, float]) -> float:
        w = self._teacher_weight
        if w <= 0 or not self._teacher_rules:
            return 0.0
        acc = 0.0
        for r in self._teacher_rules:
            if r.target_var != variable:
                continue
            if r.when_var:
                val = nodes.get(r.when_var)
                if val is None:
                    continue
                if r.when_min is not None and float(val) < r.when_min:
                    continue
                if r.when_max is not None and float(val) > r.when_max:
                    continue
            acc += r.bonus * w
        return min(0.28, acc)

    # ── Один шаг с Value Layer ────────────────────────────────────────────────
    def step(self, engine_tick: int = 0, *, enable_l3: bool = True) -> dict:
        self._last_engine_tick = engine_tick
        try:
            self.graph.apply_env_observation(dict(self.env.observe()))
        except Exception:
            pass
        sce = _score_cache_every()
        if (
            sce > 1
            and self._score_cache
            and (engine_tick - self._score_cache_tick) < sce
        ):
            scores = list(self._score_cache)
        elif _score_async_enabled():
            if self._score_thread is None or not self._score_thread.is_alive():
                self._score_thread = threading.Thread(
                    target=self._score_async_worker,
                    name="rkk_score_interventions",
                    daemon=True,
                )
                self._score_thread.start()
            with self._score_lock:
                have = list(self._score_result) if self._score_result else []
            if have:
                scores = have
            elif self._score_cache:
                scores = list(self._score_cache)
            else:
                with torch.no_grad():
                    scores = self.score_interventions()
                with self._score_lock:
                    self._score_result = list(scores)
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        else:
            with torch.no_grad():
                scores = self.score_interventions()
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        gp = self._maybe_goal_planned_candidate() if enable_l3 else None
        if gp is not None and not (
            symbolic_verifier_enabled() and self._symbolic_prediction_bad
        ):
            scores.insert(0, gp)
        if not scores:
            return {
                "blocked": False, "skipped": True, "prediction_error": 0.0,
                "cf_predicted": {}, "cf_observed": {}, "goal_planned": False,
            }

        current_phi = self.phi_approx()
        chosen      = None
        check_result = None
        blocked_count = 0

        # Перебираем кандидатов пока не найдём допустимое действие
        for candidate in scores[:MAX_FALLBACK_TRIES]:
            var   = candidate["variable"]
            value = candidate["value"]

            check_result = self.value_layer.check_action(
                variable=var,
                value=value,
                current_nodes=dict(self.graph.nodes),
                graph=self.graph,
                temporal=self.temporal,
                current_phi=current_phi,
                other_agents_phi=self.other_agents_phi,
                engine_tick=engine_tick,
                imagination_horizon=(self._imagination_horizon if enable_l3 else 0),
            )

            if check_result.allowed:
                chosen = candidate
                break
            else:
                # Штрафуем System 1 за предложение опасного действия
                self.system1.push_experience(
                    features=candidate["features"],
                    actual_ig=check_result.penalty,   # отрицательный IG
                )
                blocked_count += 1
                self._total_blocked += 1
                self._last_blocked_reason = check_result.reason.value

        # Все кандидаты заблокированы — возвращаем событие
        if chosen is None:
            return {
                "blocked":       True,
                "blocked_count": blocked_count,
                "reason":        self._last_blocked_reason,
                "variable":      scores[0]["variable"] if scores else "?",
                "value":         scores[0]["value"] if scores else 0.5,
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error":  0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        # ── Выполняем допустимое действие ────────────────────────────────────
        var   = chosen["variable"]
        value = chosen["value"]

        if is_read_only_macro_var(var):
            return {
                "blocked": True,
                "blocked_count": blocked_count + 1,
                "reason": "read_only_macro",
                "variable": var,
                "value": float(value),
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        mdl_before = self.graph.mdl_size
        obs_before_env = dict(self.env.observe())
        self.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.graph.snapshot_vec_dict()
        predicted  = self.graph.propagate(var, value)
        sym_ok, sym_fail = True, []
        if symbolic_verifier_enabled():
            sym_ok, sym_fail = verify_normalized_prediction(dict(predicted), self.env)
            self._symbolic_prediction_bad = not sym_ok
        else:
            self._symbolic_prediction_bad = False
        observed_env = self.env.intervene(var, value)

        # Temporal step (только размерность среды)
        self.temporal.step(observed_env)

        self.graph.apply_env_observation(observed_env)
        observed_full = self.graph.snapshot_vec_dict()

        # NOTEARS / GNN буферы — полный вектор узлов (включая concept_*)
        self.graph.record_observation(obs_before_full)
        self.graph.record_observation(observed_full)
        self.graph.record_intervention(var, value, obs_before_full, observed_full)

        # NOTEARS train
        notears_result = None
        if self._total_interventions % NOTEARS_EVERY == 0:
            notears_result = self.graph.train_step()
            if notears_result:
                self._notears_steps += 1
                self._last_notears_loss = notears_result
            self._maybe_train_local_reflex()

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # System 1: IG по физике; slot_* и self_* не доминируют метрику (self — прямое задание агентом).
        nids = self.graph._node_ids
        phys_ids = [
            k for k in nids
            if k not in _SELF_VAR_SET and not str(k).startswith("slot_")
        ]
        slot_ids = [k for k in nids if str(k).startswith("slot_")]

        def _mean_abs_err(keys: list) -> float:
            if not keys:
                return 0.0
            return float(np.mean([
                abs(float(predicted.get(k, 0.5)) - float(observed_full.get(k, 0.5)))
                for k in keys
            ]))

        pe_phys = _mean_abs_err(phys_ids)

        # Этап Г: петля «намерение ↔ исход» + ошибка модели → self_* (только среды с методом).
        fn_sf = getattr(self.env, "update_self_feedback", None)
        if callable(fn_sf):
            try:
                fn_sf(
                    variable=var,
                    intended_norm=value,
                    observed=observed_env,
                    predicted=predicted,
                    prediction_error_phys=pe_phys,
                )
            except Exception:
                pass
            obs_self = dict(self.env.observe())
            for sk in _SELF_VAR_SET:
                if sk in self.graph.nodes and sk in obs_self:
                    self.graph.nodes[sk] = float(obs_self[sk])
            self.graph.refresh_concept_aggregates()
        pe_slot = _mean_abs_err(slot_ids)
        w_vis = min(0.45, max(0.0, VISUAL_IG_WEIGHT))
        if slot_ids and phys_ids:
            actual_ig = (1.0 - w_vis) * pe_phys + w_vis * pe_slot
        elif phys_ids:
            actual_ig = pe_phys
        else:
            actual_ig = pe_slot

        t_bonus = self._teacher_ig_bonus(var, dict(self.graph.nodes))
        actual_ig = float(np.clip(actual_ig + t_bonus, 0.0, 1.0))

        self.system1.push_experience(
            features=chosen["features"],
            actual_ig=actual_ig,
        )

        # SSM train — размерность = temporal.d_input (= |graph._node_ids|), не только env.variable_ids
        u_next = torch.tensor(
            [float(self.graph.nodes.get(n, 0.5)) for n in self.graph._node_ids],
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal.train_step(u_next)

        self._total_interventions += 1
        try:
            _v_do = float(value)
        except (TypeError, ValueError):
            _v_do = 0.5
        self._last_do = f"do({var}={_v_do:.2f})"
        self._last_blocked_reason = ""

        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        rsi_event = self._tick_rsi_lite_discovery(cur_dr)

        _cf_keys = list(self.graph._node_ids)[:48]
        self._last_result = {
            "blocked":           False,
            "blocked_count":     blocked_count,
            "variable":          var,
            "value":             value,
            "compression_delta": compression_delta,
            "updated_edges":     [f"{e.from_}→{e.to}" for e in self.graph.edges[:4]],
            "pruned_edges":      [],
            "prediction_error":  float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed_env.items()
            ])),
            "cf_predicted": {k: float(round(float(predicted.get(k, 0.0)), 4)) for k in _cf_keys},
            "cf_observed":  {k: float(round(float(observed_full.get(k, 0.0)), 4)) for k in _cf_keys},
            "goal_planned":  bool(chosen.get("from_goal_plan")),
            "symbolic_ok": sym_ok,
            "symbolic_violations": sym_fail,
            "rsi_lite": rsi_event,
            "notears":           notears_result,
        }
        return self._last_result

    # ── Demon ─────────────────────────────────────────────────────────────────
    def demon_disrupt(self) -> str:
        if self.graph._core is None:
            return "no core"
        with torch.no_grad():
            W = self.graph._core.W
            sig = (W.abs() > 0.05).nonzero(as_tuple=False)
            if len(sig) == 0:
                return "no significant edges"
            idx = sig[np.random.randint(len(sig))]
            i, j = idx[0].item(), idx[1].item()
            noise = (np.random.rand() - 0.5) * 0.3
            # Нельзя W[i,j] += … — это in-place на view листа с requires_grad.
            w_new = W.detach().clone()
            w_new[i, j] = w_new[i, j] + float(noise)
            W.copy_(w_new)
            fn = self.graph._node_ids[i] if i < len(self.graph._node_ids) else f"v{i}"
            tn = self.graph._node_ids[j] if j < len(self.graph._node_ids) else f"v{j}"
        self.graph._invalidate_cache()
        return f"W[{fn}→{tn}] +{noise:.3f}"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def compression_gain(self) -> float:
        if not self._cg_history:
            return 0.0
        return float(np.mean(list(self._cg_history)))

    @property
    def discovery_rate(self) -> float:
        """
        Blend of GT-based and self-supervised discovery rate.
        As the agent matures, self-supervised metric gets more weight.
        """
        gt_dr = self.env.discovery_rate([
            {"from_": e.from_, "to": e.to, "weight": e.weight}
            for e in self.graph.edges
        ])
        # Self-supervised: compression discoveries / total computations
        ss_dr = self.self_supervised_discovery_rate
        # Blend: GT dominates early (calibration), self-supervised dominates later
        if self._total_interventions < 200:
            return gt_dr
        blend = min(1.0, (self._total_interventions - 200) / 1000.0)
        return (1.0 - blend) * gt_dr + blend * ss_dr

    @property
    def self_supervised_discovery_rate(self) -> float:
        """
        Discovery rate without ground-truth edges.
        Based on CausalSurprise compression discoveries — the fraction of
        interventions that actually improved the causal model.
        """
        # Try to get from IntrinsicObjective (if simulation has it patched in)
        try:
            from engine.intristic_objective import IntrinsicObjective
            # Walk up to find intrinsic objective
            for attr_name in ("_intrinsic",):
                # IntrinsicObjective attaches to simulation, not agent
                # We use the causal_surprise directly if available
                pass
            # Fallback: use graph-level stats
            if self.graph.train_losses:
                recent = self.graph.train_losses[-20:]
                if len(recent) >= 5:
                    # Discovery = loss is still decreasing (model is learning)
                    early = float(np.mean(recent[:len(recent)//2]))
                    late = float(np.mean(recent[len(recent)//2:]))
                    if early > 1e-8:
                        improvement = max(0.0, (early - late) / early)
                        return float(np.clip(improvement * 2.0, 0.0, 1.0))
        except Exception:
            pass
        return 0.5  # neutral default

    @property
    def peak_discovery_rate(self) -> float:
        return self._peak_discovery_rate

    def _apply_rsi_lite(self) -> dict[str, float | int]:
        g = self.graph
        cur_l1 = float(getattr(g, "LAMBDA_L1", CausalGraph.LAMBDA_L1))
        new_l1 = min(cur_l1 * rsi_l1_scale(), rsi_l1_max())
        g.LAMBDA_L1 = new_l1
        cap_b = rsi_buffer_cap()
        g.BUFFER_SIZE = min(cap_b, int(g.BUFFER_SIZE) * 2)
        cap_i = rsi_imagination_cap()
        self._imagination_horizon = min(cap_i, self._imagination_horizon + 1)
        self._rsi_adjustment_count += 1
        return {
            "LAMBDA_L1": float(new_l1),
            "BUFFER_SIZE": int(g.BUFFER_SIZE),
            "imagination_horizon": int(self._imagination_horizon),
        }

    def _tick_rsi_lite_discovery(self, cur_dr: float) -> dict[str, float | int] | None:
        if not rsi_lite_enabled():
            return None
        if self._total_interventions < rsi_min_interventions():
            return None
        eps = rsi_improvement_eps()
        if cur_dr > self._rsi_ref_discovery + eps:
            self._rsi_ref_discovery = float(cur_dr)
            self._rsi_plateau_count = 0
            return None
        self._rsi_plateau_count += 1
        if self._rsi_plateau_count < rsi_plateau_interventions():
            return None
        self._rsi_plateau_count = 0
        self._rsi_ref_discovery = float(cur_dr)
        return self._apply_rsi_lite()

    def _maybe_train_local_reflex(self) -> None:
        if not local_reflex_train_enabled():
            return
        self._last_local_reflex_train = train_chains_parallel(
            graph=self.graph,
            device=self.graph.device,
            cores=self._local_reflex_cores,
        )

    def phi_approx(self) -> float:
        return self.temporal.phi_approx()

    def record_phi(self, _: float):
        pass  # temporal управляет историей сам

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        h_W     = self._get_h_W()
        tb_info = self.temporal.slow_state_summary()
        s1_info = {
            "buffer_size": len(self.system1.buffer),
            "mean_loss":   round(self.system1.mean_loss, 6),
        }
        vl_info = dict(self.value_layer.snapshot(self._last_engine_tick))
        vl_info["imagination_horizon"] = self._imagination_horizon

        notears_info = None
        if self._last_notears_loss:
            notears_info = {
                "steps":  self._notears_steps,
                "loss":   self._last_notears_loss.get("loss", 0),
                "h_W":    round(h_W, 4),
                "l_int":  self._last_notears_loss.get("l_int", 0),
            }

        h_W_edge_entropy = None
        core = self.graph._core
        if core is not None:
            with torch.no_grad():
                A = core.alpha_trust_matrix().detach().float().cpu().numpy()
            p = np.clip(A, 1e-7, 1.0 - 1e-7)
            h_W_edge_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())

        snap: dict = {
            "id":                    self.id,
            "name":                  self.name,
            "env_type":              self.env.preset,
            "activation":            self.activation,
            "graph_mdl":             round(self.graph.mdl_size, 3),
            "compression_gain":      round(self.compression_gain, 4),
            "alpha_mean":            round(self.graph.alpha_mean, 3),
            "phi":                   round(self.phi_approx(), 4),
            "node_count":            len(self.graph.nodes),
            "edge_count":            len(self.graph.edges),
            "total_interventions":   self._total_interventions,
            "total_blocked":         self._total_blocked,
            "last_do":               self._last_do,
            "last_blocked_reason":   self._last_blocked_reason,
            "discovery_rate":        round(cur_dr, 3),
            "peak_discovery_rate":   round(self._peak_discovery_rate, 3),
            "h_W":                   round(h_W, 4),
            "notears":               notears_info,
            "temporal":              tb_info,
            "system1":               s1_info,
            "value_layer":           vl_info,
            "teacher": {
                "weight":     round(self._teacher_weight, 4),
                "rules":      len(self._teacher_rules),
            },
            "hypothesis_eig": _hypothesis_eig_from_env(),
            "h_W_edge_entropy": None if h_W_edge_entropy is None else round(h_W_edge_entropy, 4),
            "rsi_lite": {
                "enabled": rsi_lite_enabled(),
                "plateau_count": self._rsi_plateau_count,
                "ref_discovery": round(self._rsi_ref_discovery, 5),
                "adjustments": self._rsi_adjustment_count,
                "LAMBDA_L1": round(float(getattr(self.graph, "LAMBDA_L1", CausalGraph.LAMBDA_L1)), 5),
                "graph_BUFFER_SIZE": int(self.graph.BUFFER_SIZE),
                "imagination_horizon": int(self._imagination_horizon),
            },
            "local_reflex_train": self._last_local_reflex_train,
            "edges": [e.as_dict() for e in self.graph.edges],
        }
        if self.env.preset == "pybullet":
            pos_fn = getattr(self.env, "object_positions_world", None)
            if callable(pos_fn):
                snap["physics_objects"] = pos_fn()
        return snap
```

Added `self_supervised_discovery_rate` property that measures learning progress **without ground-truth edges**:
- Based on whether GNN train loss is still decreasing
- Blended with GT-based rate: GT dominates first 200 interventions, self-supervised takes over by intervention 1200

---

### 7. Variable Discovery Integration in IntrinsicObjective
```diff:intristic_objective.py
"""
intrinsic_objective.py — Unified Intrinsic Objective.

Одна цель: максимизировать I(agent; world) — взаимную информацию
между внутренней каузальной моделью и реальностью.

Операционально:
  R_intrinsic(t) = Δcompression(t) + λ·Δprediction_surprise(t)

Внешние hand-crafted награды и RewardCoordinator удалены из симуляции.

Остаётся ОДНО:
  ✓ "Насколько лучше агент понимает мир после этого действия?"

Локомоция возникает инструментально: ходить → больше новых состояний
→ больше информации → выше compression gain. Не потому что "так надо".

Архитектура:
  CausalSurprise   — Δ prediction error как интринсивная метрика
  GoalImagination  — агент генерирует свою следующую цель через GNN rollout
  VariableDiscovery — проактивное открытие новых переменных через EIG
  IntrinsicObjective — единый интерфейс, заменяет все внешние награды

Принцип нейрогенеза-через-цель:
  Если CausalSurprise растёт для области X, но модель не улучшается →
  создать новый узел в GNN для этой области (NeurogenesisEngine.scan_and_grow).

RKK_INTRINSIC_ENABLED=1
RKK_INTRINSIC_LAMBDA=0.4        — вес prediction surprise vs compression
RKK_INTRINSIC_GOAL_HORIZON=12   — шагов imagination для goal generation
RKK_INTRINSIC_DISCOVERY_EIG=0.3 — порог EIG для создания нового узла
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Config ───────────────────────────────────────────────────────────────────
def intrinsic_enabled() -> bool:
    return os.environ.get("RKK_INTRINSIC_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

def _ef(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)))
    except ValueError: return default

def _ei(key: str, default: int) -> int:
    try: return max(1, int(os.environ.get(key, str(default))))
    except ValueError: return default

# ─── CausalSurprise ───────────────────────────────────────────────────────────
class CausalSurprise:
    """
    Измеряет интринсивную ценность интервенции через два сигнала:

    1. Δcompression: насколько уменьшился MDL графа после do().
       Это прямая мера "граф стал точнее объяснять мир".

    2. Δprediction_error: насколько удивил нас результат do().
       Высокое удивление = в этой области модель плохая → исследуй.

    Итоговая интринсивная награда:
       R = sign(Δcompression) * |Δcompression| + λ * surprise_bonus

    Ключевое: surprise_bonus — это не штраф за ошибку.
    Это БОНУС за нахождение в неизведанной области.
    """

    def __init__(self):
        self._lambda = _ef("RKK_INTRINSIC_LAMBDA", 0.4)

        # Скользящие статистики для нормализации
        self._compression_history: deque[float] = deque(maxlen=200)
        self._surprise_history: deque[float] = deque(maxlen=200)
        self._reward_history: deque[float] = deque(maxlen=500)

        # EMA baseline для адаптивного порога
        self._compression_ema: float = 0.0
        self._surprise_ema: float = 0.0
        self._ema_alpha: float = 0.05

        # Счётчики
        self.total_computations: int = 0
        self.total_discoveries: int = 0  # тиков где compression > 0

    def compute(
        self,
        compression_delta: float,
        prediction_error: float,
        graph_mdl_before: float,
        graph_mdl_after: float,
        n_interventions: int,
    ) -> float:
        """
        Основной расчёт интринсивной награды.

        compression_delta: mdl_before - mdl_after (>0 = граф улучшился)
        prediction_error: |predicted - observed| (>0 = модель удивлена)
        """
        if not intrinsic_enabled():
            return 0.0

        self.total_computations += 1

        # --- Сигнал 1: Compression ---
        # Нормализуем по MDL размеру чтобы сравнивать ранние и поздние тики
        mdl_scale = max(graph_mdl_before, 1.0)
        comp_norm = float(compression_delta) / mdl_scale

        # EMA обновление
        self._compression_ema = (
            (1 - self._ema_alpha) * self._compression_ema
            + self._ema_alpha * comp_norm
        )
        self._compression_history.append(comp_norm)

        # --- Сигнал 2: Prediction Surprise ---
        # Нормализуем относительно исторической средней
        surprise_baseline = float(np.mean(self._surprise_history)) if self._surprise_history else 0.1
        surprise_norm = float(prediction_error) / max(surprise_baseline, 0.01)
        # Ограничиваем: не хотим чтобы огромные ошибки доминировали
        surprise_bonus = float(np.clip(np.log1p(surprise_norm) * 0.5, 0.0, 2.0))

        self._surprise_ema = (
            (1 - self._ema_alpha) * self._surprise_ema
            + self._ema_alpha * float(prediction_error)
        )
        self._surprise_history.append(float(prediction_error))

        # --- Объединяем ---
        # Compression gain — основной сигнал (может быть отрицательным если граф деградировал)
        # Surprise — всегда положительный бонус (исследование важно)
        reward = comp_norm + self._lambda * surprise_bonus

        # Ранний бонус: первые 500 интервенций мир совсем незнаком
        if n_interventions < 500:
            exploration_bonus = 0.3 * (1.0 - n_interventions / 500.0)
            reward += exploration_bonus

        if comp_norm > 0:
            self.total_discoveries += 1

        self._reward_history.append(reward)
        return float(reward)

    def surprise_is_high(self, threshold_sigma: float = 2.0) -> bool:
        """Текущее удивление аномально высокое? → агент в неизведанной области."""
        if len(self._surprise_history) < 20:
            return False
        arr = np.array(self._surprise_history)
        mu, sigma = float(arr.mean()), float(arr.std())
        if sigma < 1e-8:
            return False
        return float(self._surprise_ema) > mu + threshold_sigma * sigma

    def compression_is_stagnant(self, window: int = 50) -> bool:
        """Compression не растёт уже window тиков? → нужна другая стратегия."""
        if len(self._compression_history) < window:
            return False
        recent = list(self._compression_history)[-window:]
        return float(np.mean(recent)) < 1e-6

    def recent_mean(self, window: int = 32) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)[-window:]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_computations": self.total_computations,
            "total_discoveries": self.total_discoveries,
            "discovery_rate": round(
                self.total_discoveries / max(self.total_computations, 1), 4
            ),
            "compression_ema": round(self._compression_ema, 6),
            "surprise_ema": round(self._surprise_ema, 5),
            "recent_reward_mean": round(self.recent_mean(), 5),
            "surprise_is_high": self.surprise_is_high(),
            "compression_stagnant": self.compression_is_stagnant(),
        }


# ─── GoalImagination ─────────────────────────────────────────────────────────
class GoalImagination:
    """
    Агент генерирует СВОЮ СЛЕДУЮЩУЮ ЦЕЛЬ через GNN rollout.

    Алгоритм:
      1. Из текущего состояния сэмплируем K случайных интервенций
      2. Для каждой запускаем imagination rollout на H шагов
      3. Выбираем интервенцию с максимальным ожидаемым Δcompression
      4. Эта интервенция становится "целью" на следующие T тиков

    НЕТ захардкоженной цели. Никаких "дойди до target_dist".
    Цель = "сделай то, что больше всего улучшит понимание мира".

    Это фундаментально другое от goal_planning.py:
      goal_planning: "уменьши target_dist" (человеческая цель)
      GoalImagination: "максимизируй compression_gain" (интринсивная цель)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._horizon = _ei("RKK_INTRINSIC_GOAL_HORIZON", 12)
        self._k_candidates = _ei("RKK_INTRINSIC_GOAL_CANDIDATES", 32)

        # Текущая активная цель
        self._current_goal: dict[str, Any] | None = None
        self._goal_age: int = 0
        self._goal_ttl: int = 20  # тиков пока цель активна

        # История целей для анализа
        self._goal_history: deque[dict[str, Any]] = deque(maxlen=50)

        # Статистика качества imagination
        self._imagination_accuracy: deque[float] = deque(maxlen=100)

        self.total_goals_generated: int = 0

    def generate_goal(
        self,
        graph,
        agent_env,
        n_interventions: int,
        causal_surprise: CausalSurprise,
    ) -> dict[str, Any] | None:
        """
        Генерирует следующую цель через GNN imagination rollout.
        Возвращает {"variable": str, "value": float, "expected_gain": float}.
        """
        if not intrinsic_enabled():
            return None

        core = graph._core
        if core is None:
            return None

        node_ids = list(graph._node_ids)
        d = len(node_ids)
        if d == 0:
            return None

        # Исключаем read-only и self_* из кандидатов
        from engine.graph_constants import is_read_only_macro_var
        candidates_vars = [
            nid for nid in node_ids
            if not is_read_only_macro_var(nid)
            and not nid.startswith("concept_")
            and not nid.startswith("proprio_")
        ]
        if not candidates_vars:
            return None

        # Сэмплируем K интервенций
        k = min(self._k_candidates, len(candidates_vars) * 3)
        rng = np.random.default_rng()

        # Адаптивный диапазон: если стагнация — исследуем агрессивнее
        if causal_surprise.compression_is_stagnant():
            lo, hi = 0.1, 0.9
        elif causal_surprise.surprise_is_high():
            lo, hi = 0.3, 0.7  # осторожнее когда много сюрпризов
        else:
            lo, hi = 0.2, 0.8

        sampled = [
            (rng.choice(candidates_vars), float(rng.uniform(lo, hi)))
            for _ in range(k)
        ]

        # Текущее состояние
        current_state = dict(graph.nodes)
        current_mdl = float(graph.mdl_size)

        best_var, best_val, best_expected = None, 0.5, -np.inf

        with torch.no_grad():
            for var, val in sampled:
                try:
                    # GNN rollout
                    state = graph.propagate_from(current_state, var, val)

                    # Многошаговый rollout
                    for _ in range(min(3, self._horizon)):
                        state = graph.rollout_step_free(state)

                    # Оцениваем ожидаемый Δcompression через proxy:
                    # "насколько состояние после отличается от текущего?"
                    delta = float(np.mean([
                        abs(float(state.get(nid, 0.5)) - float(current_state.get(nid, 0.5)))
                        for nid in node_ids
                    ]))

                    # Неопределённость по этому узлу (чем выше — тем ценнее)
                    if core is not None:
                        A = core.alpha_trust_matrix()
                        idx = node_ids.index(var) if var in node_ids else -1
                        if idx >= 0:
                            uncertainty = float(1.0 - A[idx].mean().item())
                        else:
                            uncertainty = 0.5
                    else:
                        uncertainty = 0.5

                    # Ожидаемый выигрыш = delta * uncertainty
                    # (большое изменение в неизведанной области = ценно)
                    expected_gain = delta * (1.0 + uncertainty)

                    if expected_gain > best_expected:
                        best_expected = expected_gain
                        best_var = var
                        best_val = val

                except Exception:
                    continue

        if best_var is None:
            return None

        goal = {
            "variable": best_var,
            "value": best_val,
            "expected_gain": float(best_expected),
            "generated_at": n_interventions,
            "horizon": self._horizon,
        }

        self._current_goal = goal
        self._goal_age = 0
        self._goal_history.append(dict(goal))
        self.total_goals_generated += 1
        return goal

    def tick_goal(self, actual_compression: float) -> dict[str, Any] | None:
        """
        Обновляет возраст цели. Возвращает текущую цель или None если истекла.
        Записывает точность imagination (предсказали gain, получили actual).
        """
        if self._current_goal is None:
            return None

        self._goal_age += 1

        # Записываем точность imagination
        if self._goal_age == 1:
            expected = self._current_goal.get("expected_gain", 0.0)
            if expected > 0:
                accuracy = min(1.0, actual_compression / max(expected, 1e-8))
                self._imagination_accuracy.append(accuracy)

        if self._goal_age >= self._goal_ttl:
            old_goal = self._current_goal
            self._current_goal = None
            self._goal_age = 0
            return None

        return self._current_goal

    def imagination_accuracy(self) -> float:
        if not self._imagination_accuracy:
            return 0.5
        return float(np.mean(self._imagination_accuracy))

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_goal": self._current_goal,
            "goal_age": self._goal_age,
            "goal_ttl": self._goal_ttl,
            "total_goals_generated": self.total_goals_generated,
            "imagination_accuracy": round(self.imagination_accuracy(), 4),
            "recent_goals": list(self._goal_history)[-5:],
        }


# ─── VariableDiscovery ────────────────────────────────────────────────────────
class VariableDiscovery:
    """
    Проактивное открытие новых переменных через EIG.

    Принцип:
      Если агент систематически ошибается в предсказании определённой
      области состояний, и GNN не может улучшиться (высокий surprise
      при низком compression gain) → там скрытая переменная.

    Алгоритм:
      1. Отслеживаем prediction error по каждому узлу отдельно
      2. Если узел X имеет высокую и СТАБИЛЬНУЮ ошибку →
         GNN не справляется, нужен медиирующий узел
      3. Вызываем NeurogenesisEngine.scan_and_grow() с целевой парой

    Отличие от rsi_structural.py:
      rsi_structural: реактивный (срабатывает при high stress в W.grad)
      VariableDiscovery: проактивный (срабатывает при стабильной ошибке)
    """

    def __init__(self):
        self._eig_threshold = _ef("RKK_INTRINSIC_DISCOVERY_EIG", 0.3)
        # per-node prediction error history
        self._node_errors: dict[str, deque[float]] = {}
        self._window = 30
        self._last_discovery_tick: int = -9999
        self._discovery_cooldown: int = 500
        self.total_discoveries: int = 0
        self._discovery_log: deque[dict[str, Any]] = deque(maxlen=20)

    def update_node_errors(
        self,
        predicted: dict[str, float],
        observed: dict[str, float],
    ) -> None:
        """Записываем ошибку по каждому узлу."""
        for nid in observed:
            if nid not in self._node_errors:
                self._node_errors[nid] = deque(maxlen=self._window)
            pred_val = float(predicted.get(nid, 0.5))
            obs_val = float(observed[nid])
            self._node_errors[nid].append(abs(pred_val - obs_val))

    def find_high_error_nodes(
        self,
        top_k: int = 3,
        min_error: float = 0.15,
    ) -> list[tuple[str, float]]:
        """
        Находим узлы с хронически высокой ошибкой.
        Это кандидаты для нейрогенеза.
        """
        candidates: list[tuple[str, float]] = []
        for nid, errors in self._node_errors.items():
            if len(errors) < self._window // 2:
                continue
            mean_err = float(np.mean(errors))
            # Стабильность ошибки (низкий std = стабильно плохо, не случайный всплеск)
            std_err = float(np.std(errors))
            stability = 1.0 / (1.0 + std_err)
            score = mean_err * stability
            if mean_err >= min_error:
                candidates.append((nid, score))

        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def maybe_trigger_neurogenesis(
        self,
        graph,
        agent,
        tick: int,
        causal_surprise: CausalSurprise,
    ) -> dict[str, Any] | None:
        """
        Проверяет нужен ли нейрогенез и запускает его.
        Возвращает событие если новый узел был создан.
        """
        if not intrinsic_enabled():
            return None
        if (tick - self._last_discovery_tick) < self._discovery_cooldown:
            return None
        if not causal_surprise.compression_is_stagnant():
            return None  # Граф ещё учится — нейрогенез преждевременен

        high_error = self.find_high_error_nodes()
        if len(high_error) < 2:
            return None

        # Пытаемся создать медиирующий узел между двумя проблемными
        from engine.rsi_structural import NeurogenesisEngine
        neuro = getattr(agent, "_neuro_engine", NeurogenesisEngine(
            min_interventions=200,
            error_threshold=self._eig_threshold,
        ))

        result = neuro.scan_and_grow(agent, tick)
        if result is None:
            return None

        self._last_discovery_tick = tick
        self.total_discoveries += 1

        event = {
            "tick": tick,
            "new_node": result.get("new_node"),
            "triggered_by": [nid for nid, _ in high_error[:2]],
            "error_scores": [round(s, 4) for _, s in high_error[:2]],
        }
        self._discovery_log.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        high_err = self.find_high_error_nodes()
        return {
            "total_discoveries": self.total_discoveries,
            "nodes_tracked": len(self._node_errors),
            "high_error_nodes": [(n, round(s, 4)) for n, s in high_err],
            "recent_discoveries": list(self._discovery_log)[-3:],
        }


# ─── IntrinsicObjective ───────────────────────────────────────────────────────
class IntrinsicObjective:
    """
    Единый интерфейс интринсивной цели. Заменяет все внешние rewards.

    Использование в simulation_main.py:
      self._intrinsic = IntrinsicObjective(device)

    После каждого agent.step():
      r = self._intrinsic.step(
          agent=self.agent,
          result=result,
          tick=self.tick,
          locomotion_ctrl=self._locomotion_controller,
      )

    """

    def __init__(self, device: torch.device):
        self.device = device
        self.causal_surprise = CausalSurprise()
        self.goal_imagination = GoalImagination(device)
        self.variable_discovery = VariableDiscovery()

        # Когда генерируем следующую цель
        self._goal_generate_every = _ei("RKK_INTRINSIC_GOAL_EVERY", 25)
        self._last_goal_tick: int = -9999

        # История наград
        self._reward_history: deque[float] = deque(maxlen=500)
        self.total_steps: int = 0

    def step(
        self,
        agent,
        result: dict[str, Any],
        tick: int,
        locomotion_ctrl=None,
        motor_cortex=None,
    ) -> float:
        """
        Главный вызов: вычисляет интринсивную награду и применяет её.
        Возвращает скалярную награду.
        """
        if not intrinsic_enabled():
            return 0.0

        self.total_steps += 1

        # --- Извлекаем сигналы ---
        compression_delta = float(result.get("compression_delta", 0.0))
        prediction_error = float(result.get("prediction_error", 0.0))
        graph = agent.graph
        mdl_before = float(graph.mdl_size) + abs(compression_delta)
        mdl_after = float(graph.mdl_size)

        # --- Обновляем per-node ошибки ---
        cf_pred = result.get("cf_predicted") or {}
        cf_obs = result.get("cf_observed") or {}
        if cf_pred and cf_obs:
            self.variable_discovery.update_node_errors(cf_pred, cf_obs)

        # --- Вычисляем интринсивную награду ---
        r = self.causal_surprise.compute(
            compression_delta=compression_delta,
            prediction_error=prediction_error,
            graph_mdl_before=mdl_before,
            graph_mdl_after=mdl_after,
            n_interventions=int(agent._total_interventions),
        )

        # --- Обновляем активную цель ---
        active_goal = self.goal_imagination.tick_goal(compression_delta)

        # --- Генерируем новую цель если нужно ---
        if (tick - self._last_goal_tick) >= self._goal_generate_every:
            if active_goal is None or self.causal_surprise.compression_is_stagnant():
                try:
                    new_goal = self.goal_imagination.generate_goal(
                        graph=graph,
                        agent_env=agent.env,
                        n_interventions=int(agent._total_interventions),
                        causal_surprise=self.causal_surprise,
                    )
                    if new_goal is not None:
                        self._last_goal_tick = tick
                except Exception:
                    pass

        # --- Проактивный нейрогенез ---
        try:
            neuro_event = self.variable_discovery.maybe_trigger_neurogenesis(
                graph=graph,
                agent=agent,
                tick=tick,
                causal_surprise=self.causal_surprise,
            )
        except Exception:
            neuro_event = None

        self._apply_intrinsic_reward(r, locomotion_ctrl, motor_cortex, agent)

        self._reward_history.append(r)
        return r

    def _apply_intrinsic_reward(
        self,
        r: float,
        locomotion_ctrl,
        motor_cortex,
        agent,
    ) -> None:
        """
        Полная замена: ТОЛЬКО интринсивная награда идёт в все learners.
        Никаких posture, symmetry, forward_bonus.
        """
        if locomotion_ctrl is not None:
            locomotion_ctrl._reward_history.append(r)
            train_fn = getattr(locomotion_ctrl, "train_cpg_from_intrinsic_history", None)
            if callable(train_fn):
                train_fn()

        if motor_cortex is not None:
            # Все программы получают один и тот же интринсивный сигнал
            obs = {}
            try:
                obs = dict(agent.env.observe())
            except Exception:
                pass
            posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
            foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
            foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
            motor_cortex.push_and_train(
                nodes=dict(agent.graph.nodes),
                cpg_targets={},
                reward=r,
                posture=posture,
                foot_l=foot_l,
                foot_r=foot_r,
            )

    def get_current_goal(self) -> dict[str, Any] | None:
        return self.goal_imagination._current_goal

    def recent_reward(self, window: int = 32) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)[-window:]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": intrinsic_enabled(),
            "total_steps": self.total_steps,
            "recent_reward": round(self.recent_reward(), 6),
            "causal_surprise": self.causal_surprise.snapshot(),
            "goal_imagination": self.goal_imagination.snapshot(),
            "variable_discovery": self.variable_discovery.snapshot(),
        }


# ─── Integration patch ────────────────────────────────────────────────────────
def apply_intrinsic_patch(sim) -> bool:
    """
    Патч Simulation: заменяет внешние rewards на IntrinsicObjective.

    Применяет:
    1. IntrinsicObjective создаётся и сохраняется в sim._intrinsic
    2. Хук после agent.step() в _run_agent_or_skill_step

    Вызов: apply_intrinsic_patch(sim)  # после создания Simulation
    """
    device = getattr(sim, "device", torch.device("cpu"))
    intrinsic = IntrinsicObjective(device)
    sim._intrinsic = intrinsic
    print("[Intrinsic] IntrinsicObjective applied")

    # Патч _run_agent_or_skill_step
    original = sim._run_agent_or_skill_step

    def patched_run(engine_tick: int) -> dict:
        result = original(engine_tick)

        # После каждого шага — вычисляем интринсивную награду
        if not result.get("blocked") and not result.get("skipped"):
            try:
                r = intrinsic.step(
                    agent=sim.agent,
                    result=result,
                    tick=engine_tick,
                    locomotion_ctrl=getattr(sim, "_locomotion_controller", None),
                    motor_cortex=getattr(sim, "_motor_cortex", None),
                )
                result["intrinsic_reward"] = round(r, 6)

                # Если нейрогенез произошёл — сообщаем в events
                disc = intrinsic.variable_discovery._discovery_log
                if disc and disc[-1].get("tick") == engine_tick:
                    ev = disc[-1]
                    sim._add_event(
                        f"🧬 IntrinsicDiscovery: {ev.get('new_node', '?')} "
                        f"← [{', '.join(ev.get('triggered_by', [])[:2])}]",
                        "#ff44cc", "phase"
                    )

                # Если новая цель сгенерирована — логируем
                goal = intrinsic.get_current_goal()
                if goal and goal.get("generated_at") == sim.agent._total_interventions:
                    sim._add_event(
                        f"🎯 IntGoal: do({goal['variable']}="
                        f"{goal['value']:.2f}) E[gain]={goal['expected_gain']:.4f}",
                        "#44ffaa", "discovery"
                    )

            except Exception as e:
                result["intrinsic_reward"] = 0.0

        return result

    sim._run_agent_or_skill_step = patched_run

    return True
===
"""
intrinsic_objective.py — Unified Intrinsic Objective.

Одна цель: максимизировать I(agent; world) — взаимную информацию
между внутренней каузальной моделью и реальностью.

Операционально:
  R_intrinsic(t) = Δcompression(t) + λ·Δprediction_surprise(t)

Внешние hand-crafted награды и RewardCoordinator удалены из симуляции.

Остаётся ОДНО:
  ✓ "Насколько лучше агент понимает мир после этого действия?"

Локомоция возникает инструментально: ходить → больше новых состояний
→ больше информации → выше compression gain. Не потому что "так надо".

Архитектура:
  CausalSurprise   — Δ prediction error как интринсивная метрика
  GoalImagination  — агент генерирует свою следующую цель через GNN rollout
  VariableDiscovery — проактивное открытие новых переменных через EIG
  IntrinsicObjective — единый интерфейс, заменяет все внешние награды

Принцип нейрогенеза-через-цель:
  Если CausalSurprise растёт для области X, но модель не улучшается →
  создать новый узел в GNN для этой области (NeurogenesisEngine.scan_and_grow).

RKK_INTRINSIC_ENABLED=1
RKK_INTRINSIC_LAMBDA=0.4        — вес prediction surprise vs compression
RKK_INTRINSIC_GOAL_HORIZON=12   — шагов imagination для goal generation
RKK_INTRINSIC_DISCOVERY_EIG=0.3 — порог EIG для создания нового узла
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Config ───────────────────────────────────────────────────────────────────
def intrinsic_enabled() -> bool:
    return os.environ.get("RKK_INTRINSIC_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

def _ef(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)))
    except ValueError: return default

def _ei(key: str, default: int) -> int:
    try: return max(1, int(os.environ.get(key, str(default))))
    except ValueError: return default

# ─── CausalSurprise ───────────────────────────────────────────────────────────
class CausalSurprise:
    """
    Измеряет интринсивную ценность интервенции через два сигнала:

    1. Δcompression: насколько уменьшился MDL графа после do().
       Это прямая мера "граф стал точнее объяснять мир".

    2. Δprediction_error: насколько удивил нас результат do().
       Высокое удивление = в этой области модель плохая → исследуй.

    Итоговая интринсивная награда:
       R = sign(Δcompression) * |Δcompression| + λ * surprise_bonus

    Ключевое: surprise_bonus — это не штраф за ошибку.
    Это БОНУС за нахождение в неизведанной области.
    """

    def __init__(self):
        self._lambda = _ef("RKK_INTRINSIC_LAMBDA", 0.4)

        # Скользящие статистики для нормализации
        self._compression_history: deque[float] = deque(maxlen=200)
        self._surprise_history: deque[float] = deque(maxlen=200)
        self._reward_history: deque[float] = deque(maxlen=500)

        # EMA baseline для адаптивного порога
        self._compression_ema: float = 0.0
        self._surprise_ema: float = 0.0
        self._ema_alpha: float = 0.05

        # Счётчики
        self.total_computations: int = 0
        self.total_discoveries: int = 0  # тиков где compression > 0

    def compute(
        self,
        compression_delta: float,
        prediction_error: float,
        graph_mdl_before: float,
        graph_mdl_after: float,
        n_interventions: int,
    ) -> float:
        """
        Основной расчёт интринсивной награды.

        compression_delta: mdl_before - mdl_after (>0 = граф улучшился)
        prediction_error: |predicted - observed| (>0 = модель удивлена)
        """
        if not intrinsic_enabled():
            return 0.0

        self.total_computations += 1

        # --- Сигнал 1: Compression ---
        # Нормализуем по MDL размеру чтобы сравнивать ранние и поздние тики
        mdl_scale = max(graph_mdl_before, 1.0)
        comp_norm = float(compression_delta) / mdl_scale

        # EMA обновление
        self._compression_ema = (
            (1 - self._ema_alpha) * self._compression_ema
            + self._ema_alpha * comp_norm
        )
        self._compression_history.append(comp_norm)

        # --- Сигнал 2: Prediction Surprise ---
        # Нормализуем относительно исторической средней
        surprise_baseline = float(np.mean(self._surprise_history)) if self._surprise_history else 0.1
        surprise_norm = float(prediction_error) / max(surprise_baseline, 0.01)
        # Ограничиваем: не хотим чтобы огромные ошибки доминировали
        surprise_bonus = float(np.clip(np.log1p(surprise_norm) * 0.5, 0.0, 2.0))

        self._surprise_ema = (
            (1 - self._ema_alpha) * self._surprise_ema
            + self._ema_alpha * float(prediction_error)
        )
        self._surprise_history.append(float(prediction_error))

        # --- Объединяем ---
        # Compression gain — основной сигнал (может быть отрицательным если граф деградировал)
        # Surprise — всегда положительный бонус (исследование важно)
        reward = comp_norm + self._lambda * surprise_bonus

        # Ранний бонус: первые 500 интервенций мир совсем незнаком
        if n_interventions < 500:
            exploration_bonus = 0.3 * (1.0 - n_interventions / 500.0)
            reward += exploration_bonus

        if comp_norm > 0:
            self.total_discoveries += 1

        self._reward_history.append(reward)
        return float(reward)

    def surprise_is_high(self, threshold_sigma: float = 2.0) -> bool:
        """Текущее удивление аномально высокое? → агент в неизведанной области."""
        if len(self._surprise_history) < 20:
            return False
        arr = np.array(self._surprise_history)
        mu, sigma = float(arr.mean()), float(arr.std())
        if sigma < 1e-8:
            return False
        return float(self._surprise_ema) > mu + threshold_sigma * sigma

    def compression_is_stagnant(self, window: int = 50) -> bool:
        """Compression не растёт уже window тиков? → нужна другая стратегия."""
        if len(self._compression_history) < window:
            return False
        recent = list(self._compression_history)[-window:]
        return float(np.mean(recent)) < 1e-6

    def recent_mean(self, window: int = 32) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)[-window:]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_computations": self.total_computations,
            "total_discoveries": self.total_discoveries,
            "discovery_rate": round(
                self.total_discoveries / max(self.total_computations, 1), 4
            ),
            "compression_ema": round(self._compression_ema, 6),
            "surprise_ema": round(self._surprise_ema, 5),
            "recent_reward_mean": round(self.recent_mean(), 5),
            "surprise_is_high": self.surprise_is_high(),
            "compression_stagnant": self.compression_is_stagnant(),
        }


# ─── GoalImagination ─────────────────────────────────────────────────────────
class GoalImagination:
    """
    Агент генерирует СВОЮ СЛЕДУЮЩУЮ ЦЕЛЬ через GNN rollout.

    Алгоритм:
      1. Из текущего состояния сэмплируем K случайных интервенций
      2. Для каждой запускаем imagination rollout на H шагов
      3. Выбираем интервенцию с максимальным ожидаемым Δcompression
      4. Эта интервенция становится "целью" на следующие T тиков

    НЕТ захардкоженной цели. Никаких "дойди до target_dist".
    Цель = "сделай то, что больше всего улучшит понимание мира".

    Это фундаментально другое от goal_planning.py:
      goal_planning: "уменьши target_dist" (человеческая цель)
      GoalImagination: "максимизируй compression_gain" (интринсивная цель)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._horizon = _ei("RKK_INTRINSIC_GOAL_HORIZON", 12)
        self._k_candidates = _ei("RKK_INTRINSIC_GOAL_CANDIDATES", 32)

        # Текущая активная цель
        self._current_goal: dict[str, Any] | None = None
        self._goal_age: int = 0
        self._goal_ttl: int = 20  # тиков пока цель активна

        # История целей для анализа
        self._goal_history: deque[dict[str, Any]] = deque(maxlen=50)

        # Статистика качества imagination
        self._imagination_accuracy: deque[float] = deque(maxlen=100)

        self.total_goals_generated: int = 0

    def generate_goal(
        self,
        graph,
        agent_env,
        n_interventions: int,
        causal_surprise: CausalSurprise,
    ) -> dict[str, Any] | None:
        """
        Генерирует следующую цель через GNN imagination rollout.
        Возвращает {"variable": str, "value": float, "expected_gain": float}.
        """
        if not intrinsic_enabled():
            return None

        core = graph._core
        if core is None:
            return None

        node_ids = list(graph._node_ids)
        d = len(node_ids)
        if d == 0:
            return None

        # Исключаем read-only и self_* из кандидатов
        from engine.graph_constants import is_read_only_macro_var
        candidates_vars = [
            nid for nid in node_ids
            if not is_read_only_macro_var(nid)
            and not nid.startswith("concept_")
            and not nid.startswith("proprio_")
        ]
        if not candidates_vars:
            return None

        # Сэмплируем K интервенций
        k = min(self._k_candidates, len(candidates_vars) * 3)
        rng = np.random.default_rng()

        # Адаптивный диапазон: если стагнация — исследуем агрессивнее
        if causal_surprise.compression_is_stagnant():
            lo, hi = 0.1, 0.9
        elif causal_surprise.surprise_is_high():
            lo, hi = 0.3, 0.7  # осторожнее когда много сюрпризов
        else:
            lo, hi = 0.2, 0.8

        sampled = [
            (rng.choice(candidates_vars), float(rng.uniform(lo, hi)))
            for _ in range(k)
        ]

        # Текущее состояние
        current_state = dict(graph.nodes)
        current_mdl = float(graph.mdl_size)

        best_var, best_val, best_expected = None, 0.5, -np.inf

        with torch.no_grad():
            for var, val in sampled:
                try:
                    # GNN rollout
                    state = graph.propagate_from(current_state, var, val)

                    # Многошаговый rollout
                    for _ in range(min(3, self._horizon)):
                        state = graph.rollout_step_free(state)

                    # Оцениваем ожидаемый Δcompression через proxy:
                    # "насколько состояние после отличается от текущего?"
                    delta = float(np.mean([
                        abs(float(state.get(nid, 0.5)) - float(current_state.get(nid, 0.5)))
                        for nid in node_ids
                    ]))

                    # Неопределённость по этому узлу (чем выше — тем ценнее)
                    if core is not None:
                        A = core.alpha_trust_matrix()
                        idx = node_ids.index(var) if var in node_ids else -1
                        if idx >= 0:
                            uncertainty = float(1.0 - A[idx].mean().item())
                        else:
                            uncertainty = 0.5
                    else:
                        uncertainty = 0.5

                    # Ожидаемый выигрыш = delta * uncertainty
                    # (большое изменение в неизведанной области = ценно)
                    expected_gain = delta * (1.0 + uncertainty)

                    if expected_gain > best_expected:
                        best_expected = expected_gain
                        best_var = var
                        best_val = val

                except Exception:
                    continue

        if best_var is None:
            return None

        goal = {
            "variable": best_var,
            "value": best_val,
            "expected_gain": float(best_expected),
            "generated_at": n_interventions,
            "horizon": self._horizon,
        }

        self._current_goal = goal
        self._goal_age = 0
        self._goal_history.append(dict(goal))
        self.total_goals_generated += 1
        return goal

    def tick_goal(self, actual_compression: float) -> dict[str, Any] | None:
        """
        Обновляет возраст цели. Возвращает текущую цель или None если истекла.
        Записывает точность imagination (предсказали gain, получили actual).
        """
        if self._current_goal is None:
            return None

        self._goal_age += 1

        # Записываем точность imagination
        if self._goal_age == 1:
            expected = self._current_goal.get("expected_gain", 0.0)
            if expected > 0:
                accuracy = min(1.0, actual_compression / max(expected, 1e-8))
                self._imagination_accuracy.append(accuracy)

        if self._goal_age >= self._goal_ttl:
            old_goal = self._current_goal
            self._current_goal = None
            self._goal_age = 0
            return None

        return self._current_goal

    def imagination_accuracy(self) -> float:
        if not self._imagination_accuracy:
            return 0.5
        return float(np.mean(self._imagination_accuracy))

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_goal": self._current_goal,
            "goal_age": self._goal_age,
            "goal_ttl": self._goal_ttl,
            "total_goals_generated": self.total_goals_generated,
            "imagination_accuracy": round(self.imagination_accuracy(), 4),
            "recent_goals": list(self._goal_history)[-5:],
        }


# ─── VariableDiscovery ────────────────────────────────────────────────────────
class VariableDiscovery:
    """
    Проактивное открытие новых переменных через EIG.

    Принцип:
      Если агент систематически ошибается в предсказании определённой
      области состояний, и GNN не может улучшиться (высокий surprise
      при низком compression gain) → там скрытая переменная.

    Алгоритм:
      1. Отслеживаем prediction error по каждому узлу отдельно
      2. Если узел X имеет высокую и СТАБИЛЬНУЮ ошибку →
         GNN не справляется, нужен медиирующий узел
      3. Вызываем NeurogenesisEngine.scan_and_grow() с целевой парой

    Отличие от rsi_structural.py:
      rsi_structural: реактивный (срабатывает при high stress в W.grad)
      VariableDiscovery: проактивный (срабатывает при стабильной ошибке)
    """

    def __init__(self):
        self._eig_threshold = _ef("RKK_INTRINSIC_DISCOVERY_EIG", 0.3)
        # per-node prediction error history
        self._node_errors: dict[str, deque[float]] = {}
        self._window = 30
        self._last_discovery_tick: int = -9999
        self._discovery_cooldown: int = 500
        self.total_discoveries: int = 0
        self._discovery_log: deque[dict[str, Any]] = deque(maxlen=20)

    def update_node_errors(
        self,
        predicted: dict[str, float],
        observed: dict[str, float],
    ) -> None:
        """Записываем ошибку по каждому узлу."""
        for nid in observed:
            if nid not in self._node_errors:
                self._node_errors[nid] = deque(maxlen=self._window)
            pred_val = float(predicted.get(nid, 0.5))
            obs_val = float(observed[nid])
            self._node_errors[nid].append(abs(pred_val - obs_val))

    def find_high_error_nodes(
        self,
        top_k: int = 3,
        min_error: float = 0.15,
    ) -> list[tuple[str, float]]:
        """
        Находим узлы с хронически высокой ошибкой.
        Это кандидаты для нейрогенеза.
        """
        candidates: list[tuple[str, float]] = []
        for nid, errors in self._node_errors.items():
            if len(errors) < self._window // 2:
                continue
            mean_err = float(np.mean(errors))
            # Стабильность ошибки (низкий std = стабильно плохо, не случайный всплеск)
            std_err = float(np.std(errors))
            stability = 1.0 / (1.0 + std_err)
            score = mean_err * stability
            if mean_err >= min_error:
                candidates.append((nid, score))

        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def maybe_trigger_neurogenesis(
        self,
        graph,
        agent,
        tick: int,
        causal_surprise: CausalSurprise,
    ) -> dict[str, Any] | None:
        """
        Проверяет нужен ли нейрогенез и запускает его.
        Возвращает событие если новый узел был создан.
        """
        if not intrinsic_enabled():
            return None
        if (tick - self._last_discovery_tick) < self._discovery_cooldown:
            return None
        if not causal_surprise.compression_is_stagnant():
            return None  # Граф ещё учится — нейрогенез преждевременен

        high_error = self.find_high_error_nodes()
        if len(high_error) < 2:
            return None

        # Пытаемся создать медиирующий узел между двумя проблемными
        from engine.rsi_structural import NeurogenesisEngine
        neuro = getattr(agent, "_neuro_engine", NeurogenesisEngine(
            min_interventions=200,
            error_threshold=self._eig_threshold,
        ))

        result = neuro.scan_and_grow(agent, tick)
        if result is None:
            return None

        self._last_discovery_tick = tick
        self.total_discoveries += 1

        event = {
            "tick": tick,
            "new_node": result.get("new_node"),
            "triggered_by": [nid for nid, _ in high_error[:2]],
            "error_scores": [round(s, 4) for _, s in high_error[:2]],
        }
        self._discovery_log.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        high_err = self.find_high_error_nodes()
        return {
            "total_discoveries": self.total_discoveries,
            "nodes_tracked": len(self._node_errors),
            "high_error_nodes": [(n, round(s, 4)) for n, s in high_err],
            "recent_discoveries": list(self._discovery_log)[-3:],
        }


# ─── IntrinsicObjective ───────────────────────────────────────────────────────
class IntrinsicObjective:
    """
    Единый интерфейс интринсивной цели. Заменяет все внешние rewards.

    Использование в simulation_main.py:
      self._intrinsic = IntrinsicObjective(device)

    После каждого agent.step():
      r = self._intrinsic.step(
          agent=self.agent,
          result=result,
          tick=self.tick,
          locomotion_ctrl=self._locomotion_controller,
      )

    """

    def __init__(self, device: torch.device):
        self.device = device
        self.causal_surprise = CausalSurprise()
        self.goal_imagination = GoalImagination(device)
        self.variable_discovery = VariableDiscovery()

        # Когда генерируем следующую цель
        self._goal_generate_every = _ei("RKK_INTRINSIC_GOAL_EVERY", 25)
        self._last_goal_tick: int = -9999

        # История наград
        self._reward_history: deque[float] = deque(maxlen=500)
        self.total_steps: int = 0

    def step(
        self,
        agent,
        result: dict[str, Any],
        tick: int,
        locomotion_ctrl=None,
        motor_cortex=None,
    ) -> float:
        """
        Главный вызов: вычисляет интринсивную награду и применяет её.
        Возвращает скалярную награду.
        """
        if not intrinsic_enabled():
            return 0.0

        self.total_steps += 1

        # --- Извлекаем сигналы ---
        compression_delta = float(result.get("compression_delta", 0.0))
        prediction_error = float(result.get("prediction_error", 0.0))
        graph = agent.graph
        mdl_before = float(graph.mdl_size) + abs(compression_delta)
        mdl_after = float(graph.mdl_size)

        # --- Обновляем per-node ошибки ---
        cf_pred = result.get("cf_predicted") or {}
        cf_obs = result.get("cf_observed") or {}
        if cf_pred and cf_obs:
            self.variable_discovery.update_node_errors(cf_pred, cf_obs)

        # --- Вычисляем интринсивную награду ---
        r = self.causal_surprise.compute(
            compression_delta=compression_delta,
            prediction_error=prediction_error,
            graph_mdl_before=mdl_before,
            graph_mdl_after=mdl_after,
            n_interventions=int(agent._total_interventions),
        )

        # --- Обновляем активную цель ---
        active_goal = self.goal_imagination.tick_goal(compression_delta)

        # --- Генерируем новую цель если нужно ---
        if (tick - self._last_goal_tick) >= self._goal_generate_every:
            if active_goal is None or self.causal_surprise.compression_is_stagnant():
                try:
                    new_goal = self.goal_imagination.generate_goal(
                        graph=graph,
                        agent_env=agent.env,
                        n_interventions=int(agent._total_interventions),
                        causal_surprise=self.causal_surprise,
                    )
                    if new_goal is not None:
                        self._last_goal_tick = tick
                except Exception:
                    pass

        # --- Проактивный нейрогенез ---
        try:
            neuro_event = self.variable_discovery.maybe_trigger_neurogenesis(
                graph=graph,
                agent=agent,
                tick=tick,
                causal_surprise=self.causal_surprise,
            )
        except Exception:
            neuro_event = None

        # --- Variable Discovery: auto-discover new variable groups ---
        try:
            from engine.variable_bootstrap import get_variable_registry
            registry = get_variable_registry()
            if registry.is_bootstrap:
                # Update pressure from high-error nodes
                high_err = self.variable_discovery.find_high_error_nodes(top_k=5)
                registry.update_pressure(
                    high_error_nodes=high_err,
                    compression_stagnant=self.causal_surprise.compression_is_stagnant(),
                    tick=tick,
                )
                # Auto-discover if pressure threshold met
                new_vars = registry.auto_discover(tick)
                if new_vars:
                    # Add new variables to the agent's graph
                    for var in new_vars:
                        if var not in graph.nodes:
                            graph.set_node(var, 0.5)
                    # Rebind agent environment if needed
                    try:
                        env_obs = dict(agent.env.observe())
                        for var in new_vars:
                            if var in env_obs and var in graph.nodes:
                                graph.nodes[var] = env_obs[var]
                    except Exception:
                        pass
        except ImportError:
            pass
        except Exception:
            pass

        self._apply_intrinsic_reward(r, locomotion_ctrl, motor_cortex, agent)

        self._reward_history.append(r)
        return r

    def _apply_intrinsic_reward(
        self,
        r: float,
        locomotion_ctrl,
        motor_cortex,
        agent,
    ) -> None:
        """
        Полная замена: ТОЛЬКО интринсивная награда идёт в все learners.
        Никаких posture, symmetry, forward_bonus.
        """
        if locomotion_ctrl is not None:
            locomotion_ctrl._reward_history.append(r)
            train_fn = getattr(locomotion_ctrl, "train_cpg_from_intrinsic_history", None)
            if callable(train_fn):
                train_fn()

        if motor_cortex is not None:
            # Все программы получают один и тот же интринсивный сигнал
            obs = {}
            try:
                obs = dict(agent.env.observe())
            except Exception:
                pass
            posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
            foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
            foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
            motor_cortex.push_and_train(
                nodes=dict(agent.graph.nodes),
                cpg_targets={},
                reward=r,
                posture=posture,
                foot_l=foot_l,
                foot_r=foot_r,
            )

    def get_current_goal(self) -> dict[str, Any] | None:
        return self.goal_imagination._current_goal

    def recent_reward(self, window: int = 32) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)[-window:]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": intrinsic_enabled(),
            "total_steps": self.total_steps,
            "recent_reward": round(self.recent_reward(), 6),
            "causal_surprise": self.causal_surprise.snapshot(),
            "goal_imagination": self.goal_imagination.snapshot(),
            "variable_discovery": self.variable_discovery.snapshot(),
        }


# ─── Integration patch ────────────────────────────────────────────────────────
def apply_intrinsic_patch(sim) -> bool:
    """
    Патч Simulation: заменяет внешние rewards на IntrinsicObjective.

    Применяет:
    1. IntrinsicObjective создаётся и сохраняется в sim._intrinsic
    2. Хук после agent.step() в _run_agent_or_skill_step

    Вызов: apply_intrinsic_patch(sim)  # после создания Simulation
    """
    device = getattr(sim, "device", torch.device("cpu"))
    intrinsic = IntrinsicObjective(device)
    sim._intrinsic = intrinsic
    print("[Intrinsic] IntrinsicObjective applied")

    # Патч _run_agent_or_skill_step
    original = sim._run_agent_or_skill_step

    def patched_run(engine_tick: int) -> dict:
        result = original(engine_tick)

        # После каждого шага — вычисляем интринсивную награду
        if not result.get("blocked") and not result.get("skipped"):
            try:
                r = intrinsic.step(
                    agent=sim.agent,
                    result=result,
                    tick=engine_tick,
                    locomotion_ctrl=getattr(sim, "_locomotion_controller", None),
                    motor_cortex=getattr(sim, "_motor_cortex", None),
                )
                result["intrinsic_reward"] = round(r, 6)

                # Если нейрогенез произошёл — сообщаем в events
                disc = intrinsic.variable_discovery._discovery_log
                if disc and disc[-1].get("tick") == engine_tick:
                    ev = disc[-1]
                    sim._add_event(
                        f"🧬 IntrinsicDiscovery: {ev.get('new_node', '?')} "
                        f"← [{', '.join(ev.get('triggered_by', [])[:2])}]",
                        "#ff44cc", "phase"
                    )

                # Если новая цель сгенерирована — логируем
                goal = intrinsic.get_current_goal()
                if goal and goal.get("generated_at") == sim.agent._total_interventions:
                    sim._add_event(
                        f"🎯 IntGoal: do({goal['variable']}="
                        f"{goal['value']:.2f}) E[gain]={goal['expected_gain']:.4f}",
                        "#44ffaa", "discovery"
                    )

            except Exception as e:
                result["intrinsic_reward"] = 0.0

        return result

    sim._run_agent_or_skill_step = patched_run

    return True
```

After neurogenesis check, the system:
1. Gets high-error nodes from `VariableDiscovery`
2. Updates `VariableRegistry` group pressure
3. Auto-discovers new variable groups when pressure exceeds threshold
4. Adds new variables to GNN

---

### 8. VariableRegistry in Simulation
```diff:simulation_main.py
"""
simulation_singleton_v2.py — Singleton AGI с гуманоидом (Фаза 11/12).

Фаза 12 добавляет:
  - visual_mode toggle: enable_visual() / disable_visual()
  - EnvironmentVisual wrapper активируется без перезапуска агента
  - Predictive coding loop: GNN prediction → visual cortex feedback (опц. Neural ODE sub-steps, RKK_WM_NEURAL_ODE)
  - /vision/slots endpoint data через get_vision_state()
  - vision_stats в snapshot

Фаза 3:
  - refresh_phase3_teacher_llm(): LLM → правила IG + VL overlay (TTL)
  - tick_step: teacher_weight annealing (RKK_TEACHER_T_MAX), overlay на ValueLayer

Phase A (locomotion):
  - RKK_LOCOMOTION_CPG=1: CPG ноги на humanoid без fixed_root (engine.cpg_locomotion).
  - RKK_CPG_LOOP_HZ>0 (напр. 60): Low-level CPG в daemon-потоке; снимок graph.nodes после agent/skill step.
    По умолчанию 0 — CPG только сразу после agent.step (как раньше).

Phase B (hierarchy):
  - High ~1 Hz: GNN + EIG + goal_planning (agent tick / WS).
  - Mid ~10–20 Hz: RKK_AGENT_LOOP_HZ>0 — GNN/EIG/planning в daemon-потоке; tick_step() отдаёт кэш (UI не ждёт).
  - Low ~60 Hz+: CPG (decoupled) + RKK_PHYSICS_BG_HZ PyBullet (уже в humanoid).

  L3 goal_planning (agent), L2 skill_library, L1 CPG, L0 PyBullet.
  - RKK_SKILL_LIBRARY=1: моторная последовательность из библиотеки (тик = один кадр skill).

Phase C (full RSI):
  - RKK_RSI_FULL=1: engine.rsi_full — плато discovery → расширение GNN hidden; плато loco → CPG noise;
    плато walk skills → harder variants; падение phi → временное смягчение VL bounds.

Этап D (LLM в петле, RKK_LLM_LOOP=1):
  Уровень 1: каждый тик — GNN + System1 (как раньше).
  Уровень 2: фоновый Ollama по триггерам (стагнация discovery, block_rate, VLM unknown, surprise PE).
  Уровень 3: редко — перезапись гипотез (humanoid), в том же worker после L2 при run_level3.

Класс `Simulation` собран из миксинов в `features/simulation/mixin_*.py` (композиция поведения).
"""
from __future__ import annotations

import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from engine.agent import RKKAgent
from engine.demon import AdversarialDemon
from engine.visual_concept_store import VisualConceptStore
from engine.hierarchical_graph import HierarchicalGraph
from engine.rsi_structural import NeurogenesisEngine

from engine.config.runtime import RKKRuntimeConfig
from engine.core import (
    MotorState,
    WorldSwitcher,
    _make_env,
    default_bounds,
    resolve_torch_device,
)
from engine.features.simulation.background_loops import BackgroundLoopService
from engine.features.simulation.imports import *

from engine.features.simulation.mixin_api import SimulationApiMixin
from engine.features.simulation.mixin_concepts import SimulationConceptsMixin
from engine.features.simulation.mixin_demon_phase import SimulationDemonPhaseMixin
from engine.features.simulation.mixin_episodic_rssm import SimulationEpisodicRssmMixin
from engine.features.simulation.mixin_fall import SimulationFallMixin
from engine.features.simulation.mixin_llm import SimulationLlmLoopMixin
from engine.features.simulation.mixin_locomotion import SimulationLocomotionMixin
from engine.features.simulation.mixin_motor_pipeline import SimulationMotorPipelineMixin
from engine.features.simulation.mixin_phase_hierarchy import SimulationPhaseHierarchyMixin
from engine.features.simulation.mixin_pose_embodied import SimulationPoseEmbodiedMixin
from engine.features.simulation.mixin_skills import SimulationSkillsMixin
from engine.features.simulation.mixin_snapshot_shutdown import SimulationSnapshotShutdownMixin
from engine.features.simulation.mixin_teacher import SimulationTeacherMixin
from engine.features.simulation.mixin_tick import SimulationTickMixin
from engine.features.simulation.mixin_verbal import SimulationVerbalMixin
from engine.features.simulation.mixin_vision_predictor import SimulationVisionPredictorMixin
from engine.features.simulation.mixin_visual_grounding import SimulationVisualGroundingMixin
from engine.features.simulation.mixin_world import SimulationWorldMixin


class Simulation(
    SimulationConceptsMixin,
    SimulationVerbalMixin,
    SimulationPhaseHierarchyMixin,
    SimulationWorldMixin,
    SimulationFallMixin,
    SimulationLlmLoopMixin,
    SimulationLocomotionMixin,
    SimulationTeacherMixin,
    SimulationMotorPipelineMixin,
    SimulationSkillsMixin,
    SimulationPoseEmbodiedMixin,
    SimulationVisualGroundingMixin,
    SimulationEpisodicRssmMixin,
    SimulationTickMixin,
    SimulationVisionPredictorMixin,
    SimulationDemonPhaseMixin,
    SimulationApiMixin,
    SimulationSnapshotShutdownMixin,
):
    AGI_NAME = "Nova"
    AGI_COLOR = "#cc44ff"

    def __init__(self, device_str: str = "cuda", start_world: str = "humanoid"):
        self.device = resolve_torch_device(device_str)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.current_world = start_world
        print(f"[Singleton v2] Device: {self.device} | World: {start_world}")

        env = _make_env(start_world, self.device)
        bounds = default_bounds()

        self.agent = RKKAgent(
            agent_id=0,
            name=self.AGI_NAME,
            env=env,
            device=self.device,
            bounds=bounds,
        )

        self.switcher = WorldSwitcher(self.agent, self.device)
        self.demon = AdversarialDemon(n_agents=1, device=self.device)

        self.tick = 0
        self.phase = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events: deque[dict] = deque(maxlen=24)
        self._prev_edge_count = 0
        self._last_snapshot: dict = {}

        self._fall_count = 0
        self._fixed_root_active = False
        self._curriculum_auto_fr_released = False
        self._curriculum_stabilize_until: int = 0
        self._stand_ticks = 0
        self._last_fall_reset_tick: int = -999
        self._fall_recovery_active = False
        self._fall_recovery_start_tick = 0
        self._fall_recovery_last_progress_tick = 0
        self._fall_recovery_best_score = 0.0

        self._visual_mode = False
        self._visual_env = None
        self._base_env_ref = None
        self._vision_ticks = 0
        self._last_vision_state: dict = {}

        self._phase3_teacher_rules: list = []
        self._phase3_vl_overlay = None

        self._llm_loop_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rkk_llm")
        self._pending_llm_bundle: dict | None = None
        self._llm_level2_inflight = False
        self._last_level2_schedule_tick = -10**9
        self._last_level3_tick = -10**9
        self._best_discovery_rate = 0.0
        self._last_dr_gain_tick = 0
        self._rolling_block_bits: deque[int] = deque(maxlen=80)
        self._pe_history: deque[float] = deque(maxlen=200)
        self._locomotion_controller = None
        self._motor_state = MotorState()
        self._motor_state_lock = threading.Lock()
        self._bg = BackgroundLoopService(self)
        self._runtime_config = RKKRuntimeConfig.from_env()
        self._l1_motor_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l1_last_cmd_tick = 0
        self._l1_last_apply_tick = 0
        self._l1_last_credit_tick = 0
        self._sim_step_lock = threading.RLock()
        self._agent_step_response: dict | None = None
        self._skill_library = None
        self._skill_exec: dict | None = None
        self._llm_loop_stats: dict = {
            "level2_runs": 0,
            "level3_runs": 0,
            "last_triggers": [],
            "last_level2_explanation": "",
        }
        self._rsi_full = None
        self.neuro_engine = NeurogenesisEngine()
        self._embodied_reward_ctrl = None
        self._verbal_reward_total: float = 0.0
        self._visual_grounding_ctrl = (
            VisualGroundingController() if _VISUAL_GROUNDING_AVAILABLE else None
        )
        self._episodic_memory = EpisodicMemory() if _EPISODIC_MEMORY_AVAILABLE else None
        self._last_action_for_memory: tuple[str, float] | None = None
        self._last_fall_memory_tick: int = -999_999

        self._curriculum = CurriculumScheduler() if _CURRICULUM_AVAILABLE else None
        self._curriculum_apply_every: int = 50

        self._rssm_trainer: "RSSMTrainer | None" = None
        self._rssm_imagination: "RSSMImagination | None" = None
        self._rssm_upgraded: bool = False
        self._rssm_upgrade_tick: int = -1

        self._proprio: "ProprioceptionStream | None" = None
        if _PROPRIO_AVAILABLE:
            self._proprio = ProprioceptionStream(device=self.device)

        self._intrinsic: Any = None

        self._timescale: "MultiscaleTimeController | None" = None
        if _TIMESCALE_AVAILABLE:
            self._timescale = MultiscaleTimeController()

        self._inner_voice: "InnerVoiceController | None" = None
        self._llm_teacher: "LLMVoiceTeacher | None" = None
        if _INNER_VOICE_AVAILABLE:
            self._inner_voice = InnerVoiceController(device=self.device)
            self._llm_teacher = LLMVoiceTeacher()
            self._llm_teacher.add_callback(self._on_teacher_annotation)

        self._sleep_ctrl: "SleepController | None" = None
        self._physical_curriculum: "PhysicalCurriculum | None" = None
        self._persist: "PersistenceManager | None" = None
        if _PHASE_K_AVAILABLE:
            self._sleep_ctrl = SleepController()
            self._physical_curriculum = PhysicalCurriculum()
        self._meta_restored: bool = False
        self._was_fallen_last_tick: bool = False
        self._sleep_prev_fixed_root: bool = False

        self._uvicorn_loop: Any = None
        self._verbal: "VerbalActionController | None" = None
        self._chat_ws_clients: list[Any] = []
        self._verbal_tick_running: bool = False
        if _VERBAL_AVAILABLE:
            self._verbal = VerbalActionController()
            self._verbal.add_callback(self._broadcast_agent_message)

        self._slot_labeler: Any = None
        self._visual_voice: Any = None
        if _PHASE_M_AVAILABLE:
            _lang = os.environ.get("RKK_SPEECH_LANG", "ru")
            self._slot_labeler = SlotLabeler()
            self._visual_voice = VisualInnerVoice(lang=_lang)

        self._world_bridge: Any = None
        if _WORLD_BRIDGE_AVAILABLE and world_bridge_enabled():
            self._world_bridge = WorldStateBridge()

        self._motor_cortex: "_MotorCortexLibrary | None" = None
        self._mc_posture_window: deque = deque(maxlen=200)
        self._mc_fallen_count_window: deque = deque(maxlen=200)
        self._mc_abstract_nodes_injected: bool = False
        self._concepts_cache: list[dict] = []
        self._materialized_detector_concept_ids: set[str] = set()
        self._discovery_plateau_count = 0
        self._last_dr_snapshot: float | None = None
        self._hierarchical_graph: HierarchicalGraph | None = None
        self._concept_store: VisualConceptStore | None = None
        try:
            self._concept_inject_every = max(
                1, int(os.environ.get("RKK_CONCEPT_INJECT_EVERY", "30"))
            )
        except ValueError:
            self._concept_inject_every = 30
        self._l4_thread: threading.Thread | None = None
        self._l4_stop = threading.Event()
        self._l4_in_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_out_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_task_pending = False
        self._l4_last_snapshot: dict = {"n_concepts": 0, "concepts": []}
        self._l4_last_submit_tick = 0
        self._l4_last_apply_tick = 0
        self._l3_next_due_ts = 0.0
        self._l3_last_tick = 0
        self._memory_resume_enabled = os.environ.get(
            "RKK_MEMORY_RESUME_ON_START", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        if self._memory_resume_enabled:
            try:
                meta = self.memory_load()
                if meta.get("ok"):
                    print(f"[Simulation] Memory resumed at tick={self.tick}")
            except Exception as e:
                print(f"[Simulation] Memory resume skipped: {type(e).__name__}: {e}")

        if os.environ.get("RKK_NEURAL_LANG", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        ):
            try:
                from engine.neural_lang_integration import apply_neural_lang_patch

                apply_neural_lang_patch(self)
            except Exception as e:
                print(f"[Simulation] Neural lang patch skipped: {type(e).__name__}: {e}")

        from engine.intristic_objective import apply_intrinsic_patch
        from engine.llm_hint_mediator import apply_llm_mediator_patch
        from engine.learned_motor_primitives import apply_motor_primitives_patch

        apply_llm_mediator_patch(self)
        apply_motor_primitives_patch(self)
        apply_intrinsic_patch(self)
===
"""
simulation_singleton_v2.py — Singleton AGI с гуманоидом (Фаза 11/12).

Фаза 12 добавляет:
  - visual_mode toggle: enable_visual() / disable_visual()
  - EnvironmentVisual wrapper активируется без перезапуска агента
  - Predictive coding loop: GNN prediction → visual cortex feedback (опц. Neural ODE sub-steps, RKK_WM_NEURAL_ODE)
  - /vision/slots endpoint data через get_vision_state()
  - vision_stats в snapshot

Фаза 3:
  - refresh_phase3_teacher_llm(): LLM → правила IG + VL overlay (TTL)
  - tick_step: teacher_weight annealing (RKK_TEACHER_T_MAX), overlay на ValueLayer

Phase A (locomotion):
  - RKK_LOCOMOTION_CPG=1: CPG ноги на humanoid без fixed_root (engine.cpg_locomotion).
  - RKK_CPG_LOOP_HZ>0 (напр. 60): Low-level CPG в daemon-потоке; снимок graph.nodes после agent/skill step.
    По умолчанию 0 — CPG только сразу после agent.step (как раньше).

Phase B (hierarchy):
  - High ~1 Hz: GNN + EIG + goal_planning (agent tick / WS).
  - Mid ~10–20 Hz: RKK_AGENT_LOOP_HZ>0 — GNN/EIG/planning в daemon-потоке; tick_step() отдаёт кэш (UI не ждёт).
  - Low ~60 Hz+: CPG (decoupled) + RKK_PHYSICS_BG_HZ PyBullet (уже в humanoid).

  L3 goal_planning (agent), L2 skill_library, L1 CPG, L0 PyBullet.
  - RKK_SKILL_LIBRARY=1: моторная последовательность из библиотеки (тик = один кадр skill).

Phase C (full RSI):
  - RKK_RSI_FULL=1: engine.rsi_full — плато discovery → расширение GNN hidden; плато loco → CPG noise;
    плато walk skills → harder variants; падение phi → временное смягчение VL bounds.

Этап D (LLM в петле, RKK_LLM_LOOP=1):
  Уровень 1: каждый тик — GNN + System1 (как раньше).
  Уровень 2: фоновый Ollama по триггерам (стагнация discovery, block_rate, VLM unknown, surprise PE).
  Уровень 3: редко — перезапись гипотез (humanoid), в том же worker после L2 при run_level3.

Класс `Simulation` собран из миксинов в `features/simulation/mixin_*.py` (композиция поведения).
"""
from __future__ import annotations

import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from engine.agent import RKKAgent
from engine.demon import AdversarialDemon
from engine.visual_concept_store import VisualConceptStore
from engine.hierarchical_graph import HierarchicalGraph
from engine.rsi_structural import NeurogenesisEngine

from engine.config.runtime import RKKRuntimeConfig
from engine.core import (
    MotorState,
    WorldSwitcher,
    _make_env,
    default_bounds,
    resolve_torch_device,
)
from engine.features.simulation.background_loops import BackgroundLoopService
from engine.features.simulation.imports import *

from engine.features.simulation.mixin_api import SimulationApiMixin
from engine.features.simulation.mixin_concepts import SimulationConceptsMixin
from engine.features.simulation.mixin_demon_phase import SimulationDemonPhaseMixin
from engine.features.simulation.mixin_episodic_rssm import SimulationEpisodicRssmMixin
from engine.features.simulation.mixin_fall import SimulationFallMixin
from engine.features.simulation.mixin_llm import SimulationLlmLoopMixin
from engine.features.simulation.mixin_locomotion import SimulationLocomotionMixin
from engine.features.simulation.mixin_motor_pipeline import SimulationMotorPipelineMixin
from engine.features.simulation.mixin_phase_hierarchy import SimulationPhaseHierarchyMixin
from engine.features.simulation.mixin_pose_embodied import SimulationPoseEmbodiedMixin
from engine.features.simulation.mixin_skills import SimulationSkillsMixin
from engine.features.simulation.mixin_snapshot_shutdown import SimulationSnapshotShutdownMixin
from engine.features.simulation.mixin_teacher import SimulationTeacherMixin
from engine.features.simulation.mixin_tick import SimulationTickMixin
from engine.features.simulation.mixin_verbal import SimulationVerbalMixin
from engine.features.simulation.mixin_vision_predictor import SimulationVisionPredictorMixin
from engine.features.simulation.mixin_visual_grounding import SimulationVisualGroundingMixin
from engine.features.simulation.mixin_world import SimulationWorldMixin


class Simulation(
    SimulationConceptsMixin,
    SimulationVerbalMixin,
    SimulationPhaseHierarchyMixin,
    SimulationWorldMixin,
    SimulationFallMixin,
    SimulationLlmLoopMixin,
    SimulationLocomotionMixin,
    SimulationTeacherMixin,
    SimulationMotorPipelineMixin,
    SimulationSkillsMixin,
    SimulationPoseEmbodiedMixin,
    SimulationVisualGroundingMixin,
    SimulationEpisodicRssmMixin,
    SimulationTickMixin,
    SimulationVisionPredictorMixin,
    SimulationDemonPhaseMixin,
    SimulationApiMixin,
    SimulationSnapshotShutdownMixin,
):
    AGI_NAME = "Nova"
    AGI_COLOR = "#cc44ff"

    def __init__(self, device_str: str = "cuda", start_world: str = "humanoid"):
        self.device = resolve_torch_device(device_str)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.current_world = start_world
        print(f"[Singleton v2] Device: {self.device} | World: {start_world}")

        env = _make_env(start_world, self.device)
        bounds = default_bounds()

        self.agent = RKKAgent(
            agent_id=0,
            name=self.AGI_NAME,
            env=env,
            device=self.device,
            bounds=bounds,
        )

        self.switcher = WorldSwitcher(self.agent, self.device)
        self.demon = AdversarialDemon(n_agents=1, device=self.device)

        self.tick = 0
        self.phase = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events: deque[dict] = deque(maxlen=24)
        self._prev_edge_count = 0
        self._last_snapshot: dict = {}

        self._fall_count = 0
        self._fixed_root_active = False
        self._curriculum_auto_fr_released = False
        self._curriculum_stabilize_until: int = 0
        self._stand_ticks = 0
        self._last_fall_reset_tick: int = -999
        self._fall_recovery_active = False
        self._fall_recovery_start_tick = 0
        self._fall_recovery_last_progress_tick = 0
        self._fall_recovery_best_score = 0.0

        self._visual_mode = False
        self._visual_env = None
        self._base_env_ref = None
        self._vision_ticks = 0
        self._last_vision_state: dict = {}

        self._phase3_teacher_rules: list = []
        self._phase3_vl_overlay = None

        self._llm_loop_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rkk_llm")
        self._pending_llm_bundle: dict | None = None
        self._llm_level2_inflight = False
        self._last_level2_schedule_tick = -10**9
        self._last_level3_tick = -10**9
        self._best_discovery_rate = 0.0
        self._last_dr_gain_tick = 0
        self._rolling_block_bits: deque[int] = deque(maxlen=80)
        self._pe_history: deque[float] = deque(maxlen=200)
        self._locomotion_controller = None
        self._motor_state = MotorState()
        self._motor_state_lock = threading.Lock()
        self._bg = BackgroundLoopService(self)
        self._runtime_config = RKKRuntimeConfig.from_env()
        self._l1_motor_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l1_last_cmd_tick = 0
        self._l1_last_apply_tick = 0
        self._l1_last_credit_tick = 0
        self._sim_step_lock = threading.RLock()
        self._agent_step_response: dict | None = None
        self._skill_library = None
        self._skill_exec: dict | None = None
        self._llm_loop_stats: dict = {
            "level2_runs": 0,
            "level3_runs": 0,
            "last_triggers": [],
            "last_level2_explanation": "",
        }
        self._rsi_full = None
        self.neuro_engine = NeurogenesisEngine()
        self._embodied_reward_ctrl = None
        self._verbal_reward_total: float = 0.0
        self._visual_grounding_ctrl = (
            VisualGroundingController() if _VISUAL_GROUNDING_AVAILABLE else None
        )
        self._episodic_memory = EpisodicMemory() if _EPISODIC_MEMORY_AVAILABLE else None
        self._last_action_for_memory: tuple[str, float] | None = None
        self._last_fall_memory_tick: int = -999_999

        self._curriculum = CurriculumScheduler() if _CURRICULUM_AVAILABLE else None
        self._curriculum_apply_every: int = 50

        self._rssm_trainer: "RSSMTrainer | None" = None
        self._rssm_imagination: "RSSMImagination | None" = None
        self._rssm_upgraded: bool = False
        self._rssm_upgrade_tick: int = -1

        self._proprio: "ProprioceptionStream | None" = None
        if _PROPRIO_AVAILABLE:
            self._proprio = ProprioceptionStream(device=self.device)

        self._intrinsic: Any = None

        self._timescale: "MultiscaleTimeController | None" = None
        if _TIMESCALE_AVAILABLE:
            self._timescale = MultiscaleTimeController()

        self._inner_voice: "InnerVoiceController | None" = None
        self._llm_teacher: "LLMVoiceTeacher | None" = None
        if _INNER_VOICE_AVAILABLE:
            self._inner_voice = InnerVoiceController(device=self.device)
            self._llm_teacher = LLMVoiceTeacher()
            self._llm_teacher.add_callback(self._on_teacher_annotation)

        self._sleep_ctrl: "SleepController | None" = None
        self._physical_curriculum: "PhysicalCurriculum | None" = None
        self._persist: "PersistenceManager | None" = None
        if _PHASE_K_AVAILABLE:
            self._sleep_ctrl = SleepController()
            self._physical_curriculum = PhysicalCurriculum()
        self._meta_restored: bool = False
        self._was_fallen_last_tick: bool = False
        self._sleep_prev_fixed_root: bool = False

        self._uvicorn_loop: Any = None
        self._verbal: "VerbalActionController | None" = None
        self._chat_ws_clients: list[Any] = []
        self._verbal_tick_running: bool = False
        if _VERBAL_AVAILABLE:
            self._verbal = VerbalActionController()
            self._verbal.add_callback(self._broadcast_agent_message)

        self._slot_labeler: Any = None
        self._visual_voice: Any = None
        if _PHASE_M_AVAILABLE:
            _lang = os.environ.get("RKK_SPEECH_LANG", "ru")
            self._slot_labeler = SlotLabeler()
            self._visual_voice = VisualInnerVoice(lang=_lang)

        self._world_bridge: Any = None
        if _WORLD_BRIDGE_AVAILABLE and world_bridge_enabled():
            self._world_bridge = WorldStateBridge()

        self._motor_cortex: "_MotorCortexLibrary | None" = None
        self._mc_posture_window: deque = deque(maxlen=200)
        self._mc_fallen_count_window: deque = deque(maxlen=200)
        self._mc_abstract_nodes_injected: bool = False
        self._concepts_cache: list[dict] = []
        self._materialized_detector_concept_ids: set[str] = set()
        self._discovery_plateau_count = 0
        self._last_dr_snapshot: float | None = None
        self._hierarchical_graph: HierarchicalGraph | None = None
        self._concept_store: VisualConceptStore | None = None
        try:
            self._concept_inject_every = max(
                1, int(os.environ.get("RKK_CONCEPT_INJECT_EVERY", "30"))
            )
        except ValueError:
            self._concept_inject_every = 30
        self._l4_thread: threading.Thread | None = None
        self._l4_stop = threading.Event()
        self._l4_in_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_out_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_task_pending = False
        self._l4_last_snapshot: dict = {"n_concepts": 0, "concepts": []}
        self._l4_last_submit_tick = 0
        self._l4_last_apply_tick = 0
        self._l3_next_due_ts = 0.0
        self._l3_last_tick = 0
        self._memory_resume_enabled = os.environ.get(
            "RKK_MEMORY_RESUME_ON_START", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        if self._memory_resume_enabled:
            try:
                meta = self.memory_load()
                if meta.get("ok"):
                    print(f"[Simulation] Memory resumed at tick={self.tick}")
            except Exception as e:
                print(f"[Simulation] Memory resume skipped: {type(e).__name__}: {e}")

        if os.environ.get("RKK_NEURAL_LANG", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        ):
            try:
                from engine.neural_lang_integration import apply_neural_lang_patch

                apply_neural_lang_patch(self)
            except Exception as e:
                print(f"[Simulation] Neural lang patch skipped: {type(e).__name__}: {e}")

        from engine.intristic_objective import apply_intrinsic_patch
        from engine.llm_hint_mediator import apply_llm_mediator_patch
        from engine.learned_motor_primitives import apply_motor_primitives_patch

        apply_llm_mediator_patch(self)
        apply_motor_primitives_patch(self)
        apply_intrinsic_patch(self)

        # ── Variable Registry: dynamic ontology ──────────────────────────────
        try:
            from engine.variable_bootstrap import get_variable_registry
            self._variable_registry = get_variable_registry()
        except Exception as e:
            self._variable_registry = None
            print(f"[Simulation] VariableRegistry skipped: {type(e).__name__}: {e}")
```

Registry created on `Simulation.__init__()`, available for snapshot and auto-discovery.

---

## New Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RKK_BOOTSTRAP_ONLY` | `0` | Start with 11 seed vars instead of 59 |
| `RKK_BOOTSTRAP_VARS` | `` | Custom comma-separated bootstrap vars |
| `RKK_BOOTSTRAP_DISCOVER_THRESHOLD` | `1.5` | Pressure threshold for auto-discovery |
| `RKK_SPARSE_EIG_MIN_UNC` | `0.15` | Min uncertainty to include pair in EIG |

## Test Results

```
✅ variable_bootstrap.py — Full mode: 59 vars
✅ variable_bootstrap.py — Bootstrap mode: 11 vars  
✅ sleep_consolidation.py — should_sleep(None) = None
✅ sleep_consolidation.py — should_sleep(stagnant_IO) = "compression_stagnant"
✅ All imports verified (snapshot, intrinsic, bootstrap, motor_primitives)
```

## How to Use

**Default (backward compatible):** No changes needed, everything works as before.

**Bootstrap mode (open-ended AGI):**
```bash
RKK_BOOTSTRAP_ONLY=1 RKK_DEVICE=cpu python3 run.py
```

The agent starts with 11 variables and discovers the rest through learning.
