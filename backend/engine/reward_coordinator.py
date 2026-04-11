"""
reward_coordinator.py — Level 3-H: Unified Reward Coordinator.

Реализует архитектуру наград из рекомендаций по Alignment:
  НЕ сумма: R = w1*CPG + w2*LLM
  А лексикографическая иерархия с правом вето:

  Pyramid (Маслоу для робота):
    L0 SURVIVAL VETO    — физическая целостность (мгновенное вето)
    L1 EFFICIENCY       — биомеханическая эффективность
    L2 EMPOWERMENT      — число доступных будущих состояний
    L3 CURIOSITY        — prediction error (exploration drive)
    L4 TASK             — задача от LLM/человека
    L5 CONSTITUTION     — конституциональный фильтр LLM

  Итоговая формула:
    R_total = (R_efficiency + R_empowerment + R_curiosity + R_task * R_constitution)
              * survival_veto_multiplier

  survival_veto_multiplier ∈ {0, 1}:
    0 если: fallen, anomaly > threshold, joint overload, h(W) overflow
    Плавное восстановление через recovery_weight [0,1]

  Curiosity (ICM-lite):
    forward_model: (X_t, a_t) → X_{t+1}^hat
    curiosity_reward = ||X_{t+1} - X_{t+1}^hat||² (prediction error)
    Масштабирован так чтобы не доминировать над task reward

  Constitutional Filter (async LLM):
    Каждые N тиков LLM оценивает последнее действие через "Конституцию"
    Возвращает multiplier [0, 1] + verbal warning
    Кэшируется — не блокирует основной цикл

RKK_REWARD_ENABLED=1              — включить (default)
RKK_REWARD_CURIOSITY_SCALE=0.15   — масштаб curiosity reward
RKK_REWARD_EMPOWERMENT_SCALE=0.20 — масштаб empowerment reward
RKK_REWARD_TASK_SCALE=0.50        — масштаб task reward
RKK_REWARD_EFFICIENCY_SCALE=0.15  — масштаб efficiency reward
RKK_REWARD_SURVIVAL_ANOMALY=0.75  — порог anomaly для veto
RKK_REWARD_CONSTITUTION_EVERY=200 — тиков между constitutional checks
RKK_REWARD_CURIOSITY_EMA=0.95     — EMA для baseline curiosity
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reward_enabled() -> bool:
    return os.environ.get("RKK_REWARD_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


# ── Reward signal dataclass ────────────────────────────────────────────────────
@dataclass
class RewardSignal:
    """Full decomposed reward for one tick."""
    # L0: Survival veto (0 = blocked, 1 = ok)
    survival_veto: float = 1.0
    survival_reason: str = ""

    # L1: Efficiency (biomechanics quality)
    efficiency: float = 0.0
    efficiency_detail: str = ""

    # L2: Empowerment (future state diversity)
    empowerment: float = 0.0

    # L3: Curiosity (prediction surprise)
    curiosity: float = 0.0
    prediction_error: float = 0.0

    # L4: Task (from LLM / curriculum)
    task: float = 0.0
    task_source: str = ""  # "llm", "curriculum", "heuristic"

    # L5: Constitutional multiplier (from LLM safety check)
    constitution: float = 1.0
    constitution_warning: str = ""

    # Computed
    total: float = 0.0
    blocked: bool = False

    def compute_total(
        self,
        w_eff: float = 0.15,
        w_emp: float = 0.20,
        w_cur: float = 0.15,
        w_task: float = 0.50,
    ) -> float:
        """
        Lexicographic composition:
          - Survival veto multiplies everything
          - Constitution multiplies task signal only
          - Others are additive within tier
        """
        # Inner reward (everything except veto)
        inner = (
            w_eff * self.efficiency
            + w_emp * self.empowerment
            + w_cur * self.curiosity
            + w_task * self.task * self.constitution
        )
        # Apply survival veto
        self.total = float(inner * self.survival_veto)
        self.blocked = self.survival_veto < 0.1
        return self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "survival_veto": round(self.survival_veto, 4),
            "survival_reason": self.survival_reason,
            "efficiency": round(self.efficiency, 4),
            "empowerment": round(self.empowerment, 4),
            "curiosity": round(self.curiosity, 4),
            "prediction_error": round(self.prediction_error, 5),
            "task": round(self.task, 4),
            "task_source": self.task_source,
            "constitution": round(self.constitution, 4),
            "constitution_warning": self.constitution_warning[:80],
            "total": round(self.total, 5),
            "blocked": self.blocked,
        }


# ── ICM-lite: Curiosity (Intrinsic Curiosity Module) ──────────────────────────
class CuriosityICM(nn.Module):
    """
    Simplified Intrinsic Curiosity Module.

    forward_model: concat(X_t, a_t) → X_{t+1}^hat
    curiosity = ||X_{t+1} - X_{t+1}^hat||² (prediction error)

    High curiosity → agent is in unexplored territory → explore!
    Демон v2 бьёт именно сюда, поэтому curiosity reward помогает
    использовать adversarial perturbations конструктивно.
    """

    def __init__(self, d: int, hidden: int = 48, device: torch.device | None = None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.d = d

        # Forward model: [X_t; a_t] (2d) → X_{t+1} (d)
        self.forward_model = nn.Sequential(
            nn.Linear(d * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d),
        )

        for m in self.forward_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(dev)
        self.device = dev
        self.optim = torch.optim.Adam(self.parameters(), lr=3e-4)

        # EMA baseline for prediction error (curiosity = error - baseline)
        self._baseline: float = 0.0
        self._ema: float = _env_float("RKK_REWARD_CURIOSITY_EMA", 0.95)
        self._history: deque[float] = deque(maxlen=200)
        self.train_steps: int = 0

    def predict(self, X_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Predict X_{t+1} from X_t and a_t."""
        xa = torch.cat([X_t, a_t], dim=-1)
        return self.forward_model(xa)

    def compute_curiosity(
        self,
        X_t: list[float],
        a_t: list[float],
        X_tp1: list[float],
    ) -> float:
        """
        Compute curiosity reward for transition (X_t, a_t, X_{t+1}).
        Returns normalized curiosity reward [0, 1].
        """
        with torch.no_grad():
            xt = torch.tensor(X_t, dtype=torch.float32, device=self.device).unsqueeze(0)
            at = torch.tensor(a_t, dtype=torch.float32, device=self.device).unsqueeze(0)
            xtp1 = torch.tensor(X_tp1, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred = self.predict(xt, at)
            error = float(F.mse_loss(pred, xtp1).item())

        # Update baseline EMA
        self._baseline = self._ema * self._baseline + (1 - self._ema) * error
        self._history.append(error)

        # Curiosity = error above baseline, normalized
        baseline = float(np.mean(list(self._history)[-50:])) if len(self._history) >= 50 else self._baseline
        curiosity_raw = max(0.0, error - baseline * 0.8)

        # Scale by curiosity_scale
        scale = _env_float("RKK_REWARD_CURIOSITY_SCALE", 0.15)
        return float(np.clip(curiosity_raw * 5.0 * scale, 0.0, 1.0))

    def train_step(
        self,
        X_t: list[float],
        a_t: list[float],
        X_tp1: list[float],
    ) -> float | None:
        """Train forward model."""
        self.train()
        xt = torch.tensor(X_t, dtype=torch.float32, device=self.device).unsqueeze(0)
        at = torch.tensor(a_t, dtype=torch.float32, device=self.device).unsqueeze(0)
        xtp1 = torch.tensor(X_tp1, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.optim.zero_grad()
        pred = self.predict(xt, at)
        loss = F.mse_loss(pred, xtp1.detach())
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()
        self.train_steps += 1
        return float(loss.item())

    def snapshot(self) -> dict[str, Any]:
        return {
            "train_steps": self.train_steps,
            "baseline_error": round(self._baseline, 5),
            "history_mean": round(float(np.mean(list(self._history)[-50:])), 5) if self._history else 0.0,
        }


# ── Survival Veto ──────────────────────────────────────────────────────────────
class SurvivalVeto:
    """
    L0: Физическая целостность.
    Мгновенное вето при критических условиях.
    Никакой LLM reward не может перекрыть это вето.
    """

    ANOMALY_THRESHOLD = _env_float("RKK_REWARD_SURVIVAL_ANOMALY", 0.75)
    FALL_VETO = 0.0
    FALL_RECOVERY_WEIGHT = 0.3  # partial during recovery

    def __init__(self):
        self._recovery_ticks: int = 0
        self._recovery_weight: float = 1.0
        self._in_recovery: bool = False

    def evaluate(
        self,
        fallen: bool,
        anomaly_score: float,
        h_W: float,
        posture: float,
        com_z: float,
    ) -> tuple[float, str]:
        """
        Returns (veto_multiplier [0,1], reason_str).
        0 = complete block, 1 = clear.
        """
        # Hard veto: fallen
        if fallen:
            self._in_recovery = True
            self._recovery_ticks = 0
            self._recovery_weight = self.FALL_VETO
            return self.FALL_VETO, "fallen"

        # Recovery phase: gradually restore veto
        if self._in_recovery:
            if posture > 0.60 and com_z > 0.40:
                self._recovery_ticks += 1
                # Restore linearly over 30 ticks
                self._recovery_weight = min(1.0, self._recovery_ticks / 30.0)
                if self._recovery_weight >= 1.0:
                    self._in_recovery = False
            else:
                self._recovery_ticks = 0
                self._recovery_weight = self.FALL_RECOVERY_WEIGHT
            return self._recovery_weight, f"recovery({self._recovery_ticks}/30)"

        # Soft veto: high anomaly
        if anomaly_score > self.ANOMALY_THRESHOLD:
            strength = float(np.clip(1.0 - anomaly_score, 0.1, 0.8))
            return strength, f"joint_anomaly={anomaly_score:.3f}"

        # Soft veto: causal structure divergence
        if h_W > 5.0:
            strength = float(np.clip(1.0 - (h_W - 5.0) / 10.0, 0.3, 1.0))
            return strength, f"h_W_overflow={h_W:.2f}"

        return 1.0, ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "in_recovery": self._in_recovery,
            "recovery_ticks": self._recovery_ticks,
            "recovery_weight": round(self._recovery_weight, 4),
        }


# ── Constitutional Filter (async LLM) ─────────────────────────────────────────
class ConstitutionalFilter:
    """
    L5: LLM как конституциональный судья.

    Каждые N тиков проверяет последние действия на соответствие
    «Конституции» — набору базовых правил безопасности.

    Кэшируется — не блокирует основной цикл (async task).
    Constitution multiplier [0, 1]:
      1.0 = действие полностью безопасно
      0.5 = требует осторожности
      0.0 = действие нарушает конституцию (блок)
    """

    CONSTITUTION = """
ROBOT CONSTITUTION — Hard Rules:
1. SAFETY: Never take actions that could damage joints (anomaly > 0.8).
2. STABILITY: Do not increase stride when posture < 0.5 (fall risk).
3. RECOVERY: When com_z < 0.35 (fallen), activate recovery before any other goal.
4. UNCERTAINTY: If uncertain about outcome, prefer smaller actions.
5. HUMAN_OVERRIDE: If human signals stop, reduce all motor commands to neutral.
6. EFFICIENCY: Avoid repetitive blocked actions (block_rate > 0.5 for same action).
"""

    def __init__(self):
        self._multiplier: float = 1.0
        self._warning: str = ""
        self._last_check_tick: int = -999_999
        self._pending: bool = False
        self._checks: int = 0
        self._violations: int = 0
        self._every = _env_int("RKK_REWARD_CONSTITUTION_EVERY", 200)

    def get_multiplier(self) -> tuple[float, str]:
        """Returns cached (multiplier, warning)."""
        return self._multiplier, self._warning

    def should_check(self, tick: int) -> bool:
        return (
            not self._pending
            and (tick - self._last_check_tick) >= self._every
        )

    async def check_async(
        self,
        tick: int,
        last_actions: list[tuple[str, float]],
        obs: dict[str, float],
        llm_url: str,
        llm_model: str,
    ) -> None:
        """Async constitutional check. Updates cached multiplier."""
        if self._pending:
            return
        self._pending = True
        self._last_check_tick = tick

        try:
            result = await self._call_llm(last_actions, obs, llm_url, llm_model)
            self._multiplier = float(np.clip(result.get("multiplier", 1.0), 0.0, 1.0))
            self._warning = str(result.get("warning", "")).strip()[:200]
            if self._multiplier < 0.8:
                self._violations += 1
            self._checks += 1
        except Exception as e:
            # On error: maintain current multiplier (safe default)
            self._warning = f"check_failed: {e!r}"[:100]
        finally:
            self._pending = False

    async def _call_llm(
        self,
        last_actions: list[tuple[str, float]],
        obs: dict[str, float],
        llm_url: str,
        llm_model: str,
    ) -> dict[str, Any]:
        """Call LLM for constitutional evaluation."""
        from engine.ollama_env import ollama_think_disabled_payload

        # Format recent actions
        actions_str = "\n".join(
            f"  do({v}={x:.3f})" for v, x in last_actions[-5:]
        )

        # Key safety metrics
        posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.7)))
        anomaly = float(obs.get("proprio_anomaly", 0.0))
        stride = float(obs.get("intent_stride", obs.get("phys_intent_stride", 0.5)))

        prompt = f"""{self.CONSTITUTION}

Recent robot actions:
{actions_str}

Current state:
  posture_stability={posture:.3f}
  com_z={com_z:.3f} (fallen if <0.35)
  joint_anomaly={anomaly:.3f}
  intent_stride={stride:.3f}

Evaluate if recent actions comply with the Constitution.
Reply ONLY with JSON: {{"multiplier": <0.0-1.0>, "rule_violated": "<rule# or none>", "warning": "<short>"}}
multiplier: 1.0=compliant, 0.5=caution, 0.0=violation
"""
        url = llm_url.strip().rstrip("/")
        if not url.endswith("/generate"):
            url = url + "/api/generate" if "/api/" not in url else url

        payload = {
            "model": llm_model,
            "prompt": prompt,
            "stream": False,
            **ollama_think_disabled_payload(),
            "options": {"temperature": 0.05, "num_predict": 128},
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                return {"multiplier": 1.0, "warning": f"HTTP {resp.status_code}"}
            raw = (resp.json().get("response") or "").strip()

        from engine.llm_json_extract import parse_json_object_loose
        obj = parse_json_object_loose(raw)
        if not obj:
            return {"multiplier": 1.0, "warning": "parse_failed"}
        return obj

    def snapshot(self) -> dict[str, Any]:
        return {
            "multiplier": round(self._multiplier, 4),
            "warning": self._warning,
            "checks": self._checks,
            "violations": self._violations,
            "pending": self._pending,
            "every_ticks": self._every,
        }


# ── Efficiency evaluator ───────────────────────────────────────────────────────
class EfficiencyEvaluator:
    """
    L1: Биомеханическая эффективность.
    Поощряет плавные, симметричные движения с минимальной энергией.
    """

    def evaluate(self, obs: dict[str, float]) -> float:
        def g(k: str, d: float = 0.5) -> float:
            return float(obs.get(k, obs.get(f"phys_{k}", d)))

        posture = g("posture_stability", 0.5)
        gait_l = g("gait_phase_l", 0.5)
        gait_r = g("gait_phase_r", 0.5)
        foot_l = g("foot_contact_l", 0.5)
        foot_r = g("foot_contact_r", 0.5)
        bias = g("support_bias", 0.5)

        # Symmetry
        gait_sym = float(np.clip(1.0 - abs(gait_l - gait_r) * 2.0, 0.0, 1.0))
        contact_sym = float(np.clip(1.0 - abs(foot_l - foot_r) * 2.0, 0.0, 1.0))
        bias_bal = float(np.clip(1.0 - abs(bias - 0.5) * 2.0, 0.0, 1.0))

        efficiency = posture * 0.4 + gait_sym * 0.2 + contact_sym * 0.2 + bias_bal * 0.2
        return float(np.clip(efficiency, 0.0, 1.0))


# ── Main Reward Coordinator ────────────────────────────────────────────────────
class RewardCoordinator:
    """
    Unified Reward Coordinator с лексикографической иерархией.

    Интеграция в simulation.py:
      self._reward_coord = RewardCoordinator(d=graph_d, device=device)

    В конце каждого тика:
      signal = self._reward_coord.compute(
          tick=tick,
          obs=obs,
          X_t=X_t, a_t=a_t, X_tp1=X_tp1,
          fallen=fallen,
          anomaly=proprio.anomaly_score,
          empowerment_reward=proprio.get_empowerment_reward(),
          task_reward=embodied_result.combined_reward,
          h_W=agent.snapshot()['h_W'],
          agent=agent,
          locomotion_ctrl=ctrl,
          motor_cortex=mc,
      )

      # Apply total reward to all learners
      signal.total → locomotion_ctrl + motor_cortex + curiosity_training
    """

    def __init__(self, d: int = 30, device: torch.device | None = None):
        dev = device or torch.device("cpu")
        self.device = dev

        self.curiosity = CuriosityICM(d=d, hidden=48, device=dev)
        self.survival = SurvivalVeto()
        self.constitution = ConstitutionalFilter()
        self.efficiency = EfficiencyEvaluator()

        # Weights (read from env on each compute — allows hot-reload)
        self._w_eff = _env_float("RKK_REWARD_EFFICIENCY_SCALE", 0.15)
        self._w_emp = _env_float("RKK_REWARD_EMPOWERMENT_SCALE", 0.20)
        self._w_cur = _env_float("RKK_REWARD_CURIOSITY_SCALE", 0.15)
        self._w_task = _env_float("RKK_REWARD_TASK_SCALE", 0.50)

        # History
        self._signals: deque[RewardSignal] = deque(maxlen=200)
        self._action_history: deque[tuple[str, float]] = deque(maxlen=20)
        self.total_signals: int = 0

        # Constitutional check state
        self._llm_url: str = ""
        self._llm_model: str = ""

    def record_action(self, var: str, val: float) -> None:
        """Track actions for constitutional filter."""
        self._action_history.append((str(var), float(val)))

    def compute(
        self,
        tick: int,
        obs: dict[str, float],
        X_t: list[float],
        a_t: list[float],
        X_tp1: list[float],
        fallen: bool,
        anomaly: float,
        empowerment_reward: float,
        task_reward: float,
        task_source: str = "heuristic",
        h_W: float = 0.0,
        llm_url: str = "",
        llm_model: str = "",
    ) -> RewardSignal:
        """
        Compute full reward signal for one tick.
        """
        if not reward_enabled():
            sig = RewardSignal(total=task_reward)
            sig.total = task_reward
            return sig

        self._llm_url = llm_url
        self._llm_model = llm_model

        # Read current weights from env
        self._w_eff = _env_float("RKK_REWARD_EFFICIENCY_SCALE", 0.15)
        self._w_emp = _env_float("RKK_REWARD_EMPOWERMENT_SCALE", 0.20)
        self._w_cur = _env_float("RKK_REWARD_CURIOSITY_SCALE", 0.15)
        self._w_task = _env_float("RKK_REWARD_TASK_SCALE", 0.50)

        sig = RewardSignal()

        # L0: Survival veto
        posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.7)))
        sig.survival_veto, sig.survival_reason = self.survival.evaluate(
            fallen=fallen,
            anomaly_score=anomaly,
            h_W=h_W,
            posture=posture,
            com_z=com_z,
        )

        # L1: Efficiency
        sig.efficiency = self.efficiency.evaluate(obs)

        # L2: Empowerment
        sig.empowerment = float(np.clip(
            empowerment_reward * self._w_emp / max(self._w_emp, 0.01),
            0.0, 1.0,
        ))

        # L3: Curiosity
        if len(X_t) == len(X_tp1) == len(a_t) and len(X_t) > 0:
            sig.curiosity = self.curiosity.compute_curiosity(X_t, a_t, X_tp1)
            sig.prediction_error = float(np.mean(
                [(p - q) ** 2 for p, q in zip(X_tp1, X_t)]
            ) ** 0.5)
            # Train curiosity model
            if tick % 4 == 0:
                self.curiosity.train_step(X_t, a_t, X_tp1)

        # L4: Task reward (from embodied LLM or curriculum)
        sig.task = float(np.clip((task_reward + 1.0) / 2.0, 0.0, 1.0))  # normalize [-1,1] → [0,1]
        sig.task_source = task_source

        # L5: Constitutional multiplier (use cached value)
        sig.constitution, sig.constitution_warning = self.constitution.get_multiplier()

        # Schedule constitutional check
        if self.constitution.should_check(tick) and llm_url:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        self.constitution.check_async(
                            tick, list(self._action_history), obs, llm_url, llm_model
                        )
                    )
            except Exception:
                pass

        # Compute total with lexicographic composition
        sig.compute_total(
            w_eff=self._w_eff,
            w_emp=self._w_emp,
            w_cur=self._w_cur,
            w_task=self._w_task,
        )

        self._signals.append(sig)
        self.total_signals += 1

        return sig

    def apply_to_learners(
        self,
        signal: RewardSignal,
        locomotion_ctrl=None,
        motor_cortex=None,
        agent=None,
    ) -> None:
        """
        Distribute total reward signal to all learners.

        Survival veto: blocks locomotion_ctrl updates when fallen.
        Constitutional violation: reduces motor_cortex learning rate.
        """
        total = signal.total

        # Apply to locomotion controller
        if locomotion_ctrl is not None:
            if not signal.blocked:
                locomotion_ctrl._reward_history.append(total)
            # If fallen, push strong negative to avoid this state
            elif signal.survival_reason == "fallen":
                locomotion_ctrl._reward_history.append(-2.0)

        # Apply to motor cortex programs
        if motor_cortex is not None:
            mc_reward = total if not signal.blocked else -1.0
            if agent is not None:
                obs = {}
                try:
                    obs = dict(agent.env.observe())
                except Exception:
                    pass
                posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
                foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
                foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
                nodes = dict(agent.graph.nodes)
                motor_cortex.push_and_train(nodes, {}, mc_reward, posture, foot_l, foot_r)

        # Constitutional violation: add event but don't block (already in multiplier)
        if signal.constitution < 0.6 and signal.constitution_warning:
            pass  # logged externally in simulation

    def snapshot(self) -> dict[str, Any]:
        recent = list(self._signals)[-20:]
        mean_total = float(np.mean([s.total for s in recent])) if recent else 0.0
        mean_curiosity = float(np.mean([s.curiosity for s in recent])) if recent else 0.0
        mean_emp = float(np.mean([s.empowerment for s in recent])) if recent else 0.0
        last = recent[-1].to_dict() if recent else {}

        return {
            "enabled": reward_enabled(),
            "total_signals": self.total_signals,
            "mean_total_recent": round(mean_total, 4),
            "mean_curiosity_recent": round(mean_curiosity, 4),
            "mean_empowerment_recent": round(mean_emp, 4),
            "weights": {
                "efficiency": self._w_eff,
                "empowerment": self._w_emp,
                "curiosity": self._w_cur,
                "task": self._w_task,
            },
            "survival": self.survival.snapshot(),
            "constitution": self.constitution.snapshot(),
            "curiosity_icm": self.curiosity.snapshot(),
            "last_signal": last,
        }
