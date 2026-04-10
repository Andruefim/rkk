"""
embodied_llm_reward.py — Level 1-A: Embodied LLM Reward Shaping.

LLM получает реальные float-значения позы каждые N тиков и возвращает:
  1. Числовой reward → locomotion_controller + motor_cortex
  2. Verbal feedback → конвертируется в GNN seeds (causal priors)
  3. IG rules → agent.set_teacher_state() (как Phase3 Teacher, но из позы)

Архитектурный принцип: LLM знает биомеханику, накопленную человечеством.
Вместо того чтобы запрашивать его раз при старте, он консультируется
постоянно по реальным float-данным тела, давая directed learning signal.

Пример prompt → response:
  Input:  com_z=0.71, torso_pitch=-0.18, foot_l=0.82, foot_r=0.48, stride=0.62
  Output: {"posture_score":0.6,"gait_quality":0.3,"forward_progress":0.5,
           "verbal":"shift weight left, lean torso forward",
           "seeds":[{"from_":"spine_pitch","to":"com_x","weight":0.4}]}

RKK_LLM_MOTOR_REWARD_EVERY=300   — тиков между вызовами
RKK_LLM_MOTOR_REWARD_WEIGHT=0.6  — вес LLM reward vs env reward (0-1)
RKK_LLM_MOTOR_TIMEOUT=45         — HTTP timeout
RKK_LLM_MOTOR_SEEDS=1            — инжектировать seeds из verbal feedback
RKK_LLM_MOTOR_FALLBACK=1         — использовать rule-based fallback при ошибке LLM
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

from engine.llm_json_extract import parse_json_object_loose, scan_json_objects_having_any_key
from engine.ollama_env import get_ollama_generate_url, get_ollama_model, ollama_think_disabled_payload


# ── Configuration ─────────────────────────────────────────────────────────────
def _every() -> int:
    try:
        return max(50, int(os.environ.get("RKK_LLM_MOTOR_REWARD_EVERY", "300")))
    except ValueError:
        return 300


def _weight() -> float:
    try:
        return float(np.clip(float(os.environ.get("RKK_LLM_MOTOR_REWARD_WEIGHT", "0.6")), 0.0, 1.0))
    except ValueError:
        return 0.6


def _timeout() -> float:
    try:
        return max(10.0, float(os.environ.get("RKK_LLM_MOTOR_TIMEOUT", "45")))
    except ValueError:
        return 45.0


def _seeds_enabled() -> bool:
    return os.environ.get("RKK_LLM_MOTOR_SEEDS", "1").strip().lower() in ("1", "true", "yes", "on")


def _fallback_enabled() -> bool:
    return os.environ.get("RKK_LLM_MOTOR_FALLBACK", "1").strip().lower() in ("1", "true", "yes", "on")


def embodied_reward_enabled() -> bool:
    return os.environ.get("RKK_LLM_MOTOR_REWARD", "1").strip().lower() not in ("0", "false", "no", "off")


# ── Pose snapshot ─────────────────────────────────────────────────────────────
@dataclass
class PoseSnapshot:
    """Полный снимок позы гуманоида для LLM промпта."""
    tick: int
    com_z: float
    com_x: float
    torso_pitch: float
    torso_roll: float
    posture_stability: float
    foot_contact_l: float
    foot_contact_r: float
    gait_phase_l: float
    gait_phase_r: float
    support_bias: float
    intent_stride: float
    intent_torso_forward: float
    lhip: float
    rhip: float
    lknee: float
    rknee: float
    lankle: float
    rankle: float
    fallen: bool
    fall_count: int
    # Motor cortex state
    cpg_weight: float = 1.0
    mc_quality_ema: float = 0.0
    # History stats
    recent_fall_rate: float = 0.0
    mean_posture_recent: float = 0.5

    def to_prompt_lines(self) -> list[str]:
        """Форматируем для LLM."""
        lines = [
            f"tick={self.tick}",
            f"",
            f"BODY STATE:",
            f"  com_z={self.com_z:.3f}  (1.0=standing, 0.0=floor)",
            f"  com_x={self.com_x:.3f}  (0.5=neutral, >0.5=forward lean)",
            f"  torso_pitch={self.torso_pitch:.3f}  (neg=leaning backward, pos=forward)",
            f"  torso_roll={self.torso_roll:.3f}   (±roll tilt)",
            f"  posture_stability={self.posture_stability:.3f}  (1.0=perfect balance)",
            f"",
            f"FOOT CONTACT (0=no contact, 1=full contact):",
            f"  foot_contact_l={self.foot_contact_l:.3f}",
            f"  foot_contact_r={self.foot_contact_r:.3f}",
            f"  support_bias={self.support_bias:.3f}  (0.5=balanced, 0=left, 1=right)",
            f"",
            f"GAIT (phase 0..1):",
            f"  gait_phase_l={self.gait_phase_l:.3f}",
            f"  gait_phase_r={self.gait_phase_r:.3f}",
            f"",
            f"INTENTS (agent's current motor commands, 0.5=neutral):",
            f"  intent_stride={self.intent_stride:.3f}      (0.5=stand, 1.0=max forward stride)",
            f"  intent_torso_forward={self.intent_torso_forward:.3f}",
            f"",
            f"JOINTS (normalized 0..1):",
            f"  lhip={self.lhip:.3f}  rhip={self.rhip:.3f}",
            f"  lknee={self.lknee:.3f}  rknee={self.rknee:.3f}",
            f"  lankle={self.lankle:.3f}  rankle={self.rankle:.3f}",
            f"",
            f"HISTORY:",
            f"  fallen={self.fallen}  fall_count={self.fall_count}",
            f"  recent_fall_rate={self.recent_fall_rate:.3f}  (falls / last 200 ticks)",
            f"  mean_posture_last_200={self.mean_posture_recent:.3f}",
            f"  cpg_weight={self.cpg_weight:.3f}  (1.0=full CPG, 0.08=cortex dominant)",
        ]
        return lines

    @classmethod
    def from_obs_and_graph(
        cls,
        obs: dict[str, float],
        graph_nodes: dict[str, float],
        tick: int,
        fallen: bool,
        fall_count: int,
        cpg_weight: float = 1.0,
        mc_quality_ema: float = 0.0,
        recent_fall_rate: float = 0.0,
        mean_posture_recent: float = 0.5,
    ) -> "PoseSnapshot":
        def g(key: str, default: float = 0.5) -> float:
            v = obs.get(key)
            if v is None:
                v = obs.get(f"phys_{key}")
            if v is None:
                v = graph_nodes.get(key)
            if v is None:
                v = graph_nodes.get(f"phys_{key}")
            return float(v if v is not None else default)

        return cls(
            tick=tick,
            com_z=g("com_z", 0.7),
            com_x=g("com_x", 0.5),
            torso_pitch=g("torso_pitch", 0.0),
            torso_roll=g("torso_roll", 0.0),
            posture_stability=g("posture_stability", 0.5),
            foot_contact_l=g("foot_contact_l", 0.5),
            foot_contact_r=g("foot_contact_r", 0.5),
            gait_phase_l=g("gait_phase_l", 0.5),
            gait_phase_r=g("gait_phase_r", 0.5),
            support_bias=g("support_bias", 0.5),
            intent_stride=g("intent_stride", 0.5),
            intent_torso_forward=g("intent_torso_forward", 0.5),
            lhip=g("lhip", 0.5),
            rhip=g("rhip", 0.5),
            lknee=g("lknee", 0.5),
            rknee=g("rknee", 0.5),
            lankle=g("lankle", 0.5),
            rankle=g("rankle", 0.5),
            fallen=fallen,
            fall_count=fall_count,
            cpg_weight=cpg_weight,
            mc_quality_ema=mc_quality_ema,
            recent_fall_rate=recent_fall_rate,
            mean_posture_recent=mean_posture_recent,
        )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_embodied_reward_prompt(pose: PoseSnapshot, valid_vars: list[str]) -> str:
    pose_str = "\n".join(pose.to_prompt_lines())
    vars_sample = ", ".join(sorted(valid_vars)[:60])
    if len(valid_vars) > 60:
        vars_sample += " ..."

    return f"""You are an expert robotics biomechanics coach for a simulated bipedal humanoid.
The robot is learning to walk from scratch. Your job is to evaluate its current body state
and provide numerical feedback that directly trains its motor controllers.

CURRENT BODY STATE:
{pose_str}

BIOMECHANICS KNOWLEDGE TO APPLY:
- For forward walking: CoM (center of mass) must be slightly AHEAD of the support foot (com_x > 0.47)
- Backward falling happens when torso_pitch is too negative (leaning back) during stride
- Good gait: left and right phases should be anti-phase (gait_phase_l and gait_phase_r differ by ~0.5)
- Stable stance needs both foot contacts > 0.55 before attempting stride
- After falling (com_z < 0.35): activate recovery — bend knees, stabilize, then rise slowly
- Torso forward lean is critical for walking: spine must pitch forward when stride > 0.5

AVAILABLE VARIABLE NAMES (use EXACT strings for seeds):
{vars_sample}

Respond with ONLY a valid JSON object, no markdown:
{{
  "posture_score": <float 0-1, how well-balanced is the robot>,
  "gait_quality": <float 0-1, smoothness and symmetry of gait>,
  "forward_progress": <float 0-1, progress toward walking forward>,
  "verbal": "<1-2 sentences: specific actionable biomechanics advice>",
  "priority_issue": "<single most critical problem right now>",
  "seeds": [
    {{"from_": "<exact variable name>", "to": "<exact variable name>", "weight": <0.2-0.6>, "reason": "<short>"}}
  ],
  "intent_adjustments": {{
    "<intent_var_name>": <suggested target value 0-1>
  }}
}}

Rules:
- seeds: 2-6 causal edges based on biomechanics. Only use variable names from the list above.
- intent_adjustments: optional, suggest 0-3 intent_* variable target values.
- weight in seeds: 0.2=weak prior, 0.6=strong biomechanical fact.
- Be SPECIFIC about what's wrong, not generic ("lean torso forward" not "improve balance").
"""


# ── Response parser ───────────────────────────────────────────────────────────
@dataclass
class EmbodiedRewardResult:
    posture_score: float = 0.5
    gait_quality: float = 0.5
    forward_progress: float = 0.5
    verbal: str = ""
    priority_issue: str = ""
    seeds: list[dict[str, Any]] = field(default_factory=list)
    intent_adjustments: dict[str, float] = field(default_factory=dict)
    combined_reward: float = 0.0
    ok: bool = False
    error: str = ""

    def compute_combined(self, weights: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
        """Compute combined motor reward from LLM scores."""
        w_p, w_g, w_f = weights
        r = (
            w_p * self.posture_score
            + w_g * self.gait_quality
            + w_f * self.forward_progress
        )
        self.combined_reward = float(np.clip(r * 2.0 - 0.5, -1.0, 1.5))
        return self.combined_reward


def parse_embodied_reward_response(
    raw: str,
    valid_vars: set[str],
) -> EmbodiedRewardResult:
    result = EmbodiedRewardResult()
    _KEYS = frozenset({"posture_score", "gait_quality", "verbal", "seeds"})
    obj = parse_json_object_loose(raw)
    if not obj:
        obj = scan_json_objects_having_any_key(raw, _KEYS)
    if not obj:
        result.error = f"no JSON in response (len={len(raw)})"
        return result

    def _f(key: str, default: float = 0.5) -> float:
        try:
            return float(np.clip(float(obj.get(key, default)), 0.0, 1.0))
        except (TypeError, ValueError):
            return default

    result.posture_score = _f("posture_score", 0.5)
    result.gait_quality = _f("gait_quality", 0.5)
    result.forward_progress = _f("forward_progress", 0.5)
    result.verbal = str(obj.get("verbal", "")).strip()[:300]
    result.priority_issue = str(obj.get("priority_issue", "")).strip()[:200]

    # Parse seeds
    for s in (obj.get("seeds") or []):
        if not isinstance(s, dict):
            continue
        fr = str(s.get("from_") or s.get("from") or "").strip()
        to = str(s.get("to") or "").strip()
        if not fr or not to or fr == to:
            continue
        if fr not in valid_vars or to not in valid_vars:
            continue
        try:
            w = float(np.clip(float(s.get("weight", 0.3)), 0.1, 0.7))
        except (TypeError, ValueError):
            w = 0.3
        result.seeds.append({"from_": fr, "to": to, "weight": w, "alpha": 0.06})

    # Parse intent adjustments
    for k, v in (obj.get("intent_adjustments") or {}).items():
        ks = str(k).strip()
        if not ks.startswith("intent_"):
            continue
        if ks not in valid_vars:
            continue
        try:
            result.intent_adjustments[ks] = float(np.clip(float(v), 0.05, 0.95))
        except (TypeError, ValueError):
            pass

    result.compute_combined()
    result.ok = True
    return result


# ── Rule-based fallback ───────────────────────────────────────────────────────
def rule_based_embodied_reward(pose: PoseSnapshot) -> EmbodiedRewardResult:
    """
    Fast rule-based fallback when LLM is unavailable.
    Encodes basic biomechanics heuristics.
    """
    result = EmbodiedRewardResult()

    # Posture score
    upright = float(np.clip(pose.com_z, 0.0, 1.0))
    stable = float(np.clip(pose.posture_stability, 0.0, 1.0))
    tilt = float(np.clip(1.0 - abs(pose.torso_pitch) * 3.0, 0.0, 1.0))
    result.posture_score = float(upright * 0.4 + stable * 0.4 + tilt * 0.2)

    # Gait quality
    contact_sym = float(np.clip(1.0 - abs(pose.foot_contact_l - pose.foot_contact_r), 0.0, 1.0))
    phase_anti = float(np.clip(abs(pose.gait_phase_l - pose.gait_phase_r) * 2.0, 0.0, 1.0))
    bias_bal = float(np.clip(1.0 - abs(pose.support_bias - 0.5) * 2.0, 0.0, 1.0))
    result.gait_quality = float(contact_sym * 0.4 + phase_anti * 0.3 + bias_bal * 0.3)

    # Forward progress
    com_fwd = float(np.clip((pose.com_x - 0.40) * 5.0, 0.0, 1.0))
    stride_active = float(np.clip((pose.intent_stride - 0.5) * 2.0, 0.0, 1.0))
    result.forward_progress = float(com_fwd * 0.5 + stride_active * 0.5)

    # Verbal
    issues = []
    if pose.torso_pitch < -0.08:
        issues.append("lean torso forward (torso_pitch too negative)")
    if pose.com_x < 0.44 and pose.intent_stride > 0.55:
        issues.append("shift CoM forward before increasing stride")
    if abs(pose.foot_contact_l - pose.foot_contact_r) > 0.25:
        issues.append("uneven foot contact — stabilize support leg")
    if pose.com_z < 0.35:
        issues.append("critical: robot fell, activate recovery sequence")
    result.verbal = "; ".join(issues) if issues else "movement looks reasonable"
    result.priority_issue = issues[0] if issues else ""

    # Basic biomechanics seeds
    result.seeds = [
        {"from_": "spine_pitch", "to": "com_x", "weight": 0.35, "alpha": 0.05},
        {"from_": "intent_stride", "to": "intent_torso_forward", "weight": 0.30, "alpha": 0.05},
        {"from_": "foot_contact_l", "to": "support_bias", "weight": 0.25, "alpha": 0.05},
        {"from_": "foot_contact_r", "to": "support_bias", "weight": 0.25, "alpha": 0.05},
    ]

    # Intent adjustments
    if pose.torso_pitch < -0.08 and pose.intent_torso_forward < 0.58:
        result.intent_adjustments["intent_torso_forward"] = min(0.72, pose.intent_torso_forward + 0.12)
    if pose.com_z < 0.35 and pose.intent_stride > 0.52:
        result.intent_adjustments["intent_stride"] = 0.50
        result.intent_adjustments["intent_stop_recover"] = 0.78

    result.compute_combined()
    result.ok = True
    return result


# ── Main async fetcher ────────────────────────────────────────────────────────
async def fetch_embodied_reward(
    pose: PoseSnapshot,
    valid_vars: list[str],
    llm_url: str | None = None,
    llm_model: str | None = None,
) -> EmbodiedRewardResult:
    """
    Async LLM call для embodied reward. Falls back to rule-based on error.
    """
    url = (llm_url or get_ollama_generate_url()).strip()
    model = (llm_model or get_ollama_model()).strip()

    if not url:
        return rule_based_embodied_reward(pose) if _fallback_enabled() else EmbodiedRewardResult(error="no llm_url")

    # Normalize URL
    if not url.endswith("/generate"):
        if "/api/" not in url:
            url = url.rstrip("/") + "/api/generate"
        elif not url.endswith("/generate"):
            url = url.rsplit("/", 1)[0] + "/generate"

    prompt = build_embodied_reward_prompt(pose, list(valid_vars))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        **ollama_think_disabled_payload(),
        "options": {"temperature": 0.1, "num_predict": 512},
    }

    timeout = _timeout()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            raw = (resp.json().get("response") or "").strip()
            if not raw:
                raise RuntimeError("empty response from LLM")
    except Exception as e:
        if _fallback_enabled():
            fb = rule_based_embodied_reward(pose)
            fb.error = f"llm_failed: {e!r}"
            return fb
        return EmbodiedRewardResult(error=str(e))

    result = parse_embodied_reward_response(raw, set(valid_vars))
    if not result.ok and _fallback_enabled():
        fb = rule_based_embodied_reward(pose)
        fb.error = f"parse_failed: {result.error}"
        return fb
    return result


# ── EmbodiedRewardController ──────────────────────────────────────────────────
class EmbodiedRewardController:
    """
    Контроллер для интеграции в Simulation.

    Использование в simulation.py:
      self._embodied_reward_ctrl = EmbodiedRewardController()

    В _run_single_agent_timestep_inner:
      if self._embodied_reward_ctrl.should_run(self.tick):
          asyncio.create_task(self._run_embodied_reward_async())

    В отдельном async методе:
      result = await self._embodied_reward_ctrl.run(pose, sim)
    """

    def __init__(self):
        self._last_tick: int = -999_999
        self._results: deque[EmbodiedRewardResult] = deque(maxlen=50)
        self._pending: bool = False
        self._last_result: EmbodiedRewardResult | None = None

        # Статистика
        self.total_runs: int = 0
        self.total_seeds_injected: int = 0
        self.total_llm_ok: int = 0
        self.total_fallback: int = 0

    def should_run(self, tick: int) -> bool:
        if not embodied_reward_enabled():
            return False
        if self._pending:
            return False
        return (tick - self._last_tick) >= _every()

    async def run(
        self,
        pose: PoseSnapshot,
        agent,                          # RKKAgent
        locomotion_ctrl=None,           # LocomotionController
        motor_cortex=None,              # MotorCortexLibrary
        llm_url: str | None = None,
        llm_model: str | None = None,
    ) -> EmbodiedRewardResult:
        """
        Полный цикл:
          1. Fetch LLM reward
          2. Apply reward to locomotion_ctrl and motor_cortex
          3. Inject seeds into GNN
          4. Apply intent adjustments to agent env
        """
        self._pending = True
        self._last_tick = pose.tick
        valid_vars = list(agent.graph.nodes.keys())

        try:
            result = await fetch_embodied_reward(pose, valid_vars, llm_url, llm_model)
        except Exception as e:
            result = EmbodiedRewardResult(error=str(e))
        finally:
            self._pending = False

        self.total_runs += 1
        if result.ok:
            if result.error:
                self.total_fallback += 1
            else:
                self.total_llm_ok += 1
        self._results.append(result)
        self._last_result = result

        if not result.ok:
            return result

        w = _weight()

        # Apply reward to locomotion controller
        if locomotion_ctrl is not None and result.combined_reward != 0.0:
            # Blend LLM reward into the CPG reward history
            blended = w * result.combined_reward + (1.0 - w) * (
                float(np.mean(locomotion_ctrl._reward_history[-8:])) if locomotion_ctrl._reward_history else 0.0
            )
            locomotion_ctrl._reward_history.append(blended)

        # Apply reward to motor cortex programs
        if motor_cortex is not None and len(motor_cortex.programs) > 0:
            nodes = dict(agent.graph.nodes)
            cpg_out = {}  # no CPG targets available here — use nodes
            posture = float(nodes.get("posture_stability", nodes.get("phys_posture_stability", 0.5)))
            foot_l = float(nodes.get("foot_contact_l", nodes.get("phys_foot_contact_l", 0.5)))
            foot_r = float(nodes.get("foot_contact_r", nodes.get("phys_foot_contact_r", 0.5)))
            mc_reward = result.combined_reward * w
            motor_cortex.push_and_train(nodes, cpg_out, mc_reward, posture, foot_l, foot_r)

        # Inject causal seeds from LLM biomechanics knowledge
        if _seeds_enabled() and result.seeds:
            inj = agent.inject_text_priors(result.seeds)
            self.total_seeds_injected += int(inj.get("injected", 0))

        # Apply intent adjustments to the environment
        if result.intent_adjustments:
            env = agent.env
            fn = getattr(env, "intervene", None)
            if not callable(fn):
                # Try base_env
                base = getattr(env, "base_env", None)
                if base is not None:
                    fn = getattr(base, "intervene", None)
            if callable(fn):
                for intent_var, target_val in result.intent_adjustments.items():
                    if intent_var in agent.graph.nodes:
                        try:
                            fn(intent_var, float(target_val), count_intervention=False)
                        except TypeError:
                            try:
                                fn(intent_var, float(target_val))
                            except Exception:
                                pass

        return result

    def snapshot(self) -> dict[str, Any]:
        lr = self._last_result
        return {
            "enabled": embodied_reward_enabled(),
            "every_ticks": _every(),
            "llm_weight": _weight(),
            "total_runs": self.total_runs,
            "total_seeds_injected": self.total_seeds_injected,
            "total_llm_ok": self.total_llm_ok,
            "total_fallback": self.total_fallback,
            "pending": self._pending,
            "last_tick": self._last_tick,
            "last_result": {
                "ok": lr.ok if lr else False,
                "posture_score": round(lr.posture_score, 3) if lr else 0.0,
                "gait_quality": round(lr.gait_quality, 3) if lr else 0.0,
                "forward_progress": round(lr.forward_progress, 3) if lr else 0.0,
                "combined_reward": round(lr.combined_reward, 4) if lr else 0.0,
                "verbal": (lr.verbal[:100] if lr else ""),
                "priority_issue": (lr.priority_issue[:100] if lr else ""),
                "n_seeds": len(lr.seeds) if lr else 0,
                "n_intents": len(lr.intent_adjustments) if lr else 0,
                "error": (lr.error[:100] if lr else ""),
            } if lr else None,
        }
