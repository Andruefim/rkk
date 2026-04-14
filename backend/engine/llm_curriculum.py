"""
llm_curriculum.py — Level 2-E: LLM Curriculum Generator.

LLM динамически генерирует учебный план для гуманоида на основе
текущих навыков, истории падений и фазы обучения.

Проблема с hardcoded curriculum:
  stand → walk — это слишком грубо. Реальная биомеханика требует:
  1. Стоять неподвижно
  2. Небольшой вес на левую ногу
  3. Шаг правой
  4. Перенос веса
  5. ... каждый этап с конкретными intent_* targets

Решение:
  LLMCurriculumGenerator опрашивает LLM каждые N тиков плато:
    Input: текущий skill stats + fall history + pose metrics
    Output: JSON stage с конкретными targets и условиями перехода

  CurriculumStage: один этап с:
    - required_conditions: dict условий для перехода на следующий
    - intent_targets: dict[str, float] — intent_* которые нужно удерживать
    - success_metric: какой metric мерить
    - description: описание для UI

  CurriculumScheduler: управляет переходами между этапами,
    обновляет SkillLibrary, инжектирует seeds в GNN.

RKK_LLM_CURRICULUM=1              — включить (default)
RKK_LLM_CURRICULUM_EVERY=800      — тиков плато до следующего этапа
RKK_LLM_CURRICULUM_TIMEOUT=60     — HTTP timeout
RKK_LLM_CURRICULUM_MAX_STAGES=12  — максимум этапов в плане
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

from engine.llm_json_extract import parse_json_object_loose, parse_json_array_loose
from engine.ollama_env import get_ollama_generate_url, get_ollama_model, ollama_think_disabled_payload


def curriculum_enabled() -> bool:
    return os.environ.get("RKK_LLM_CURRICULUM", "1").strip().lower() not in (
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


# ── Stage definitions ─────────────────────────────────────────────────────────
@dataclass
class CurriculumStage:
    """Один этап обучения."""
    stage_id: int
    name: str
    description: str
    # Intent targets to maintain during this stage (agent will try to achieve these)
    intent_targets: dict[str, float] = field(default_factory=dict)
    # Conditions to advance to next stage (metric: threshold)
    advance_conditions: dict[str, float] = field(default_factory=dict)
    # GNN seeds to inject when entering this stage
    seeds: list[dict] = field(default_factory=list)
    # Skill goals to activate
    skill_goals: list[str] = field(default_factory=lambda: ["stand"])
    # Min ticks before we can advance (even if conditions met)
    min_ticks: int = 200
    # VL bounds adjustments
    vl_warmup_override: int | None = None

    # Runtime state
    entered_tick: int = 0
    ticks_in_stage: int = 0
    advance_checks_passed: int = 0

    def is_ready_to_advance(
        self,
        tick: int,
        metrics: dict[str, float],
    ) -> bool:
        """Check if all advance conditions are met."""
        if self.ticks_in_stage < self.min_ticks:
            return False
        for metric, threshold in self.advance_conditions.items():
            val = float(metrics.get(metric, 0.0))
            if val < threshold:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "description": self.description,
            "intent_targets": self.intent_targets,
            "advance_conditions": self.advance_conditions,
            "skill_goals": self.skill_goals,
            "min_ticks": self.min_ticks,
            "ticks_in_stage": self.ticks_in_stage,
        }


# ── Без ручных этапов: прогрессия из intrinsic objective; этапы только от LLM при необходимости ──
_INTRINSIC_PLACEHOLDER_STAGE = CurriculumStage(
    stage_id=-1,
    name="intrinsic",
    description="No hand-authored stages; progression from compression (intrinsic objective).",
    intent_targets={},
    advance_conditions={},
    seeds=[],
    skill_goals=[],
    min_ticks=999999,
)

DEFAULT_CURRICULUM: list[CurriculumStage] = []


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_curriculum_prompt(
    current_stage: CurriculumStage,
    skill_stats: dict[str, Any],
    fall_summary: str,
    pose_metrics: dict[str, float],
    valid_intent_vars: list[str],
    valid_graph_vars: list[str],
) -> str:
    pose_str = "\n".join(
        f"  {k}={v:.3f}" for k, v in sorted(pose_metrics.items())
        if v != 0.5
    )
    skills_str = json.dumps(skill_stats, indent=2)[:800]
    intents_str = ", ".join(valid_intent_vars)
    vars_str = ", ".join(sorted(valid_graph_vars)[:50])

    return f"""You are designing a step-by-step motor learning curriculum for a PyBullet humanoid robot.

CURRENT LEARNING STAGE: "{current_stage.name}" (stage {current_stage.stage_id})
  Description: {current_stage.description}
  Ticks in stage: {current_stage.ticks_in_stage}
  Status: {"PLATEAU — generating next stage" if current_stage.ticks_in_stage > 0 else "just entered"}

CURRENT PERFORMANCE:
{pose_str}

SKILL LIBRARY STATS:
{skills_str}

FALL HISTORY SUMMARY:
{fall_summary or "No falls recorded yet."}

AVAILABLE INTENT VARIABLES (use these for intent_targets):
{intents_str}

AVAILABLE GRAPH VARIABLES (use these for seeds):
{vars_str}

TASK: Design the NEXT curriculum stage after "{current_stage.name}".
Follow biomechanics principles:
  - Each stage should build on the previous with one new challenge
  - intent_targets should be specific float values (0.0..1.0)  
  - advance_conditions should use observable metrics: posture_stability, com_x_mean, gait_symmetry,
    foot_contact_min, fall_rate_recent (all 0..1)
  - seeds should encode the key causal relationship being learned

Respond with ONLY valid JSON (no markdown):
{{
  "name": "<stage name, snake_case>",
  "description": "<1 sentence what the robot learns>",
  "reasoning": "<why this is the right next step given current performance>",
  "intent_targets": {{
    "<intent_var>": <float 0-1>
  }},
  "advance_conditions": {{
    "<metric>": <threshold float 0-1>
  }},
  "seeds": [
    {{"from_": "<var>", "to": "<var>", "weight": <0.2-0.6>}}
  ],
  "skill_goals": ["<walk|stand>"],
  "min_ticks": <int 200-800>
}}

Rules:
- 2-5 intent_targets, values should be achievable given current performance
- 1-3 advance_conditions with realistic thresholds
- 2-5 seeds encoding the key causal relationship
- Be PROGRESSIVE: one step harder than current stage
- If fall_rate is high, make the stage easier first
"""


# ── Response parser ────────────────────────────────────────────────────────────
def parse_curriculum_stage(
    raw: str,
    stage_id: int,
    valid_intent_vars: set[str],
    valid_graph_vars: set[str],
) -> CurriculumStage | None:
    obj = parse_json_object_loose(raw)
    if not obj:
        return None

    name = str(obj.get("name", f"stage_{stage_id}")).strip()[:60]
    description = str(obj.get("description", "")).strip()[:200]

    # Parse intent_targets
    intent_targets: dict[str, float] = {}
    for k, v in (obj.get("intent_targets") or {}).items():
        if str(k) in valid_intent_vars:
            try:
                intent_targets[str(k)] = float(np.clip(float(v), 0.05, 0.95))
            except (TypeError, ValueError):
                pass

    # Parse advance_conditions
    advance_conditions: dict[str, float] = {}
    valid_metrics = {
        "posture_stability", "com_x_mean", "gait_symmetry",
        "foot_contact_min", "fall_rate_recent",
    }
    for k, v in (obj.get("advance_conditions") or {}).items():
        if str(k) in valid_metrics:
            try:
                advance_conditions[str(k)] = float(np.clip(float(v), 0.0, 1.0))
            except (TypeError, ValueError):
                pass

    # Parse seeds
    seeds: list[dict] = []
    for s in (obj.get("seeds") or []):
        if not isinstance(s, dict):
            continue
        fr = str(s.get("from_") or s.get("from") or "").strip()
        to = str(s.get("to") or "").strip()
        if not fr or not to or fr not in valid_graph_vars or to not in valid_graph_vars:
            continue
        try:
            w = float(np.clip(float(s.get("weight", 0.3)), 0.1, 0.7))
        except (TypeError, ValueError):
            w = 0.3
        seeds.append({"from_": fr, "to": to, "weight": w, "alpha": 0.06})

    skill_goals = [s for s in (obj.get("skill_goals") or ["walk"]) if s in ("walk", "stand")]
    if not skill_goals:
        skill_goals = ["walk"]

    try:
        min_ticks = int(np.clip(int(obj.get("min_ticks", 400)), 100, 1200))
    except (TypeError, ValueError):
        min_ticks = 400

    if not intent_targets and not advance_conditions:
        return None

    return CurriculumStage(
        stage_id=stage_id,
        name=name,
        description=description,
        intent_targets=intent_targets,
        advance_conditions=advance_conditions,
        seeds=seeds,
        skill_goals=skill_goals,
        min_ticks=min_ticks,
    )


# ── Curriculum Scheduler ──────────────────────────────────────────────────────
class CurriculumScheduler:
    """
    Управляет последовательностью этапов обучения.

    Интеграция в simulation.py:
      scheduler.tick(tick, obs, skill_stats, fall_summary)
      → возвращает (current_stage, advanced: bool)

    При переходе этапа:
      - Инжектирует seeds в GNN
      - Обновляет skill_goals для SkillLibrary
      - Регулирует intent_* через agent.env.intervene()
    """

    def __init__(self):
        self._stages: list[CurriculumStage] = list(DEFAULT_CURRICULUM)
        self._current_idx: int = 0
        self._generating: bool = False
        self._last_generate_tick: int = -999_999
        self._plateau_ticks: int = 0
        self._last_advance_tick: int = 0

        # History of metrics for plateau detection
        self._metrics_history: deque[dict[str, float]] = deque(maxlen=100)

        self.total_stages_generated: int = 0
        self.total_advances: int = 0
        self._last_generated_stage: CurriculumStage | None = None

    @property
    def current_stage(self) -> CurriculumStage:
        if not self._stages:
            return _INTRINSIC_PLACEHOLDER_STAGE
        return self._stages[min(self._current_idx, len(self._stages) - 1)]

    def compute_metrics(self, obs: dict[str, float]) -> dict[str, float]:
        """Compute advance condition metrics from observation."""
        def g(k: str, d: float = 0.5) -> float:
            return float(obs.get(k, obs.get(f"phys_{k}", d)))

        posture = g("posture_stability", 0.5)
        com_x = g("com_x", 0.5)
        foot_l = g("foot_contact_l", 0.5)
        foot_r = g("foot_contact_r", 0.5)
        gait_l = g("gait_phase_l", 0.5)
        gait_r = g("gait_phase_r", 0.5)
        bias = g("support_bias", 0.5)

        gait_sym = float(np.clip(1.0 - abs(gait_l - gait_r) * 2.0, 0.0, 1.0))
        foot_min = min(foot_l, foot_r)
        bias_range = abs(bias - 0.5)  # how much bias shifts

        return {
            "posture_stability": posture,
            "com_x_mean": com_x,
            "gait_symmetry": gait_sym,
            "foot_contact_min": foot_min,
            "support_bias_range": bias_range,
            "fall_rate_recent": 0.0,  # updated externally
        }

    def tick(
        self,
        tick: int,
        obs: dict[str, float],
        fallen: bool,
        fall_rate: float = 0.0,
    ) -> tuple[CurriculumStage, bool]:
        """
        Update scheduler. Returns (current_stage, advanced).
        """
        if not curriculum_enabled() or not self._stages:
            return self.current_stage, False

        stage = self.current_stage
        stage.ticks_in_stage += 1
        self._metrics_history.append(self.compute_metrics(obs))

        if fallen:
            return stage, False

        # Compute mean metrics over recent history
        if len(self._metrics_history) < 20:
            return stage, False
        recent = list(self._metrics_history)[-20:]
        mean_metrics = {
            k: float(np.mean([m.get(k, 0.0) for m in recent]))
            for k in recent[0].keys()
        }
        mean_metrics["fall_rate_recent"] = fall_rate

        # Check advance conditions
        if stage.is_ready_to_advance(tick, mean_metrics):
            advanced = self._advance_stage(tick)
            if advanced:
                return self.current_stage, True

        return stage, False

    def _advance_stage(self, tick: int) -> bool:
        """Move to next stage. Returns True if advanced."""
        if self._current_idx + 1 >= len(self._stages):
            # No more predefined stages
            return False

        self._current_idx += 1
        self._last_advance_tick = tick
        self.total_advances += 1
        new_stage = self.current_stage
        new_stage.entered_tick = tick
        new_stage.ticks_in_stage = 0
        print(f"[Curriculum] Stage {self._current_idx}: {new_stage.name}")
        return True

    def inject_stage_seeds(self, agent) -> int:
        """Inject current stage seeds into GNN."""
        stage = self.current_stage
        if not stage.seeds:
            return 0
        result = agent.inject_text_priors(stage.seeds)
        return int(result.get("injected", 0))

    def apply_stage_intents(self, agent_env) -> None:
        """Softly apply stage intent_targets to environment."""
        stage = self.current_stage
        if not stage.intent_targets:
            return
        fn = getattr(agent_env, "intervene", None)
        if not callable(fn):
            base = getattr(agent_env, "base_env", None)
            if base:
                fn = getattr(base, "intervene", None)
        if not callable(fn):
            return
        # Apply softly — only move 20% toward target per application
        for var, target in stage.intent_targets.items():
            try:
                obs = dict(agent_env.observe())
                current = float(obs.get(var, obs.get(f"phys_{var}", 0.5)))
                soft_target = current + 0.2 * (target - current)
                fn(var, float(np.clip(soft_target, 0.05, 0.95)), count_intervention=False)
            except Exception:
                pass

    async def maybe_generate_next_stage_llm(
        self,
        tick: int,
        skill_stats: dict[str, Any],
        fall_summary: str,
        pose_metrics: dict[str, float],
        valid_intent_vars: list[str],
        valid_graph_vars: list[str],
        llm_url: str | None = None,
        llm_model: str | None = None,
    ) -> CurriculumStage | None:
        """
        Generate next stage via LLM when we reach end of predefined curriculum.
        """
        every = _env_int("RKK_LLM_CURRICULUM_EVERY", 800)
        max_stages = _env_int("RKK_LLM_CURRICULUM_MAX_STAGES", 12)

        if not curriculum_enabled():
            return None
        if self._generating:
            return None
        if len(self._stages) >= max_stages:
            return None
        if self._current_idx < len(self._stages) - 2:
            return None  # still have predefined stages ahead
        if tick - self._last_generate_tick < every:
            return None

        url = (llm_url or get_ollama_generate_url()).strip()
        model = (llm_model or get_ollama_model()).strip()
        if not url:
            return None

        self._generating = True
        self._last_generate_tick = tick
        stage_id = len(self._stages)

        # Normalize URL
        if not url.endswith("/generate"):
            if "/api/" not in url:
                url = url.rstrip("/") + "/api/generate"
            elif not url.endswith("/generate"):
                url = url.rsplit("/", 1)[0] + "/generate"

        prompt = build_curriculum_prompt(
            current_stage=self.current_stage,
            skill_stats=skill_stats,
            fall_summary=fall_summary,
            pose_metrics=pose_metrics,
            valid_intent_vars=valid_intent_vars,
            valid_graph_vars=valid_graph_vars,
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **ollama_think_disabled_payload(),
            "options": {"temperature": 0.15, "num_predict": 600},
        }

        try:
            timeout = _env_float("RKK_LLM_CURRICULUM_TIMEOUT", 60.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                raw = (resp.json().get("response") or "").strip()
                stage = parse_curriculum_stage(
                    raw, stage_id, set(valid_intent_vars), set(valid_graph_vars)
                )
                if stage is not None:
                    self._stages.append(stage)
                    self._last_generated_stage = stage
                    self.total_stages_generated += 1
                    print(f"[Curriculum] LLM generated stage {stage_id}: {stage.name}")
                    return stage
        except Exception as e:
            print(f"[Curriculum] LLM generation failed: {e}")
        finally:
            self._generating = False

        return None

    def snapshot(self) -> dict[str, Any]:
        stage = self.current_stage
        return {
            "enabled": curriculum_enabled(),
            "current_stage_idx": self._current_idx,
            "current_stage_name": stage.name,
            "current_stage_description": stage.description,
            "ticks_in_stage": stage.ticks_in_stage,
            "n_stages_total": len(self._stages),
            "n_stages_generated": self.total_stages_generated,
            "total_advances": self.total_advances,
            "intent_targets": dict(stage.intent_targets),
            "advance_conditions": dict(stage.advance_conditions),
            "skill_goals": list(stage.skill_goals),
            "generating": self._generating,
        }
