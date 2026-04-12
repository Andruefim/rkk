"""
physical_curriculum.py — расширенный физический учебный план.

25+ этапов охватывающих полный репертуар человеческих движений.
Организован в Tiers — каждый Tier требует мастерства предыдущего.

TIER 0 — Базовый контроль (стоять, контролировать тело)
  0.0 static_stance      → стоять неподвижно
  0.1 weight_shift       → переносить вес между ногами
  0.2 forward_lean       → устойчивый наклон вперёд
  0.3 arm_balance        → руки как балансировочный механизм

TIER 1 — Простая локомоция
  1.0 slow_step          → один медленный шаг
  1.1 continuous_walk    → непрерывная ходьба
  1.2 walk_backward      → ходьба назад
  1.3 sidestep_left      → шаги в сторону влево
  1.4 sidestep_right     → шаги в сторону вправо
  1.5 turn_in_place      → поворот на месте

TIER 2 — Вертикальные переходы
  2.0 controlled_lower   → контролируемое приседание
  2.1 squat_hold         → удержание приседания
  2.2 squat_to_stand     → встать из приседа
  2.3 sit_down           → сесть (сгиб в тазобедренных)
  2.4 sit_to_stand       → встать из сидячего положения
  2.5 kneel_down         → встать на одно колено
  2.6 kneel_to_stand     → встать с колена

TIER 3 — Наклоны и манипуляция
  3.0 forward_bend       → наклон вперёд (дотянуться до объекта)
  3.1 reach_up           → тянуться вверх
  3.2 reach_forward      → тянуться вперёд

TIER 4 — Сложная локомоция
  4.0 walk_fast          → быстрая ходьба / медленный бег
  4.1 balance_one_leg    → стоять на одной ноге
  4.2 crouch_walk        → ходьба в полуприседе

TIER 5 — Восстановление
  5.0 fall_recovery_back → встать после падения назад
  5.1 fall_recovery_fwd  → встать после падения вперёд
  5.2 roll_recovery      → перекат и подъём

Каждый Tier автоматически разблокируется при мастерстве предыдущего.
LLM Teacher генерирует дополнительные этапы при достижении Tier 5.

Структура advance_conditions совместима с CurriculumScheduler из Level 2-E.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.llm_curriculum import CurriculumStage


# ── Tier definitions ──────────────────────────────────────────────────────────
@dataclass
class PhysicalSkill:
    """Один физический навык с метаданными."""
    tier: int
    skill_id: str           # "1.1_continuous_walk"
    name: str
    description: str
    prerequisites: list[str]  # skill_ids которые нужны до
    stage: CurriculumStage
    estimated_ticks: int    # приблизительно сколько тиков для обучения
    mastery_threshold: float = 0.75  # advance_condition порог

    def is_unlocked(self, mastered: set[str]) -> bool:
        return all(p in mastered for p in self.prerequisites)


def _make_stage(stage_id: int, **kwargs) -> CurriculumStage:
    return CurriculumStage(stage_id=stage_id, **kwargs)


# ── Tier 0: Basic control ─────────────────────────────────────────────────────
TIER0_SKILLS: list[PhysicalSkill] = [
    PhysicalSkill(
        tier=0, skill_id="0.0_static_stance", name="static_stance",
        description="Stand still. Both feet planted, minimal sway.",
        prerequisites=[],
        estimated_ticks=500,
        stage=_make_stage(0,
            name="static_stance",
            description="Stand still with both feet firmly planted.",
            intent_targets={
                "intent_stride": 0.50,
                "intent_stop_recover": 0.65,
                "intent_support_left": 0.58,
                "intent_support_right": 0.58,
                "intent_torso_forward": 0.52,
            },
            advance_conditions={"posture_stability": 0.72, "foot_contact_min": 0.62},
            seeds=[
                {"from_": "foot_contact_l", "to": "support_bias", "weight": 0.30, "alpha": 0.05},
                {"from_": "foot_contact_r", "to": "support_bias", "weight": 0.30, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=400,
        ),
    ),
    PhysicalSkill(
        tier=0, skill_id="0.1_weight_shift", name="weight_shift",
        description="Shift weight between legs without losing balance.",
        prerequisites=["0.0_static_stance"],
        estimated_ticks=600,
        stage=_make_stage(1,
            name="weight_shift",
            description="Shift weight left and right while maintaining balance.",
            intent_targets={
                "intent_stride": 0.50,
                "intent_support_left": 0.68,
                "intent_support_right": 0.42,
                "intent_torso_forward": 0.54,
            },
            advance_conditions={"posture_stability": 0.67, "support_bias_range": 0.28},
            seeds=[
                {"from_": "support_bias", "to": "torso_roll", "weight": -0.22, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=400,
        ),
    ),
    PhysicalSkill(
        tier=0, skill_id="0.2_forward_lean", name="forward_lean",
        description="Maintain stable forward lean with CoM ahead of feet.",
        prerequisites=["0.1_weight_shift"],
        estimated_ticks=500,
        stage=_make_stage(2,
            name="forward_lean",
            description="Lean torso forward, keep CoM stable ahead of feet.",
            intent_targets={
                "intent_torso_forward": 0.65,
                "intent_gait_coupling": 0.80,
                "intent_arm_counterbalance": 0.55,
            },
            advance_conditions={"posture_stability": 0.64, "com_x_mean": 0.47},
            seeds=[
                {"from_": "spine_pitch", "to": "com_x", "weight": 0.38, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=400,
        ),
    ),
    PhysicalSkill(
        tier=0, skill_id="0.3_arm_balance", name="arm_balance",
        description="Use arms actively for counterbalancing.",
        prerequisites=["0.2_forward_lean"],
        estimated_ticks=400,
        stage=_make_stage(3,
            name="arm_balance",
            description="Arms counterbalance torso movements actively.",
            intent_targets={
                "intent_arm_counterbalance": 0.70,
                "intent_torso_forward": 0.58,
                "intent_gait_coupling": 0.75,
            },
            advance_conditions={"posture_stability": 0.66, "foot_contact_min": 0.58},
            seeds=[
                {"from_": "proprio_arm_counterbalance", "to": "support_bias", "weight": 0.28, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=300,
        ),
    ),
]

# ── Tier 1: Simple locomotion ──────────────────────────────────────────────────
TIER1_SKILLS: list[PhysicalSkill] = [
    PhysicalSkill(
        tier=1, skill_id="1.0_slow_step", name="slow_step",
        description="Take slow deliberate single steps with full weight transfer.",
        prerequisites=["0.3_arm_balance"],
        estimated_ticks=700,
        stage=_make_stage(4,
            name="slow_step",
            description="One slow step with full weight transfer. No rushing.",
            intent_targets={
                "intent_stride": 0.58,
                "intent_torso_forward": 0.62,
                "intent_gait_coupling": 0.85,
                "intent_arm_counterbalance": 0.60,
            },
            advance_conditions={"posture_stability": 0.62, "gait_symmetry": 0.52},
            seeds=[
                {"from_": "intent_stride", "to": "com_x", "weight": 0.28, "alpha": 0.05},
                {"from_": "gait_phase_l", "to": "foot_contact_l", "weight": 0.32, "alpha": 0.05},
                {"from_": "gait_phase_r", "to": "foot_contact_r", "weight": 0.32, "alpha": 0.05},
            ],
            skill_goals=["walk"], min_ticks=600,
        ),
    ),
    PhysicalSkill(
        tier=1, skill_id="1.1_continuous_walk", name="continuous_walk",
        description="Continuous rhythmic walking gait, symmetric.",
        prerequisites=["1.0_slow_step"],
        estimated_ticks=1000,
        stage=_make_stage(5,
            name="continuous_walk",
            description="Continuous walking. Rhythm, symmetry, forward progress.",
            intent_targets={
                "intent_stride": 0.65,
                "intent_torso_forward": 0.66,
                "intent_gait_coupling": 0.90,
                "intent_arm_counterbalance": 0.62,
            },
            advance_conditions={"posture_stability": 0.64, "gait_symmetry": 0.62, "com_x_mean": 0.48},
            skill_goals=["walk"], min_ticks=800,
        ),
    ),
    PhysicalSkill(
        tier=1, skill_id="1.2_walk_backward", name="walk_backward",
        description="Walk backward with stable balance. CoM shifts back.",
        prerequisites=["1.1_continuous_walk"],
        estimated_ticks=800,
        stage=_make_stage(6,
            name="walk_backward",
            description="Walk backward. Torso upright, weight slightly back.",
            intent_targets={
                "intent_stride": 0.55,
                "intent_torso_forward": 0.44,  # lean back
                "intent_gait_coupling": 0.82,
                "intent_support_left": 0.52,
                "intent_support_right": 0.52,
            },
            advance_conditions={"posture_stability": 0.60, "gait_symmetry": 0.55},
            seeds=[
                {"from_": "intent_torso_forward", "to": "com_x", "weight": -0.30, "alpha": 0.05},
            ],
            skill_goals=["walk"], min_ticks=600,
        ),
    ),
    PhysicalSkill(
        tier=1, skill_id="1.3_sidestep", name="sidestep",
        description="Step sideways left and right. Lateral weight transfer.",
        prerequisites=["1.1_continuous_walk"],
        estimated_ticks=700,
        stage=_make_stage(7,
            name="sidestep",
            description="Step sideways. Strong lateral weight shifts, torso upright.",
            intent_targets={
                "intent_stride": 0.52,
                "intent_torso_forward": 0.52,
                "intent_support_left": 0.72,
                "intent_support_right": 0.34,
                "intent_gait_coupling": 0.70,
            },
            advance_conditions={"posture_stability": 0.62, "support_bias_range": 0.30},
            seeds=[
                {"from_": "support_bias", "to": "gait_phase_l", "weight": 0.35, "alpha": 0.05},
            ],
            skill_goals=["walk"], min_ticks=500,
        ),
    ),
    PhysicalSkill(
        tier=1, skill_id="1.4_turn_in_place", name="turn_in_place",
        description="Rotate body in place with minimal CoM displacement.",
        prerequisites=["1.1_continuous_walk"],
        estimated_ticks=600,
        stage=_make_stage(8,
            name="turn_in_place",
            description="Rotate in place. Feet alternate, torso stays centered.",
            intent_targets={
                "intent_stride": 0.50,
                "intent_torso_forward": 0.52,
                "intent_gait_coupling": 0.78,
                "intent_support_left": 0.60,
                "intent_support_right": 0.45,
            },
            advance_conditions={"posture_stability": 0.62, "foot_contact_min": 0.55},
            skill_goals=["walk"], min_ticks=400,
        ),
    ),
]

# ── Tier 2: Vertical transitions ──────────────────────────────────────────────
TIER2_SKILLS: list[PhysicalSkill] = [
    PhysicalSkill(
        tier=2, skill_id="2.0_controlled_lower", name="controlled_lower",
        description="Slowly lower body (controlled squat descent).",
        prerequisites=["1.1_continuous_walk"],
        estimated_ticks=700,
        stage=_make_stage(9,
            name="controlled_lower",
            description="Slowly bend knees, lower CoM. Control the descent.",
            intent_targets={
                "intent_stride": 0.45,
                "intent_torso_forward": 0.58,
                "intent_stop_recover": 0.60,
                "intent_support_left": 0.60,
                "intent_support_right": 0.60,
            },
            advance_conditions={"posture_stability": 0.60, "com_z_low": 0.45},
            seeds=[
                {"from_": "lknee", "to": "com_z", "weight": -0.35, "alpha": 0.05},
                {"from_": "rknee", "to": "com_z", "weight": -0.35, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=500,
        ),
    ),
    PhysicalSkill(
        tier=2, skill_id="2.1_squat_hold", name="squat_hold",
        description="Hold squat position. Both knees bent, stable.",
        prerequisites=["2.0_controlled_lower"],
        estimated_ticks=600,
        stage=_make_stage(10,
            name="squat_hold",
            description="Hold squat. Knees bent ~90deg, stable for >40 ticks.",
            intent_targets={
                "intent_stride": 0.42,
                "intent_stop_recover": 0.70,
                "intent_torso_forward": 0.62,
                "intent_support_left": 0.65,
                "intent_support_right": 0.65,
            },
            advance_conditions={"posture_stability": 0.58, "squat_hold_duration": 0.60},
            skill_goals=["stand"], min_ticks=500,
        ),
    ),
    PhysicalSkill(
        tier=2, skill_id="2.2_squat_to_stand", name="squat_to_stand",
        description="Rise from squat to standing. Controlled extension.",
        prerequisites=["2.1_squat_hold"],
        estimated_ticks=600,
        stage=_make_stage(11,
            name="squat_to_stand",
            description="Rise from squat. Extend knees and hips simultaneously.",
            intent_targets={
                "intent_stride": 0.50,
                "intent_torso_forward": 0.56,
                "intent_stop_recover": 0.58,
                "intent_support_left": 0.60,
                "intent_support_right": 0.60,
            },
            advance_conditions={"posture_stability": 0.68, "com_z_mean": 0.62},
            seeds=[
                {"from_": "com_z", "to": "posture_stability", "weight": 0.40, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=400,
        ),
    ),
    PhysicalSkill(
        tier=2, skill_id="2.3_sit_to_stand", name="sit_to_stand",
        description="Rise from seated position. Hip extension, forward lean, push up.",
        prerequisites=["2.2_squat_to_stand"],
        estimated_ticks=700,
        stage=_make_stage(12,
            name="sit_to_stand",
            description="Get up from seated. Lean forward, push through feet.",
            intent_targets={
                "intent_torso_forward": 0.68,
                "intent_stride": 0.50,
                "intent_gait_coupling": 0.72,
                "intent_stop_recover": 0.55,
            },
            advance_conditions={"posture_stability": 0.68, "com_z_mean": 0.65},
            skill_goals=["stand"], min_ticks=500,
        ),
    ),
    PhysicalSkill(
        tier=2, skill_id="2.4_kneel_to_stand", name="kneel_to_stand",
        description="Rise from kneeling. Hip drive forward, use momentum.",
        prerequisites=["2.2_squat_to_stand"],
        estimated_ticks=700,
        stage=_make_stage(13,
            name="kneel_to_stand",
            description="Get up from kneel. Step forward with lead foot, drive up.",
            intent_targets={
                "intent_torso_forward": 0.64,
                "intent_stride": 0.56,
                "intent_support_left": 0.70,
                "intent_gait_coupling": 0.80,
            },
            advance_conditions={"posture_stability": 0.65, "com_z_mean": 0.63},
            skill_goals=["stand"], min_ticks=500,
        ),
    ),
]

# ── Tier 3: Reach and manipulation ────────────────────────────────────────────
TIER3_SKILLS: list[PhysicalSkill] = [
    PhysicalSkill(
        tier=3, skill_id="3.0_forward_bend", name="forward_bend",
        description="Bend forward at hips while keeping balance. Reach low.",
        prerequisites=["1.1_continuous_walk", "0.3_arm_balance"],
        estimated_ticks=600,
        stage=_make_stage(14,
            name="forward_bend",
            description="Hinge at hips, reach down. Keep knees soft, back flat.",
            intent_targets={
                "intent_torso_forward": 0.75,
                "intent_stride": 0.44,
                "intent_arm_counterbalance": 0.68,
                "intent_stop_recover": 0.60,
            },
            advance_conditions={"posture_stability": 0.58, "com_x_mean": 0.52},
            seeds=[
                {"from_": "spine_pitch", "to": "com_x", "weight": 0.45, "alpha": 0.05},
                {"from_": "proprio_arm_counterbalance", "to": "posture_stability", "weight": 0.30, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=400,
        ),
    ),
    PhysicalSkill(
        tier=3, skill_id="3.1_reach_up", name="reach_up",
        description="Extend arms upward. Slight torso extension, heel rise possible.",
        prerequisites=["0.3_arm_balance"],
        estimated_ticks=500,
        stage=_make_stage(15,
            name="reach_up",
            description="Arms overhead. Slight back extension. Weight on toes.",
            intent_targets={
                "intent_torso_forward": 0.48,  # slight back lean
                "intent_arm_counterbalance": 0.72,
                "intent_stride": 0.46,
                "intent_support_left": 0.55,
                "intent_support_right": 0.55,
            },
            advance_conditions={"posture_stability": 0.62, "foot_contact_min": 0.52},
            skill_goals=["stand"], min_ticks=400,
        ),
    ),
]

# ── Tier 4: Advanced locomotion ────────────────────────────────────────────────
TIER4_SKILLS: list[PhysicalSkill] = [
    PhysicalSkill(
        tier=4, skill_id="4.0_walk_fast", name="walk_fast",
        description="Faster walking cadence. Higher stride, more momentum.",
        prerequisites=["1.1_continuous_walk"],
        estimated_ticks=800,
        stage=_make_stage(16,
            name="walk_fast",
            description="Faster walk. Stride 0.75+, strong arm swing.",
            intent_targets={
                "intent_stride": 0.76,
                "intent_torso_forward": 0.70,
                "intent_gait_coupling": 0.92,
                "intent_arm_counterbalance": 0.68,
            },
            advance_conditions={"posture_stability": 0.62, "gait_symmetry": 0.65, "com_x_mean": 0.50},
            skill_goals=["walk"], min_ticks=700,
        ),
    ),
    PhysicalSkill(
        tier=4, skill_id="4.1_balance_one_leg", name="balance_one_leg",
        description="Stand on one leg for >20 ticks without falling.",
        prerequisites=["1.1_continuous_walk", "0.3_arm_balance"],
        estimated_ticks=1000,
        stage=_make_stage(17,
            name="balance_one_leg",
            description="Single-leg stance. Ankle stabilization, arms out.",
            intent_targets={
                "intent_stride": 0.45,
                "intent_support_left": 0.85,
                "intent_support_right": 0.20,
                "intent_torso_forward": 0.54,
                "intent_arm_counterbalance": 0.75,
            },
            advance_conditions={"posture_stability": 0.58, "single_leg_hold": 0.55},
            seeds=[
                {"from_": "lankle", "to": "support_bias", "weight": 0.40, "alpha": 0.05},
                {"from_": "proprio_arm_counterbalance", "to": "proprio_balance", "weight": 0.35, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=800,
        ),
    ),
    PhysicalSkill(
        tier=4, skill_id="4.2_crouch_walk", name="crouch_walk",
        description="Walk while crouched. Low CoM, slow, controlled.",
        prerequisites=["1.1_continuous_walk", "2.1_squat_hold"],
        estimated_ticks=900,
        stage=_make_stage(18,
            name="crouch_walk",
            description="Walk with bent knees. Low and slow. No falling.",
            intent_targets={
                "intent_stride": 0.52,
                "intent_torso_forward": 0.62,
                "intent_stop_recover": 0.62,
                "intent_support_left": 0.62,
                "intent_support_right": 0.62,
            },
            advance_conditions={"posture_stability": 0.58, "com_z_low": 0.40, "gait_symmetry": 0.55},
            skill_goals=["walk"], min_ticks=700,
        ),
    ),
]

# ── Tier 5: Fall recovery ─────────────────────────────────────────────────────
TIER5_SKILLS: list[PhysicalSkill] = [
    PhysicalSkill(
        tier=5, skill_id="5.0_fall_recovery_back", name="fall_recovery_back",
        description="Get up after falling backward. Roll to side, push up.",
        prerequisites=["1.0_slow_step"],
        estimated_ticks=800,
        stage=_make_stage(19,
            name="fall_recovery_back",
            description="Fallen backward. Roll, use arms, get to kneeling, then stand.",
            intent_targets={
                "intent_stop_recover": 0.85,
                "intent_torso_forward": 0.68,
                "intent_arm_counterbalance": 0.75,
                "intent_stride": 0.45,
            },
            advance_conditions={"recovery_success_rate": 0.60},
            seeds=[
                {"from_": "intent_stop_recover", "to": "posture_stability", "weight": 0.42, "alpha": 0.05},
            ],
            skill_goals=["stand"], min_ticks=600,
        ),
    ),
    PhysicalSkill(
        tier=5, skill_id="5.1_fall_recovery_fwd", name="fall_recovery_fwd",
        description="Get up after falling forward. Push up, plant feet, extend.",
        prerequisites=["1.0_slow_step"],
        estimated_ticks=800,
        stage=_make_stage(20,
            name="fall_recovery_fwd",
            description="Fallen forward. Push with arms, tuck legs, stand.",
            intent_targets={
                "intent_stop_recover": 0.85,
                "intent_torso_forward": 0.45,
                "intent_arm_counterbalance": 0.80,
                "intent_stride": 0.45,
            },
            advance_conditions={"recovery_success_rate": 0.60},
            skill_goals=["stand"], min_ticks=600,
        ),
    ),
]


# ── Physical Curriculum Manager ───────────────────────────────────────────────
ALL_SKILLS: list[PhysicalSkill] = (
    TIER0_SKILLS + TIER1_SKILLS + TIER2_SKILLS +
    TIER3_SKILLS + TIER4_SKILLS + TIER5_SKILLS
)
ALL_SKILLS_BY_ID = {s.skill_id: s for s in ALL_SKILLS}


class PhysicalCurriculum:
    """
    Manages the full physical skill progression tree.

    Интеграция в simulation.py:
      self._physical_curriculum = PhysicalCurriculum()

    В CurriculumScheduler: при достижении конца DEFAULT_CURRICULUM,
    передаём управление PhysicalCurriculum вместо LLM генерации.

    Unlock sequence:
      mastered = {"0.0_static_stance", "0.1_weight_shift"}
      next_skills = physical_curriculum.get_unlocked(mastered)
      → ["0.2_forward_lean"]  (следующий разблокованный)
    """

    def __init__(self):
        self.mastered: set[str] = set()
        self.failed: set[str] = set()
        self._active_skill_id: str | None = None
        self._active_ticks: int = 0
        self._skill_metrics: dict[str, list[float]] = {}

    def get_unlocked(self) -> list[PhysicalSkill]:
        """Get skills that are unlocked but not yet mastered."""
        return [
            s for s in ALL_SKILLS
            if s.is_unlocked(self.mastered) and s.skill_id not in self.mastered
        ]

    def get_next_recommended(self) -> PhysicalSkill | None:
        """Get lowest-tier unlocked skill to attempt next."""
        unlocked = self.get_unlocked()
        if not unlocked:
            return None
        # Prefer lowest tier, then lowest stage_id
        unlocked.sort(key=lambda s: (s.tier, s.stage.stage_id))
        return unlocked[0]

    def mark_mastered(self, skill_id: str) -> list[PhysicalSkill]:
        """Mark skill as mastered. Returns newly unlocked skills."""
        self.mastered.add(skill_id)
        newly_unlocked = [
            s for s in ALL_SKILLS
            if s.is_unlocked(self.mastered)
            and s.skill_id not in self.mastered
            and not any(p == skill_id and p not in self.mastered
                       for p in s.prerequisites)
        ]
        return newly_unlocked

    def mark_failed(self, skill_id: str) -> None:
        self.failed.add(skill_id)

    def inject_into_scheduler(self, scheduler) -> int:
        """
        Inject next unlocked PhysicalSkill stages into CurriculumScheduler.
        Returns number of stages added.
        """
        next_skill = self.get_next_recommended()
        if next_skill is None:
            return 0
        if len(scheduler._stages) > scheduler._current_idx + 3:
            return 0  # enough stages ahead

        # Add this skill's stage to scheduler
        stage = next_skill.stage
        stage.stage_id = len(scheduler._stages)
        scheduler._stages.append(stage)
        self._active_skill_id = next_skill.skill_id
        return 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "mastered": sorted(self.mastered),
            "failed": sorted(self.failed),
            "active_skill": self._active_skill_id,
            "unlocked_count": len(self.get_unlocked()),
            "total_skills": len(ALL_SKILLS),
            "progress_pct": round(len(self.mastered) / len(ALL_SKILLS) * 100, 1),
        }
