"""
skill_library.py — Phase B: иерархия действий, уровень 2 (Skill Library).

Скилл = (предусловие, последовательность (var, value), постусловие, теги целей).
Встроенные шаблоны для humanoid; статистика success_rate; история для будущего auto-discovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


StateFn = Callable[[dict], bool]


@dataclass
class Skill:
    name: str
    precondition: StateFn
    action_sequence: list[tuple[str, float]]
    postcondition: StateFn
    goals: frozenset[str] = field(default_factory=lambda: frozenset({"walk"}))
    success_rate: float = 0.5
    uses: int = 0


def _cz(s: dict) -> float:
    return float(s.get("com_z", s.get("phys_com_z", 0.5)))


class SkillLibrary:
    """
    Переиспользуемые моторные последовательности.
    select_skill(state, goal) — фильтр по goals + precondition.
    """

    BUILT_IN_SKILLS: list[Skill] = [
        Skill(
            name="stand_up",
            goals=frozenset({"stand"}),
            precondition=lambda s: _cz(s) < 0.38,
            action_sequence=[
                ("lhip", 0.7),
                ("rhip", 0.7),
                ("lknee", 0.3),
                ("rknee", 0.3),
                ("lhip", 0.5),
                ("rhip", 0.5),
            ],
            postcondition=lambda s: _cz(s) > 0.42,
        ),
        Skill(
            name="step_forward_L",
            goals=frozenset({"walk"}),
            precondition=lambda s: _cz(s) > 0.34,
            action_sequence=[
                ("lhip", 0.65),
                ("lknee", 0.4),
                ("lankle", 0.55),
                ("lhip", 0.48),
                ("lknee", 0.5),
            ],
            postcondition=lambda s: True,
        ),
        Skill(
            name="step_forward_R",
            goals=frozenset({"walk"}),
            precondition=lambda s: _cz(s) > 0.34,
            action_sequence=[
                ("rhip", 0.65),
                ("rknee", 0.4),
                ("rankle", 0.55),
                ("rhip", 0.48),
                ("rknee", 0.5),
            ],
            postcondition=lambda s: True,
        ),
    ]

    def __init__(self) -> None:
        self.skills: list[Skill] = list(self.BUILT_IN_SKILLS)
        self._execution_history: list[dict] = []

    def select_skill(self, state: dict, goal: str = "walk") -> Skill | None:
        g = (goal or "walk").strip().lower()
        applicable: list[Skill] = []
        for sk in self.skills:
            if sk.goals and g not in sk.goals:
                continue
            try:
                if sk.precondition(state):
                    applicable.append(sk)
            except Exception:
                continue
        if not applicable:
            return None
        noise = np.random.normal(0.0, 0.08, size=len(applicable))
        best_i = max(
            range(len(applicable)),
            key=lambda i: applicable[i].success_rate + float(noise[i]),
        )
        return applicable[best_i]

    def ensure_harder_walk_variants(self) -> str | None:
        """
        Self-curriculum: если базовые шаги освоены, добавляем более длинные / резкие варианты.
        Возвращает краткое описание или None.
        """
        names = {s.name for s in self.skills}
        if "step_forward_L_hard" in names or "step_forward_R_hard" in names:
            return None
        base = [s for s in self.skills if s.name in ("step_forward_L", "step_forward_R")]
        if len(base) < 2:
            return None
        if not all(s.uses >= 25 and s.success_rate >= 0.82 for s in base):
            return None

        self.skills.append(
            Skill(
                name="step_forward_L_hard",
                goals=frozenset({"walk"}),
                precondition=lambda s: _cz(s) > 0.36,
                action_sequence=[
                    ("lhip", 0.72),
                    ("lknee", 0.32),
                    ("lankle", 0.58),
                    ("lhip", 0.42),
                    ("lknee", 0.55),
                    ("lankle", 0.48),
                    ("lhip", 0.50),
                ],
                postcondition=lambda s: _cz(s) > 0.36,
            )
        )
        self.skills.append(
            Skill(
                name="step_forward_R_hard",
                goals=frozenset({"walk"}),
                precondition=lambda s: _cz(s) > 0.36,
                action_sequence=[
                    ("rhip", 0.72),
                    ("rknee", 0.32),
                    ("rankle", 0.58),
                    ("rhip", 0.42),
                    ("rknee", 0.55),
                    ("rankle", 0.48),
                    ("rhip", 0.50),
                ],
                postcondition=lambda s: _cz(s) > 0.36,
            )
        )
        return "step_forward_L_hard, step_forward_R_hard"

    def record_outcome(
        self, skill: Skill, state_after: dict, reward: float
    ) -> None:
        skill.uses += 1
        try:
            success = bool(skill.postcondition(state_after))
        except Exception:
            success = False
        skill.success_rate = 0.9 * skill.success_rate + 0.1 * (1.0 if success else 0.0)
        self._execution_history.append(
            {
                "skill": skill.name,
                "reward": float(reward),
                "success": success,
                "uses": skill.uses,
            }
        )

    def snapshot(self) -> dict:
        return {
            "n_skills": len(self.skills),
            "skills": [
                {
                    "name": s.name,
                    "uses": s.uses,
                    "success_rate": round(s.success_rate, 3),
                    "goals": sorted(s.goals),
                }
                for s in self.skills
            ],
            "history_len": len(self._execution_history),
        }
