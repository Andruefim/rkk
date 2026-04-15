"""
variable_bootstrap.py — Динамическая онтология переменных.

Заменяет статический VAR_NAMES из constants.py на VariableRegistry,
которая начинает с ~10 физически необходимых переменных и растёт
через VariableDiscovery.

Принцип:
  Мозг младенца не рождается со списком из 60+ переменных.
  Он начинает с сырых сенсорных потоков (проприоцепция, контакт)
  и СТРОИТ свою онтологию. RKK теперь делает то же самое.

Bootstrap переменные (SEED_VARS):
  - com_z:          высота центра масс (необходимо для определения падения)
  - torso_roll:     крен торса (базовое чувство ориентации)
  - torso_pitch:    тангаж торса
  - foot_contact_l: контакт левой стопы (нужно для баланса)
  - foot_contact_r: контакт правой стопы
  - posture_stability: интегральная устойчивость (derived, но необходимо)
  - lhip, rhip:     бёдра (минимум для начала ходьбы)
  - lknee, rknee:   колени
  - intent_stride:  базовый моторный intent

Всё остальное (руки, кубы, голова, sandbox, self_*) — открывается
через VariableDiscovery когда агент обнаруживает что модель
систематически ошибается и нужны новые переменные.

RKK_BOOTSTRAP_ONLY=1  — начинать с SEED_VARS (default: 0 для обратной совместимости)
RKK_BOOTSTRAP_VARS=   — переопределить список Bootstrap через запятую
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.features.humanoid.constants import (
    _RANGES,
    ARM_VARS,
    CUBE_VARS,
    FIXED_BASE_VARS,
    FOOT_VARS,
    HEAD_VARS,
    LEG_VARS,
    MOTOR_INTENT_VARS,
    MOTOR_OBSERVABLE_VARS,
    SANDBOX_VARS,
    SELF_VARS,
    SPINE_VARS,
    TORSO_VARS,
    VAR_NAMES as FULL_VAR_NAMES,
)


# ─── Seed Variables ──────────────────────────────────────────────────────────
# Минимальный набор для начала жизни гуманоида
SEED_VARS: list[str] = [
    # Проприоцепция (где я в пространстве?)
    "com_z",
    "torso_roll",
    "torso_pitch",
    # Контакт с полом (стою ли я?)
    "foot_contact_l",
    "foot_contact_r",
    "posture_stability",
    # Минимальные суставы для локомоции
    "lhip",
    "rhip",
    "lknee",
    "rknee",
    # Базовый моторный intent
    "intent_stride",
]

# Группы переменных для поэтапного открытия
DISCOVERABLE_GROUPS: dict[str, list[str]] = {
    "ankles":     ["lankle", "rankle"],
    "feet":       ["lfoot_z", "rfoot_z"],
    "torso_xy":   ["com_x", "com_y"],
    "spine":      list(SPINE_VARS),
    "head":       list(HEAD_VARS),
    "arms":       list(ARM_VARS),
    "cubes":      list(CUBE_VARS),
    "sandbox":    list(SANDBOX_VARS),
    "motor_intents": [v for v in MOTOR_INTENT_VARS if v != "intent_stride"],
    "motor_obs":  [v for v in MOTOR_OBSERVABLE_VARS
                   if v not in ("foot_contact_l", "foot_contact_r", "posture_stability")],
    "self":       list(SELF_VARS),
}


def _bootstrap_enabled() -> bool:
    """Режим bootstrap: начинаем с SEED_VARS вместо полного VAR_NAMES."""
    v = os.environ.get("RKK_BOOTSTRAP_ONLY", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _custom_bootstrap_vars() -> list[str] | None:
    """Переопределённый список bootstrap переменных через env."""
    raw = os.environ.get("RKK_BOOTSTRAP_VARS", "").strip()
    if not raw:
        return None
    return [v.strip() for v in raw.split(",") if v.strip()]


# ─── VariableRegistry ────────────────────────────────────────────────────────
@dataclass
class DiscoveryEvent:
    """Событие добавления переменной/группы."""
    tick: int
    group: str
    variables: list[str]
    trigger_reason: str      # "high_error", "neurogenesis", "compression_stagnant", "manual"
    target_node: str = ""    # узел с высокой ошибкой, спровоцировавший открытие


class VariableRegistry:
    """
    Реестр активных переменных. Заменяет статический VAR_NAMES.

    Два режима:
      1. bootstrap=False (default): все VAR_NAMES активны (обратная совместимость)
      2. bootstrap=True: начинаем с SEED_VARS, остальные открываются

    API:
      registry.active_vars       → текущий список активных переменных
      registry.discover_group()  → добавить группу переменных
      registry.should_discover() → проверить нужно ли открывать новые
      registry.auto_discover()   → автоматическое открытие по сигналам
    """

    def __init__(self):
        self._bootstrap = _bootstrap_enabled()

        # Определяем начальный набор
        custom = _custom_bootstrap_vars()
        if custom:
            self._active: list[str] = list(custom)
        elif self._bootstrap:
            self._active = list(SEED_VARS)
        else:
            self._active = list(FULL_VAR_NAMES)

        self._active_set: set[str] = set(self._active)
        self._all_possible: set[str] = set(FULL_VAR_NAMES)

        # Какие группы уже открыты
        self._discovered_groups: set[str] = set()
        self._discovery_log: deque[DiscoveryEvent] = deque(maxlen=50)

        # Для auto_discover: счётчики ошибок по потенциальным группам
        self._group_pressure: dict[str, float] = {g: 0.0 for g in DISCOVERABLE_GROUPS}

        # Cooldown между открытиями
        self._last_discovery_tick: int = -9999
        self._discovery_cooldown: int = 300

        self.total_discoveries: int = 0

        if self._bootstrap:
            print(f"[VariableRegistry] Bootstrap mode: {len(self._active)} vars "
                  f"({', '.join(self._active[:5])}...)")
        else:
            print(f"[VariableRegistry] Full mode: {len(self._active)} vars")

    @property
    def active_vars(self) -> list[str]:
        """Текущий список активных переменных (замена VAR_NAMES)."""
        return list(self._active)

    @property
    def active_var_count(self) -> int:
        return len(self._active)

    @property
    def is_bootstrap(self) -> bool:
        return self._bootstrap

    @property
    def undiscovered_count(self) -> int:
        return len(self._all_possible - self._active_set)

    def is_active(self, var: str) -> bool:
        return var in self._active_set

    def get_range(self, var: str) -> tuple[float, float]:
        """Диапазон нормализации для переменной."""
        return _RANGES.get(var, (-1.0, 1.0))

    def discover_group(
        self,
        group_name: str,
        tick: int,
        reason: str = "manual",
        target_node: str = "",
    ) -> list[str]:
        """
        Активировать группу переменных. Возвращает список новых переменных.
        Пустой список если группа уже активна или не существует.
        """
        if group_name in self._discovered_groups:
            return []
        if group_name not in DISCOVERABLE_GROUPS:
            return []

        new_vars = [v for v in DISCOVERABLE_GROUPS[group_name]
                    if v not in self._active_set]
        if not new_vars:
            self._discovered_groups.add(group_name)
            return []

        self._active.extend(new_vars)
        self._active_set.update(new_vars)
        self._discovered_groups.add(group_name)
        self._last_discovery_tick = tick
        self.total_discoveries += 1

        event = DiscoveryEvent(
            tick=tick,
            group=group_name,
            variables=new_vars,
            trigger_reason=reason,
            target_node=target_node,
        )
        self._discovery_log.append(event)

        print(f"[VariableRegistry] 🔓 Discovered group '{group_name}': "
              f"+{len(new_vars)} vars ({', '.join(new_vars[:3])}{'...' if len(new_vars) > 3 else ''}), "
              f"total={len(self._active)}, reason={reason}")

        return new_vars

    def discover_single(
        self,
        var_name: str,
        tick: int,
        reason: str = "neurogenesis",
    ) -> bool:
        """Добавить одну переменную (для NeurogenesisEngine)."""
        if var_name in self._active_set:
            return False
        if var_name not in self._all_possible:
            # Полностью новая переменная (latent node)
            self._all_possible.add(var_name)

        self._active.append(var_name)
        self._active_set.add(var_name)
        self._last_discovery_tick = tick

        event = DiscoveryEvent(
            tick=tick,
            group="single",
            variables=[var_name],
            trigger_reason=reason,
        )
        self._discovery_log.append(event)

        print(f"[VariableRegistry] 🔓 Discovered single var '{var_name}', "
              f"total={len(self._active)}, reason={reason}")
        return True

    def update_pressure(
        self,
        high_error_nodes: list[tuple[str, float]],
        compression_stagnant: bool,
        tick: int,
    ) -> None:
        """
        Обновляет давление на группы по сигналам от VariableDiscovery.

        Идея: если агент систематически ошибается на узлах, которые
        связаны с ещё не открытой группой, давление на эту группу растёт.
        Когда давление превышает порог → группа открывается.
        """
        if not self._bootstrap:
            return  # В полном режиме нечего открывать

        decay = 0.95
        for g in self._group_pressure:
            self._group_pressure[g] *= decay

        # Ищем связи между проблемными узлами и неоткрытыми группами
        for node, score in high_error_nodes:
            for group_name, group_vars in DISCOVERABLE_GROUPS.items():
                if group_name in self._discovered_groups:
                    continue
                # Есть ли связь proблемного узла с переменными из группы?
                relevance = self._compute_relevance(node, group_vars)
                self._group_pressure[group_name] += score * relevance

        # Стагнация компрессии → давление на все неоткрытые группы
        if compression_stagnant:
            for g in self._group_pressure:
                if g not in self._discovered_groups:
                    self._group_pressure[g] += 0.15

    def _compute_relevance(self, node: str, group_vars: list[str]) -> float:
        """
        Оценивает связь проблемного узла с группой переменных.
        Используем эвристику по именам (слабо, но достаточно для bootstrap).
        """
        node_parts = set(node.replace("_", " ").split())
        total = 0.0
        for gv in group_vars:
            gv_parts = set(gv.replace("_", " ").split())
            overlap = len(node_parts & gv_parts)
            if overlap > 0:
                total += 0.5
            # Специальные связи
            if "hip" in node and ("ankle" in gv or "foot" in gv):
                total += 0.3
            if "knee" in node and ("ankle" in gv or "foot" in gv):
                total += 0.4
            if "shoulder" in node and ("elbow" in gv or "cube" in gv):
                total += 0.3
            if "intent" in node and "intent" in gv:
                total += 0.2
            if "com" in node and ("com" in gv or "torso" in gv):
                total += 0.3
        return min(1.0, total / max(len(group_vars), 1))

    def auto_discover(self, tick: int) -> list[str]:
        """
        Автоматическое открытие: если давление на группу превышает порог.
        Возвращает список всех новых переменных.
        """
        if not self._bootstrap:
            return []
        if (tick - self._last_discovery_tick) < self._discovery_cooldown:
            return []

        threshold = float(os.environ.get("RKK_BOOTSTRAP_DISCOVER_THRESHOLD", "1.5"))
        all_new: list[str] = []

        # Сортируем по давлению, открываем наиболее нужную
        sorted_groups = sorted(
            [(g, p) for g, p in self._group_pressure.items()
             if g not in self._discovered_groups],
            key=lambda x: -x[1],
        )

        for group_name, pressure in sorted_groups:
            if pressure >= threshold:
                new_vars = self.discover_group(
                    group_name, tick,
                    reason="pressure",
                    target_node=f"pressure={pressure:.2f}",
                )
                all_new.extend(new_vars)
                break  # Одна группа за раз

        return all_new

    def get_fixed_base_vars(self) -> list[str]:
        """Аналог FIXED_BASE_VARS но только из активных."""
        if not self._bootstrap:
            return list(FIXED_BASE_VARS)
        # fixed_base = всё кроме торса XYZ, ног и стоп
        exclude = {"com_x", "com_y", "com_z", "torso_roll", "torso_pitch",
                    "lhip", "lknee", "lankle", "rhip", "rknee", "rankle",
                    "lfoot_z", "rfoot_z"}
        return [v for v in self._active if v not in exclude]

    def snapshot(self) -> dict[str, Any]:
        """Для UI."""
        return {
            "bootstrap_mode": self._bootstrap,
            "active_count": len(self._active),
            "undiscovered_count": self.undiscovered_count,
            "total_discoveries": self.total_discoveries,
            "discovered_groups": sorted(self._discovered_groups),
            "group_pressure": {
                g: round(p, 3)
                for g, p in sorted(self._group_pressure.items(), key=lambda x: -x[1])
                if p > 0.01
            },
            "recent_discoveries": [
                {
                    "tick": e.tick,
                    "group": e.group,
                    "n_vars": len(e.variables),
                    "reason": e.trigger_reason,
                }
                for e in list(self._discovery_log)[-5:]
            ],
        }


# ─── Глобальный singleton ────────────────────────────────────────────────────
_REGISTRY: VariableRegistry | None = None


def get_variable_registry() -> VariableRegistry:
    """Получить или создать глобальный реестр переменных."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = VariableRegistry()
    return _REGISTRY


def reset_variable_registry() -> VariableRegistry:
    """Пересоздать реестр (для тестов)."""
    global _REGISTRY
    _REGISTRY = VariableRegistry()
    return _REGISTRY
