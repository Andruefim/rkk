"""Общие правила имён узлов графа (Фаза 1+)."""

from engine.features.humanoid.constants import MOTOR_OBSERVABLE_VARS

READ_ONLY_MACRO_PREFIX = "concept_"
READ_ONLY_INTERO_PREFIX = "intero_"


def is_read_only_macro_var(variable: str) -> bool:
    """
    Узлы без прямого do(): aggregate/concept, интеро, производные моторные наблюдаемые
    (posture_stability, gait_phase_*, foot_contact_*, phys_*-зеркала только для них).

    intent_*, phys_intent_*, суставы (в т.ч. phys_lhip если есть в графе) — не read-only.
    """
    v = str(variable)
    if v.startswith(READ_ONLY_MACRO_PREFIX) or v.startswith(READ_ONLY_INTERO_PREFIX):
        return True
    if v in MOTOR_OBSERVABLE_VARS:
        return True
    if v.startswith("phys_"):
        suf = v[5:]
        if suf in MOTOR_OBSERVABLE_VARS:
            return True
    return False
