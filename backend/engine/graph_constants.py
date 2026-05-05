"""Общие правила имён узлов графа (Фаза 1+)."""

from engine.features.humanoid.constants import MOTOR_INTENT_VARS, MOTOR_OBSERVABLE_VARS

READ_ONLY_MACRO_PREFIX = "concept_"
READ_ONLY_INTERO_PREFIX = "intero_"


def is_read_only_macro_var(variable: str) -> bool:
    """
    Узлы без прямого do(): concept_*, intero_*, производные MOTOR_OBSERVABLE_*.

    phys_* в графе — зеркала физики; do() только по коротким именам (lshoulder, intent_stride)
    или явному phys_intent_* / суффиксу из MOTOR_INTENT_VARS (намерение).
    Остальные phys_lhip, phys_posture_stability, … — read-only (иначе агент «крутит исход»).
    """
    v = str(variable)
    if v.startswith(READ_ONLY_MACRO_PREFIX) or v.startswith(READ_ONLY_INTERO_PREFIX):
        return True
    if v in MOTOR_OBSERVABLE_VARS:
        return True
    if v.startswith("phys_intent_"):
        return False
    if v.startswith("phys_"):
        suf = v[5:]
        if suf in MOTOR_OBSERVABLE_VARS:
            return True
        if suf in MOTOR_INTENT_VARS or suf.startswith("intent_"):
            return False
        return True
    return False
