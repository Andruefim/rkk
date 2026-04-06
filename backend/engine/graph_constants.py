"""Общие правила имён узлов графа (Фаза 1+)."""

READ_ONLY_MACRO_PREFIX = "concept_"


def is_read_only_macro_var(variable: str) -> bool:
    """Макро-узлы concept_* — только агрегаты; do()/intervene запрещены."""
    return str(variable).startswith(READ_ONLY_MACRO_PREFIX)
