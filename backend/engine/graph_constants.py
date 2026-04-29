"""Общие правила имён узлов графа (Фаза 1+)."""

READ_ONLY_MACRO_PREFIX = "concept_"
READ_ONLY_INTERO_PREFIX = "intero_"


def is_read_only_macro_var(variable: str) -> bool:
    """Macro nodes (concept_*) and interoceptive sensors (intero_*) — read-only, no do()/intervene."""
    v = str(variable)
    return v.startswith(READ_ONLY_MACRO_PREFIX) or v.startswith(READ_ONLY_INTERO_PREFIX)
