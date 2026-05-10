"""
Pearl-style API поверх CausalGraph (Фаза A плана Strong AI).

L1 observe_predict — ассоциативное одношаговое обновление без интервенции.
L2 intervene_predict — do(var=val) через propagate_from.
L3 counterfactual — интервенция с последующим clamp экзогенных узлов U к значениям из base
(Pearl: фиксируем экзогенные причины, меняем только эндогенное действие).

Включение: ``RKK_PEARL_API=1`` (по умолчанию выключено в ``pearl_constants``).
"""
from __future__ import annotations

from engine.pearl_constants import default_exogenous_node_ids


class PearlCausalFacade:
    """Тонкая обёртка над CausalGraph с явной семантикой Pearl-уровней."""

    def __init__(self, graph: Any):
        self._graph = graph

    def observe_predict(self, base: dict[str, float]) -> dict[str, float]:
        """L1: одношаговая свободная динамика (ассоциативное предсказание)."""
        return self._graph.rollout_step_free(dict(base))

    def intervene_predict(
        self,
        base: dict[str, float],
        variable: str,
        value: float,
    ) -> dict[str, float]:
        """L2: предсказание после явной интервенции do(variable=value)."""
        return self._graph.propagate_from(dict(base), variable, value)

    def counterfactual(
        self,
        base: dict[str, float],
        alternative_var: str,
        alternative_val: float,
        *,
        exogenous_ids: frozenset[str] | None = None,
    ) -> dict[str, float]:
        """
        L3: контрфакт «что было бы, если бы» с альтернативным действием.

        После интервенционного шага значения узлов из множества U принудительно
        возвращаются к базовому снимку base — упрощённый SCM-clamp для экзогены.
        """
        U = exogenous_ids if exogenous_ids is not None else default_exogenous_node_ids()
        pred = self._graph.propagate_from(
            dict(base), alternative_var, alternative_val
        )
        out = dict(pred)
        for uid in U:
            if uid in base:
                try:
                    out[uid] = float(base[uid])
                except (TypeError, ValueError):
                    continue
        return out
