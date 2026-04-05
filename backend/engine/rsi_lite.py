"""
Этап G — RSI lite: агент отслеживает плато discovery_rate и сам подкручивает обучение графа.

Не полный Nova RSI; только локальные хуки: L1 на W, размер буфера интервенций, imagination horizon.
"""
from __future__ import annotations

import os


def rsi_lite_enabled() -> bool:
    return os.environ.get("RKK_RSI_LITE", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def rsi_plateau_interventions() -> int:
    """Сколько подряд шагов без роста discovery относительно внутреннего референса."""
    try:
        return max(8, int(os.environ.get("RKK_RSI_PLATEAU_TICKS", "96")))
    except ValueError:
        return 96


def rsi_min_interventions() -> int:
    """Не вызывать RSI до стольки успешных интервенций (избегаем ранних ложных плато)."""
    try:
        return max(0, int(os.environ.get("RKK_RSI_MIN_INTERVENTIONS", "40")))
    except ValueError:
        return 40


def rsi_l1_scale() -> float:
    try:
        return max(1.0, float(os.environ.get("RKK_RSI_L1_SCALE", "1.2")))
    except ValueError:
        return 1.2


def rsi_l1_max() -> float:
    try:
        return max(0.01, float(os.environ.get("RKK_RSI_L1_MAX", "0.35")))
    except ValueError:
        return 0.35


def rsi_buffer_cap() -> int:
    try:
        return max(64, int(os.environ.get("RKK_RSI_BUFFER_CAP", "256")))
    except ValueError:
        return 256


def rsi_imagination_cap() -> int:
    try:
        return max(0, int(os.environ.get("RKK_RSI_IMAGINATION_CAP", "4")))
    except ValueError:
        return 4


def rsi_improvement_eps() -> float:
    try:
        return max(1e-7, float(os.environ.get("RKK_RSI_IMPROVE_EPS", "1e-5")))
    except ValueError:
        return 1e-5
