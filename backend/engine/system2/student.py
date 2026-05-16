from __future__ import annotations

from typing import Any

import numpy as np


def choose_macro_from_obs(obs: dict[str, float]) -> str:
    """
    Эвристический «студент» без LLM: выбор макроса по proprio/homeostatic осям.
    Совместимо с phys_* и прямыми ключами.
    """
    cz = float(
        obs.get("com_z", obs.get("phys_com_z", 0.5)),
    )
    ps = float(
        obs.get(
            "posture_stability",
            obs.get("phys_posture_stability", 0.5),
        ),
    )
    td = float(obs.get("target_dist", obs.get("phys_target_dist", 0.5)))
    fl = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
    fr = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))

    grounded = min(fl, fr)
    if cz < 0.48 or ps < 0.40 or grounded < 0.35:
        return "RECOVER_POSTURE"
    if td > 0.52 and ps > 0.42:
        return "LOCOMOTE_DELIVERY"
    if ps > 0.48 and abs(float(obs.get("com_x", obs.get("phys_com_x", 0.41))) - 0.41) < 0.08:
        return "EXPLORE"
    return "IDLE"


class MacroStudent:
    """Счётчик успехов макросов для будущей дистилляции (минимальная память)."""

    def __init__(self) -> None:
        self._wins: dict[str, float] = {}
        self._trials: dict[str, float] = {}

    def record_outcome(self, macro: str, success: bool, weight: float = 1.0) -> None:
        m = str(macro or "IDLE").upper()
        self._trials[m] = self._trials.get(m, 0.0) + weight
        if success:
            self._wins[m] = self._wins.get(m, 0.0) + weight

    def preference_vector(self, macros: list[str]) -> np.ndarray:
        out = []
        for m in macros:
            t = max(1.0, self._trials.get(m, 0.0))
            w = self._wins.get(m, 0.0) / t
            out.append(float(np.clip(w, 0.0, 1.0)))
        arr = np.array(out, dtype=np.float64)
        s = float(arr.sum())
        if s < 1e-6:
            return np.ones(len(macros), dtype=np.float64) / max(1, len(macros))
        return arr / s
