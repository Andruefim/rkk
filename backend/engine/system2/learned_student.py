from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from engine.system2.student import choose_macro_from_obs

_MACRO_ORDER = ("IDLE", "RECOVER_POSTURE", "LOCOMOTE_DELIVERY", "EXPLORE")


def _feat_vec(obs: dict[str, float]) -> np.ndarray:
    cz = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
    ps = float(
        obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
    )
    td = float(obs.get("target_dist", obs.get("phys_target_dist", 0.5)))
    fl = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
    fr = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
    cx = float(obs.get("com_x", obs.get("phys_com_x", 0.41)))
    return np.array(
        [cz, ps, td, fl, fr, cx, cz * ps, abs(cz - ps)],
        dtype=np.float64,
    )


class LearnedMacroStudent:
    """
    Лёгкий онлайн-студент: линейные логиты по фичам + обновление по исходу макроса.
    При низкой уверенности — откат к эвристике choose_macro_from_obs.
    """

    def __init__(self) -> None:
        self._d = 8
        self._k = len(_MACRO_ORDER)
        self._W = np.zeros((self._k, self._d + 1), dtype=np.float64)  # + bias
        self._bootstrapped = False

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - float(np.max(logits))
        e = np.exp(np.clip(z, -40.0, 40.0))
        s = float(np.sum(e)) + 1e-9
        return e / s

    def predict(self, obs_f: dict[str, float]) -> tuple[str, float]:
        x = _feat_vec(obs_f)
        aug = np.append(x, 1.0)
        logits = self._W @ aug
        p = self._softmax(logits)
        j = int(np.argmax(p))
        conf = float(p[j])
        try:
            th = float(os.environ.get("RKK_SYSTEM2_STUDENT_CONF", "0.38"))
        except ValueError:
            th = 0.38
        if conf < th:
            return choose_macro_from_obs(obs_f), 0.0
        return _MACRO_ORDER[j], conf

    def learn(self, macro: str, success: bool, obs_f: dict[str, float]) -> None:
        try:
            lr = float(os.environ.get("RKK_SYSTEM2_STUDENT_LR", "0.06"))
        except ValueError:
            lr = 0.06
        lr = float(np.clip(lr, 0.0, 0.25))
        if macro not in _MACRO_ORDER:
            return
        y = _MACRO_ORDER.index(macro)
        x = _feat_vec(obs_f)
        aug = np.append(x, 1.0)
        logits = self._W @ aug
        p = self._softmax(logits)
        grad = p.copy()
        grad[y] -= 1.0
        reward = 1.0 if success else -0.35
        self._W -= lr * reward * grad[:, None] @ aug[None, :]

    def bootstrap_from_log(self, path: Path, *, max_lines: int = 1500) -> int:
        if not path.is_file():
            return 0
        n = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return 0
        for line in lines[-max_lines:]:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            macro = str(row.get("macro", "IDLE")).upper()
            if macro not in _MACRO_ORDER:
                continue
            ok = bool(row.get("success", False))
            delta = row.get("delta") or {}
            cz = float(delta.get("d_com_z", 0.0))
            ps = float(delta.get("d_posture", 0.0))
            x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.41, 0.25, 0.0], dtype=np.float64)
            x[0] = float(np.clip(0.5 + 2.0 * cz, 0.05, 0.95))
            x[1] = float(np.clip(0.5 + 2.0 * ps, 0.05, 0.95))
            aug = np.append(x, 1.0)
            y = _MACRO_ORDER.index(macro)
            logits = self._W @ aug
            p = self._softmax(logits)
            grad = p.copy()
            grad[y] -= 1.0
            reward = 1.0 if ok else -0.4
            self._W -= 0.04 * reward * grad[:, None] @ aug[None, :]
            n += 1
        self._bootstrapped = n > 0
        return n

    def enabled(self) -> bool:
        return os.environ.get("RKK_SYSTEM2_STUDENT", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
