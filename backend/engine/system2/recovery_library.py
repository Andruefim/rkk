"""
k-NN recovery plan library from distill traces + bootstrap fallback seed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from engine.system2.learned_student import snapshot_obs_for_distill
from engine.system2.recovery_schedule import (
    default_recovery_fallback_steps,
    enrich_recovery_steps,
    prepare_scripted_getup_steps,
    recovery_scripted_enabled,
)


def recovery_library_enabled() -> bool:
    return os.environ.get("RKK_S2_RECOVERY_LIBRARY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _library_k() -> int:
    try:
        return max(1, int(os.environ.get("RKK_S2_RECOVERY_LIBRARY_K", "8")))
    except ValueError:
        return 8


def _library_max_entries() -> int:
    try:
        return max(8, int(os.environ.get("RKK_S2_RECOVERY_LIBRARY_MAX", "32")))
    except ValueError:
        return 32


def _feat_vec(obs: dict[str, Any]) -> list[float]:
    o = snapshot_obs_for_distill(obs) or obs
    return [
        float(o.get("phys_com_z", o.get("com_z", 0.5))),
        float(o.get("phys_posture_stability", o.get("posture_stability", 0.5))),
        float(o.get("phys_foot_contact_l", o.get("foot_contact_l", 0.5))),
        float(o.get("phys_foot_contact_r", o.get("foot_contact_r", 0.5))),
        float(o.get("phys_target_dist", o.get("target_dist", 0.5))),
    ]


def _l2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


class RecoveryLibrary:
    """In-memory k-NN index of successful recovery step schedules."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._seed_bootstrap()

    def _seed_bootstrap(self) -> None:
        if recovery_scripted_enabled():
            steps = prepare_scripted_getup_steps()
            skill = "recovery_scripted_seed"
            feat = [0.12, 0.08, 0.55, 0.55, 0.95]
        else:
            steps = enrich_recovery_steps(default_recovery_fallback_steps())
            skill = "recovery_fallback_seed"
            feat = [0.45, 0.5, 0.55, 0.55, 0.95]
        if not steps:
            return
        self._entries.append(
            {
                "feat": feat,
                "steps": steps,
                "skill_id": skill,
                "source": "bootstrap",
            }
        )

    def add_success(
        self,
        obs0: dict[str, Any],
        steps: list[dict[str, Any]],
        *,
        skill_id: str = "recovery_library",
    ) -> None:
        if not steps:
            return
        entry = {
            "feat": _feat_vec(obs0),
            "steps": enrich_recovery_steps(steps),
            "skill_id": skill_id,
            "source": "distill",
        }
        self._entries.append(entry)
        cap = _library_max_entries()
        if len(self._entries) > cap:
            self._entries = self._entries[-cap:]

    def lookup(
        self, obs: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, float], float | None, str] | None:
        if not self._entries:
            return None
        q = _feat_vec(obs)
        ranked = sorted(
            self._entries,
            key=lambda e: _l2(q, list(e.get("feat") or [])),
        )
        best = ranked[0]
        dist = _l2(q, list(best.get("feat") or []))
        try:
            max_dist = float(os.environ.get("RKK_S2_RECOVERY_LIBRARY_MAX_DIST", "0.12"))
        except ValueError:
            max_dist = 0.12
        if dist > max_dist and str(best.get("source")) != "bootstrap":
            return None
        steps = [dict(s) for s in (best.get("steps") or [])]
        if not steps:
            return None
        es: dict[str, float] = {}
        mx: float | None = None
        return (steps, es, mx, str(best.get("skill_id", "recovery_library")))

    def load_distill_jsonl(self, path: str | Path) -> int:
        p = Path(path)
        if not p.is_file():
            return 0
        added = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("macro", "")).upper() != "RECOVER_POSTURE":
                    continue
                if not row.get("success"):
                    continue
                steps = row.get("recovery_steps")
                if not isinstance(steps, list) or not steps:
                    continue
                obs0 = row.get("obs0")
                if not isinstance(obs0, dict):
                    continue
                self.add_success(
                    obs0,
                    steps,
                    skill_id=str(row.get("skill_id", "recovery_library")),
                )
                added += 1
        return added


_LIB: RecoveryLibrary | None = None


def get_recovery_library() -> RecoveryLibrary:
    global _LIB
    if _LIB is None:
        _LIB = RecoveryLibrary()
        path = os.environ.get("RKK_S2_RECOVERY_LIBRARY_DISTILL", "").strip()
        if path:
            _LIB.load_distill_jsonl(path)
    return _LIB
