"""
Track G Phase 5: autonomous subgoal generation via CausalNoveltyScore + W_meta filter.
"""
from __future__ import annotations

import os
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from engine.meta_causal import WMetaEnsemble, meta_causal_enabled


def goal_gen_enabled() -> bool:
    default = "1"
    try:
        from engine.intention_cortex import intention_cortex_enabled

        if not intention_cortex_enabled():
            default = "0"
    except ImportError:
        default = "0"
    return os.environ.get("RKK_GOAL_GEN_ENABLED", default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _ei(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def causal_novelty_score(graph: Any, role_map: dict[str, str] | None = None) -> dict[str, float]:
    """EIG discovery map + role-cluster entropy weighting."""
    eig_map = graph.edge_discovery_eig()
    role_ent = graph.role_cluster_entropy()
    w = _ef("RKK_GOAL_ROLE_ENT_W", 0.30)
    if role_map:
        for vid in list(eig_map.keys()):
            rt = role_map.get(vid, "")
            if rt and vid in role_ent:
                eig_map[vid] = eig_map[vid] + w * role_ent[vid]
    else:
        for vid in eig_map:
            eig_map[vid] = eig_map[vid] + w * role_ent.get(vid, 0.0)
    return eig_map


@dataclass
class GoalCandidate:
    var_id: str
    score: float
    target_val: float = 0.62
    world_id: str = "humanoid"
    source: str = "generated"
    meta_success_pred: float = 0.0
    tick_proposed: int = 0
    tick_completed: int | None = None
    success_rate: float | None = None
    status: str = "active"  # active | completed | rejected | blocked

    def key(self) -> str:
        return self.var_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_id": self.var_id,
            "score": round(self.score, 4),
            "target_val": self.target_val,
            "world_id": self.world_id,
            "source": self.source,
            "meta_success_pred": round(self.meta_success_pred, 4),
            "tick_proposed": self.tick_proposed,
            "tick_completed": self.tick_completed,
            "success_rate": self.success_rate,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoalCandidate:
        return cls(
            var_id=str(d.get("var_id", "")),
            score=float(d.get("score", 0.0)),
            target_val=float(d.get("target_val", 0.62)),
            world_id=str(d.get("world_id", "humanoid")),
            source=str(d.get("source", "generated")),
            meta_success_pred=float(d.get("meta_success_pred", 0.0)),
            tick_proposed=int(d.get("tick_proposed", 0)),
            tick_completed=(
                int(d["tick_completed"]) if d.get("tick_completed") is not None else None
            ),
            success_rate=(
                float(d["success_rate"]) if d.get("success_rate") is not None else None
            ),
            status=str(d.get("status", "active")),
        )


class GoalGenerator:
    def __init__(self) -> None:
        self._recent: deque[str] = deque(maxlen=_ei("RKK_GOAL_DIVERSITY_WINDOW", 10))
        self._counts: Counter[str] = Counter()
        self._active: list[GoalCandidate] = []
        self._completed: list[GoalCandidate] = []
        self._last_propose_tick: int = -9999
        self._blocked_log: list[dict[str, Any]] = []
        self._world_goals: dict[str, list[GoalCandidate]] = {}

    def _cooldown_max(self) -> int:
        return _ei("RKK_GOAL_COOLDOWN_MAX", 3)

    def _saturation_frac(self) -> float:
        return _ef("RKK_GOAL_SATURATION_FRAC", 0.50)

    def _wmeta_min_success(self) -> float:
        return _ef("RKK_GOAL_WMETA_MIN_SUCCESS", 0.30)

    def _max_active(self) -> int:
        return _ei("RKK_GOAL_MAX_ACTIVE", 3)

    def _is_saturated(self, key: str) -> bool:
        if not self._recent:
            return False
        freq = sum(1 for p in self._recent if p == key) / len(self._recent)
        return freq > self._saturation_frac()

    def on_tick(self, tick: int) -> None:
        decay_every = _ei("RKK_GOAL_COUNT_DECAY_EVERY", 1000)
        if tick > 0 and tick % decay_every == 0:
            self._counts = Counter({k: max(0, v - 1) for k, v in self._counts.items()})

    def propose(
        self,
        graph: Any,
        w_meta: WMetaEnsemble | None,
        *,
        role_map: dict[str, str] | None = None,
        tick: int = 0,
        world_id: str = "humanoid",
    ) -> GoalCandidate | None:
        if len(self._active) >= self._max_active():
            self._log_blocked("max_active", tick)
            return None
        candidates = sorted(
            causal_novelty_score(graph, role_map).items(),
            key=lambda x: -x[1],
        )
        min_succ = self._wmeta_min_success()
        for var_id, score in candidates:
            if not var_id or var_id.startswith("concept_"):
                continue
            key = var_id
            if self._counts[key] > self._cooldown_max():
                continue
            if self._is_saturated(key):
                continue
            pred = 0.5
            if w_meta is not None and meta_causal_enabled():
                pred = w_meta.predict_success(goal_var=var_id, goal_score=score)
            if pred < min_succ:
                self._log_blocked("w_meta_reject", tick, var_id=var_id, pred=pred)
                continue
            target_val = 0.62
            if var_id in graph.nodes:
                v = float(graph.nodes[var_id])
                target_val = max(0.35, min(0.85, v))
            cand = GoalCandidate(
                var_id=var_id,
                score=score,
                target_val=target_val,
                world_id=world_id,
                meta_success_pred=pred,
                tick_proposed=tick,
            )
            self._recent.append(key)
            self._counts[key] += 1
            self._active.append(cand)
            self._last_propose_tick = tick
            wl = self._world_goals.setdefault(world_id, [])
            wl.append(cand)
            return cand
        self._log_blocked("goal_gen_blocked", tick)
        return None

    def complete_goal(
        self,
        var_id: str,
        *,
        success_rate: float,
        tick: int,
    ) -> bool:
        for i, g in enumerate(self._active):
            if g.var_id == var_id:
                g.status = "completed"
                g.tick_completed = tick
                g.success_rate = success_rate
                self._completed.append(g)
                self._active.pop(i)
                return True
        return False

    def reject_goal(self, var_id: str, reason: str = "rejected") -> None:
        for i, g in enumerate(self._active):
            if g.var_id == var_id:
                g.status = reason
                self._active.pop(i)
                return

    def _log_blocked(self, reason: str, tick: int, **extra: Any) -> None:
        entry = {"reason": reason, "tick": tick, **extra}
        self._blocked_log.append(entry)
        if len(self._blocked_log) > 32:
            self._blocked_log = self._blocked_log[-32:]

    def autonomous_goals_metrics(self) -> dict[str, Any]:
        """Scorecard #7: cross-world autonomous goals."""
        worlds = set(self._world_goals.keys()) | {g.world_id for g in self._completed}
        by_world: dict[str, list[GoalCandidate]] = {}
        for g in self._completed:
            by_world.setdefault(g.world_id, []).append(g)
        min_sr = _ef("RKK_GOAL_TRANSFER_MIN_SUCCESS", 0.40)
        passed_worlds = [
            w
            for w, gs in by_world.items()
            if len(gs) >= 1
            and sum(1 for g in gs if (g.success_rate or 0.0) >= min_sr) >= 1
        ]
        n_goals = len(self._completed)
        sr_vals = [float(g.success_rate or 0.0) for g in self._completed]
        mean_sr = float(sum(sr_vals) / len(sr_vals)) if sr_vals else 0.0
        crossworld_pass = n_goals >= 3 and mean_sr >= min_sr and len(passed_worlds) >= 2
        return {
            "autonomous_goals_count": n_goals,
            "autonomous_goals_mean_success": round(mean_sr, 4),
            "autonomous_goals_worlds": sorted(worlds),
            "autonomous_goals_crossworld_pass": crossworld_pass,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": goal_gen_enabled(),
            "active": [g.to_dict() for g in self._active],
            "completed": [g.to_dict() for g in self._completed[-12:]],
            "blocked_log": list(self._blocked_log)[-8:],
            "last_propose_tick": self._last_propose_tick,
            **self.autonomous_goals_metrics(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": [g.to_dict() for g in self._active],
            "completed": [g.to_dict() for g in self._completed],
            "recent": list(self._recent),
            "counts": dict(self._counts),
            "world_goals": {
                w: [g.to_dict() for g in gs]
                for w, gs in self._world_goals.items()
            },
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if not data:
            return
        self._active = [GoalCandidate.from_dict(x) for x in data.get("active") or []]
        self._completed = [GoalCandidate.from_dict(x) for x in data.get("completed") or []]
        self._recent = deque(data.get("recent") or [], maxlen=_ei("RKK_GOAL_DIVERSITY_WINDOW", 10))
        self._counts = Counter(data.get("counts") or {})
        self._world_goals = {}
        for w, gs in (data.get("world_goals") or {}).items():
            self._world_goals[str(w)] = [GoalCandidate.from_dict(x) for x in gs]

