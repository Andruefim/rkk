"""
Lightweight knowledge graph — triples + simple ontology inference (Layer 4).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    tick_added: int = 0
    learned: bool = False


@dataclass
class KnowledgeGraph:
    triples: list[Triple] = field(default_factory=list)
    _index: dict[str, list[Triple]] = field(default_factory=dict)

    def add(
        self,
        s: str,
        p: str,
        o: str,
        confidence: float = 1.0,
        *,
        tick: int = 0,
        learned: bool = False,
    ) -> None:
        t = Triple(s, p, o, float(confidence), int(tick), bool(learned))
        self.triples.append(t)
        self._index.setdefault(s, []).append(t)

    def query(self, subject: str, predicate: str | None = None) -> list[Triple]:
        rows = self._index.get(subject, [])
        if predicate is None:
            return list(rows)
        return [t for t in rows if t.predicate == predicate]

    def infer_subclass(self, entity: str, visited: set[str] | None = None) -> set[str]:
        """Transitive closure over subclass_of."""
        visited = visited or set()
        out: set[str] = {entity}
        if entity in visited:
            return out
        visited.add(entity)
        for t in self.query(entity, "subclass_of"):
            out |= self.infer_subclass(t.obj, visited)
        return out

    def effects_of(self, entity: str) -> list[str]:
        classes = self.infer_subclass(entity)
        effects: list[str] = []
        for cls in classes:
            for t in self.query(cls, "effect_of"):
                effects.append(t.obj)
        return effects

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_triples": len(self.triples),
            "runtime_facts": dict(getattr(self, "_runtime_facts", {})),
            "sample": [
                {"s": t.subject, "p": t.predicate, "o": t.obj, "c": round(t.confidence, 3)}
                for t in self.triples[:24]
            ],
        }

    def set_runtime_fact(self, name: str, value: float) -> None:
        if not hasattr(self, "_runtime_facts"):
            self._runtime_facts: dict[str, float] = {}
        self._runtime_facts[str(name)] = float(value)

    def get_runtime_fact(self, name: str, default: float = 1.0) -> float:
        return float(getattr(self, "_runtime_facts", {}).get(name, default))

    def learn_from_surprise(
        self,
        predicate: str,
        observed: float,
        expected: float,
        *,
        tick: int = 0,
        threshold: float = 0.22,
    ) -> bool:
        """Surprise-driven rule: large |obs−exp| → new causes/triggered_by triple."""
        delta = abs(float(observed) - float(expected))
        if delta < threshold:
            return False
        conf = float(min(0.92, 0.45 + delta))
        self.add(
            predicate,
            "triggered_by",
            f"Surprise_{predicate}",
            confidence=conf,
            tick=tick,
            learned=True,
        )
        self.add(
            f"Surprise_{predicate}",
            "requires",
            "Replan",
            confidence=conf * 0.9,
            tick=tick,
            learned=True,
        )
        return True

    def learn_from_plan_failure(
        self,
        macro: str,
        failed_predicate: str,
        *,
        tick: int = 0,
    ) -> None:
        """Planning error → blocks relation for future STRIPS."""
        self.add(
            failed_predicate,
            "blocks",
            str(macro).upper(),
            confidence=0.62,
            tick=tick,
            learned=True,
        )

    def is_blocked(self, macro: str, predicate: str) -> bool:
        for t in self.query(predicate, "blocks"):
            if t.obj == str(macro).upper() and t.confidence >= 0.5:
                return True
        return False

    def forget_stale(self, tick: int, *, max_age_ticks: int | None = None) -> int:
        """Forgetting curve: decay learned triple confidence; drop when below floor."""
        try:
            age = int(
                max_age_ticks
                if max_age_ticks is not None
                else int(os.environ.get("RKK_KG_FORGET_TICKS", "60000"))
            )
        except ValueError:
            age = 60000
        floor = 0.12
        decay = 0.985
        removed = 0
        kept: list[Triple] = []
        new_index: dict[str, list[Triple]] = {}
        for t in self.triples:
            if not t.learned:
                kept.append(t)
                new_index.setdefault(t.subject, []).append(t)
                continue
            if tick - t.tick_added > age:
                removed += 1
                continue
            t.confidence *= decay
            if t.confidence < floor:
                removed += 1
                continue
            kept.append(t)
            new_index.setdefault(t.subject, []).append(t)
        self.triples = kept
        self._index = new_index
        return removed


def bootstrap_humanoid_ontology() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add("Human", "subclass_of", "Agent")
    kg.add("FragileMaterial", "subclass_of", "Material")
    kg.add("Glass", "subclass_of", "FragileMaterial")
    kg.add("FragileMaterial", "effect_of", "HandleGently")
    kg.add("Human", "effect_of", "MaintainSafeDistance")
    kg.add("HumanProximity", "requires", "MaintainSafeDistance")
    kg.add("distance_to_human", "measures", "HumanProximity")
    kg.add("LocomoteGoal", "subclass_of", "MotorGoal")
    kg.add("LocomoteGoal", "requires", "IsStable")
    kg.add("RecoverGoal", "subclass_of", "MotorGoal")
    kg.add("RecoverGoal", "triggered_by", "IsFallen")
    kg.add("PathBlocked", "subclass_of", "SpatialConstraint")
    kg.add("StepForward", "requires", "NOT PathBlocked")
    kg.add("ApproachTarget", "requires", "NOT PathBlocked")
    kg.add("Turn", "effect_of", "NOT PathBlocked")
    kg.add("Turn", "triggered_by", "PathBlocked")
    return kg


def human_proximity_threshold() -> float:
    try:
        return float(os.environ.get("RKK_NS_HUMAN_DIST_THRESH", "0.70"))
    except ValueError:
        return 0.70
