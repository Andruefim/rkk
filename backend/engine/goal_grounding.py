"""
Language → TaskGoal via embedding similarity over a declarative predicate ontology.

No verb-branching heuristics: command text is matched to predicate *descriptions*
by cosine similarity. Composite goals (e.g. contact + reduce_distance prerequisite)
follow physical constraints, not keyword tables.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from engine.grounded_language import command_tag_for_text
from engine.task_goal import GoalPredicate, TaskGoal

EmbedFn = Callable[[str], np.ndarray | None]

_KINDS_NEEDING_TARGET = frozenset({"reduce_distance", "contact", "displace"})
_MANIPULATION_KINDS = frozenset({"contact", "displace"})
_EXCLUSIVE_KIND_GROUPS: tuple[frozenset[str], ...] = (frozenset({"contact", "displace"}),)
_CONJUNCTION_RE = re.compile(r"\s+(?:и|and|,)\s+", flags=re.IGNORECASE)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def goal_near_m() -> float:
    """Legacy alias; manipulation goals use task_observation.nav_stop_m()."""
    from engine.task_observation import nav_stop_m

    return nav_stop_m()


def manip_min_disp_m() -> float:
    return _env_float("RKK_MANIP_MIN_DISP", 0.12)


def composite_kind_margin() -> float:
    """Kinds within this cosine margin of a clause top score join the composite goal."""
    return _env_float("RKK_GOAL_COMPOSITE_MARGIN", 0.025)


@dataclass(frozen=True)
class _CatalogEntry:
    kind: str
    description: str
    key: str | None = None
    target_value: float | None = None
    tolerance: float | None = None
    weight: float = 1.0


# Declarative goal ontology — extend by adding paraphrases, not branching code.
_PREDICATE_CATALOG: tuple[_CatalogEntry, ...] = (
    # reduce_distance
    _CatalogEntry("reduce_distance", "подойти к объекту"),
    _CatalogEntry("reduce_distance", "приблизиться к цели"),
    _CatalogEntry("reduce_distance", "подойди ближе"),
    _CatalogEntry("reduce_distance", "walk up to the object"),
    _CatalogEntry("reduce_distance", "go to the target"),
    _CatalogEntry("reduce_distance", "approach the object"),
    _CatalogEntry("reduce_distance", "get closer"),
    _CatalogEntry("reduce_distance", "подойди к объекту перед тобой"),
    _CatalogEntry("reduce_distance", "go to the object in front"),
    # contact
    _CatalogEntry("contact", "дотронуться до объекта"),
    _CatalogEntry("contact", "дотронься до объекта перед тобой"),
    _CatalogEntry("contact", "touch the object in front of you"),
    _CatalogEntry("contact", "коснуться"),
    _CatalogEntry("contact", "дотронься"),
    _CatalogEntry("contact", "touch the object"),
    _CatalogEntry("contact", "make contact"),
    _CatalogEntry("contact", "reach and touch"),
  # displace
    _CatalogEntry("displace", "передвинуть объект"),
    _CatalogEntry("displace", "сдвинуть"),
    _CatalogEntry("displace", "толкни"),
    _CatalogEntry("displace", "push the object"),
    _CatalogEntry("displace", "move the object"),
    _CatalogEntry("displace", "shift it"),
    # locomotion / postural state_key
    _CatalogEntry(
        "state_key", "иди вперёд", key="intent_stride", target_value=0.66, tolerance=0.15
    ),
    _CatalogEntry(
        "state_key", "step forward", key="intent_stride", target_value=0.66, tolerance=0.15
    ),
    _CatalogEntry(
        "state_key", "walk forward", key="com_x_vel", target_value=0.55, tolerance=0.2
    ),
    _CatalogEntry(
        "state_key",
        "иду вперёд",
        key="intent_torso_forward",
        target_value=0.58,
        tolerance=0.15,
    ),
    _CatalogEntry(
        "state_key", "повернись", key="intent_gait_coupling", target_value=0.72, tolerance=0.15
    ),
    _CatalogEntry(
        "state_key",
        "повернись налево",
        key="intent_gait_coupling",
        target_value=0.72,
        tolerance=0.15,
    ),
    _CatalogEntry(
        "state_key",
        "повернись направо",
        key="intent_gait_coupling",
        target_value=0.72,
        tolerance=0.15,
    ),
    _CatalogEntry(
        "state_key", "turn around", key="intent_gait_coupling", target_value=0.72, tolerance=0.15
    ),
    _CatalogEntry(
        "state_key", "turn left", key="intent_gait_coupling", target_value=0.72, tolerance=0.15
    ),
    _CatalogEntry(
        "state_key", "turn right", key="intent_gait_coupling", target_value=0.72, tolerance=0.15
    ),
    _CatalogEntry(
        "state_key",
        "встань",
        key="intent_stop_recover",
        target_value=0.72,
        tolerance=0.15,
    ),
    _CatalogEntry(
        "state_key",
        "get up",
        key="intent_stop_recover",
        target_value=0.72,
        tolerance=0.15,
    ),
    _CatalogEntry(
        "state_key",
        "стабилизируйся",
        key="posture_stability",
        target_value=0.5,
        tolerance=0.12,
    ),
    _CatalogEntry(
        "state_key", "stand stable", key="posture_stability", target_value=0.5, tolerance=0.12
    ),
)

# Cached catalog embeddings keyed by description text.
_catalog_emb_cache: dict[str, np.ndarray] = {}


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def _align_dim(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    da = np.asarray(a, dtype=np.float32).reshape(-1)
    db = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(da.size, db.size)
    return da[:n], db[:n]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    da, db = _align_dim(a, b)
    return float(np.dot(_normalize(da), _normalize(db)))


def clear_catalog_cache() -> None:
    """Test helper: reset cached predicate-description embeddings."""
    _catalog_emb_cache.clear()


def _ensure_catalog_embeddings(embed_fn: EmbedFn) -> list[tuple[_CatalogEntry, np.ndarray]]:
    out: list[tuple[_CatalogEntry, np.ndarray]] = []
    probe = embed_fn("probe")
    dim = int(probe.reshape(-1).size) if probe is not None else 0
    for entry in _PREDICATE_CATALOG:
        desc = entry.description
        cache_key = f"{dim}:{desc}"
        if cache_key not in _catalog_emb_cache:
            emb = embed_fn(desc)
            if emb is None:
                continue
            _catalog_emb_cache[cache_key] = _normalize(emb)
        out.append((entry, _catalog_emb_cache[cache_key]))
    return out


def _softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr - arr.max()
    ex = np.exp(arr)
    s = ex.sum()
    if s < 1e-12:
        return [1.0 / len(xs)] * len(xs)
    return (ex / s).tolist()


def _kind_scores(
    cmd_emb: np.ndarray, catalog: list[tuple[_CatalogEntry, np.ndarray]]
) -> dict[str, float]:
    """Max cosine per kind across all paraphrases."""
    scores: dict[str, float] = {}
    for entry, emb in catalog:
        sim = _cosine(cmd_emb, emb)
        prev = scores.get(entry.kind)
        if prev is None or sim > prev:
            scores[entry.kind] = sim
    return scores


def _split_command_clauses(text: str) -> list[str]:
    """Syntactic conjunction split — not verb-specific branching."""
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [p.strip(" .,;") for p in _CONJUNCTION_RE.split(raw) if p.strip(" .,;")]
    return parts if parts else [raw]


def _merge_kind_scores(scores_list: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for scores in scores_list:
        for kind, score in scores.items():
            prev = merged.get(kind)
            if prev is None or score > prev:
                merged[kind] = float(score)
    return merged


def _select_kinds_for_clause(
    kind_scores: dict[str, float],
    *,
    min_kind_score: float,
    margin: float,
) -> set[str]:
    if not kind_scores:
        return set()
    ranked = sorted(kind_scores.items(), key=lambda kv: -kv[1])
    top_score = ranked[0][1]
    if top_score < min_kind_score:
        return set()
    floor = top_score - float(margin)
    return {k for k, s in ranked if s >= min_kind_score and s >= floor}


# Ontology tie-break among manipulation terminals (less invasive wins on embedding tie).
_MANIPULATION_TIE_ORDER: tuple[str, ...] = ("contact", "displace")


def _resolve_exclusive_kinds(
    selected: set[str],
    kind_scores: dict[str, float],
    *,
    margin: float,
) -> set[str]:
    """Mutually exclusive manipulation predicates — score winner, ontology tie-break."""
    out = set(selected)
    for group in _EXCLUSIVE_KIND_GROUPS:
        overlap = out & group
        if len(overlap) <= 1:
            continue
        top_score = max(kind_scores.get(k, -2.0) for k in overlap)
        tied = [k for k in overlap if kind_scores.get(k, -2.0) >= top_score - float(margin)]
        if len(tied) > 1:
            winner = next((k for k in _MANIPULATION_TIE_ORDER if k in tied), tied[0])
        else:
            winner = max(overlap, key=lambda k: kind_scores.get(k, -2.0))
        out = (out - overlap) | {winner}
    return out


def _apply_physical_prerequisites(kinds: set[str]) -> list[str]:
    """Order predicates by physical dependency (approach before contact/displace)."""
    ordered: list[str] = []
    if "reduce_distance" in kinds or kinds & _MANIPULATION_KINDS:
        ordered.append("reduce_distance")
    if "contact" in kinds:
        ordered.append("contact")
    if "displace" in kinds:
        ordered.append("displace")
    if "state_key" in kinds:
        ordered.append("state_key")
    return ordered


def warm_predicate_catalog(embed_fn: EmbedFn) -> int:
    """Pre-cache predicate-description embeddings (avoids multi-second first command)."""
    return len(_ensure_catalog_embeddings(embed_fn))


def _best_entry_for_kind(
    kind: str, cmd_emb: np.ndarray, catalog: list[tuple[_CatalogEntry, np.ndarray]]
) -> _CatalogEntry | None:
    best: _CatalogEntry | None = None
    best_sim = -2.0
    for entry, emb in catalog:
        if entry.kind != kind:
            continue
        sim = _cosine(cmd_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best = entry
    return best


def _fallback_state_key_predicate(text: str) -> GoalPredicate:
    """No embed_fn: single state_key from legacy command_tag_for_text."""
    tag = command_tag_for_text(text)
    tag_map: dict[str, tuple[str, float]] = {
        "locomote": ("intent_stride", 0.66),
        "recover": ("intent_stop_recover", 0.72),
        "turn": ("intent_gait_coupling", 0.72),
        "stable": ("posture_stability", 0.5),
        "manipulate": ("intent_grasp", 0.6),
    }
    key, val = tag_map.get(tag, ("posture_stability", 0.5))
    return GoalPredicate(
        kind="state_key",
        key=key,
        target_value=val,
        tolerance=0.15,
        weight=1.0,
    )


def _build_predicate(kind: str, entry: _CatalogEntry | None) -> GoalPredicate:
    from engine.task_observation import nav_stop_m

    near = nav_stop_m()
    disp = manip_min_disp_m()
    if kind == "reduce_distance":
        return GoalPredicate(
            kind="reduce_distance",
            target_ref=None,
            target_value=near,
            tolerance=0.25,
            weight=1.0,
        )
    if kind == "contact":
        return GoalPredicate(
            kind="contact",
            target_ref=None,
            target_value=1.0,
            tolerance=0.5,
            weight=1.0,
        )
    if kind == "displace":
        return GoalPredicate(
            kind="displace",
            target_ref=None,
            target_value=disp,
            tolerance=0.05,
            weight=1.0,
        )
    if kind == "state_key" and entry is not None:
        return GoalPredicate(
            kind="state_key",
            key=entry.key,
            target_value=float(entry.target_value if entry.target_value is not None else 0.5),
            tolerance=float(entry.tolerance if entry.tolerance is not None else 0.15),
            weight=float(entry.weight),
        )
    return GoalPredicate(kind="state_key", key="posture_stability", target_value=0.5, tolerance=0.15)


def ground_command(
    text: str,
    embed_fn: EmbedFn | None,
    *,
    target_resolver: Any | None = None,
    top_k: int = 2,
    min_kind_score: float = 0.25,
) -> TaskGoal:
    """
    Ground natural-language command into observable predicates.

    ``target_resolver`` is accepted for API symmetry; target_ref is left None —
    callers attach it after object resolution.
    """
    raw = str(text or "").strip()
    _ = target_resolver  # resolved by caller, not here

    if not raw:
        return TaskGoal(text="", confidence=0.0, diagnostics={"needs_target": False})

    if embed_fn is None:
        pred = _fallback_state_key_predicate(raw)
        return TaskGoal(
            text=raw,
            predicates=[pred],
            confidence=0.0,
            wm_trusted=False,
            diagnostics={"needs_target": False, "fallback": "command_tag"},
        )

    catalog = _ensure_catalog_embeddings(embed_fn)
    if not catalog:
        pred = _fallback_state_key_predicate(raw)
        return TaskGoal(
            text=raw,
            predicates=[pred],
            confidence=0.0,
            wm_trusted=False,
            diagnostics={"needs_target": False, "fallback": "empty_catalog"},
        )

    clauses = _split_command_clauses(raw)
    margin = composite_kind_margin()
    clause_kind_scores: list[dict[str, float]] = []
    selected: set[str] = set()
    clause_diag: list[dict[str, Any]] = []

    for clause in clauses:
        clause_emb = embed_fn(clause)
        if clause_emb is None:
            continue
        clause_emb = _normalize(clause_emb)
        scores = _kind_scores(clause_emb, catalog)
        if not scores:
            continue
        clause_kind_scores.append(scores)
        picked = _select_kinds_for_clause(
            scores, min_kind_score=min_kind_score, margin=margin
        )
        selected |= picked
        clause_diag.append(
            {
                "clause": clause[:80],
                "kind_scores": {k: round(v, 4) for k, v in scores.items()},
                "selected": sorted(picked),
            }
        )

    if not selected:
        pred = _fallback_state_key_predicate(raw)
        merged_scores = _merge_kind_scores(clause_kind_scores)
        return TaskGoal(
            text=raw,
            predicates=[pred],
            confidence=0.0,
            wm_trusted=False,
            diagnostics={
                "needs_target": False,
                "fallback": "below_min_kind_score",
                "kind_scores": {k: round(v, 4) for k, v in merged_scores.items()},
                "clauses": clause_diag,
            },
        )

    kind_scores = _merge_kind_scores(clause_kind_scores)
    selected = _resolve_exclusive_kinds(selected, kind_scores, margin=margin)
    if (selected & _MANIPULATION_KINDS or "reduce_distance" in selected) and "state_key" in selected:
        selected.discard("state_key")
    ordered_kinds = _apply_physical_prerequisites(selected)

    full_emb = embed_fn(raw)
    cmd_emb = _normalize(full_emb) if full_emb is not None else None

    predicates: list[GoalPredicate] = []
    for kind in ordered_kinds:
        entry = _best_entry_for_kind(kind, cmd_emb, catalog) if cmd_emb is not None else None
        predicates.append(_build_predicate(kind, entry))

    ranked = sorted(kind_scores.items(), key=lambda kv: -kv[1])
    primary_kind = ordered_kinds[-1] if ordered_kinds else ranked[0][0]
    if ordered_kinds:
        manip = [k for k in ordered_kinds if k in _MANIPULATION_KINDS]
        if manip:
            primary_kind = manip[-1]
        elif "reduce_distance" in ordered_kinds:
            primary_kind = "reduce_distance"
    primary_score = kind_scores.get(primary_kind, ranked[0][1] if ranked else 0.0)
    confidence = float(np.clip((primary_score + 1.0) * 0.5, 0.0, 1.0))

    needs_target = any(p.kind in _KINDS_NEEDING_TARGET for p in predicates)
    kind_probs = dict(zip([k for k, _ in ranked], _softmax([s for _, s in ranked])))

    return TaskGoal(
        text=raw,
        predicates=predicates,
        confidence=confidence,
        wm_trusted=False,
        diagnostics={
            "needs_target": needs_target,
            "kind_scores": {k: round(v, 4) for k, v in kind_scores.items()},
            "kind_probs": {k: round(v, 4) for k, v in kind_probs.items()},
            "selected_kinds": list(ordered_kinds),
            "primary_kind": primary_kind,
            "clauses": clause_diag,
            "composite": len(clauses) > 1,
        },
    )


def goal_observation_keys(goal: TaskGoal | None) -> list[str]:
    """Observation keys relevant to a TaskGoal (task-conditioned + state_key preds)."""
    from engine.task_observation import task_observation_keys_for_goal

    return task_observation_keys_for_goal(goal)


# --- Manipulation push direction (geometry default + embedding text override) ---

_DIRECTION_PARAPHRASES: tuple[tuple[str, str], ...] = (
    ("forward", "вперёд"),
    ("forward", "вперед"),
    ("forward", "forward"),
    ("forward", "ahead"),
    ("backward", "назад"),
    ("backward", "back"),
    ("backward", "backward"),
    ("left", "влево"),
    ("left", "left"),
    ("right", "вправо"),
    ("right", "right"),
)

_direction_emb_cache: dict[str, np.ndarray] = {}


def clear_direction_cache() -> None:
    """Test helper: reset cached direction-description embeddings."""
    _direction_emb_cache.clear()


def _direction_vector(
    label: str,
    *,
    agent_forward: tuple[float, float],
) -> tuple[float, float]:
    fx, fy = float(agent_forward[0]), float(agent_forward[1])
    n = float(np.hypot(fx, fy)) + 1e-9
    fx, fy = fx / n, fy / n
    if label == "forward":
        return fx, fy
    if label == "backward":
        return -fx, -fy
    if label == "left":
        return fy, -fx
    if label == "right":
        return -fy, fx
    return fx, fy


def _geometry_direction(
    agent_xy: tuple[float, float],
    target_xy: tuple[float, float] | None,
    agent_forward: tuple[float, float] | None,
) -> tuple[float, float]:
    fwd = agent_forward if agent_forward is not None else (1.0, 0.0)
    if target_xy is None:
        n = math.hypot(fwd[0], fwd[1])
        if n < 1e-9:
            return 1.0, 0.0
        return fwd[0] / n, fwd[1] / n
    dx = float(target_xy[0]) - float(agent_xy[0])
    dy = float(target_xy[1]) - float(agent_xy[1])
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        n = math.hypot(fwd[0], fwd[1])
        if n < 1e-9:
            return 1.0, 0.0
        return fwd[0] / n, fwd[1] / n
    return dx / dist, dy / dist


def _substring_direction(
    text: str,
    agent_forward: tuple[float, float] | None,
) -> tuple[float, float] | None:
    low = str(text or "").lower()
    fwd = agent_forward if agent_forward is not None else (1.0, 0.0)
    if any(k in low for k in ("назад", "back", "backward")):
        return _direction_vector("backward", agent_forward=fwd)
    if any(k in low for k in ("влево", "left")):
        return _direction_vector("left", agent_forward=fwd)
    if any(k in low for k in ("вправо", "right")):
        return _direction_vector("right", agent_forward=fwd)
    if any(k in low for k in ("вперёд", "вперед", "forward", "ahead")):
        return _direction_vector("forward", agent_forward=fwd)
    return None


def _embed_direction_override(
    text: str,
    embed_fn: EmbedFn,
    agent_forward: tuple[float, float],
    *,
    min_score: float | None = None,
) -> tuple[float, float] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    thresh = min_score if min_score is not None else _env_float("RKK_DIRECTION_EMBED_MIN", 0.38)
    cmd_emb = embed_fn(raw)
    if cmd_emb is None:
        return None
    cmd_emb = _normalize(cmd_emb)
    best_label = ""
    best_sim = -2.0
    for label, phrase in _DIRECTION_PARAPHRASES:
        cache_key = f"{phrase}"
        if cache_key not in _direction_emb_cache:
            emb = embed_fn(phrase)
            if emb is None:
                continue
            _direction_emb_cache[cache_key] = _normalize(emb)
        sim = _cosine(cmd_emb, _direction_emb_cache[cache_key])
        if sim > best_sim:
            best_sim = sim
            best_label = label
    if best_sim < thresh:
        return None
    return _direction_vector(best_label, agent_forward=agent_forward)


def infer_manip_direction(
    text: str,
    *,
    agent_xy: tuple[float, float],
    target_xy: tuple[float, float] | None = None,
    agent_forward: tuple[float, float] | None = None,
    embed_fn: EmbedFn | None = None,
    min_embed_score: float | None = None,
) -> tuple[float, float]:
    """
    Push direction: agent→target geometry by default; text override via embedding
    (or substring when embed_fn is unavailable).
    """
    fwd = agent_forward if agent_forward is not None else (1.0, 0.0)
    default = _geometry_direction(agent_xy, target_xy, fwd)

    if embed_fn is not None:
        override = _embed_direction_override(
            text, embed_fn, fwd, min_score=min_embed_score
        )
        if override is not None:
            return override
        return default

    substring = _substring_direction(text, fwd)
    if substring is not None:
        return substring
    return default
