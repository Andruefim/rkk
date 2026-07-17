"""Resolve manipulation targets from scene extras/registry (no VLM/Ollama)."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

MOVE_VERBS = frozenset({"передвинь", "передвинуть", "move", "push", "толкни", "сдвинь", "shift"})
INTERACTION_VERBS = MOVE_VERBS | frozenset({
    "дотронься",
    "дотронуться",
    "коснись",
    "коснуться",
    "touch",
    "reach",
    "подойди",
    "подойти",
    "approach",
    "иди",
    "идти",
    "walk",
})

STOP_WORDS = frozenset({
    "перед", "тобой", "тебя", "вперёд", "вперед", "ближе", "меня", "мной", "туда", "сюда",
    "немного", "чуть", "пожалуйста", "please", "now", "сейчас", "до",
    "the", "a", "an", "forward", "ahead", "toward", "towards", "me", "you", "your", "my",
    "of", "to", "in", "on", "at", "by", "for", "and", "or",
})

GENERIC_POINTERS = frozenset({
    "это", "object", "объект", "prop", "предмет", "thing", "item", "it",
})

_RU_SUFFIXES = (
    "ами", "ями", "ов", "ев", "ей", "ий", "ый", "ая", "ое", "ую", "юю",
    "ом", "ем", "ам", "ям", "ах", "ях", "ы", "и", "а", "я", "о", "е", "у", "ю",
)

_DEFAULT_FORWARD_CONE_COS = 0.35
_ARM_REACH_M = 0.95
_LEXICAL_MATCH_THRESHOLD = 0.35
_EMBED_MATCH_THRESHOLD = 0.45

# Deictic reference phrases — embedding similarity, not keyword routing.
_DEICTIC_FORWARD_PHRASES: tuple[str, ...] = (
    "перед тобой",
    "перед собой",
    "впереди",
    "прямо перед",
    "in front of you",
    "in front of me",
    "ahead of you",
    "straight ahead",
    "object in front",
)

_deictic_emb_cache: dict[str, Any] = {}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def deictic_forward_min_score() -> float:
    return _env_float("RKK_DEICTIC_FORWARD_MIN", 0.38)


@dataclass(frozen=True)
class ResolvedObject:
    ref: str
    obj_id: str
    body_id: int | None
    semantic: str
    position: tuple[float, float, float]
    mass: float
    movable: bool
    source: str


def _tokenize(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-zа-яё0-9]+", str(query or "").lower(), flags=re.IGNORECASE) if t]


def _ru_stem(token: str) -> str:
    t = str(token or "").lower()
    if not re.search(r"[а-яё]", t):
        return t
    for suf in _RU_SUFFIXES:
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _norm_token(token: str) -> str:
    return _ru_stem(str(token or "").lower())


def _is_generic_token(token: str) -> bool:
    nt = _norm_token(str(token or ""))
    return nt in GENERIC_POINTERS or nt in {"object", "obj", "thing", "item", "prop"}


def extract_content_tokens(query: str) -> list[str]:
    """Content tokens from a command (verbs and stop-words removed)."""
    tokens = _tokenize(query)
    return [
        t
        for t in tokens
        if t not in INTERACTION_VERBS and t not in STOP_WORDS
    ]


def has_generic_object_pointer(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    tokens = _tokenize(q)
    if any(_is_generic_token(t) for t in tokens):
        return True
    return any(ptr in q for ptr in GENERIC_POINTERS)


def has_move_verb(query: str) -> bool:
    q = str(query or "").strip().lower()
    return any(v in q for v in MOVE_VERBS)


def _candidate_fields(row: dict) -> list[str]:
    out: list[str] = []
    for key in ("semantic", "ref", "id", "label"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            text = val.strip().lower()
            out.append(text)
            for part in re.split(r"[_\-\s]+", text):
                if part.strip():
                    out.append(part.strip())
    return out


def _lexical_match_score(token: str, field: str) -> float:
    if not token or not field:
        return 0.0
    nt = _norm_token(token)
    nf = _norm_token(field)
    if not nt or not nf:
        return 0.0
    if nt == nf:
        return 1.0
    if len(nt) >= 3 and nt in nf:
        return 0.85
    if len(nf) >= 3 and nf in nt:
        return 0.85
    field_parts = [_norm_token(p) for p in re.findall(r"[a-zа-яё0-9]+", field.lower())]
    for fp in field_parts:
        if not fp:
            continue
        if nt == fp:
            return 1.0
        if len(nt) >= 3 and (nt in fp or fp in nt):
            return 0.8
    return 0.0


def _match_row_lexical(row: dict, noun_tokens: list[str]) -> float:
    if not noun_tokens:
        return 0.0
    fields = _candidate_fields(row)
    best = 0.0
    for tok in noun_tokens:
        for field in fields:
            best = max(best, _lexical_match_score(tok, field))
    return best


def _cosine(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    try:
        import numpy as np

        if isinstance(a, np.ndarray):
            a = a.reshape(-1)
        if isinstance(b, np.ndarray):
            b = b.reshape(-1)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.size == 0 or b.size == 0 or a.size != b.size:
                return 0.0
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
    except Exception:
        pass
    if a is None or b is None:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0 or la != lb:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def infer_semantic_from_query(query: str) -> str | None:
    """
    Infer target hint from utterance alone (no scene).

    Returns a noun stem, ``object`` for generic deictic commands, or None.
    """
    if not has_move_verb(query):
        return None
    tokens = extract_content_tokens(query)
    noun_tokens = [t for t in tokens if not _is_generic_token(t)]
    if noun_tokens:
        return _norm_token(noun_tokens[0])
    if has_generic_object_pointer(query):
        return "object"
    return None


def _is_movable(row: dict) -> bool:
    if "movable" in row:
        return bool(row.get("movable"))
    if "static" in row:
        return not bool(row.get("static"))
    if row.get("body_id") is not None:
        return True
    return False


def _has_coords(row: dict) -> bool:
    pos = row.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return True
    for key in ("x", "y", "hx", "tx"):
        if key in row:
            try:
                float(row[key])
                return True
            except (TypeError, ValueError):
                pass
    return False


def _norm_semantic(key: str) -> str:
    return re.sub(r"[_\-\s]+", "_", str(key or "").strip().lower()).strip("_") or "object"


def _candidate_row(
    row: dict,
    *,
    ref: str,
    source: str,
    semantic_fallback: str | None = None,
) -> dict:
    pos = _obj_xyz(row)
    body_id = row.get("body_id")
    sem = _semantic_of(row, semantic_fallback or _norm_semantic(ref))
    return {
        "ref": ref,
        "id": str(row.get("id") or ref),
        "body_id": body_id,
        "semantic": sem,
        "label": str(row.get("label") or ""),
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "mass": float(row.get("mass", 1.0 if body_id is not None else 0.0)),
        "movable": _is_movable(row),
        "source": source,
    }


def _append_extra_candidates(out: list[dict], extras: dict) -> None:
    for key, val in extras.items():
        if key in _HANDLED_EXTRA_KEYS:
            continue
        source = str(key)
        sem_fallback = _norm_semantic(key)
        if isinstance(val, dict):
            if _has_coords(val):
                ref = str(val.get("ref") or val.get("id") or key)
                out.append(_candidate_row(val, ref=ref, source=source, semantic_fallback=sem_fallback))
                continue
            for subkey, row in val.items():
                if not isinstance(row, dict) or not _has_coords(row):
                    continue
                ref = str(row.get("ref") or row.get("id") or f"{key}_{subkey}")
                out.append(
                    _candidate_row(row, ref=ref, source=source, semantic_fallback=sem_fallback)
                )
        elif isinstance(val, list):
            for i, row in enumerate(val):
                if not isinstance(row, dict) or not _has_coords(row):
                    continue
                ref = str(row.get("ref") or row.get("id") or f"{key}_{i}")
                out.append(
                    _candidate_row(row, ref=ref, source=source, semantic_fallback=sem_fallback)
                )


_HANDLED_EXTRA_KEYS = frozenset({"registry", "props", "static_geometry"})


def _obj_xyz(row: dict) -> tuple[float, float, float]:
    pos = row.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        z = float(pos[2]) if len(pos) > 2 else 0.5
        return float(pos[0]), float(pos[1]), z
    x = float(row.get("x", row.get("hx", row.get("tx", 0.0))))
    y = float(row.get("y", row.get("hy", row.get("ty", 0.0))))
    z = float(row.get("z", row.get("hz", row.get("tz", 0.5))))
    return x, y, z


def _semantic_of(row: dict, fallback: str = "object") -> str:
    for key in ("semantic", "type", "kind", "style", "label"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return fallback


def _dedup_candidates_by_ref(candidates: list[dict]) -> list[dict]:
    """Keep one row per ref; prefer movable entries when registry/extras duplicate."""
    by_ref: dict[str, dict] = {}
    for row in candidates:
        ref = str(row["ref"])
        prev = by_ref.get(ref)
        if prev is None:
            by_ref[ref] = row
            continue
        if bool(row.get("movable")) and not bool(prev.get("movable")):
            by_ref[ref] = row
    return list(by_ref.values())


def collect_scene_candidates(scene_extras: dict | None) -> list[dict]:
    """Flatten scene extras (registry, props, static, and generic keyed objects)."""
    extras = dict(scene_extras or {})
    out: list[dict] = []

    registry = extras.get("registry")
    if isinstance(registry, dict):
        registry = list(registry.values())
    if isinstance(registry, list):
        for i, row in enumerate(registry):
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref") or row.get("id") or f"registry_{i}")
            out.append(_candidate_row(row, ref=ref, source=str(row.get("source") or "registry"), semantic_fallback="object"))

    props = extras.get("props") or []
    if isinstance(props, list):
        for i, row in enumerate(props):
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref") or row.get("id") or f"prop_{i}")
            out.append(_candidate_row(row, ref=ref, source=str(row.get("source") or "props"), semantic_fallback="prop"))

    static = extras.get("static_geometry") or []
    if isinstance(static, list):
        for i, row in enumerate(static):
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref") or row.get("id") or f"static_{i}")
            row_static = dict(row)
            row_static.setdefault("static", True)
            out.append(
                _candidate_row(
                    row_static,
                    ref=ref,
                    source="static_geometry",
                    semantic_fallback="static",
                )
            )

    _append_extra_candidates(out, extras)
    return _dedup_candidates_by_ref(out)


def _normalize_xy(v: tuple[float, float]) -> tuple[float, float]:
    x, y = float(v[0]), float(v[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return 1.0, 0.0
    return x / n, y / n


def _forward_alignment(
    row: dict,
    *,
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float] | None,
) -> float:
    if agent_forward is None:
        return 0.0
    ox, oy = float(row["x"]), float(row["y"])
    dx, dy = ox - agent_xy[0], oy - agent_xy[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 1.0
    fx, fy = _normalize_xy(agent_forward)
    return float((dx * fx + dy * fy) / dist)


def clear_deictic_cache() -> None:
    """Test helper: reset cached deictic phrase embeddings."""
    _deictic_emb_cache.clear()


def deictic_forward_strength(
    query: str,
    embed_fn: Callable[[str], list[float] | tuple[float, ...] | None] | None,
    *,
    min_score: float | None = None,
) -> float:
    """Max cosine between query and deictic forward-reference paraphrases."""
    if embed_fn is None:
        return 0.0
    raw = str(query or "").strip()
    if not raw:
        return 0.0
    q_vec = embed_fn(raw)
    if q_vec is None:
        return 0.0
    thresh = float(min_score if min_score is not None else deictic_forward_min_score())
    best = -2.0
    for phrase in _DEICTIC_FORWARD_PHRASES:
        if phrase not in _deictic_emb_cache:
            emb = embed_fn(phrase)
            if emb is None:
                continue
            _deictic_emb_cache[phrase] = emb
        sim = _cosine(q_vec, _deictic_emb_cache[phrase])
        best = max(best, sim)
    return float(best) if best >= thresh else 0.0


def _pool_for_interaction(
    candidates: list[dict],
    *,
    require_movable: bool,
    interaction_kinds: frozenset[str] | None,
) -> list[dict]:
    movable_only = bool(require_movable)
    if interaction_kinds is not None:
        if "displace" in interaction_kinds:
            movable_only = True
        elif interaction_kinds & {"contact", "reduce_distance"}:
            movable_only = False
    if movable_only:
        return [c for c in candidates if bool(c.get("movable"))]
    return list(candidates)


def _filter_forward_cone(
    pool: list[dict],
    *,
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float] | None,
    forward_cone_cos: float,
) -> list[tuple[dict, float]]:
    if agent_forward is None or not pool:
        return [(c, _forward_alignment(c, agent_xy=agent_xy, agent_forward=agent_forward)) for c in pool]
    cos_min = float(forward_cone_cos)
    for _ in range(4):
        in_cone: list[tuple[dict, float]] = []
        for c in pool:
            cos_a = _forward_alignment(c, agent_xy=agent_xy, agent_forward=agent_forward)
            if cos_a >= cos_min:
                in_cone.append((c, cos_a))
        if in_cone:
            return in_cone
        cos_min = max(0.0, cos_min - 0.12)
    return [(c, _forward_alignment(c, agent_xy=agent_xy, agent_forward=agent_forward)) for c in pool]


def _score_candidate(
    row: dict,
    *,
    lexical_match: float,
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float] | None,
    forward_cone_cos: float,
) -> float:
    ox, oy = float(row["x"]), float(row["y"])
    dx, dy = ox - agent_xy[0], oy - agent_xy[1]
    dist = math.hypot(dx, dy)
    score = -dist

    if lexical_match > 0.0:
        score += 2.0 * lexical_match

    if bool(row.get("movable")):
        score += 5.0
    else:
        score -= 8.0

    if agent_forward is not None and dist > 1e-6:
        fx, fy = _normalize_xy(agent_forward)
        cos_a = (dx * fx + dy * fy) / dist
        if cos_a >= forward_cone_cos:
            score += 1.5 + 0.5 * cos_a
        else:
            score -= 1.0

    if dist <= _ARM_REACH_M:
        score += 0.4

    return score


def _score_candidate_deictic(
    row: dict,
    *,
    agent_xy: tuple[float, float],
    forward_cos: float,
    lexical_match: float,
    prefer_movable: bool,
) -> float:
    ox, oy = float(row["x"]), float(row["y"])
    dist = math.hypot(ox - agent_xy[0], oy - agent_xy[1])
    score = -dist + 2.5 * float(forward_cos)
    if lexical_match > 0.0:
        score += 1.5 * lexical_match
    if prefer_movable and bool(row.get("movable")):
        score += 0.6
    if dist <= _ARM_REACH_M:
        score += 0.25
    return score


def resolve_manipulation_target(
    query: str,
    scene_extras: dict | None,
    *,
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float] | None = None,
    require_movable: bool = True,
    forward_cone_cos: float = _DEFAULT_FORWARD_CONE_COS,
    embed_fn: Callable[[str], list[float] | tuple[float, ...] | None] | None = None,
    interaction_kinds: frozenset[str] | None = None,
) -> tuple[ResolvedObject | None, dict[str, Any]]:
    """
    Resolve a manipulation target from scene data.

    Returns (ResolvedObject | None, diagnostics) where diagnostics is JSON-safe.
    """
    q = str(query or "")
    content_tokens = extract_content_tokens(q)
    noun_tokens = [t for t in content_tokens if not _is_generic_token(t)]
    has_noun = bool(noun_tokens)
    generic_ptr = has_generic_object_pointer(q)
    deictic_strength = deictic_forward_strength(q, embed_fn)
    deictic_forward = (
        deictic_strength > 0.0
        or (generic_ptr and not has_noun and not has_move_verb(q))
        or (
            generic_ptr
            and interaction_kinds is not None
            and "displace" not in interaction_kinds
            and not has_noun
        )
    )

    candidates = collect_scene_candidates(scene_extras)
    diag: dict[str, Any] = {
        "query": q,
        "content_tokens": content_tokens,
        "noun_tokens": noun_tokens,
        "has_noun": has_noun,
        "generic_pointer": generic_ptr,
        "deictic_forward": bool(deictic_forward),
        "deictic_strength": round(float(deictic_strength), 4) if deictic_strength else 0.0,
        "interaction_kinds": sorted(interaction_kinds) if interaction_kinds else [],
        "candidate_count": len(candidates),
        "agent_xy": [float(agent_xy[0]), float(agent_xy[1])],
        "reason": "",
    }

    if not candidates:
        diag["reason"] = "no_scene_candidates"
        return None, diag

    lexical_scores: dict[str, float] = {}
    for c in candidates:
        ref = str(c["ref"])
        lexical_scores[ref] = _match_row_lexical(c, noun_tokens)

    embed_scores: dict[str, float] = {}

    if has_noun and embed_fn is not None:
        query_text = " ".join(noun_tokens)
        q_vec = embed_fn(query_text)
        if q_vec is not None:
            for c in candidates:
                sem = str(c.get("semantic") or "")
                if not sem:
                    continue
                s_vec = embed_fn(sem)
                if s_vec is not None:
                    embed_scores[str(c["ref"])] = _cosine(q_vec, s_vec)

    def _effective_match(ref: str) -> float:
        return max(lexical_scores.get(ref, 0.0), embed_scores.get(ref, 0.0))

    if has_noun:
        matched = [c for c in candidates if _effective_match(str(c["ref"])) >= _LEXICAL_MATCH_THRESHOLD]
        if embed_fn is not None and not matched:
            matched = [
                c for c in candidates
                if embed_scores.get(str(c["ref"]), 0.0) >= _EMBED_MATCH_THRESHOLD
            ]
        if not matched:
            diag["reason"] = "no_semantic_match"
            diag["scene_semantics"] = sorted({
                str(c.get("semantic") or "")
                for c in candidates
                if str(c.get("semantic") or "").strip()
            })
            return None, diag

        if require_movable:
            movable_matched = [c for c in matched if bool(c.get("movable"))]
            if not movable_matched:
                diag["reason"] = "target_not_movable"
                diag["matched_static_count"] = len(matched)
                return None, diag
            pool = movable_matched
        else:
            pool = matched
    else:
        pool = _pool_for_interaction(
            candidates,
            require_movable=require_movable,
            interaction_kinds=interaction_kinds,
        )
        if require_movable and not pool and interaction_kinds is None:
            static_matched = [c for c in candidates if not bool(c.get("movable"))]
            if static_matched and (generic_ptr or has_move_verb(q)):
                diag["reason"] = "target_not_movable"
                diag["matched_static_count"] = len(static_matched)
                return None, diag
        if not pool:
            diag["reason"] = "no_movable_candidates" if require_movable else "no_matching_candidates"
            return None, diag

    if not pool:
        diag["reason"] = "no_matching_candidates"
        return None, diag

    prefer_movable = bool(
        require_movable
        or (interaction_kinds is not None and "displace" in interaction_kinds)
    )

    if deictic_forward:
        cone_pool = _filter_forward_cone(
            pool,
            agent_xy=agent_xy,
            agent_forward=agent_forward,
            forward_cone_cos=forward_cone_cos,
        )
        ranked = sorted(
            cone_pool,
            key=lambda item: _score_candidate_deictic(
                item[0],
                agent_xy=agent_xy,
                forward_cos=item[1],
                lexical_match=_effective_match(str(item[0]["ref"])),
                prefer_movable=prefer_movable,
            ),
            reverse=True,
        )
        best = ranked[0][0]
        diag["resolution_mode"] = "deictic_forward_cone"
    else:
        ranked = sorted(
            pool,
            key=lambda c: _score_candidate(
                c,
                lexical_match=_effective_match(str(c["ref"])),
                agent_xy=agent_xy,
                agent_forward=agent_forward,
                forward_cone_cos=forward_cone_cos,
            ),
            reverse=True,
        )
        best = ranked[0]
        diag["resolution_mode"] = "spatial_lexical"
    pos = _obj_xyz(best)
    body_id = best.get("body_id")
    resolved = ResolvedObject(
        ref=str(best["ref"]),
        obj_id=str(best.get("id") or best["ref"]),
        body_id=int(body_id) if body_id is not None else None,
        semantic=str(best.get("semantic") or "object"),
        position=pos,
        mass=float(best.get("mass", 1.0)),
        movable=bool(best.get("movable")),
        source=str(best.get("source") or "unknown"),
    )
    diag["reason"] = "resolved"
    diag["chosen_ref"] = resolved.ref
    diag["chosen_source"] = resolved.source
    diag["chosen_movable"] = resolved.movable
    diag["chosen_distance_m"] = round(
        math.hypot(pos[0] - agent_xy[0], pos[1] - agent_xy[1]), 4
    )
    diag["chosen_forward_cos"] = round(
        _forward_alignment(best, agent_xy=agent_xy, agent_forward=agent_forward), 4
    )
    diag["lexical_match"] = round(_effective_match(resolved.ref), 4)
    return resolved, diag
