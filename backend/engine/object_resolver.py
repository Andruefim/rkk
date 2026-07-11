"""Resolve manipulation targets from scene extras/registry (no VLM/Ollama)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

MOVE_VERBS = frozenset({"передвинь", "передвинуть", "move", "push", "толкни", "сдвинь", "shift"})

STOP_WORDS = frozenset({
    "перед", "тобой", "тебя", "вперёд", "вперед", "ближе", "меня", "мной", "туда", "сюда",
    "немного", "чуть", "пожалуйста", "please", "now", "сейчас",
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


def extract_content_tokens(query: str) -> list[str]:
    """Content tokens from a command (verbs and stop-words removed)."""
    tokens = _tokenize(query)
    return [t for t in tokens if t not in MOVE_VERBS and t not in STOP_WORDS]


def has_generic_object_pointer(query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    tokens = set(_tokenize(q))
    if tokens & GENERIC_POINTERS:
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
    noun_tokens = [t for t in tokens if t not in GENERIC_POINTERS]
    if noun_tokens:
        return _norm_token(noun_tokens[0])
    if has_generic_object_pointer(query):
        return "object"
    return None


def _obj_xyz(row: dict) -> tuple[float, float, float]:
    x = float(row.get("x", row.get("hx", row.get("tx", 0.0))))
    y = float(row.get("y", row.get("hy", row.get("ty", 0.0))))
    z = float(row.get("z", row.get("hz", row.get("tz", 0.5))))
    return x, y, z


def _is_movable(row: dict) -> bool:
    if "movable" in row:
        return bool(row.get("movable"))
    if "static" in row:
        return not bool(row.get("static"))
    mass = float(row.get("mass", 1.0))
    return mass > 0.0


def _semantic_of(row: dict, fallback: str = "object") -> str:
    for key in ("semantic", "type", "kind", "style", "label"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return fallback


def collect_scene_candidates(scene_extras: dict | None) -> list[dict]:
    """Flatten registry, props, and static geometry into candidate rows."""
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
            pos = _obj_xyz(row)
            out.append({
                "ref": ref,
                "id": ref,
                "body_id": row.get("body_id"),
                "semantic": _semantic_of(row, "object"),
                "label": str(row.get("label") or ""),
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
                "mass": float(row.get("mass", 1.0)),
                "movable": _is_movable(row),
                "source": str(row.get("source") or "registry"),
            })

    props = extras.get("props") or []
    if isinstance(props, list):
        for i, row in enumerate(props):
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref") or row.get("id") or f"prop_{i}")
            pos = _obj_xyz(row)
            out.append({
                "ref": ref,
                "id": ref,
                "body_id": row.get("body_id"),
                "semantic": _semantic_of(row, "prop"),
                "label": str(row.get("label") or ""),
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
                "mass": float(row.get("mass", 0.3)),
                "movable": _is_movable(row),
                "source": str(row.get("source") or "props"),
            })

    static = extras.get("static_geometry") or []
    if isinstance(static, list):
        for i, row in enumerate(static):
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref") or row.get("id") or f"static_{i}")
            pos = _obj_xyz(row)
            out.append({
                "ref": ref,
                "id": ref,
                "body_id": None,
                "semantic": _semantic_of(row, "static"),
                "label": str(row.get("label") or ""),
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
                "mass": 0.0,
                "movable": False,
                "source": "static_geometry",
            })

    return out


def _normalize_xy(v: tuple[float, float]) -> tuple[float, float]:
    x, y = float(v[0]), float(v[1])
    n = math.hypot(x, y)
    if n < 1e-9:
        return 1.0, 0.0
    return x / n, y / n


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


def resolve_manipulation_target(
    query: str,
    scene_extras: dict | None,
    *,
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float] | None = None,
    require_movable: bool = True,
    forward_cone_cos: float = _DEFAULT_FORWARD_CONE_COS,
    embed_fn: Callable[[str], list[float] | tuple[float, ...] | None] | None = None,
) -> tuple[ResolvedObject | None, dict[str, Any]]:
    """
    Resolve a manipulation target from scene data.

    Returns (ResolvedObject | None, diagnostics) where diagnostics is JSON-safe.
    """
    q = str(query or "")
    content_tokens = extract_content_tokens(q)
    noun_tokens = [t for t in content_tokens if t not in GENERIC_POINTERS]
    has_noun = bool(noun_tokens)
    generic_ptr = has_generic_object_pointer(q)

    candidates = collect_scene_candidates(scene_extras)
    diag: dict[str, Any] = {
        "query": q,
        "content_tokens": content_tokens,
        "noun_tokens": noun_tokens,
        "has_noun": has_noun,
        "generic_pointer": generic_ptr,
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

    def _combined_match(ref: str) -> float:
        return lexical_scores.get(ref, 0.0)

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
        lex = _combined_match(ref)
        if lex >= _LEXICAL_MATCH_THRESHOLD:
            return lex
        return embed_scores.get(ref, 0.0)

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
        pool = [c for c in candidates if bool(c.get("movable"))] if require_movable else list(candidates)
        if require_movable and not pool:
            static_matched = [c for c in candidates if not bool(c.get("movable"))]
            if static_matched and (generic_ptr or has_move_verb(q)):
                diag["reason"] = "target_not_movable"
                diag["matched_static_count"] = len(static_matched)
                return None, diag
            diag["reason"] = "no_movable_candidates"
            return None, diag

    if not pool:
        diag["reason"] = "no_matching_candidates"
        return None, diag

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
    diag["lexical_match"] = round(_effective_match(resolved.ref), 4)
    return resolved, diag
