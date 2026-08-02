"""Declarative visual-referent ontology for language↔slot binding.

Same pattern as goal_grounding: match command text to natural-language
descriptions by embedding cosine — no verb/keyword branching.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EmbedFn = Callable[[str], np.ndarray | None]

_emb_cache: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class VisualReferentEntry:
    """A visual referent the agent can bind a slot to."""

    key: str
    description: str
    weight: float = 1.0


# Extend by adding paraphrases — not if/elif on command text.
_VISUAL_REFERENT_CATALOG: tuple[VisualReferentEntry, ...] = (
    VisualReferentEntry(
        "cylinder",
        "цилиндр колонна бочка цилиндрическая форма",
        weight=1.25,
    ),
    VisualReferentEntry(
        "cylinder",
        "cylindrical column pillar barrel shape",
        weight=1.25,
    ),
    VisualReferentEntry(
        "cylinder",
        "white cylinder planter pillar",
        weight=1.2,
    ),
    VisualReferentEntry("object", "объект перед агентом в поле зрения камеры"),
    VisualReferentEntry("object", "предмет который видно впереди"),
    VisualReferentEntry("object", "цель для подхода и касания"),
    VisualReferentEntry("object", "an object visible in front of the agent"),
    VisualReferentEntry("object", "the object in front of you"),
    VisualReferentEntry("object", "something to approach and touch"),
    VisualReferentEntry("object", "a physical object in the camera view"),
    VisualReferentEntry(
        "cylinder",
        "подойди к цилиндрическому объекту перед тобой и дотронься",
        weight=1.3,
    ),
    VisualReferentEntry("chair", "стул в сцене"),
    VisualReferentEntry("chair", "a chair"),
    VisualReferentEntry("chair", "chair in front of you"),
    VisualReferentEntry("prop", "подвижный предмет"),
    VisualReferentEntry("prop", "a movable prop or object"),
    VisualReferentEntry("ball", "мяч"),
    VisualReferentEntry("ball", "a ball"),
    VisualReferentEntry("ball", "жёлтый шар сфера"),
    VisualReferentEntry("ball", "большой шар перед тобой"),
    VisualReferentEntry("ball", "этот большой шар"),
    VisualReferentEntry("ball", "big ball in front of you"),
    VisualReferentEntry("ball", "подойди к этому большому шару и дотронься"),
    VisualReferentEntry("ball", "круглый объект перед тобой"),
    VisualReferentEntry("ball", "круглому объекту перед тобой"),
    VisualReferentEntry("ball", "круглый обьект перед тобой"),
    VisualReferentEntry("cylinder", "цилиндрический большой объект перед тобой"),
    VisualReferentEntry("cylinder", "большой цилиндр перед тобой"),
)


def clear_visual_referent_cache() -> None:
    _emb_cache.clear()


def _normalize(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a)) + 1e-9
    return a / n


def _ensure_catalog(embed_fn: EmbedFn) -> list[tuple[VisualReferentEntry, np.ndarray]]:
    out: list[tuple[VisualReferentEntry, np.ndarray]] = []
    for entry in _VISUAL_REFERENT_CATALOG:
        key = entry.description
        if key not in _emb_cache:
            emb = embed_fn(key)
            if emb is None:
                continue
            _emb_cache[key] = _normalize(emb)
        out.append((entry, _emb_cache[key]))
    return out


def match_visual_referent(
    command_text: str,
    embed_fn: EmbedFn,
) -> tuple[VisualReferentEntry | None, float, dict]:
    """
    Best ontology referent for the command (embedding cosine).
    Returns (entry, score, diagnostics).
    """
    text = str(command_text or "").strip()
    diag: dict = {"catalog_size": 0}
    if not text:
        return None, 0.0, {**diag, "reason": "empty_command"}
    cmd = embed_fn(text)
    if cmd is None:
        return None, 0.0, {**diag, "reason": "command_embed_failed"}
    cmd_n = _normalize(cmd)
    catalog = _ensure_catalog(embed_fn)
    diag["catalog_size"] = len(catalog)
    if not catalog:
        return None, 0.0, {**diag, "reason": "empty_catalog"}

    best_entry: VisualReferentEntry | None = None
    best_sc = -1.0
    for entry, emb in catalog:
        sc = float(np.clip(np.dot(cmd_n, emb), -1.0, 1.0)) * float(entry.weight)
        if sc > best_sc:
            best_sc = sc
            best_entry = entry
    diag["best_key"] = best_entry.key if best_entry else None
    diag["best_description"] = best_entry.description if best_entry else None
    diag["best_score"] = float(best_sc)
    return best_entry, float(max(0.0, best_sc)), diag
