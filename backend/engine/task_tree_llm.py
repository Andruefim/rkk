"""Optional LLM-assisted task-tree stage decomposition (predicate-safe)."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from engine.task_goal import TaskGoal

# Allowed stage kinds the LLM may emit (must match task_tree executor).
_ALLOWED_STAGES = frozenset(
    {
        "resolve_target",
        "approach",
        "reach_contact",
        "reach_target",
        "push_target",
        "verify_target",
        "imagine_goal",
        "execute_goal",
        "verify_goal",
    }
)


def llm_decompose_enabled() -> bool:
    return os.environ.get("RKK_TASK_TREE_LLM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _parse_stages_json(raw: str) -> list[str] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    # Extract JSON array if wrapped in prose
    m = re.search(r"\[[^\]]+\]", text, flags=re.DOTALL)
    blob = m.group(0) if m else text
    try:
        data = json.loads(blob)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    out: list[str] = []
    for item in data:
        s = str(item).strip()
        if s in _ALLOWED_STAGES and s not in out:
            out.append(s)
    return out or None


def llm_decompose_stages(
    goal: TaskGoal,
    *,
    command_text: str = "",
    needs_target: bool = False,
) -> tuple[str, ...] | None:
    """
    Ask local LLM for stage list. Returns None → caller uses ontology fallback.
    Never invents verbs outside allowlist.
    """
    if not llm_decompose_enabled():
        return None

    pred_kinds = [str(p.kind) for p in (goal.predicates or [])]
    prompt = (
        "You decompose a humanoid robot task into ordered stage names.\n"
        f"Command: {command_text!r}\n"
        f"Predicates: {pred_kinds}\n"
        f"needs_target: {bool(needs_target)}\n"
        "Reply with ONLY a JSON array of stage names from this allowlist:\n"
        f"{sorted(_ALLOWED_STAGES)}\n"
        "Typical contact task: [\"resolve_target\",\"approach\",\"reach_contact\",\"verify_goal\"]\n"
        "No prose."
    )

    raw = None
    try:
        import urllib.request

        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        model = os.environ.get(
            "RKK_OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        )
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 120, "temperature": 0.1},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            raw = payload.get("response") or payload.get("text")
    except Exception:
        return None

    stages = _parse_stages_json(str(raw or ""))
    if not stages:
        return None
    if needs_target and "resolve_target" not in stages:
        stages = ["resolve_target", *stages]
    return tuple(stages)
