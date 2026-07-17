"""Tests for embedding-based goal grounding, WM trust, and predicate verification."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from engine.causal_graph import CausalGraph
from engine.grounded_language import FallbackEmbeddingClient
from engine.goal_grounding import ground_command
from engine.success_predicates import evaluate_goal
from engine.task_binding import TaskBindingController
from engine.task_goal import GoalPredicate, TaskGoal


@pytest.fixture(autouse=True)
def _reset_catalog_cache():
    from engine.goal_grounding import clear_catalog_cache, clear_direction_cache
    from engine.object_resolver import clear_deictic_cache

    clear_catalog_cache()
    clear_direction_cache()
    clear_deictic_cache()
    yield
    clear_catalog_cache()
    clear_direction_cache()
    clear_deictic_cache()


def _hash_vec(text: str, dim: int = 32) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        vec[bucket] += 1.0
    phrase_digest = hashlib.sha256(text.encode("utf-8")).digest()
    for i in range(min(6, dim)):
        off = (i * 3) % max(1, len(phrase_digest) - 2)
        vec[i] += int.from_bytes(phrase_digest[off : off + 2], "little") / 65535.0
    n = float(np.linalg.norm(vec)) + 1e-9
    return vec / n


class _PhraseEmbedder:
    """Deterministic embed_fn: explicit phrase→vector dictionary for tests."""

    def __init__(self, *, dim: int = 32) -> None:
        self.dim = int(dim)
        kinds = ("reduce_distance", "contact", "displace", "state_key")
        self._kind_dirs = {
            k: _normalize(np.eye(self.dim, dtype=np.float32)[i])
            for i, k in enumerate(kinds)
        }

        phrase_kind: dict[str, str] = {
            # commands
            "передвинь стул": "displace",
            "передвинуть стул": "displace",
            "дотронься до шара": "contact",
            "подойди": "reduce_distance",
            "дотронься до объекта перед тобой": "contact",
            "подойди и дотронься до объекта перед тобой": "contact",
            "подойди к кубу": "reduce_distance",
            "передвинь стул": "displace",
            "иди вперёд": "state_key",
            # catalog RU
            "подойти к объекту": "reduce_distance",
            "приблизиться к цели": "reduce_distance",
            "подойди ближе": "reduce_distance",
            "дотронуться до объекта": "contact",
            "коснуться": "contact",
            "дотронься": "contact",
            "передвинуть объект": "displace",
            "сдвинуть": "displace",
            "толкни": "displace",
            "иду вперёд": "state_key",
            "повернись": "state_key",
            "повернись налево": "state_key",
            "повернись направо": "state_key",
            "встань": "state_key",
            "стабилизируйся": "state_key",
            # catalog EN
            "walk up to the object": "reduce_distance",
            "go to the target": "reduce_distance",
            "approach the object": "reduce_distance",
            "get closer": "reduce_distance",
            "touch the object": "contact",
            "make contact": "contact",
            "reach and touch": "contact",
            "push the object": "displace",
            "move the object": "displace",
            "shift it": "displace",
            "step forward": "state_key",
            "walk forward": "state_key",
            "turn around": "state_key",
            "turn left": "state_key",
            "turn right": "state_key",
            "get up": "state_key",
            "stand stable": "state_key",
        }
        self._anchors: dict[str, np.ndarray] = {}
        for phrase, kind in phrase_kind.items():
            noise = _hash_vec(phrase, dim=self.dim) * 0.08
            self._anchors[phrase] = _normalize(self._kind_dirs[kind] + noise)

    def embed(self, text: str) -> np.ndarray | None:
        t = str(text or "").strip()
        if not t:
            return None
        if t in self._anchors:
            return self._anchors[t]
        return _hash_vec(t, dim=self.dim)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


class _AnchorGraph:
    """Graph with live obs anchoring and non-degenerate rollout."""

    def __init__(self) -> None:
        self.nodes = {
            "intent_stride": 0.5,
            "intent_stop_recover": 0.5,
            "posture_stability": 0.62,
            "target_dist": 1.4,
            "com_z": 0.55,
        }

    def propagate_from(self, base: dict[str, float], variable: str, value: float) -> dict[str, float]:
        out = dict(base)
        out[str(variable)] = float(value)
        if variable == "intent_stride":
            out["posture_stability"] = float(np.clip(0.35 + 0.55 * float(value), 0.05, 0.95))
            out["target_dist"] = float(out.get("target_dist", 1.0)) * 0.92
        if variable == "intent_stop_recover":
            out["com_z"] = float(np.clip(0.35 + 0.35 * float(value), 0.05, 0.95))
        return out

    def rollout_step_free(self, base: dict[str, float]) -> dict[str, float]:
        out = dict(base)
        out["posture_stability"] = float(out.get("posture_stability", 0.5)) * 0.99 + 0.005
        out["target_dist"] = float(out.get("target_dist", 1.0)) * 0.97
        return out


class _DegenerateGraph:
    """Rollout collapses all keys to a constant — wm_trusted must be False."""

    nodes = {"posture_stability": 0.5, "target_dist": 0.5}

    def propagate_from(self, base: dict[str, float], variable: str, value: float) -> dict[str, float]:
        return dict(base)

    def rollout_step_free(self, base: dict[str, float]) -> dict[str, float]:
        return {k: 0.111 for k in base}


def test_ground_touch_ball_contact_and_approach() -> None:
    emb = _PhraseEmbedder()
    goal = ground_command("дотронься до шара", emb.embed)
    kinds = [p.kind for p in goal.predicates]
    assert "contact" in kinds
    assert "reduce_distance" in kinds
    assert goal.diagnostics.get("needs_target") is True
    assert goal.confidence > 0.0


def test_ground_composite_approach_and_touch() -> None:
    emb = _PhraseEmbedder()
    cmd = "подойди и дотронься до объекта перед тобой"
    goal = ground_command(cmd, emb.embed)
    kinds = [p.kind for p in goal.predicates]
    assert "reduce_distance" in kinds
    assert "contact" in kinds
    assert "displace" not in kinds
    assert goal.diagnostics.get("primary_kind") == "contact"
    assert goal.diagnostics.get("composite") is True
    assert len(goal.diagnostics.get("clauses") or []) >= 2


def test_ground_approach_cube() -> None:
    emb = _PhraseEmbedder()
    goal = ground_command("подойди к кубу", emb.embed)
    kinds = [p.kind for p in goal.predicates]
    assert "reduce_distance" in kinds
    assert "contact" not in kinds
    assert goal.diagnostics.get("needs_target") is True


def test_ground_displace_chair() -> None:
    emb = _PhraseEmbedder()
    goal = ground_command("передвинь стул", emb.embed)
    kinds = [p.kind for p in goal.predicates]
    assert "displace" in kinds
    assert goal.diagnostics.get("needs_target") is True


def test_ground_locomote_no_target() -> None:
    emb = _PhraseEmbedder()
    goal = ground_command("иди вперёд", emb.embed)
    kinds = [p.kind for p in goal.predicates]
    assert "state_key" in kinds
    assert goal.diagnostics.get("needs_target") is False
    assert any(p.key == "intent_stride" for p in goal.predicates)


def test_ground_turn_left_state_key_only() -> None:
    emb = _PhraseEmbedder()
    goal = ground_command("повернись налево", emb.embed)
    kinds = [p.kind for p in goal.predicates]
    assert kinds == ["state_key"]
    assert goal.diagnostics.get("needs_target") is False
    assert any(p.key == "intent_gait_coupling" for p in goal.predicates)


def test_ground_fallback_without_embed_fn() -> None:
    goal = ground_command("иди вперёд", None)
    assert goal.confidence == 0.0
    assert len(goal.predicates) == 1
    assert goal.predicates[0].kind == "state_key"


def test_imagination_anchored_nonzero() -> None:
    g = _AnchorGraph()
    obs = {"posture_stability": 0.62, "target_dist": 1.4, "com_z": 0.55}
    tb = TaskBindingController()
    expected, wm_trusted, diag = tb.imagine_expected_state(g, obs, horizon=4)
    assert expected
    assert expected.get("posture_stability", 0) > 0.1
    assert expected.get("target_dist", 0) > 0.1
    assert diag.get("anchor_pe_mean", 1.0) < 0.05


def test_wm_trust_gate_degenerate() -> None:
    g = _DegenerateGraph()
    obs = {"posture_stability": 0.62, "target_dist": 1.4}
    tb = TaskBindingController()
    expected, wm_trusted, diag = tb.imagine_expected_state(g, obs, horizon=3)
    assert wm_trusted is False
    assert diag.get("degenerate") is True or diag.get("wm_trust_reason") in (
        "degenerate_imagination",
        "step1_drift",
    )


def test_bind_command_sets_goal_and_wm_trust() -> None:
    g = CausalGraph(device=torch.device("cpu"))
    g.set_node("posture_stability", 0.6)
    g.set_node("target_dist", 0.9)
    obs = {"posture_stability": 0.6, "target_dist": 0.9, "com_z": 0.55}
    tb = TaskBindingController()
    emb = _PhraseEmbedder()
    task = tb.bind_command(g, obs, "иди вперёд", tick=1, embed_fn=emb.embed)
    assert task.goal is not None
    assert task.goal.text == "иди вперёд"
    assert isinstance(task.goal.wm_trusted, bool)


def test_evaluate_goal_reduce_distance_satisfied() -> None:
    goal = TaskGoal(
        text="подойди",
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.9, tolerance=0.25, weight=1.0),
        ],
    )
    obs = {"target_dist": 0.85}
    ctx = {"distance_m": 0.8}
    ok, score, detail = evaluate_goal(goal, obs, ctx)
    assert score > 0.85
    assert ok
    assert detail["predicates"][0]["kind"] == "reduce_distance"


def test_evaluate_goal_state_key() -> None:
    goal = TaskGoal(
        text="иди",
        predicates=[
            GoalPredicate(
                kind="state_key",
                key="intent_stride",
                target_value=0.66,
                tolerance=0.15,
            ),
        ],
    )
    obs = {"intent_stride": 0.65}
    ok, score, _ = evaluate_goal(goal, obs, {})
    assert ok
    assert score > 0.9


def test_verify_predicate_path_when_wm_untrusted() -> None:
    from engine.task_binding import HumanTask

    tb = TaskBindingController()
    goal = TaskGoal(
        text="подойди",
        wm_trusted=False,
        predicates=[
            GoalPredicate(kind="reduce_distance", target_value=0.9, tolerance=0.25),
        ],
    )
    task = HumanTask(
        text="подойди",
        expected_state={},
        max_prediction_error=0.5,
        tick_started=0,
        tick_deadline=100,
        goal=goal,
    )
    ok, pe, diag = tb.verify({"target_dist": 0.5}, task, ctx={"distance_m": 0.7})
    assert diag.get("score") is not None
    assert ok


def test_imagine_uses_predicate_interventions_contact() -> None:
    from engine.goal_interventions import interventions_for_goal
    from engine.task_goal import TaskGoal

    g = _AnchorGraph()
    obs = {"posture_stability": 0.62, "target_dist": 1.4}
    goal = TaskGoal(
        text="дотронься",
        predicates=[
            GoalPredicate(kind="contact", target_value=1.0),
            GoalPredicate(kind="reduce_distance", target_value=0.9),
        ],
    )
    motor = interventions_for_goal(
        goal, agent_xy=(0.0, 0.0), target_xy=(1.0, -0.2), agent_forward=(1.0, 0.0)
    )
    assert "intent_reach_right" in motor
    assert "intent_grasp" in motor

    tb = TaskBindingController()
    _, _, diag = tb.imagine_expected_state(
        g,
        obs,
        horizon=4,
        goal=goal,
        agent_xy=(0.0, 0.0),
        target_xy=(1.0, -0.2),
        agent_forward=(1.0, 0.0),
    )
    assert "intent_reach_right" in diag.get("interventions", [])
    assert "intent_grasp" in diag.get("interventions", [])


def test_infer_manip_direction_geometry_default() -> None:
    from engine.goal_grounding import infer_manip_direction

    direction = infer_manip_direction(
        "передвинь стул",
        agent_xy=(0.0, 0.0),
        target_xy=(1.0, 0.0),
        agent_forward=(0.0, 1.0),
        embed_fn=None,
    )
    assert direction[0] > 0.9
    assert abs(direction[1]) < 0.1


def test_infer_manip_direction_embed_override() -> None:
    from engine.goal_grounding import infer_manip_direction

    fb = FallbackEmbeddingClient(embed_dim=64)

    def _dir_embed(text: str):
        vec = fb.embed(text)
        if vec is None:
            return None
        low = text.lower()
        if "назад" in low or "back" in low:
            return fb.embed("назад")
        return vec

    direction = infer_manip_direction(
        "толкни назад",
        agent_xy=(0.0, 0.0),
        target_xy=(1.0, 0.0),
        agent_forward=(1.0, 0.0),
        embed_fn=_dir_embed,
        min_embed_score=0.5,
    )
    assert direction[0] < -0.5
