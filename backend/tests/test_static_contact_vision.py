"""Static body contact resolution from vision/OWM (no full PyBullet)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from engine.features.humanoid.pybullet_humanoid import _PyBulletHumanoid
from engine.features.simulation.mixin_grounded_language import SimulationGroundedLanguageMixin
from engine.object_working_memory import LatentSceneMemory, ObjectWorkingMemory, SceneEntity


def _make_env_with_registry() -> _PyBulletHumanoid:
    env = _PyBulletHumanoid.__new__(_PyBulletHumanoid)
    env._static_body_registry = [
        {
            "body_id": 101,
            "kind": "cylinder",
            "style": "planter",
            "x": 2.0,
            "y": 0.0,
            "radius": 0.3,
            "height": 1.0,
        },
        {
            "body_id": 102,
            "kind": "cylinder",
            "style": "default",
            "x": 5.0,
            "y": 0.0,
            "radius": 0.2,
            "height": 0.8,
        },
        {
            "body_id": 103,
            "kind": "box",
            "style": "default",
            "x": 0.0,
            "y": 3.0,
            "hx": 0.5,
            "hy": 0.5,
            "hz": 0.5,
        },
    ]
    return env


def test_find_static_contact_body_cylinder_surface() -> None:
    env = _make_env_with_registry()
    # Near planter surface (center 2.0 + radius 0.3)
    bid = env.find_static_contact_body((2.25, 0.0), kind="cylinder")
    assert bid == 101
    # Farther cylinder wins when closer in surface distance
    bid2 = env.find_static_contact_body((4.9, 0.0))
    assert bid2 == 102


def test_find_static_contact_body_respects_kind_filter() -> None:
    env = _make_env_with_registry()
    assert env.find_static_contact_body((0.1, 3.0), kind="box") == 103
    assert env.find_static_contact_body((0.1, 3.0), kind="cylinder") is None


def test_find_static_contact_body_max_dist() -> None:
    env = _make_env_with_registry()
    assert env.find_static_contact_body((20.0, 0.0), max_dist_m=0.5) is None


class _StaticContactHarness(SimulationGroundedLanguageMixin):
    def __init__(self, *, contact_body_id: int | None = None) -> None:
        self._contact_body_id = contact_body_id
        self._owm_cached = None
        scene = LatentSceneMemory()
        ent = SceneEntity(entity_id="bound_0")
        ent.seed_from_bearing_range(
            bearing=0.0,
            range_m=2.0,
            tick=1,
            label="planter",
            confidence=0.8,
            slot_id="slot_0",
        )
        scene.entities["bound_0"] = ent
        scene.focus("bound_0", exclusive=True)
        self._obj_working_memory = ObjectWorkingMemory(scene)
        self._base = _MockBaseEnv(contact_body_id=contact_body_id)
        self.agent = SimpleNamespace(env=SimpleNamespace(base_env=self._base, _contact_flag=False))


class _MockBaseEnv:
    def __init__(self, *, contact_body_id: int | None) -> None:
        self._static_body_registry = [
            {
                "body_id": 201,
                "kind": "cylinder",
                "style": "planter",
                "x": 2.0,
                "y": 0.0,
                "radius": 0.3,
                "height": 1.0,
            }
        ]
        self._contact_body_id = contact_body_id

    def get_task_agent_pose(self) -> dict:
        return {"xy": (0.0, 0.0), "forward": (1.0, 0.0), "yaw": 0.0}

    def find_static_contact_body(
        self,
        world_xy: tuple[float, float],
        *,
        kind: str | None = None,
        style: str | None = None,
        max_dist_m: float = 1.2,
    ) -> int | None:
        wx, wy = float(world_xy[0]), float(world_xy[1])
        best_id: int | None = None
        best_dist = float(max_dist_m)
        for row in self._static_body_registry:
            if kind is not None and str(row.get("kind")) != str(kind):
                continue
            bx = float(row.get("x", 0.0))
            by = float(row.get("y", 0.0))
            d = float(math.hypot(wx - bx, wy - by))
            if str(row.get("kind")) == "cylinder":
                d -= float(row.get("radius", 0.0))
            if d < best_dist:
                best_dist = d
                best_id = int(row["body_id"])
        return best_id

    def _manip_has_contact(self, body_id: int) -> bool:
        return self._contact_body_id is not None and int(body_id) == int(
            self._contact_body_id
        )


def test_world_xy_from_owm_ahead_of_agent() -> None:
    h = _StaticContactHarness()
    owm = h._obj_working_memory
    xy = h._world_xy_from_owm(owm)
    assert xy is not None
    assert xy[0] == pytest.approx(2.0, abs=0.05)
    assert abs(xy[1]) < 0.05


def test_contact_body_id_from_owm_without_resolved() -> None:
    h = _StaticContactHarness()
    bid = h._contact_body_id_for_task(None)
    assert bid == 201


def test_manip_has_contact_via_static_body_without_resolved() -> None:
    h = _StaticContactHarness(contact_body_id=201)
    assert h._manip_has_contact(None) is True


def test_manip_has_contact_false_when_no_pybullet_contact() -> None:
    h = _StaticContactHarness(contact_body_id=None)
    assert h._manip_has_contact(None) is False
