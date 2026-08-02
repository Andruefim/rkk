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


def test_contact_body_id_uses_cylinder_when_label_is_backward_lean() -> None:
    """Ontology/command cylinder must win over BACKWARD_LEAN OWM label."""

    class _Harness(_StaticContactHarness):
        def _task_ontology_best_key(self) -> str | None:
            return "cylinder"

    h = _Harness()
    ent = h._obj_working_memory.scene.entities["bound_0"]
    ent.label = "BACKWARD_LEAN"
    ent.range_m = 2.0
    bid = h._contact_body_id_for_task(None)
    assert bid == 201


def test_contact_body_id_near_planter_from_agent_com_when_ontology_cylinder() -> None:
    """Close range + cylinder ontology: probe from agent COM near planter rim."""

    class _Harness(_StaticContactHarness):
        def _task_ontology_best_key(self) -> str | None:
            return "cylinder"

        def _agent_xy_forward(self) -> tuple[tuple[float, float], tuple[float, float]]:
            return (1.95, 0.0), (1.0, 0.0)

    h = _Harness()
    ent = h._obj_working_memory.scene.entities["bound_0"]
    ent.label = "BACKWARD_LEAN"
    ent.range_m = 0.28
    ent.bearing = 0.0
    bid = h._contact_body_id_for_task(None)
    assert bid == 201


def test_contact_body_id_from_agent_com_when_range_within_contact_reach() -> None:
    """Within contact_reach_m, agent COM wins over stale OWM world XY."""

    class _Harness(_StaticContactHarness):
        def _task_ontology_best_key(self) -> str | None:
            return "cylinder"

        def _agent_xy_forward(self) -> tuple[tuple[float, float], tuple[float, float]]:
            return (1.95, 0.0), (1.0, 0.0)

    h = _Harness()
    ent = h._obj_working_memory.scene.entities["bound_0"]
    ent.label = "cylinder"
    ent.range_m = 0.85
    ent.bearing = 0.0
    bid = h._contact_body_id_for_task(None)
    assert bid == 201


def test_manip_has_contact_scans_all_cylinders_near_com() -> None:
    """When primary body misses, scan all cylinder registry entries near COM."""

    class _Harness(_StaticContactHarness):
        def _task_ontology_best_key(self) -> str | None:
            return "cylinder"

        def _agent_xy_forward(self) -> tuple[tuple[float, float], tuple[float, float]]:
            return (1.95, 0.0), (1.0, 0.0)

    h = _Harness(contact_body_id=201)
    ent = h._obj_working_memory.scene.entities["bound_0"]
    ent.label = "cylinder"
    ent.range_m = 0.45
    assert h._manip_has_contact(None) is True


def test_manip_has_contact_via_static_body_without_resolved() -> None:
    h = _StaticContactHarness(contact_body_id=201)
    assert h._manip_has_contact(None) is True


def test_manip_has_contact_false_when_no_pybullet_contact() -> None:
    h = _StaticContactHarness(contact_body_id=None)
    assert h._manip_has_contact(None) is False


def test_physics_range_to_locked_body_cylinder_surface() -> None:
    """COM near planter rim → surface distance, not inflated OWM range."""

    class _Harness(_StaticContactHarness):
        def _agent_xy_forward(self) -> tuple[tuple[float, float], tuple[float, float]]:
            return (1.95, 0.0), (1.0, 0.0)

    h = _Harness()
    h._task_locked_body_id = 201
    r = h._physics_range_to_locked_body()
    assert r is not None
    assert r == pytest.approx(0.0, abs=0.02)


def test_physics_range_to_locked_body_from_registry_xy() -> None:
    h = _StaticContactHarness()
    h._task_locked_body_id = 201
    r = h._physics_range_to_locked_body()
    assert r is not None
    # agent at origin, planter center (2,0) r=0.3 → surface ~1.7m
    assert r == pytest.approx(1.7, abs=0.05)


def test_blend_dist_uses_min_physics_when_owm_inflated() -> None:
    """Approach gate: min(vision/OWM, physics) completes when physically near."""

    h = _StaticContactHarness()
    h._task_locked_body_id = 201
    h.tick = 100
    blended = h._blend_dist_with_physics_range(3.8, 3.8, 100)
    assert blended == pytest.approx(1.7, abs=0.05)


def test_lock_task_contact_body_on_bind_from_cylinder_label() -> None:
    from engine.vision_target import VisualTarget

    h = _StaticContactHarness()
    vt = VisualTarget(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="planter",
        confidence=0.8,
        bearing=0.0,
        range_m=3.5,
    )
    h._lock_task_contact_body_on_bind(vt)
    assert h._task_locked_body_id == 201


def test_clear_object_working_memory_clears_locked_body() -> None:
    h = _StaticContactHarness()
    h._task_locked_body_id = 201
    h._clear_object_working_memory()
    assert h._task_locked_body_id is None


def test_humanoid_physics_sim_unwraps_nested_sim() -> None:
    """Contact registry lives on PyBullet _sim, not HumanoidEnv wrapper."""

    class _Sim:
        def __init__(self) -> None:
            self._static_body_registry = [
                {
                    "body_id": 301,
                    "kind": "cylinder",
                    "style": "planter",
                    "x": 1.0,
                    "y": 0.0,
                    "radius": 0.3,
                    "height": 1.0,
                }
            ]

        def find_static_contact_body(self, world_xy, *, kind=None, style=None, max_dist_m=1.2):
            return 301

        def _manip_has_contact(self, body_id: int) -> bool:
            return int(body_id) == 301

    class _Env:
        def __init__(self) -> None:
            self._sim = _Sim()

        def get_task_agent_pose(self) -> dict:
            return {"xy": (0.7, 0.0), "forward": (1.0, 0.0), "yaw": 0.0}

    class _Harness(SimulationGroundedLanguageMixin):
        def __init__(self) -> None:
            self.agent = SimpleNamespace(env=SimpleNamespace(base_env=_Env(), _contact_flag=False))
            self._obj_working_memory = None
            self._owm_cached = None

        def _task_ontology_best_key(self) -> str:
            return "cylinder"

        def _agent_xy_forward(self):
            return (0.7, 0.0), (1.0, 0.0)

    h = _Harness()
    phys = h._humanoid_physics_sim()
    assert phys is not None
    assert getattr(phys, "_static_body_registry", None) is not None
    assert h._contact_body_id_for_task(None) == 301
    h._task_locked_body_id = 301
    assert h._physics_range_to_locked_body() == pytest.approx(0.0, abs=0.02)
    assert h._manip_has_contact(None) is True


def test_forward_cylinder_matches_vision_range_to_central_planter() -> None:
    """Spawn facing +X must still lock the nearby large planter when depth agrees."""

    class _Harness(_StaticContactHarness):
        def _agent_xy_forward(self):
            # Facing +X; central planter is toward -X (behind).
            return (2.55, -1.5), (1.0, 0.0)

        def _task_ontology_best_key(self) -> str:
            return "cylinder"

    h = _Harness()
    h._base._static_body_registry = [
        {
            "body_id": 7,
            "kind": "cylinder",
            "style": "planter",
            "x": 0.0,
            "y": 0.0,
            "radius": 1.42,
            "height": 0.22,
        },
        {
            "body_id": 98,
            "kind": "cylinder",
            "style": "planter",
            "x": 7.2,
            "y": -2.5,
            "radius": 0.55,
            "height": 0.9,
        },
    ]
    # Depth ~1.85 matches central surface (hypot(2.55,1.5)-1.42 ≈ 1.86)
    bid = h._forward_cylinder_contact_body(vision_range=1.85, prefer_planter=True)
    assert bid == 7


def test_forward_cylinder_prefers_near_planter_over_far() -> None:
    class _Harness(_StaticContactHarness):
        def _agent_xy_forward(self):
            return (0.0, 0.0), (1.0, 0.0)

        def _task_ontology_best_key(self) -> str:
            return "cylinder"

    h = _Harness()
    h._base._static_body_registry = [
        {
            "body_id": 201,
            "kind": "cylinder",
            "style": "planter",
            "x": 2.0,
            "y": 0.0,
            "radius": 0.3,
            "height": 1.0,
        },
        {
            "body_id": 299,
            "kind": "cylinder",
            "style": "planter",
            "x": 7.0,
            "y": 0.0,
            "radius": 0.5,
            "height": 1.0,
        },
        {
            "body_id": 25,
            "kind": "cylinder",
            "style": "chrome",
            "x": 0.5,
            "y": 0.0,
            "radius": 0.015,
            "height": 0.4,
        },
    ]
    bid = h._forward_cylinder_contact_body(vision_range=1.7, prefer_planter=True)
    assert bid == 201
    h._lock_task_contact_body_on_bind()
    assert h._task_locked_body_id == 201
    # Optimistic vision must not beat physics for approach gates.
    blended = h._blend_dist_with_physics_range(0.3, 0.3, 50)
    assert blended == pytest.approx(1.7, abs=0.05)
    # Mild optimism still defers to physics when gap > 0.25m.
    blended2 = h._blend_dist_with_physics_range(1.4, 1.4, 80)
    assert blended2 == pytest.approx(1.7, abs=0.05)
    # Sticky planter lock across rebinds.
    h._lock_task_contact_body_on_bind()
    assert h._task_locked_body_id == 201


def test_fall_assist_near_goal_blocks_teleport() -> None:
    h = _StaticContactHarness()
    h._task_locked_body_id = 201
    h._task_fall_start_range = 3.5
    h._owm_bind_range_m = 3.5

    class _Near(_StaticContactHarness):
        def _agent_xy_forward(self):
            return (1.95, 0.0), (1.0, 0.0)

        def _task_fall_approach_range_m(self):
            return 0.4

    n = _Near()
    n._task_locked_body_id = 201
    n._task_fall_start_range = 0.45  # re-fall near goal must not erase progress
    n._owm_bind_range_m = 3.5
    assert n._task_fall_assist_near_goal() is True
    assert n._task_fall_assist_progress_blocks_reset() is True
    n._capture_task_fall_approach_baseline(0.4)
    # Baseline stays at bind/far reference, not the near re-fall distance.
    assert float(n._task_fall_start_range) >= 3.0
