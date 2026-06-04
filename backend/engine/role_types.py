"""
Track B Phase 1: semantic role_type tags on causal graph nodes (same topology transfer).
"""
from __future__ import annotations

import os
from typing import Iterable

from engine.features.humanoid.constants import (
    ARM_VARS,
    CUBE_VARS,
    FOOT_VARS,
    HEAD_VARS,
    INTERO_VARS,
    LEG_VARS,
    MOTOR_INTENT_VARS,
    MOTOR_OBSERVABLE_VARS,
    SANDBOX_VARS,
    SELF_VARS,
    SPINE_VARS,
    TORSO_VARS,
    VAR_NAMES,
    VESTIBULAR_VARS,
)

ROLE_MOTOR = "motor"
ROLE_POSTURE = "posture"
ROLE_CONTACT = "contact"
ROLE_PROPRIOCEPTIVE = "proprioceptive"
ROLE_INTENT = "intent"
ROLE_CONCEPT = "concept"

VALID_ROLE_TYPES = frozenset(
    {
        ROLE_MOTOR,
        ROLE_POSTURE,
        ROLE_CONTACT,
        ROLE_PROPRIOCEPTIVE,
        ROLE_INTENT,
        ROLE_CONCEPT,
    }
)

# Roles included in offline genome compression / cross-world W init (Track B2).
TRANSFER_ROLE_TYPES = frozenset(
    {
        ROLE_MOTOR,
        ROLE_POSTURE,
        ROLE_CONTACT,
        ROLE_PROPRIOCEPTIVE,
        ROLE_INTENT,
    }
)

_HUMANOID_ROLE_OVERRIDES: dict[str, str] = {}

for _v in MOTOR_INTENT_VARS:
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_INTENT
for _v in ("gait_phase_l", "gait_phase_r", "motor_drive_l", "motor_drive_r"):
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_MOTOR
for _v in ("foot_contact_l", "foot_contact_r", "lfoot_z", "rfoot_z"):
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_CONTACT
for _v in ("posture_stability", "support_bias"):
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_POSTURE
for _v in TORSO_VARS:
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_POSTURE
for _v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS + list(VESTIBULAR_VARS):
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_PROPRIOCEPTIVE
for _v in CUBE_VARS + SANDBOX_VARS:
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_PROPRIOCEPTIVE
for _v in INTERO_VARS:
    _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_POSTURE
for _v in SELF_VARS:
    if _v.startswith("self_intention"):
        _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_INTENT
    else:
        _HUMANOID_ROLE_OVERRIDES[_v] = ROLE_POSTURE


def role_type_enabled() -> bool:
    return os.environ.get("RKK_ROLE_TYPE_ENABLED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def role_type_strict() -> bool:
    return os.environ.get("RKK_ROLE_TYPE_STRICT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def infer_role_type(node_id: str, *, env_preset: str = "humanoid") -> str:
    """
    Map a graph node id to a semantic role_type.
    humanoid and humanoid_variant share the same variable_ids / role map.
    """
    nid = str(node_id)
    if nid.startswith("concept_"):
        return ROLE_CONCEPT
    if nid.startswith("phys_intent_"):
        return ROLE_INTENT
    preset = (env_preset or "humanoid").strip().lower()
    if preset in ("humanoid", "humanoid_variant", "pybullet"):
        if nid in _HUMANOID_ROLE_OVERRIDES:
            return _HUMANOID_ROLE_OVERRIDES[nid]
    if nid.startswith("intent_"):
        return ROLE_INTENT
    if nid.startswith("slot_") or nid.startswith("phys_"):
        return ROLE_PROPRIOCEPTIVE
    if "contact" in nid or nid.endswith("foot_z"):
        return ROLE_CONTACT
    if "posture" in nid or "support" in nid or nid.startswith("torso_") or nid.startswith("com_"):
        return ROLE_POSTURE
    if "motor" in nid or "gait" in nid:
        return ROLE_MOTOR
    return ROLE_PROPRIOCEPTIVE


def build_role_map(
    variable_ids: Iterable[str],
    *,
    env_preset: str = "humanoid",
    strict: bool | None = None,
) -> dict[str, str]:
    """Build {var_id: role_type} for all variables; strict raises on gaps."""
    use_strict = role_type_strict() if strict is None else bool(strict)
    out: dict[str, str] = {}
    for vid in variable_ids:
        role = infer_role_type(vid, env_preset=env_preset)
        if role not in VALID_ROLE_TYPES:
            if use_strict:
                raise ValueError(f"invalid role_type {role!r} for {vid!r}")
            role = ROLE_PROPRIOCEPTIVE
        out[str(vid)] = role
    if use_strict:
        missing = [v for v in variable_ids if str(v) not in out]
        if missing:
            raise ValueError(f"role map missing variables: {missing[:8]}")
    return out


def validate_role_map(variable_ids: Iterable[str], role_map: dict[str, str]) -> None:
    """Raise if any variable lacks a valid role (RKK_ROLE_TYPE_STRICT)."""
    if not role_type_strict():
        return
    for vid in variable_ids:
        role = role_map.get(str(vid))
        if role not in VALID_ROLE_TYPES:
            raise ValueError(f"variable {vid!r} missing role_type (got {role!r})")


def humanoid_variable_ids_for_roles() -> list[str]:
    """Canonical humanoid VAR_NAMES used in phase-1 gate tests."""
    return list(VAR_NAMES)
