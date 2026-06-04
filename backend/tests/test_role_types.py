"""Track B0: role_type mapping for humanoid topology."""
from __future__ import annotations

import pytest

from engine.features.humanoid.constants import VAR_NAMES
from engine.role_types import (
    ROLE_INTENT,
    build_role_map,
    humanoid_variable_ids_for_roles,
    infer_role_type,
    validate_role_map,
)


def test_humanoid_var_names_all_have_role_type():
    role_map = build_role_map(VAR_NAMES, env_preset="humanoid", strict=True)
    assert len(role_map) == len(VAR_NAMES)
    for vid in VAR_NAMES:
        assert role_map[vid]
    validate_role_map(VAR_NAMES, role_map)


def test_humanoid_variant_shares_role_map():
    ids = humanoid_variable_ids_for_roles()
    a = build_role_map(ids, env_preset="humanoid")
    b = build_role_map(ids, env_preset="humanoid_variant")
    assert a == b


def test_intent_and_concept_roles():
    assert infer_role_type("intent_stride") == ROLE_INTENT
    assert infer_role_type("concept_balance") == "concept"
