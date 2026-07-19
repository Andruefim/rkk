"""Shared stable-posture gate for initiating locomotion macros and walk skills."""
from __future__ import annotations

import os
from typing import Mapping


def stable_locomote_ps_threshold() -> float:
    try:
        return float(os.environ.get("RKK_STABLE_LOCOMOTE_PS", "0.90"))
    except ValueError:
        return 0.90


def stable_locomote_contact_min() -> float:
    try:
        return float(os.environ.get("RKK_STABLE_LOCOMOTE_CONTACT", "0.52"))
    except ValueError:
        return 0.52


def stable_locomote_com_z_min() -> float:
    try:
        return float(os.environ.get("RKK_STABLE_LOCOMOTE_COM_Z", "0.20"))
    except ValueError:
        return 0.20


def stable_locomote_enabled() -> bool:
    return os.environ.get("RKK_STABLE_LOCOMOTE_GATE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def stable_locomote_ready(
    obs: Mapping[str, float],
    *,
    fallen: bool = False,
) -> bool:
    """True when upright humanoid should command forward locomotion (not stance-hold)."""
    if fallen or not stable_locomote_enabled():
        return False
    ps = float(
        obs.get("posture_stability", obs.get("phys_posture_stability", 0.5))
    )
    cz = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
    fl = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
    fr = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
    grounded = max(fl, fr)
    if ps < stable_locomote_ps_threshold():
        return False
    if cz < stable_locomote_com_z_min():
        return False
    if grounded < stable_locomote_contact_min():
        return False
    return True
