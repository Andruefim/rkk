"""Text priors (golden edges) для humanoid и merge с LLM."""
from __future__ import annotations


def humanoid_hardcoded_seeds() -> list[dict]:
    """Биомеханические text priors для полного режима (суставы → COM, стопы)."""
    return [
        {"from_": "intent_stride", "to": "intent_gait_coupling", "weight": 0.24, "alpha": 0.05},
        {"from_": "gait_phase_l", "to": "lhip", "weight": 0.40, "alpha": 0.05},
        {"from_": "gait_phase_l", "to": "lknee", "weight": -0.45, "alpha": 0.05},
        {"from_": "gait_phase_r", "to": "rhip", "weight": 0.40, "alpha": 0.05},
        {"from_": "gait_phase_r", "to": "rknee", "weight": -0.45, "alpha": 0.05},
        {"from_": "posture_stability", "to": "intent_gait_coupling", "weight": 0.20, "alpha": 0.05},
        {"from_": "intent_stop_recover", "to": "intent_gait_coupling", "weight": -0.20, "alpha": 0.05},
        {"from_": "intent_stride", "to": "intent_torso_forward", "weight": 0.24, "alpha": 0.05},
        {"from_": "intent_stride", "to": "spine_pitch", "weight": 0.20, "alpha": 0.05},
        {"from_": "intent_stride", "to": "com_x", "weight": 0.18, "alpha": 0.05},
        {"from_": "intent_torso_forward", "to": "com_x", "weight": 0.16, "alpha": 0.05},
        {"from_": "lhip", "to": "com_x", "weight": 0.22, "alpha": 0.05},
        {"from_": "rhip", "to": "com_x", "weight": 0.22, "alpha": 0.05},
        {"from_": "lknee", "to": "lfoot_z", "weight": 0.25, "alpha": 0.05},
        {"from_": "rknee", "to": "rfoot_z", "weight": 0.25, "alpha": 0.05},
        {"from_": "lhip", "to": "com_z", "weight": 0.18, "alpha": 0.05},
        {"from_": "rhip", "to": "com_z", "weight": 0.18, "alpha": 0.05},
        {"from_": "lshoulder", "to": "cube0_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "rshoulder", "to": "cube1_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "com_z", "to": "torso_roll", "weight": -0.15, "alpha": 0.05},
        {"from_": "self_intention_larm", "to": "lshoulder", "weight": 0.18, "alpha": 0.05},
        {"from_": "self_intention_rarm", "to": "rshoulder", "weight": 0.18, "alpha": 0.05},
        {"from_": "self_attention", "to": "neck_yaw", "weight": 0.12, "alpha": 0.04},
        {"from_": "self_energy", "to": "lshoulder", "weight": 0.10, "alpha": 0.04},
    ]


def fixed_root_seeds() -> list[dict]:
    """
    Text priors для fixed_root mode.
    Фокус: arms → cubes (3 куба × 3 оси), цепочка плечо→локоть→куб,
    позвоночник → cube2.

    Веса слабые (0.18–0.28) — как обычно для seeds.
    """
    return [
        {"from_": "lshoulder", "to": "cube0_x", "weight": 0.24, "alpha": 0.05},
        {"from_": "lshoulder", "to": "cube0_y", "weight": 0.18, "alpha": 0.05},
        {"from_": "lelbow", "to": "cube0_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "lelbow", "to": "cube0_z", "weight": 0.18, "alpha": 0.05},
        {"from_": "rshoulder", "to": "cube1_x", "weight": 0.24, "alpha": 0.05},
        {"from_": "rshoulder", "to": "cube1_y", "weight": 0.18, "alpha": 0.05},
        {"from_": "relbow", "to": "cube1_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "relbow", "to": "cube1_z", "weight": 0.18, "alpha": 0.05},
        {"from_": "lshoulder", "to": "lelbow", "weight": 0.28, "alpha": 0.05},
        {"from_": "rshoulder", "to": "relbow", "weight": 0.28, "alpha": 0.05},
        {"from_": "spine_yaw", "to": "cube2_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "spine_pitch", "to": "cube2_z", "weight": 0.16, "alpha": 0.05},
        {"from_": "neck_yaw", "to": "cube0_x", "weight": 0.12, "alpha": 0.04},
        {"from_": "self_intention_larm", "to": "lshoulder", "weight": 0.22, "alpha": 0.05},
        {"from_": "self_intention_larm", "to": "cube0_x", "weight": 0.16, "alpha": 0.04},
        {"from_": "self_intention_rarm", "to": "rshoulder", "weight": 0.22, "alpha": 0.05},
        {"from_": "self_attention", "to": "neck_yaw", "weight": 0.14, "alpha": 0.04},
        {"from_": "self_energy", "to": "lshoulder", "weight": 0.12, "alpha": 0.04},
    ]


def merge_humanoid_golden_with_llm_edges(llm_edges: list[dict]) -> list[dict]:
    golden = humanoid_hardcoded_seeds()
    seen = {(e["from_"], e["to"]) for e in golden}
    out = list(golden)
    for e in llm_edges:
        from_ = e.get("from_") or e.get("from")
        to = e.get("to")
        if not from_ or not to:
            continue
        from_, to = str(from_).strip(), str(to).strip()
        key = (from_, to)
        if key in seen:
            continue
        seen.add(key)
        row = {
            "from_": from_,
            "to": to,
            "weight": float(e.get("weight", 0.25)),
            "alpha": float(e.get("alpha", 0.05)),
        }
        out.append(row)
    return out
