"""
Опциональные подсистемы: импорты с graceful fallback.
Используется `simulation.py` и `features/simulation/snapshot.py`.
"""
from __future__ import annotations

# Level 1-A: Embodied LLM Reward
try:
    from engine.embodied_llm_reward import (
        EmbodiedRewardController,
        PoseSnapshot,
        embodied_reward_enabled,
    )

    _EMBODIED_REWARD_AVAILABLE = True
except ImportError:
    _EMBODIED_REWARD_AVAILABLE = False
    print("[Simulation] embodied_llm_reward.py not found")

# Level 1-B: Visual Grounding
try:
    from engine.visual_grounding import (
        VisualGroundingController,
        get_pybullet_state_from_humanoid_env,
    )

    _VISUAL_GROUNDING_AVAILABLE = True
except ImportError:
    _VISUAL_GROUNDING_AVAILABLE = False
    print("[Simulation] visual_grounding.py not found")

# Level 2-D: Episodic Fall Memory
try:
    from engine.episodic_memory import EpisodicMemory, episode_memory_enabled

    _EPISODIC_MEMORY_AVAILABLE = True
except ImportError:
    _EPISODIC_MEMORY_AVAILABLE = False
    print("[Simulation] episodic_memory.py not found")

# Level 2-E: LLM Curriculum Generator
try:
    from engine.llm_curriculum import CurriculumScheduler, curriculum_enabled

    _CURRICULUM_AVAILABLE = True
except ImportError:
    _CURRICULUM_AVAILABLE = False
    print("[Simulation] llm_curriculum.py not found")

# Level 2-F: RSSM Temporal World Model
try:
    from engine.temporal_world_model import (
        RSSMImagination,
        RSSMTrainer,
        maybe_upgrade_graph_to_rssm,
        rssm_enabled,
    )

    _RSSM_AVAILABLE = True
except ImportError:
    _RSSM_AVAILABLE = False
    print("[Simulation] temporal_world_model.py not found")

# Level 3-G: Proprioception Stream
try:
    from engine.proprioception import ProprioceptionStream

    _PROPRIO_AVAILABLE = True
except ImportError:
    _PROPRIO_AVAILABLE = False
    print("[Simulation] proprioception.py not found")

# Level 3-H: Unified Reward Coordinator
try:
    from engine.reward_coordinator import RewardCoordinator

    _REWARD_COORD_AVAILABLE = True
except ImportError:
    _REWARD_COORD_AVAILABLE = False
    print("[Simulation] reward_coordinator.py not found")

# Level 3-I: Multi-scale Time
try:
    from engine.multiscale_time import (
        LEVEL_COGNIT,
        LEVEL_MOTOR,
        LEVEL_REFLECT,
        LEVEL_REFLEX,
        MultiscaleTimeController,
        timescale_enabled,
    )

    _TIMESCALE_AVAILABLE = True
except ImportError:
    _TIMESCALE_AVAILABLE = False
    print("[Simulation] multiscale_time.py not found")

# Phase J: Inner Voice (GRU) + LLM teacher (τ3 only)
try:
    from engine.inner_voice_net import InnerVoiceController
    from engine.llm_voice_teacher import LLMVoiceTeacher, TeacherAnnotation

    _INNER_VOICE_AVAILABLE = True
except ImportError:
    _INNER_VOICE_AVAILABLE = False
    TeacherAnnotation = None  # type: ignore
    print("[Simulation] inner_voice / llm_voice_teacher not found")

# Phase K: Sleep + Physical Curriculum + Persistence
try:
    from engine.persistent_state import (
        PersistenceManager,
        collect_meta_from_simulation,
        restore_meta_to_simulation,
    )
    from engine.physical_curriculum import PhysicalCurriculum
    from engine.sleep_consolidation import SleepController

    _PHASE_K_AVAILABLE = True
except ImportError:
    _PHASE_K_AVAILABLE = False
    print(
        "[Simulation] Phase K not found — copy sleep_consolidation.py, "
        "physical_curriculum.py, persistent_state.py"
    )

# Phase L: Verbal Action (chat / speech)
try:
    from engine.verbal_action import VerbalActionController, speech_enabled

    _VERBAL_AVAILABLE = True
except ImportError:
    _VERBAL_AVAILABLE = False
    print("[Simulation] verbal_action.py not found")

# Phase M: Visual Grounding (SlotLabeler + VisualInnerVoice → GNN + speech)
try:
    from engine.slot_labeler import SlotLabeler
    from engine.visual_inner_voice import VisualInnerVoice

    _PHASE_M_AVAILABLE = True
except ImportError:
    _PHASE_M_AVAILABLE = False
    print(
        "[Simulation] Phase M not found — copy visual_concepts.py, "
        "slot_labeler.py, visual_inner_voice.py"
    )

# Unified world ↔ semantic bridge
try:
    from engine.world_state_bridge import WorldStateBridge, world_bridge_enabled

    _WORLD_BRIDGE_AVAILABLE = True
except ImportError:
    _WORLD_BRIDGE_AVAILABLE = False
    world_bridge_enabled = lambda: False  # type: ignore[misc, assignment]

# Motor Cortex import (lazy — инициализируется при первом вызове)
try:
    from engine.motor_cortex import MotorCortexLibrary as _MotorCortexLibrary

    _MOTOR_CORTEX_AVAILABLE = True
except ImportError:
    _MOTOR_CORTEX_AVAILABLE = False
    print("[Simulation] motor_cortex.py not found — motor cortex disabled")

# `import *` подхватывает только имена из `__all__`; флаги `_FOO_AVAILABLE` нужны в `Simulation.__init__`.
__all__ = [n for n in list(globals()) if not n.startswith("__")]
