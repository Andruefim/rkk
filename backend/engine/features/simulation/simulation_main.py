"""
simulation_singleton_v2.py — Singleton AGI с гуманоидом (Фаза 11/12).

Фаза 12 добавляет:
  - visual_mode toggle: enable_visual() / disable_visual()
  - EnvironmentVisual wrapper активируется без перезапуска агента
  - Predictive coding loop: GNN prediction → visual cortex feedback (опц. Neural ODE sub-steps, RKK_WM_NEURAL_ODE)
  - /vision/slots endpoint data через get_vision_state()
  - vision_stats в snapshot

Фаза 3:
  - refresh_phase3_teacher_llm(): LLM → правила IG + VL overlay (TTL)
  - tick_step: teacher_weight annealing (RKK_TEACHER_T_MAX), overlay на ValueLayer

Phase A (locomotion):
  - RKK_LOCOMOTION_CPG=1: CPG ноги на humanoid без fixed_root (engine.cpg_locomotion).
  - RKK_CPG_LOOP_HZ>0 (напр. 60): Low-level CPG в daemon-потоке; снимок graph.nodes после agent/skill step.
    По умолчанию 0 — CPG только сразу после agent.step (как раньше).

Phase B (hierarchy):
  - High ~1 Hz: GNN + EIG + goal_planning (agent tick / WS).
  - Mid ~10–20 Hz: RKK_AGENT_LOOP_HZ>0 — GNN/EIG/planning в daemon-потоке; tick_step() отдаёт кэш (UI не ждёт).
  - Low ~60 Hz+: CPG (decoupled) + RKK_PHYSICS_BG_HZ PyBullet (уже в humanoid).

  L3 goal_planning (agent), L2 skill_library, L1 CPG, L0 PyBullet.
  - RKK_SKILL_LIBRARY=1: моторная последовательность из библиотеки (тик = один кадр skill).

Phase C (full RSI):
  - RKK_RSI_FULL=1: engine.rsi_full — плато discovery → расширение GNN hidden; плато loco → CPG noise;
    плато walk skills → harder variants; падение phi → временное смягчение VL bounds.

Этап D (LLM в петле, RKK_LLM_LOOP=1):
  Уровень 1: каждый тик — GNN + System1 (как раньше).
  Уровень 2: фоновый Ollama по триггерам (стагнация discovery, block_rate, VLM unknown, surprise PE).
  Уровень 3: редко — перезапись гипотез (humanoid), в том же worker после L2 при run_level3.

Класс `Simulation` собран из миксинов в `features/simulation/mixin_*.py` (композиция поведения).
"""
from __future__ import annotations

import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from engine.agent import RKKAgent
from engine.demon import AdversarialDemon
from engine.visual_concept_store import VisualConceptStore
from engine.hierarchical_graph import HierarchicalGraph
from engine.rsi_structural import NeurogenesisEngine

from engine.config.runtime import RKKRuntimeConfig
from engine.core import (
    MotorState,
    WorldSwitcher,
    _make_env,
    default_bounds,
    resolve_torch_device,
)
from engine.features.simulation.background_loops import BackgroundLoopService
from engine.features.simulation.imports import *

from engine.features.simulation.mixin_api import SimulationApiMixin
from engine.features.simulation.mixin_concepts import SimulationConceptsMixin
from engine.features.simulation.mixin_demon_phase import SimulationDemonPhaseMixin
from engine.features.simulation.mixin_episodic_rssm import SimulationEpisodicRssmMixin
from engine.features.simulation.mixin_fall import SimulationFallMixin
from engine.features.simulation.mixin_llm import SimulationLlmLoopMixin
from engine.features.simulation.mixin_locomotion import SimulationLocomotionMixin
from engine.features.simulation.mixin_motor_pipeline import SimulationMotorPipelineMixin
from engine.features.simulation.mixin_phase_hierarchy import SimulationPhaseHierarchyMixin
from engine.features.simulation.mixin_pose_embodied import SimulationPoseEmbodiedMixin
from engine.features.simulation.mixin_skills import SimulationSkillsMixin
from engine.features.simulation.mixin_snapshot_shutdown import SimulationSnapshotShutdownMixin
from engine.features.simulation.mixin_teacher import SimulationTeacherMixin
from engine.features.simulation.mixin_tick import SimulationTickMixin
from engine.features.simulation.mixin_verbal import SimulationVerbalMixin
from engine.features.simulation.mixin_vision_predictor import SimulationVisionPredictorMixin
from engine.features.simulation.mixin_visual_grounding import SimulationVisualGroundingMixin
from engine.features.simulation.mixin_world import SimulationWorldMixin


class Simulation(
    SimulationConceptsMixin,
    SimulationVerbalMixin,
    SimulationPhaseHierarchyMixin,
    SimulationWorldMixin,
    SimulationFallMixin,
    SimulationLlmLoopMixin,
    SimulationLocomotionMixin,
    SimulationTeacherMixin,
    SimulationMotorPipelineMixin,
    SimulationSkillsMixin,
    SimulationPoseEmbodiedMixin,
    SimulationVisualGroundingMixin,
    SimulationEpisodicRssmMixin,
    SimulationTickMixin,
    SimulationVisionPredictorMixin,
    SimulationDemonPhaseMixin,
    SimulationApiMixin,
    SimulationSnapshotShutdownMixin,
):
    AGI_NAME = "Nova"
    AGI_COLOR = "#cc44ff"

    def __init__(self, device_str: str = "cuda", start_world: str = "humanoid"):
        self.device = resolve_torch_device(device_str)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.current_world = start_world
        print(f"[Singleton v2] Device: {self.device} | World: {start_world}")

        env = _make_env(start_world, self.device)
        bounds = default_bounds()

        self.agent = RKKAgent(
            agent_id=0,
            name=self.AGI_NAME,
            env=env,
            device=self.device,
            bounds=bounds,
        )

        self.switcher = WorldSwitcher(self.agent, self.device)
        self.demon = AdversarialDemon(n_agents=1, device=self.device)

        self.tick = 0
        self.phase = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events: deque[dict] = deque(maxlen=24)
        self._prev_edge_count = 0
        self._last_snapshot: dict = {}

        self._fall_count = 0
        self._fixed_root_active = False
        self._curriculum_auto_fr_released = False
        self._curriculum_stabilize_until: int = 0
        self._stand_ticks = 0
        self._last_fall_reset_tick: int = -999
        self._fall_recovery_active = False
        self._fall_recovery_start_tick = 0
        self._fall_recovery_last_progress_tick = 0
        self._fall_recovery_best_score = 0.0

        self._visual_mode = False
        self._visual_env = None
        self._base_env_ref = None
        self._vision_ticks = 0
        self._last_vision_state: dict = {}

        self._phase3_teacher_rules: list = []
        self._phase3_vl_overlay = None

        self._llm_loop_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rkk_llm")
        self._pending_llm_bundle: dict | None = None
        self._llm_level2_inflight = False
        self._last_level2_schedule_tick = -10**9
        self._last_level3_tick = -10**9
        self._best_discovery_rate = 0.0
        self._last_dr_gain_tick = 0
        self._rolling_block_bits: deque[int] = deque(maxlen=80)
        self._pe_history: deque[float] = deque(maxlen=200)
        self._locomotion_controller = None
        self._motor_state = MotorState()
        self._motor_state_lock = threading.Lock()
        self._bg = BackgroundLoopService(self)
        self._runtime_config = RKKRuntimeConfig.from_env()
        self._l1_motor_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l1_last_cmd_tick = 0
        self._l1_last_apply_tick = 0
        self._l1_last_credit_tick = 0
        self._sim_step_lock = threading.RLock()
        self._agent_step_response: dict | None = None
        self._skill_library = None
        self._skill_exec: dict | None = None
        self._llm_loop_stats: dict = {
            "level2_runs": 0,
            "level3_runs": 0,
            "last_triggers": [],
            "last_level2_explanation": "",
        }
        self._rsi_full = None
        self.neuro_engine = NeurogenesisEngine()
        self._embodied_reward_ctrl = None
        self._verbal_reward_total: float = 0.0
        self._visual_grounding_ctrl = (
            VisualGroundingController() if _VISUAL_GROUNDING_AVAILABLE else None
        )
        self._episodic_memory = EpisodicMemory() if _EPISODIC_MEMORY_AVAILABLE else None
        self._last_action_for_memory: tuple[str, float] | None = None
        self._last_fall_memory_tick: int = -999_999

        self._curriculum = CurriculumScheduler() if _CURRICULUM_AVAILABLE else None
        self._curriculum_apply_every: int = 50

        self._rssm_trainer: "RSSMTrainer | None" = None
        self._rssm_imagination: "RSSMImagination | None" = None
        self._rssm_upgraded: bool = False
        self._rssm_upgrade_tick: int = -1

        self._proprio: "ProprioceptionStream | None" = None
        if _PROPRIO_AVAILABLE:
            self._proprio = ProprioceptionStream(device=self.device)

        self._intrinsic: Any = None

        self._timescale: "MultiscaleTimeController | None" = None
        if _TIMESCALE_AVAILABLE:
            self._timescale = MultiscaleTimeController()

        self._inner_voice: "InnerVoiceController | None" = None
        self._llm_teacher: "LLMVoiceTeacher | None" = None
        if _INNER_VOICE_AVAILABLE:
            self._inner_voice = InnerVoiceController(device=self.device)
            self._llm_teacher = LLMVoiceTeacher()
            self._llm_teacher.add_callback(self._on_teacher_annotation)

        self._sleep_ctrl: "SleepController | None" = None
        self._physical_curriculum: "PhysicalCurriculum | None" = None
        self._persist: "PersistenceManager | None" = None
        if _PHASE_K_AVAILABLE:
            self._sleep_ctrl = SleepController()
            self._physical_curriculum = PhysicalCurriculum()
        self._meta_restored: bool = False
        self._was_fallen_last_tick: bool = False
        self._sleep_prev_fixed_root: bool = False

        self._uvicorn_loop: Any = None
        self._verbal: "VerbalActionController | None" = None
        self._chat_ws_clients: list[Any] = []
        self._verbal_tick_running: bool = False
        if _VERBAL_AVAILABLE:
            self._verbal = VerbalActionController()
            self._verbal.add_callback(self._broadcast_agent_message)

        self._slot_labeler: Any = None
        self._visual_voice: Any = None
        if _PHASE_M_AVAILABLE:
            _lang = os.environ.get("RKK_SPEECH_LANG", "ru")
            self._slot_labeler = SlotLabeler()
            self._visual_voice = VisualInnerVoice(lang=_lang)

        self._world_bridge: Any = None
        if _WORLD_BRIDGE_AVAILABLE and world_bridge_enabled():
            self._world_bridge = WorldStateBridge()

        self._motor_cortex: "_MotorCortexLibrary | None" = None
        self._mc_posture_window: deque = deque(maxlen=200)
        self._mc_fallen_count_window: deque = deque(maxlen=200)
        self._mc_abstract_nodes_injected: bool = False
        self._concepts_cache: list[dict] = []
        self._materialized_detector_concept_ids: set[str] = set()
        self._discovery_plateau_count = 0
        self._last_dr_snapshot: float | None = None
        self._hierarchical_graph: HierarchicalGraph | None = None
        self._concept_store: VisualConceptStore | None = None
        try:
            self._concept_inject_every = max(
                1, int(os.environ.get("RKK_CONCEPT_INJECT_EVERY", "30"))
            )
        except ValueError:
            self._concept_inject_every = 30
        self._l4_thread: threading.Thread | None = None
        self._l4_stop = threading.Event()
        self._l4_in_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_out_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_task_pending = False
        self._l4_last_snapshot: dict = {"n_concepts": 0, "concepts": []}
        self._l4_last_submit_tick = 0
        self._l4_last_apply_tick = 0
        self._l3_next_due_ts = 0.0
        self._l3_last_tick = 0
        self._memory_resume_enabled = os.environ.get(
            "RKK_MEMORY_RESUME_ON_START", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        if self._memory_resume_enabled:
            try:
                meta = self.memory_load()
                if meta.get("ok"):
                    print(f"[Simulation] Memory resumed at tick={self.tick}")
            except Exception as e:
                print(f"[Simulation] Memory resume skipped: {type(e).__name__}: {e}")

        if os.environ.get("RKK_NEURAL_LANG", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        ):
            try:
                from engine.neural_lang_integration import apply_neural_lang_patch

                apply_neural_lang_patch(self)
            except Exception as e:
                print(f"[Simulation] Neural lang patch skipped: {type(e).__name__}: {e}")

        from engine.intristic_objective import apply_intrinsic_patch
        from engine.llm_hint_mediator import apply_llm_mediator_patch
        from engine.learned_motor_primitives import apply_motor_primitives_patch

        apply_llm_mediator_patch(self)
        apply_motor_primitives_patch(self)
        apply_intrinsic_patch(self)

        # ── Variable Registry: dynamic ontology ──────────────────────────────
        try:
            from engine.variable_bootstrap import get_variable_registry
            self._variable_registry = get_variable_registry()
        except Exception as e:
            self._variable_registry = None
            print(f"[Simulation] VariableRegistry skipped: {type(e).__name__}: {e}")
