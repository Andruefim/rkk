"""
simulation_singleton_v2.py — Singleton AGI с гуманоидом (Фаза 11/12).

Фаза 12 добавляет:
  - visual_mode toggle: enable_visual() / disable_visual()
  - EnvironmentVisual wrapper активируется без перезапуска агента
  - Predictive coding loop: GNN prediction → visual cortex feedback (опц. Neural ODE sub-steps, RKK_WM_NEURAL_ODE)
  - /vision/slots endpoint data через get_vision_state()
  - vision_stats в snapshot

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

Класс `Simulation` собран из миксинов в `features/simulation/mixin_*.py` (композиция поведения).
"""
from __future__ import annotations

from engine.core.world import is_humanoid_topology

import os
import queue
import threading
from collections import deque
from typing import Any

import torch

from engine.agent import RKKAgent
from engine.demon import AdversarialDemon
from engine.visual_concept_store import VisualConceptStore
from engine.hierarchical_graph import HierarchicalGraph
from engine.behavioral_tracker import BehavioralTracker
from engine.neurogenesis_coordinator import NeurogenesisCoordinator
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
from engine.features.simulation.mixin_locomotion import SimulationLocomotionMixin
from engine.features.simulation.mixin_motor_pipeline import SimulationMotorPipelineMixin
from engine.features.simulation.mixin_phase_hierarchy import SimulationPhaseHierarchyMixin
from engine.features.simulation.mixin_grounded_language import SimulationGroundedLanguageMixin
from engine.features.simulation.mixin_neuro_symbolic import SimulationNeuroSymbolicMixin
from engine.features.simulation.mixin_phase5 import SimulationPhase5Mixin
from engine.features.simulation.mixin_phase6 import SimulationPhase6Mixin
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
    SimulationGroundedLanguageMixin,
    SimulationPhaseHierarchyMixin,
    SimulationNeuroSymbolicMixin,
    SimulationPhase5Mixin,
    SimulationPhase6Mixin,
    SimulationWorldMixin,
    SimulationFallMixin,
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
        self._wire_rkk_sim_ref(self.agent.env)

        self.switcher = WorldSwitcher(self.agent, self.device)
        if hasattr(self, "on_world_switch_phase6"):
            self.switcher._on_switch_hook = self.on_world_switch_phase6
        self.demon = AdversarialDemon(n_agents=1, device=self.device)

        self.tick = 0
        self._cached_scene: dict = {}
        self._cached_scene_tick = -1
        self.phase = 1
        self.max_phase = 1

        self._phase_hold_counter = 0
        self._candidate_phase = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events: deque[dict] = deque(maxlen=24)
        self._prev_edge_count = 0
        self._last_snapshot: dict = {}

        self._fall_count = 0
        self._prev_fallen: bool = False
        self._post_fr_last_release_tick: int = -1
        self._fixed_root_active = False
        self._curriculum_auto_fr_released = False
        self._curriculum_stabilize_until: int = 0
        self._fr_posture_streak: int = 0
        self._fr_support_bias_hist: deque[float] = deque(maxlen=48)
        self._fr_reattach_active: bool = False
        self._fr_reattach_until: int = 0
        self._fr_reattach_count: int = 0
        self._fr_fallen_ticks_accum: int = 0
        self._fr_release_blocked_until: int = 0
        self._fr_soft_release_deadline: int = 0
        self._fr_soft_release_start: int = 0
        self._fr_soft_release_initial_ratio: float = 1.0
        self._fr_soft_release_reason: str = ""
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

        self._best_discovery_rate = 0.0
        self._last_dr_gain_tick = 0
        self._rolling_block_bits: deque[int] = deque(maxlen=80)
        self._hai_prev_com_x: float | None = None
        self._hai_pe_fwd_ema: float = 0.0
        self._hai_pe_vert_ema: float = 0.0
        self._hai_pe_lat_ema: float = 0.0
        self._hai_pe_ema: float = 0.0  # mirror of _hai_pe_fwd_ema (compat)
        self._hai_last_diag: dict | None = None
        self._context_posterior = None
        self._context_posterior_d = 0
        self._pe_history: deque[float] = deque(maxlen=200)
        self._locomotion_controller = None
        self._reflex_stabilizer = None
        self._reflex_posture_prev = 0.5
        self._reflex_stabilizer_logged = False
        self._cerebellum = None
        self._cerebellum_obs_prev = None
        self._cerebellum_logged = False
        self._causal_motor_executor = None
        self._last_joint_cmd_applied: dict[str, float] = {}
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
        self._public_state_cache: dict | None = None
        self._public_state_cache_at: float = 0.0
        self._skill_library = None
        self._skill_exec: dict | None = None
        self._skill_chain: list[str] = []
        self.neuro_engine = NeurogenesisEngine()
        self.neuro_coordinator = NeurogenesisCoordinator(self.neuro_engine)
        from engine.latent_confounder import LatentConfounderManager

        self._latent_confounder = LatentConfounderManager()
        self._latent_confounder_last: dict = {}
        self.behavioral_tracker = BehavioralTracker()
        from engine.motor_arbiter import MotorArbiter
        from engine.locomotion_mastery import LocomotionEval
        from engine.interaction_eval import InteractionEval
        from engine.scene_graph import SceneGraphObserver

        self._motor_arbiter = MotorArbiter()
        self._locomotion_eval = LocomotionEval()
        self._interaction_eval = InteractionEval()
        self._scene_graph = SceneGraphObserver()
        self._s2_episodes_collected_total = 0
        self._edge_delta_hist: deque[int] = deque(maxlen=256)
        self._wm_warmup_until: int = 0
        self._neuro_pending: bool = False
        self._edge_growth_blocked: bool = False
        self._tick_log_prev_edges: int = 0
        self._prev_behavioral_score: float = 0.5
        self._embodied_reward_ctrl = None
        self._verbal_reward_total: float = 0.0
        self._visual_grounding_ctrl = (
            VisualGroundingController() if _VISUAL_GROUNDING_AVAILABLE else None
        )
        self._episodic_memory = EpisodicMemory() if _EPISODIC_MEMORY_AVAILABLE else None
        self._last_action_for_memory: tuple[str, float] | None = None
        self._last_fall_memory_tick: int = -999_999
        self._pending_fall_obs_for_memory: dict[str, float] | None = None

        self._system2: Any = None
        self._system2_last: dict | None = None

        self._proprio: "ProprioceptionStream | None" = None
        if _PROPRIO_AVAILABLE:
            self._proprio = ProprioceptionStream(device=self.device)

        self._intrinsic: Any = None

        self._timescale: "MultiscaleTimeController | None" = None
        if _TIMESCALE_AVAILABLE:
            self._timescale = MultiscaleTimeController()

        self._inner_voice: "InnerVoiceController | None" = None
        if _INNER_VOICE_AVAILABLE:
            self._inner_voice = InnerVoiceController(device=self.device)

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

        self._grounded_lang: Any = None
        self._grounded_lang_ready = False
        try:
            from engine.grounded_language import grounded_language_enabled

            if grounded_language_enabled():
                self._schedule_grounded_lang_bootstrap()
        except ImportError:
            pass

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
        from engine.learned_motor_primitives import apply_motor_primitives_patch

        apply_motor_primitives_patch(self)
        apply_intrinsic_patch(self)

        if is_humanoid_topology(self.current_world):
            try:
                from engine.genome.priors import bootstrap_innate_genome

                n_g = bootstrap_innate_genome(self.agent.graph, self.agent)
                print(
                    f"[Genome] Innate bootstrap (no LLM): {n_g} edges/priors on d={self.agent.graph._d}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[Genome] Innate bootstrap failed: {type(e).__name__}: {e}",
                    flush=True,
                )

        # ── Variable Registry: dynamic ontology ──────────────────────────────
        try:
            from engine.variable_bootstrap import get_variable_registry
            self._variable_registry = get_variable_registry()
        except Exception as e:
            self._variable_registry = None
            print(f"[Simulation] VariableRegistry skipped: {type(e).__name__}: {e}")

    def _wire_rkk_sim_ref(self, env: Any) -> None:
        """Back-ref so RKKAgent can reach Simulation.behavioral_tracker."""
        try:
            env._rkk_sim = self
        except Exception:
            pass
        base = getattr(env, "base_env", None)
        if base is not None:
            try:
                base._rkk_sim = self
            except Exception:
                pass
