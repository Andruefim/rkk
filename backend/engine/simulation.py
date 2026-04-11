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
"""
from __future__ import annotations

import copy
import os
import queue
import threading
import time
from dataclasses import dataclass, field
import torch
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from engine.agent       import RKKAgent
from engine.demon       import AdversarialDemon
from engine.environment import Environment
from engine.concept_store import ConceptStore
from engine.hierarchical_graph import HierarchicalGraph, hierarchical_graph_enabled
from engine.ollama_env  import get_ollama_generate_url, get_ollama_model
from engine.value_layer import HomeostaticBounds
from engine.wm_neural_ode import integrate_world_model_step
from engine.rsi_structural import NeurogenesisEngine

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
        maybe_upgrade_graph_to_rssm,
        RSSMTrainer,
        RSSMImagination,
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
        LEVEL_REFLEX,
        MultiscaleTimeController,
        timescale_enabled,
    )

    _TIMESCALE_AVAILABLE = True
except ImportError:
    _TIMESCALE_AVAILABLE = False
    print("[Simulation] multiscale_time.py not found")

# Motor Cortex import (lazy — инициализируется при первом вызове)
try:
    from engine.motor_cortex import MotorCortexLibrary as _MotorCortexLibrary
    _MOTOR_CORTEX_AVAILABLE = True
except ImportError:
    _MOTOR_CORTEX_AVAILABLE = False
    print("[Simulation] motor_cortex.py not found — motor cortex disabled")

PHASE_THRESHOLDS = [0.0, 0.15, 0.30, 0.50, 0.70, 0.88]
PHASE_HOLD_TICKS = 12
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]

# Visual mode: полный GNN→cortex на каждом тике дорог; предсказание для PC обновляем реже
VISION_GNN_FEED_EVERY = 2


def _cpg_loop_hz_from_env() -> float:
    """0 = CPG синхронно с тиком агента; >0 = отдельный поток (часто 60)."""
    try:
        hz = float(os.environ.get("RKK_CPG_LOOP_HZ", "0"))
    except ValueError:
        hz = 0.0
    return max(0.0, min(hz, 240.0))


def _agent_loop_hz_from_env() -> float:
    """0 = полный tick_step в вызывающем потоке (WS); >0 = high-level в daemon (часто 10–20)."""
    try:
        hz = float(os.environ.get("RKK_AGENT_LOOP_HZ", "0"))
    except ValueError:
        hz = 0.0
    return max(0.0, min(hz, 60.0))


def _l3_loop_hz_from_env() -> float:
    """
    Cadence для L3 planning/imagination.
    0 = L3 выполняется на каждом тике agent.step (legacy).
    >0 = L3 выполняется реже по wall-clock.
    """
    try:
        hz = float(os.environ.get("RKK_L3_LOOP_HZ", "0"))
    except ValueError:
        hz = 0.0
    return max(0.0, min(hz, 30.0))


def _l4_worker_enabled() -> bool:
    v = os.environ.get("RKK_L4_WORKER", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass
class MotorCommandLog:
    tick: int
    source: str
    intents: dict[str, float] = field(default_factory=dict)
    joint_targets: dict[str, float] = field(default_factory=dict)
    support_bias: float = 0.5
    gait_phase_l: float = 0.5
    gait_phase_r: float = 0.5
    foot_contact_l: float = 0.5
    foot_contact_r: float = 0.5
    posture_stability: float = 0.5


@dataclass
class MotorState:
    tick: int = 0
    source: str = "init"
    intents: dict[str, float] = field(default_factory=lambda: {
        "intent_stride": 0.5,
        "intent_support_left": 0.5,
        "intent_support_right": 0.5,
        "intent_torso_forward": 0.5,
        "intent_gait_coupling": 0.88,
        "intent_arm_counterbalance": 0.5,
        "intent_stop_recover": 0.5,
    })
    joint_targets: dict[str, float] = field(default_factory=dict)
    gait_phase_l: float = 0.5
    gait_phase_r: float = 0.5
    foot_contact_l: float = 0.5
    foot_contact_r: float = 0.5
    support_bias: float = 0.5
    motor_drive_l: float = 0.5
    motor_drive_r: float = 0.5
    posture_stability: float = 0.5
    support_leg: str = "balanced"
    history: list[MotorCommandLog] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "source": self.source,
            "intents": dict(self.intents),
            "joint_targets": dict(self.joint_targets),
            "gait_phase_l": float(self.gait_phase_l),
            "gait_phase_r": float(self.gait_phase_r),
            "foot_contact_l": float(self.foot_contact_l),
            "foot_contact_r": float(self.foot_contact_r),
            "support_bias": float(self.support_bias),
            "motor_drive_l": float(self.motor_drive_l),
            "motor_drive_r": float(self.motor_drive_r),
            "posture_stability": float(self.posture_stability),
            "support_leg": self.support_leg,
            "history_len": len(self.history),
        }

    def update_from_observation(self, obs: dict[str, float], *, tick: int | None = None, source: str | None = None) -> None:
        if tick is not None:
            self.tick = int(tick)
        if source is not None:
            self.source = str(source)
        self.gait_phase_l = float(obs.get("gait_phase_l", self.gait_phase_l))
        self.gait_phase_r = float(obs.get("gait_phase_r", self.gait_phase_r))
        self.foot_contact_l = float(obs.get("foot_contact_l", self.foot_contact_l))
        self.foot_contact_r = float(obs.get("foot_contact_r", self.foot_contact_r))
        self.support_bias = float(obs.get("support_bias", self.support_bias))
        self.motor_drive_l = float(obs.get("motor_drive_l", self.motor_drive_l))
        self.motor_drive_r = float(obs.get("motor_drive_r", self.motor_drive_r))
        self.posture_stability = float(obs.get("posture_stability", self.posture_stability))
        if self.foot_contact_l > self.foot_contact_r + 0.08:
            self.support_leg = "left"
        elif self.foot_contact_r > self.foot_contact_l + 0.08:
            self.support_leg = "right"
        else:
            self.support_leg = "balanced"

    def update_from_command(
        self,
        *,
        tick: int,
        source: str,
        intents: dict[str, float] | None = None,
        joint_targets: dict[str, float] | None = None,
        obs: dict[str, float] | None = None,
    ) -> MotorCommandLog:
        if intents:
            self.intents.update({k: float(v) for k, v in intents.items()})
        if joint_targets is not None:
            self.joint_targets = {k: float(v) for k, v in joint_targets.items()}
        self.tick = int(tick)
        self.source = str(source)
        if obs:
            self.update_from_observation(obs, tick=tick, source=source)
        log = MotorCommandLog(
            tick=int(tick),
            source=str(source),
            intents=dict(self.intents),
            joint_targets=dict(self.joint_targets),
            support_bias=float(self.support_bias),
            gait_phase_l=float(self.gait_phase_l),
            gait_phase_r=float(self.gait_phase_r),
            foot_contact_l=float(self.foot_contact_l),
            foot_contact_r=float(self.foot_contact_r),
            posture_stability=float(self.posture_stability),
        )
        self.history.append(log)
        if len(self.history) > 160:
            self.history = self.history[-160:]
        return log


def resolve_torch_device(requested: str | None = None) -> torch.device:
    """
    Выбор устройства для GNN, демона, temporal и CausalVisualCortex.
    Переменная окружения RKK_DEVICE перекрывает аргумент (например cuda, cuda:0, mps, cpu).
    """
    req = (os.environ.get("RKK_DEVICE") or requested or "cuda").strip().lower()
    if req in ("mps", "mps:0"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[RKK] RKK_DEVICE=mps, но MPS недоступен → CPU")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            dev = torch.device(req)
            return dev
        print(
            f"[RKK] Запрошено {req}, но torch.cuda.is_available()=False "
            "(поставьте PyTorch с CUDA или задайте RKK_DEVICE=cpu) → CPU"
        )
        return torch.device("cpu")
    try:
        return torch.device(req)
    except Exception:
        print(f"[RKK] Неизвестное RKK_DEVICE={req!r} → CPU")
        return torch.device("cpu")

WORLDS = {
    "humanoid":  {"label": "Humanoid",          "color": "#cc44ff"},
    "robot":     {"label": "Robot Arm",          "color": "#aa22dd"},
    "pybullet":  {"label": "3D Physics",         "color": "#ff44aa"},
    "physics":   {"label": "Thermodynamics",     "color": "#00ff99"},
    "chemistry": {"label": "Chemical Kinetics",  "color": "#0099ff"},
    "logic":     {"label": "Logic Gates",        "color": "#ff9900"},
}


def _make_env(world: str, device: torch.device):
    if world == "humanoid":
        from engine.environment_humanoid import EnvironmentHumanoid
        return EnvironmentHumanoid(device=device)
    if world == "robot":
        from engine.environment_robot import EnvironmentRobot
        return EnvironmentRobot(device=device)
    if world == "pybullet":
        from engine.environment_pybullet import EnvironmentPyBullet
        return EnvironmentPyBullet(n_objects=3, device=device, use_pybullet=True)
    return Environment(world, device)


def _default_bounds() -> HomeostaticBounds:
    return HomeostaticBounds(
        var_min=0.05, var_max=0.95,
        phi_min=0.01,
        h_slow_max=14.0,
        env_entropy_max_delta=0.96,
        warmup_ticks=3000,
        blend_ticks=800,
        phi_min_steady=0.03,
        env_entropy_max_delta_steady=0.85,
        h_slow_max_steady=12.0,
        predict_band_edge_steady=0.015,
    )


class WorldSwitcher:
    def __init__(self, agent: RKKAgent, device: torch.device):
        self.agent   = agent
        self.device  = device
        self.history: list[dict] = []

    def switch(self, new_world: str) -> dict:
        old = self.agent.env.preset
        if old == new_world:
            return {"switched": False, "world": new_world}

        new_env = _make_env(new_world, self.device)

        old_nodes = set(self.agent.graph.nodes.keys())
        init_obs  = new_env.observe()
        new_vars  = new_env.variable_ids
        new_nodes = [v for v in new_vars if v not in old_nodes]

        for var_id in new_vars:
            self.agent.graph.set_node(var_id, init_obs.get(var_id, 0.5))

        self.agent.env = new_env

        from engine.temporal import TemporalBlankets
        new_d = len(new_vars)
        if self.agent.temporal.d_input != new_d:
            self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)

        self.agent.temporal.step(init_obs)
        self.agent.graph.record_observation(init_obs)

        rec = {
            "from_world":  old,
            "to_world":    new_world,
            "new_nodes":   new_nodes,
            "total_nodes": len(self.agent.graph.nodes),
            "gnn_d":       self.agent.graph._d,
        }
        self.history.append(rec)
        print(f"[WorldSwitch] {old} → {new_world} | +{len(new_nodes)} nodes | d={self.agent.graph._d}")
        return {"switched": True, **rec}


# ─── Simulation ───────────────────────────────────────────────────────────────
class Simulation:
    AGI_NAME  = "Nova"
    AGI_COLOR = "#cc44ff"

    def __init__(self, device_str: str = "cuda", start_world: str = "humanoid"):
        self.device = resolve_torch_device(device_str)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.current_world = start_world
        print(f"[Singleton v2] Device: {self.device} | World: {start_world}")

        env    = _make_env(start_world, self.device)
        bounds = _default_bounds()

        self.agent = RKKAgent(
            agent_id=0, name=self.AGI_NAME,
            env=env, device=self.device, bounds=bounds,
        )

        self.switcher    = WorldSwitcher(self.agent, self.device)
        self.demon       = AdversarialDemon(n_agents=1, device=self.device)

        self.tick        = 0
        self.phase       = 1
        self.max_phase   = 1

        self._phase_hold_counter = 0
        self._candidate_phase    = 1
        self._dr_window: deque[float] = deque(maxlen=20)
        self.events:    deque[dict]   = deque(maxlen=24)
        self._prev_edge_count = 0
        self._last_snapshot: dict = {}

        self._fall_count  = 0
        self._fixed_root_active = False
        self._curriculum_auto_fr_released = False
        self._curriculum_stabilize_until: int = 0
        self._stand_ticks = 0
        self._last_fall_reset_tick: int = -999
        self._fall_recovery_active = False
        self._fall_recovery_start_tick = 0
        self._fall_recovery_last_progress_tick = 0
        self._fall_recovery_best_score = 0.0

        # ── Фаза 12: Visual Cortex ────────────────────────────────────────────
        self._visual_mode   = False     # выкл по умолчанию
        self._visual_env    = None      # EnvironmentVisual instance
        self._base_env_ref  = None      # оригинальный env (до visual wrap)
        self._vision_ticks  = 0         # тиков с включённым зрением
        self._last_vision_state: dict = {}

        # Фаза 3: виртуальный учитель (LLM → правила + VL overlay)
        self._phase3_teacher_rules: list = []
        self._phase3_vl_overlay = None

        # Этап D: LLM консультации в петле (фоновый sync HTTP, не блокирует тик надолго)
        self._llm_loop_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rkk_llm")
        self._pending_llm_bundle: dict | None = None
        self._llm_level2_inflight = False
        self._last_level2_schedule_tick = -10**9
        self._last_level3_tick = -10**9
        self._best_discovery_rate = 0.0
        self._last_dr_gain_tick = 0
        self._rolling_block_bits: deque[int] = deque(maxlen=80)
        self._pe_history: deque[float] = deque(maxlen=200)
        # Phase A: CPG locomotion (humanoid, не fixed_root)
        self._locomotion_controller = None
        self._motor_state = MotorState()
        self._motor_state_lock = threading.Lock()
        self._cpg_loop_thread: threading.Thread | None = None
        self._cpg_stop = threading.Event()
        self._cpg_snapshot_lock = threading.Lock()
        self._cpg_node_snapshot: dict[str, float] = {}
        self._l1_motor_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l1_last_cmd_tick = 0
        self._l1_last_apply_tick = 0
        self._l1_last_credit_tick = 0
        # High-level agent tick (GNN/EIG/do) vs HTTP/WS: одна критическая секция на симуляцию.
        self._sim_step_lock = threading.RLock()
        self._agent_loop_thread: threading.Thread | None = None
        self._agent_stop = threading.Event()
        self._agent_step_response: dict | None = None
        # Phase B: skill library (L2), один кадр последовательности за тик
        self._skill_library = None
        self._skill_exec: dict | None = None
        self._llm_loop_stats: dict = {
            "level2_runs": 0,
            "level3_runs": 0,
            "last_triggers": [],
            "last_level2_explanation": "",
        }
        # Phase C: full RSI (GNN NAS-lite, CPG perturb, skill curriculum, VL relax)
        self._rsi_full = None
        self.neuro_engine = NeurogenesisEngine()
        # Level 1-A: Embodied LLM Reward
        self._embodied_reward_ctrl = (
            EmbodiedRewardController() if _EMBODIED_REWARD_AVAILABLE else None
        )
        # Level 1-B: Visual Grounding
        self._visual_grounding_ctrl = (
            VisualGroundingController() if _VISUAL_GROUNDING_AVAILABLE else None
        )
        # Level 2-D: Episodic Fall Memory
        self._episodic_memory = (
            EpisodicMemory() if _EPISODIC_MEMORY_AVAILABLE else None
        )
        self._last_action_for_memory: tuple[str, float] | None = None
        self._last_fall_memory_tick: int = -999_999

        # Level 2-E: LLM Curriculum
        self._curriculum = (
            CurriculumScheduler() if _CURRICULUM_AVAILABLE else None
        )
        self._curriculum_apply_every: int = 50

        # Level 2-F: RSSM
        self._rssm_trainer: "RSSMTrainer | None" = None
        self._rssm_imagination: "RSSMImagination | None" = None
        self._rssm_upgraded: bool = False
        self._rssm_upgrade_tick: int = -1

        # Level 3-G: Proprioception Stream
        self._proprio: "ProprioceptionStream | None" = None
        if _PROPRIO_AVAILABLE:
            self._proprio = ProprioceptionStream(device=self.device)

        # Level 3-H: Unified Reward Coordinator (lazy-init after graph d known)
        self._reward_coord: "RewardCoordinator | None" = None
        self._reward_X_prev: list[float] = []
        self._reward_a_prev: list[float] = []
        self._reward_action_var: str = ""
        self._reward_action_val: float = 0.5
        self._was_blocked: bool = False

        # Level 3-I: Multi-scale Time
        self._timescale: "MultiscaleTimeController | None" = None
        if _TIMESCALE_AVAILABLE:
            self._timescale = MultiscaleTimeController()

        # Phase D: Motor Cortex (learned motor programs + CPG annealing)
        self._motor_cortex: "_MotorCortexLibrary | None" = None
        self._mc_posture_window: deque = deque(maxlen=200)
        self._mc_fallen_count_window: deque = deque(maxlen=200)
        self._mc_abstract_nodes_injected: bool = False
        # Фаза 1: зачатки понятий (кэш детектора), автосохранение памяти
        self._concepts_cache: list[dict] = []
        self._materialized_detector_concept_ids: set[str] = set()
        self._discovery_plateau_count = 0
        self._last_dr_snapshot: float | None = None
        # Фаза 2 (часть 1): L1 агрегаты → виртуальные узлы в основном GNN
        self._hierarchical_graph: HierarchicalGraph | None = None
        # Фаза 2 (часть 3): концепты из SlotAttention → узлы concept_* в GNN (visual humanoid)
        self._concept_store: ConceptStore | None = None
        try:
            self._concept_inject_every = max(
                1, int(os.environ.get("RKK_CONCEPT_INJECT_EVERY", "30"))
            )
        except ValueError:
            self._concept_inject_every = 30
        # L4 worker (концепты): single-writer graph apply в основном тике.
        self._l4_thread: threading.Thread | None = None
        self._l4_stop = threading.Event()
        self._l4_in_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_out_q: queue.SimpleQueue = queue.SimpleQueue()
        self._l4_task_pending = False
        self._l4_last_snapshot: dict = {"n_concepts": 0, "concepts": []}
        self._l4_last_submit_tick = 0
        self._l4_last_apply_tick = 0
        # L3 cadence без отдельного writer-потока (минимум гонок).
        self._l3_next_due_ts = 0.0
        self._l3_last_tick = 0
        # Фаза 1: автозагрузка .rkk на старте (если есть файл).
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

    # ── Фаза 1: память и концепты ─────────────────────────────────────────────
    def _annotate_concepts_with_graph_nodes(self) -> None:
        for c in self._concepts_cache:
            did = str(c.get("id", ""))
            gn = None
            for nid, meta in self.agent.graph._concept_meta.items():
                if str(meta.get("detector_id", "")) == did:
                    gn = nid
                    break
            c["graph_node"] = gn

    def _maybe_materialize_concept_macros(self, concepts: list[dict]) -> None:
        try:
            max_n = int(os.environ.get("RKK_CONCEPT_MACRO_MAX", "3"))
        except ValueError:
            max_n = 3
        if max_n <= 0:
            return
        try:
            dplat = int(os.environ.get("RKK_CONCEPT_DISCOVERY_PLATEAU_TICKS", "0"))
        except ValueError:
            dplat = 0
        if dplat > 0 and self._discovery_plateau_count < dplat:
            return
        try:
            amin_edge = float(os.environ.get("RKK_CONCEPT_EDGE_ALPHA_MIN", "0"))
        except ValueError:
            amin_edge = 0.0
        n_macro = sum(1 for x in self.agent.graph._node_ids if str(x).startswith("concept_"))
        g = self.agent.graph
        for c in concepts:
            if n_macro >= max_n:
                break
            did = str(c.get("id", ""))
            if not did or did in self._materialized_detector_concept_ids:
                continue
            members = list(c.get("pattern_nodes_example") or [])
            if not members:
                continue
            if amin_edge > 0.0 and g._core is not None:
                if g.path_min_alpha_trust_on_path(members) < amin_edge:
                    continue
            if did.startswith("c") and did[1:].isdigit():
                macro = f"concept_{int(did[1:])}"
            else:
                macro = f"concept_{n_macro}"
            if macro in g.nodes:
                self._materialized_detector_concept_ids.add(did)
                continue
            ok = g.materialize_concept_macro(
                macro,
                members,
                detector_id=did,
                pattern=list(c.get("pattern") or []),
            )
            if ok:
                self._materialized_detector_concept_ids.add(did)
                n_macro += 1
                self._add_event(
                    f"🧩 Macro-node {macro} ← {len(members)} vars (detector {did})",
                    "#88aaff",
                    "phase",
                )

    def _maybe_refresh_concepts_cache(self) -> None:
        try:
            every = int(os.environ.get("RKK_CONCEPT_EVERY", "24"))
        except ValueError:
            every = 24
        if every <= 0 or self.tick % every != 0:
            return
        from engine.concept_detector import detect_proto_concepts

        try:
            pt = int(os.environ.get("RKK_CONCEPT_PLATEAU_TICKS", "0"))
        except ValueError:
            pt = 0
        self._concepts_cache = detect_proto_concepts(
            self.agent.graph,
            agent_plateau_counter=self.agent._rsi_plateau_count,
            plateau_ticks_required=pt,
        )
        self._maybe_materialize_concept_macros(self._concepts_cache)
        self._annotate_concepts_with_graph_nodes()

    def _maybe_autosave_memory(self) -> None:
        from engine.persistence import autosave_every_ticks, default_memory_path, save_simulation

        n = autosave_every_ticks()
        if n <= 0 or self.tick <= 0 or self.tick % n != 0:
            return
        try:
            save_simulation(self, default_memory_path())
        except Exception as e:
            print(f"[RKK] memory autosave: {e}")

    def memory_save(self, path: str | None = None) -> dict:
        from pathlib import Path

        from engine.persistence import save_simulation

        with self._sim_step_lock:
            return save_simulation(self, Path(path) if path else None)

    def memory_load(self, path: str | None = None) -> dict:
        from pathlib import Path

        from engine.persistence import default_memory_path, load_simulation

        p = Path(path) if path else default_memory_path()
        target_world = None
        if p.is_file():
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(payload, dict):
                    cw = payload.get("current_world")
                    if isinstance(cw, str) and cw in WORLDS:
                        target_world = cw
            except Exception:
                target_world = None

        if target_world and target_world != self.current_world:
            sw = self.switch_world(target_world)
            if sw.get("error"):
                return {
                    "ok": False,
                    "error": f"failed to switch world to {target_world!r} before load: {sw.get('error')}",
                }

        with self._sim_step_lock:
            # Worker-safe load: останавливаем L4 воркер и чистим его очереди,
            # чтобы не применить устаревшие концепты после миграции графа.
            self._stop_l4_worker()
            self._l4_last_snapshot = {"n_concepts": 0, "concepts": []}
            self._l4_last_submit_tick = 0
            self._l4_last_apply_tick = 0
            self._l3_next_due_ts = 0.0
            self._motor_state = MotorState()
            self._clear_fall_recovery()
            self._drain_simple_queue(self._l1_motor_q)
            self._l1_last_cmd_tick = 0
            self._l1_last_apply_tick = 0
            out = load_simulation(self, p)
            if out.get("ok"):
                self._annotate_concepts_with_graph_nodes()
                self._ensure_phase2()
                try:
                    auto_fr = int(os.environ.get("RKK_AUTO_FIXED_ROOT_TICKS", "0"))
                except ValueError:
                    auto_fr = 0
                if (
                    auto_fr > 0
                    and self.current_world == "humanoid"
                    and self.tick >= auto_fr
                ):
                    self._curriculum_auto_fr_released = True
                    if self._fixed_root_active:
                        self.disable_fixed_root()
            return out

    def concepts_list_payload(self) -> dict:
        phase2 = (
            dict(self._l4_last_snapshot)
            if _l4_worker_enabled()
            else (
                self._concept_store.snapshot()
                if self._concept_store is not None
                else {"n_concepts": 0, "concepts": []}
            )
        )
        return {
            "concepts": list(self._concepts_cache),  # legacy / Phase 1
            "concept_store": phase2,                 # Phase 2 Part 3
            "phase2_concepts": list(phase2.get("concepts", [])),
        }

    def concept_subgraph_payload(self, cid: str) -> dict:
        from engine.concept_detector import concept_by_id

        c = concept_by_id(self._concepts_cache, cid)
        if c is None:
            return {"ok": False, "error": f"unknown concept {cid!r}"}
        nodes: list[str] = []
        for e in c.get("edges", []):
            for k in ("from_", "to"):
                if k in e and e[k] not in nodes:
                    nodes.append(e[k])
        out = {k: v for k, v in c.items()}
        out["ok"] = True
        out["nodes"] = nodes
        gn = c.get("graph_node")
        if gn and gn in self.agent.graph.nodes:
            out["graph_node_value"] = round(float(self.agent.graph.nodes[gn]), 4)
        return out

    def _memory_snapshot_meta(self) -> dict:
        try:
            from engine.persistence import autosave_every_ticks, default_memory_path

            return {
                "autosave_every": autosave_every_ticks(),
                "default_path": str(default_memory_path().resolve()),
            }
        except Exception:
            return {"autosave_every": 0, "default_path": ""}

    def _tick_discovery_plateau(self, dr: float) -> None:
        try:
            eps = float(os.environ.get("RKK_CONCEPT_DR_EPS", "0.0015"))
        except ValueError:
            eps = 0.0015
        ref = self._last_dr_snapshot
        if ref is not None and abs(float(dr) - float(ref)) < eps:
            self._discovery_plateau_count += 1
        else:
            self._discovery_plateau_count = 0
        self._last_dr_snapshot = float(dr)

    def _phase1_snapshot_meta(self) -> dict:
        from engine.local_reflex import snapshot_chains_metadata

        try:
            dpt = int(os.environ.get("RKK_CONCEPT_DISCOVERY_PLATEAU_TICKS", "0"))
        except ValueError:
            dpt = 0
        try:
            amin = float(os.environ.get("RKK_CONCEPT_EDGE_ALPHA_MIN", "0"))
        except ValueError:
            amin = 0.0
        dag_mask_frozen = os.environ.get("RKK_DAG_MASK_FROZEN", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        return {
            "read_only_macro_prefix": "concept_",
            "discovery_plateau_ticks": self._discovery_plateau_count,
            "discovery_plateau_required": dpt,
            "concept_edge_alpha_min": amin,
            "urdf_frozen_pairs": len(
                getattr(self.agent.graph, "_frozen_edge_set", set()) or set()
            ),
            "dag_mask_frozen": dag_mask_frozen,
            "local_reflex": snapshot_chains_metadata(list(self.agent.graph._node_ids)),
            "local_reflex_train": getattr(self.agent, "_last_local_reflex_train", None),
        }

    def _phase2_snapshot_meta(self) -> dict:
        hg = self._hierarchical_graph
        base = {
            "hierarchical_graph_env": hierarchical_graph_enabled(),
            "active": hg is not None,
        }
        concept_snapshot = (
            dict(self._l4_last_snapshot)
            if _l4_worker_enabled()
            else (
                self._concept_store.snapshot()
                if self._concept_store is not None
                else {"n_concepts": 0, "concepts": []}
            )
        )
        if hg is None:
            base["concept_store"] = concept_snapshot
            if _l4_worker_enabled():
                base["l4_worker"] = {
                    "enabled": True,
                    "pending": bool(self._l4_task_pending),
                    "last_submit_tick": int(self._l4_last_submit_tick),
                    "last_apply_tick": int(self._l4_last_apply_tick),
                }
            return base
        out = {**base, **hg.snapshot()}
        out["concept_store"] = concept_snapshot
        if _l4_worker_enabled():
            out["l4_worker"] = {
                "enabled": True,
                "pending": bool(self._l4_task_pending),
                "last_submit_tick": int(self._l4_last_submit_tick),
                "last_apply_tick": int(self._l4_last_apply_tick),
            }
        return out

    def _ensure_phase2(self) -> None:
        """Ленивая инициализация компонентов фазы 2 (humanoid)."""
        if self.current_world != "humanoid":
            return
        if hierarchical_graph_enabled():
            if self._hierarchical_graph is None:
                self._hierarchical_graph = HierarchicalGraph(self.agent.graph, self.device)
        cs_en = os.environ.get("RKK_CONCEPT_STORE", "1").strip().lower()
        if cs_en in ("0", "false", "no", "off"):
            self._stop_l4_worker()
            self._concept_store = None
            return
        if not self._visual_mode or self._visual_env is None:
            self._stop_l4_worker()
            return
        if _l4_worker_enabled():
            self._ensure_l4_worker()
            self._concept_store = None
            return
        if self._concept_store is None:
            vids = [
                v for v in self.agent.graph._node_ids
                if not str(v).startswith("slot_") and not str(v).startswith("concept_")
            ]
            self._concept_store = ConceptStore(
                n_slots=int(self._visual_env.n_slots),
                variable_ids=vids,
            )

    def _maybe_step_hierarchical_l1(self) -> None:
        if not hierarchical_graph_enabled():
            return
        if self.current_world != "humanoid":
            return
        if self._hierarchical_graph is None:
            return
        if self._visual_mode and self._visual_env is not None:
            base_env = getattr(self._visual_env, "base_env", None)
            if base_env is None:
                return
            raw_obs = dict(base_env.observe())
        else:
            raw_obs = dict(self.agent.env.observe())
        self._hierarchical_graph.step_l1(raw_obs)
        self._hierarchical_graph.inject_l1_virtual_nodes()

    def _l3_planning_due(self) -> bool:
        """
        Разрешение запуска L3 (goal_planning + imagination horizon) в текущем тике.
        Single-writer: только флаг для agent.step, без фоновых мутаций graph/env.
        """
        hz = _l3_loop_hz_from_env()
        if hz <= 0.0:
            self._l3_last_tick = self.tick
            return True
        now = time.perf_counter()
        if now >= self._l3_next_due_ts:
            self._l3_next_due_ts = now + (1.0 / hz)
            self._l3_last_tick = self.tick
            return True
        return False

    def _ensure_l4_worker(self) -> None:
        if self._l4_thread is not None and self._l4_thread.is_alive():
            return
        self._l4_stop.clear()
        self._l4_task_pending = False
        self._l4_thread = threading.Thread(
            target=self._l4_worker_loop,
            daemon=True,
            name="rkk-l4-concepts",
        )
        self._l4_thread.start()
        print("[Simulation] L4 concept worker enabled (single-writer apply)")

    def _stop_l4_worker(self) -> None:
        self._l4_stop.set()
        th = self._l4_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.5)
        self._l4_thread = None
        self._l4_stop.clear()
        self._l4_task_pending = False
        self._drain_simple_queue(self._l4_in_q)
        self._drain_simple_queue(self._l4_out_q)

    @staticmethod
    def _drain_simple_queue(q: queue.SimpleQueue) -> None:
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    def _enqueue_l4_task(
        self,
        *,
        slot_vecs: torch.Tensor,
        slot_values: list[float],
        variability: list[float],
        phys_obs: dict[str, float],
    ) -> None:
        if self._l4_task_pending:
            return
        payload = {
            "tick": int(self.tick),
            "slot_vecs": slot_vecs.detach().cpu().float().numpy(),
            "slot_values": [float(x) for x in list(slot_values)],
            "variability": [float(x) for x in list(variability)],
            "phys_obs": {str(k): float(v) for k, v in dict(phys_obs).items()},
            "graph_node_ids": list(self.agent.graph._node_ids),
            "n_slots": int(self._visual_env.n_slots) if self._visual_env is not None else 8,
        }
        self._l4_in_q.put(payload)
        self._l4_task_pending = True
        self._l4_last_submit_tick = self.tick

    @staticmethod
    def _serialize_l4_concept(c) -> dict:
        return {
            "cid": str(c.cid),
            "label": c.label,
            "slot_idx": int(c.slot_idx),
            "phys_vars": list(c.phys_vars),
            "corr_scores": {str(k): float(v) for k, v in dict(c.corr_scores).items()},
            "uses": int(c.uses),
            "stable_frames": int(c.stable_frames),
            "created_tick": int(c.created_tick),
        }

    def _l4_worker_loop(self) -> None:
        store: ConceptStore | None = None
        while not self._l4_stop.is_set():
            try:
                task = self._l4_in_q.get(timeout=0.05)
            except Exception:
                continue
            try:
                n_slots = int(task.get("n_slots", 8))
                graph_node_ids = list(task.get("graph_node_ids") or [])
                if store is None or store.n_slots != n_slots:
                    vids = [
                        v for v in graph_node_ids
                        if not str(v).startswith("slot_") and not str(v).startswith("concept_")
                    ]
                    store = ConceptStore(n_slots=n_slots, variable_ids=vids)
                slot_vecs_np = task.get("slot_vecs")
                slot_vecs = torch.from_numpy(slot_vecs_np).float()
                new_concepts = store.update(
                    slot_vecs=slot_vecs,
                    slot_values=list(task.get("slot_values") or []),
                    variability=list(task.get("variability") or []),
                    phys_obs=dict(task.get("phys_obs") or {}),
                    tick=int(task.get("tick", 0)),
                    graph_node_ids=graph_node_ids,
                )
                self._l4_out_q.put({
                    "tick": int(task.get("tick", 0)),
                    "snapshot": store.snapshot(),
                    "new_concepts": [self._serialize_l4_concept(c) for c in new_concepts],
                })
            except Exception as ex:
                self._l4_out_q.put({"error": str(ex)})

    def _apply_l4_concepts(self, concepts: list[dict]) -> int:
        added = 0
        for c in concepts:
            cid = str(c.get("cid", ""))
            if not cid:
                continue
            node_name = f"concept_{cid[:4]}"
            if node_name in self.agent.graph.nodes:
                continue
            uses = int(c.get("uses", 0))
            val = float(uses / (uses + 10)) if uses >= 0 else 0.0
            self.agent.graph.set_node(node_name, val)
            slot_idx = int(c.get("slot_idx", -1))
            slot_key = f"slot_{slot_idx}"
            if slot_key in self.agent.graph.nodes:
                self.agent.graph.set_edge(slot_key, node_name, 0.15, 0.05)
            corrs = dict(c.get("corr_scores") or {})
            for phys_var, corr in corrs.items():
                if phys_var not in self.agent.graph.nodes:
                    continue
                corr_f = float(corr)
                w = float(np.clip(abs(corr_f) * 0.5, 0.06, 0.4))
                sign = 1.0 if corr_f > 0 else -1.0
                self.agent.graph.set_edge(node_name, phys_var, sign * w, 0.06)
            added += 1
        return added

    def _drain_l4_results(self) -> None:
        while True:
            try:
                msg = self._l4_out_q.get_nowait()
            except Exception:
                break
            self._l4_task_pending = False
            if not isinstance(msg, dict):
                continue
            err = msg.get("error")
            if err:
                print(f"[Simulation] L4 worker: {err}")
                continue
            snap = msg.get("snapshot")
            if isinstance(snap, dict):
                self._l4_last_snapshot = snap
            new_concepts = list(msg.get("new_concepts") or [])
            if new_concepts:
                added = self._apply_l4_concepts(new_concepts)
                c0 = new_concepts[0]
                self._add_event(
                    f"Concept formed: {str(c0.get('cid',''))[:4]}, "
                    f"slot_{int(c0.get('slot_idx', -1))}, +{added} nodes",
                    "#EF9F27",
                    "phase",
                )
                self._l4_last_apply_tick = self.tick

    # ── World switch ──────────────────────────────────────────────────────────
    def switch_world(self, new_world: str) -> dict:
        if new_world not in WORLDS:
            return {"error": f"unknown world: {new_world}"}

        self._stop_rkk_agent_loop_thread()

        # Если visual mode — сначала отключаем; switch + сброс — под одним lock с тиком агента
        with self._sim_step_lock:
            was_visual = self._visual_mode
            if was_visual:
                self._disable_visual_internal()

            result = self.switcher.switch(new_world)
            if result.get("switched"):
                self._stop_cpg_background_loop()
                self.current_world = new_world
                self._locomotion_controller = None
                self._skill_library = None
                self._skill_exec = None
                self._rsi_full = None
                self._fixed_root_active = False
                self._phase3_teacher_rules = []
                self._phase3_vl_overlay = None
                self.agent.value_layer.set_teacher_vl_overlay(None)
                self._pending_llm_bundle = None
                self._llm_level2_inflight = False
                winfo = WORLDS[new_world]
                self._add_event(
                    f"🌍 → {winfo['label']} "
                    f"(+{len(result.get('new_nodes',[]))} vars, d={result.get('gnn_d')})",
                    winfo["color"], "phase"
                )
                if new_world == "humanoid":
                    self._curriculum_auto_fr_released = False
                    fr = os.environ.get("RKK_FREEZE_URDF", "1").strip().lower()
                    if fr not in ("0", "false", "no", "off") and "lhip" in self.agent.graph.nodes:
                        self.agent.graph.freeze_kinematic_priors()
                else:
                    self.agent.graph._frozen_edge_set.clear()
                    nv = list(self.agent.env.variable_ids)
                    obs = dict(self.agent.env.observe())
                    vals = {
                        k: float(obs.get(k, self.agent.graph.nodes.get(k, 0.5)))
                        for k in nv
                    }
                    self.agent.graph.rebind_variables(nv, vals)
                self._materialized_detector_concept_ids.clear()
                self._discovery_plateau_count = 0
                self._last_dr_snapshot = None
                self._hierarchical_graph = None
                self._concept_store = None
                self._stop_l4_worker()
                self._l4_last_snapshot = {"n_concepts": 0, "concepts": []}
                self._l3_next_due_ts = 0.0
                self._l3_last_tick = 0
                self._motor_state = MotorState()
                # Reset Level 1 controllers on world switch
                if _EMBODIED_REWARD_AVAILABLE and self._embodied_reward_ctrl is not None:
                    self._embodied_reward_ctrl = EmbodiedRewardController()
                if _VISUAL_GROUNDING_AVAILABLE and self._visual_grounding_ctrl is not None:
                    self._visual_grounding_ctrl = VisualGroundingController()
                if hasattr(self, "_mc_posture_window"):
                    self._mc_posture_window.clear()
                if hasattr(self, "_mc_fallen_count_window"):
                    self._mc_fallen_count_window.clear()
                # Reset Level 2 controllers
                if _EPISODIC_MEMORY_AVAILABLE and self._episodic_memory is not None:
                    from engine.episodic_memory import EpisodicMemory

                    self._episodic_memory = EpisodicMemory()
                if _CURRICULUM_AVAILABLE and self._curriculum is not None:
                    from engine.llm_curriculum import CurriculumScheduler

                    self._curriculum = CurriculumScheduler()
                self._last_action_for_memory = None
                self._last_fall_memory_tick = -999_999
                self._rssm_upgraded = False
                self._rssm_trainer = None
                self._rssm_imagination = None
                # Reset Level 3 controllers
                if _PROPRIO_AVAILABLE:
                    self._proprio = ProprioceptionStream(device=self.device)
                if _TIMESCALE_AVAILABLE:
                    self._timescale = MultiscaleTimeController()
                self._reward_coord = None
                self._reward_X_prev = []
                self._reward_a_prev = []
                self._was_blocked = False
                # Motor Cortex reset on world switch
                self._motor_cortex = None
                self._mc_abstract_nodes_injected = False
                self._clear_fall_recovery()
                self._drain_simple_queue(self._l1_motor_q)
                self._l1_last_cmd_tick = 0
                self._l1_last_apply_tick = 0

        # Восстанавливаем visual mode если был (вне lock: enable_visual сам берёт lock)
        if was_visual and result.get("switched"):
            self.enable_visual()

        return result

    # ── Фаза 12: Visual mode ──────────────────────────────────────────────────
    def enable_visual(self, n_slots: int = 8, mode: str = "hybrid") -> dict:
        """
        Включаем Causal Visual Cortex.
        Текущая среда оборачивается в EnvironmentVisual.
        По умолчанию mode="hybrid": слоты для наблюдения + phys_* моторы — иначе VL/физика
        блокируют do(slot_k) без реального сустава.
        """
        if self._visual_mode:
            return {"visual": True, "already_enabled": True}

        try:
            from engine.environment_visual import EnvironmentVisual
        except ImportError:
            return {"error": "causal_vision module not available (install: opencv-python, scipy)"}

        with self._sim_step_lock:
            if self._visual_mode:
                return {"visual": True, "already_enabled": True}

            # Сохраняем оригинальный env
            self._base_env_ref = self.agent.env

            # Оборачиваем
            vis_env = EnvironmentVisual(
                self._base_env_ref,
                device=self.device,
                n_slots=n_slots,
                mode=mode,
            )
            self._visual_env = vis_env

            # Меняем среду агента: граф только под variable_ids обёртки (не 26+K узлов)
            new_vars  = list(vis_env.variable_ids)
            init_obs  = vis_env.observe()
            self.agent.graph.rebind_variables(new_vars, init_obs)

            self.agent.env = vis_env

            # Пересоздаём Temporal для нового d
            from engine.temporal import TemporalBlankets
            new_d = len(new_vars)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)

            self.agent.temporal.step(init_obs)
            self.agent.graph.record_observation(init_obs)

            # Инжектируем слабые seeds между слотами
            seeds = vis_env.hardcoded_seeds()
            self.agent.inject_text_priors(seeds)

            self._visual_mode = True
            self._vision_ticks = 0

            self._add_event(
                f"👁 Visual Cortex ENABLED: {n_slots} slots · {mode} mode",
                "#44ffcc", "phase"
            )
            cd = str(next(vis_env.cortex.parameters()).device)
            print(
                f"[Simulation] Visual mode ON: {n_slots} slots, d={self.agent.graph._d}, "
                f"cortex={cd}"
            )

            return {
                "visual": True,
                "n_slots": n_slots,
                "mode": mode,
                "new_vars": new_vars,
                "gnn_d": self.agent.graph._d,
                "cortex_device": cd,
            }

    def _disable_visual_internal(self):
        """Внутреннее отключение без event."""
        if not self._visual_mode:
            return
        if self._base_env_ref is not None:
            self.agent.env = self._base_env_ref
            base_ids = list(self._base_env_ref.variable_ids)
            base_obs = self._base_env_ref.observe()
            self.agent.graph.rebind_variables(base_ids, base_obs)
            from engine.temporal import TemporalBlankets
            new_d = len(base_ids)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)
            self.agent.temporal.step(base_obs)
            self.agent.graph.record_observation(base_obs)
        self._visual_mode = False
        self._visual_env  = None
        self._concept_store = None
        self._stop_l4_worker()
        self._l3_next_due_ts = 0.0

    def disable_visual(self) -> dict:
        """Отключаем Visual Cortex, возвращаемся к ручным переменным."""
        if not self._visual_mode:
            return {"visual": False, "was_enabled": False}
        with self._sim_step_lock:
            if not self._visual_mode:
                return {"visual": False, "was_enabled": False}
            self._disable_visual_internal()
            self._add_event("👁 Visual Cortex DISABLED", "#cc44ff", "phase")
        return {"visual": False, "was_enabled": True}

    def enable_fixed_root(self) -> dict:
        """
        Включаем fixed_root mode:
          1. PyBullet JOINT_FIXED constraint фиксирует базу
          2. variable_ids → FIXED_BASE_VARS (+ sandbox vars)
          3. GNN rebind; при visual — через EnvironmentVisual.set_fixed_root
          4. HomeostaticBounds → for_fixed_root()
          5. inject fixed_root_seeds() (для slot-графа большинство рёбер может быть skipped)
        """
        from engine.environment_humanoid import EnvironmentHumanoid, fixed_root_seeds

        base_env = (
            self._base_env_ref
            if (self._visual_mode and getattr(self, "_base_env_ref", None) is not None)
            else self.agent.env
        )
        if not isinstance(base_env, EnvironmentHumanoid):
            return {"error": "fixed_root требует humanoid world"}

        if base_env.fixed_root:
            return {"fixed_root": True, "already_enabled": True}

        with self._sim_step_lock:
            if self._visual_mode and self._visual_env is not None:
                self._visual_env.set_fixed_root(True)
            else:
                base_env.set_fixed_root(True)

            env = self.agent.env
            new_vars = list(env.variable_ids)
            init_obs = env.observe()
            self.agent.graph.rebind_variables(new_vars, init_obs)

            from engine.temporal import TemporalBlankets
            new_d = len(new_vars)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)
            self.agent.temporal.step(init_obs)
            self.agent.graph.record_observation(init_obs)

            from engine.value_layer import HomeostaticBounds
            self.agent.value_layer.bounds = HomeostaticBounds.for_fixed_root()

            seeds = fixed_root_seeds()
            result = self.agent.inject_text_priors(seeds)

            self._fixed_root_active = True
            self._fall_count = 0
            self._stop_cpg_background_loop()

            self._add_event(
                f"📌 FIXED ROOT ON: d={self.agent.graph._d}, "
                f"{len(new_vars)} vars, +{result.get('injected',0)} seeds",
                "#ffcc44", "phase"
            )
            print(
                f"[Simulation] fixed_root ON: vars={len(new_vars)}, "
                f"d={self.agent.graph._d}, seeds={result.get('injected',0)}"
            )
            return {
                "fixed_root": True,
                "gnn_d":      self.agent.graph._d,
                "new_vars":   new_vars,
                "seeds_injected": result.get("injected", 0),
            }

    def disable_fixed_root(self) -> dict:
        """
        Отключаем fixed_root mode:
          1. Снимаем JOINT_FIXED constraint
          2. variable_ids → полные VAR_NAMES
          3. GNN rebind; при visual — через EnvironmentVisual.set_fixed_root(False)
          4. HomeostaticBounds → default (строгие, но с warmup)
        """
        from engine.environment_humanoid import EnvironmentHumanoid, humanoid_hardcoded_seeds

        base_env = (
            self._base_env_ref
            if (self._visual_mode and getattr(self, "_base_env_ref", None) is not None)
            else self.agent.env
        )
        if not isinstance(base_env, EnvironmentHumanoid):
            return {"error": "не humanoid world"}

        if not base_env.fixed_root:
            return {"fixed_root": False, "was_enabled": False}

        with self._sim_step_lock:
            if self._visual_mode and self._visual_env is not None:
                self._visual_env.set_fixed_root(False)
            else:
                base_env.set_fixed_root(False)

            env = self.agent.env
            new_vars = list(env.variable_ids)
            init_obs = env.observe()
            self.agent.graph.rebind_variables(new_vars, init_obs)

            from engine.temporal import TemporalBlankets
            new_d = len(new_vars)
            if self.agent.temporal.d_input != new_d:
                self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)
            self.agent.temporal.step(init_obs)
            self.agent.graph.record_observation(init_obs)

            from engine.value_layer import HomeostaticBounds
            self.agent.value_layer.bounds = HomeostaticBounds(
                var_min=0.05, var_max=0.95,
                phi_min=0.005,
                h_slow_max=18.0,
                env_entropy_max_delta=1.2,
                s1_penalty=-0.2,
                warmup_ticks=800,
                blend_ticks=400,
                phi_min_steady=0.03,
                env_entropy_max_delta_steady=0.65,
                h_slow_max_steady=14.0,
                predict_band_edge_steady=0.015,
                fixed_root_mode=False,
            )

            result = self.agent.inject_text_priors(humanoid_hardcoded_seeds())

            self._fixed_root_active = False
            self._add_event(
                f"📌 FIXED ROOT OFF: d={self.agent.graph._d}, {len(new_vars)} vars",
                "#cc44ff", "phase"
            )
            return {
                "fixed_root": False,
                "gnn_d":      self.agent.graph._d,
                "new_vars":   new_vars,
                "seeds_injected": result.get("injected", 0),
            }

    # ── Seeds ─────────────────────────────────────────────────────────────────
    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        with self._sim_step_lock:
            result = self.agent.inject_text_priors(edges)
            n = result.get("injected", 0)
            self._add_event(f"💉 Seeds → Nova: {n} edges (α=0.05)", "#886600", "discovery")
        return {"injected": n, "agent": self.AGI_NAME,
                "skipped": result.get("skipped", []),
                "node_ids": result.get("node_ids", [])}

    def agent_seed_context(self, agent_id: int = 0) -> dict | None:
        return {
            "name":      self.AGI_NAME,
            "preset":    self.current_world,
            "variables": list(self.agent.graph.nodes.keys()),
        }

    def _try_reset_pose_after_fall(self) -> bool:
        """Сброс позы гуманоида (база PyBullet), чтобы выйти из ловушки fallen + VL block."""
        env = self.agent.env
        fn = getattr(env, "reset_stance", None)
        if not callable(fn):
            return False
        if self.tick - self._last_fall_reset_tick < 4:
            return False
        fn()
        self.agent.graph._obs_buffer.clear()
        self.agent.graph._int_buffer.clear()
        self._last_fall_reset_tick = self.tick
        self._add_event("🔄 Сброс позы после падения", "#44aaff", "value")
        return True

    @staticmethod
    def _fall_recovery_score(obs: dict) -> float:
        cz = float(obs.get("com_z", obs.get("phys_com_z", 0.0)))
        posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.0)))
        foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.0)))
        foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.0)))
        return 0.45 * cz + 0.35 * posture + 0.20 * min(foot_l, foot_r)

    def _clear_fall_recovery(self) -> None:
        self._fall_recovery_active = False
        self._fall_recovery_start_tick = 0
        self._fall_recovery_last_progress_tick = 0
        self._fall_recovery_best_score = 0.0

    def _maybe_recover_or_reset_after_fall(self, obs: dict) -> bool:
        """
        Recovery-first policy:
        - give the agent time to stand up on its own,
        - hard-reset only when recovery stalls for too long.
        Returns True if a hard reset was performed.
        """
        score = self._fall_recovery_score(obs)
        try:
            max_ticks = int(os.environ.get("RKK_FALL_RECOVERY_TICKS", "40"))
        except ValueError:
            max_ticks = 40
        try:
            stall_ticks = int(os.environ.get("RKK_FALL_RECOVERY_STALL_TICKS", "12"))
        except ValueError:
            stall_ticks = 12
        try:
            min_gain = float(os.environ.get("RKK_FALL_RECOVERY_MIN_GAIN", "0.02"))
        except ValueError:
            min_gain = 0.02
        max_ticks = max(8, min(max_ticks, 600))
        stall_ticks = max(4, min(stall_ticks, max_ticks))
        min_gain = float(np.clip(min_gain, 0.0, 0.25))

        if not self._fall_recovery_active:
            self._fall_recovery_active = True
            self._fall_recovery_start_tick = self.tick
            self._fall_recovery_last_progress_tick = self.tick
            self._fall_recovery_best_score = score
            self._add_event("🦿 Recovery window after fall", "#ffbb66", "value")
            return False

        if score > self._fall_recovery_best_score + min_gain:
            self._fall_recovery_best_score = score
            self._fall_recovery_last_progress_tick = self.tick

        total_elapsed = self.tick - self._fall_recovery_start_tick
        stalled_for = self.tick - self._fall_recovery_last_progress_tick
        if total_elapsed < max_ticks and stalled_for < stall_ticks:
            return False

        if self._try_reset_pose_after_fall():
            self._clear_fall_recovery()
            return True
        return False

    # ── Этап D: LLM в петле ────────────────────────────────────────────────────
    def _llm_loop_enabled(self) -> bool:
        return os.environ.get("RKK_LLM_LOOP", "").strip().lower() in ("1", "true", "yes", "on")

    def _apply_pending_llm_bundle(self) -> None:
        b = self._pending_llm_bundle
        if not b:
            return
        self._pending_llm_bundle = None
        l2 = b.get("l2") or {}
        if l2.get("ok"):
            edges = l2.get("candidate_edges") or []
            ex = (l2.get("explanation") or "").strip()
            next_probe = l2.get("next_probe") or {}
            if ex or edges or next_probe:
                edge_preview = [
                    f"{e.get('from_')}->{e.get('to')}({float(e.get('weight', 0.0)):+.2f})"
                    for e in list(edges)[:6]
                    if isinstance(e, dict)
                ]
                print(
                    "[LLM L2] "
                    f"explanation={ex or '-'} | "
                    f"next_probe={next_probe or {}} | "
                    f"edges={len(edges)} | "
                    f"preview={edge_preview}"
                )
            if edges:
                inj = self.agent.inject_text_priors(edges)
                n = inj.get("injected", 0)
                self._add_event(f"🧠 LLM L2 (+{n} priors): {ex}", "#9bdcff", "phase")
            self._llm_loop_stats["level2_runs"] = int(self._llm_loop_stats.get("level2_runs", 0)) + 1
            self._llm_loop_stats["last_level2_explanation"] = (l2.get("explanation") or "").strip()
        elif l2.get("error"):
            print(f"[LLM L2] error={l2.get('error')}")
            self._add_event(f"LLM L2 error: {l2.get('error')}", "#cc6666", "phase")

        l3 = b.get("l3")
        if isinstance(l3, dict) and l3.get("ok"):
            edges3 = l3.get("edges") or []
            if edges3:
                edge_preview3 = [
                    f"{e.get('from_')}->{e.get('to')}({float(e.get('weight', 0.0)):+.2f})"
                    for e in list(edges3)[:8]
                    if isinstance(e, dict)
                ]
                print(
                    "[LLM L3] "
                    f"edges={len(edges3)} | "
                    f"preview={edge_preview3}"
                )
            if edges3:
                inj3 = self.agent.inject_text_priors(edges3)
                self._add_event(
                    f"🧬 LLM L3 restructure +{inj3.get('injected', 0)} hypotheses",
                    "#66ddff",
                    "phase",
                )
            self._last_level3_tick = self.tick
            self._llm_loop_stats["level3_runs"] = int(self._llm_loop_stats.get("level3_runs", 0)) + 1
        elif isinstance(l3, dict) and l3.get("error"):
            print(f"[LLM L3] error={l3.get('error')}")

    def _rolling_block_rate(self) -> float:
        if len(self._rolling_block_bits) < 24:
            return float(self.agent.value_layer.block_rate)
        return float(np.mean(self._rolling_block_bits))

    def _prediction_surprise_trigger(self, result: dict) -> bool:
        if result.get("skipped") or result.get("blocked"):
            return False
        pe = float(result.get("prediction_error", 0.0))
        self._pe_history.append(pe)
        if len(self._pe_history) < 40:
            return False
        arr = np.array(self._pe_history, dtype=np.float64)
        mu, sd = float(arr.mean()), float(arr.std())
        if sd < 1e-8:
            return False
        return pe > mu + 3.0 * sd

    def _vlm_unknown_slot_trigger(self) -> bool:
        """Слот с заметной динамикой, но без привязки к phys / подозрительный лейбл."""
        if not self._visual_mode or self._visual_env is None:
            return False
        try:
            vis = self._visual_env.get_slot_visualization()
        except Exception:
            return False
        lex = getattr(self._visual_env, "_slot_lexicon", None) or {}
        varib = list(vis.get("variability") or [])
        n = int(getattr(self._visual_env, "n_slots", 0) or 0)
        bad_labs = frozenset({"?", "unknown", "unlabeled", "", "thing", "object"})
        for i in range(n):
            sk = f"slot_{i}"
            entry = lex.get(sk) or {}
            lab = str(entry.get("label", "")).strip().lower()
            likely = entry.get("likely_phys") or []
            vscore = float(varib[i]) if i < len(varib) else 0.0
            if vscore > 0.32 and not likely:
                return True
            if vscore > 0.25 and lab in bad_labs:
                return True
        return False

    def _should_run_level3(self, triggers: list[str]) -> bool:
        if self.current_world != "humanoid":
            return False
        try:
            interval = int(os.environ.get("RKK_LLM_LEVEL3_INTERVAL", "4200"))
        except ValueError:
            interval = 4200
        if self.tick - self._last_level3_tick < interval:
            return False
        tset = " ".join(triggers)
        return ("stagnation" in tset) or ("block_rate" in tset)

    def _maybe_schedule_llm_loop(self, result: dict, snap: dict) -> None:
        if not self._llm_loop_enabled():
            return
        if self._llm_level2_inflight or self._pending_llm_bundle is not None:
            return
        try:
            cooldown = int(os.environ.get("RKK_LLM_LEVEL2_COOLDOWN", "1200"))
        except ValueError:
            cooldown = 1200
        if self.tick - self._last_level2_schedule_tick < cooldown:
            return

        try:
            stagnation_ticks = int(os.environ.get("RKK_LLM_STAGNATION_TICKS", "1400"))
        except ValueError:
            stagnation_ticks = 1400
        try:
            min_iv = int(os.environ.get("RKK_LLM_MIN_INTERVENTIONS", "96"))
        except ValueError:
            min_iv = 96

        triggers: list[str] = []
        if (
            self.agent._total_interventions >= min_iv
            and (self.tick - self._last_dr_gain_tick) >= stagnation_ticks
        ):
            triggers.append("discovery_stagnation")

        vl = self.agent.value_layer
        if (
            vl.total_checked >= 128
            and len(self.agent.graph.edges) >= 6
            and self._rolling_block_rate() > 0.72
        ):
            triggers.append("block_rate")

        if self._vlm_unknown_slot_trigger():
            triggers.append("vlm_unknown_object")

        if self._prediction_surprise_trigger(result) and self.tick >= max(240, min_iv):
            triggers.append("prediction_surprise_3sigma")

        if not triggers:
            return

        self._last_level2_schedule_tick = self.tick
        self._llm_level2_inflight = True
        self._llm_loop_stats["last_triggers"] = list(triggers)

        from engine.phase3_teacher import _slot_lexicon_summary

        ctx = {
            "variable_ids": list(self.agent.graph.nodes.keys()),
            "triggers": triggers,
            "variable": result.get("variable"),
            "value": float(result.get("value", 0.0)),
            "prediction_error": float(result.get("prediction_error", 0.0)),
            "discovery_rate": float(snap.get("discovery_rate", 0.0)),
            "block_rate": float(vl.block_rate),
            "cf_predicted": result.get("cf_predicted") or {},
            "cf_observed": result.get("cf_observed") or {},
            "slot_lexicon": _slot_lexicon_summary(self._visual_env),
            "fall_history": (
                self._episodic_memory.get_llm_context_block(max_falls=5)
                if self._episodic_memory is not None
                else ""
            ),
            "curriculum_stage": (
                self._curriculum.current_stage.name
                if self._curriculum is not None
                else ""
            ),
            "temporal_context": (
                self._timescale.build_llm_temporal_context()
                if self._timescale is not None
                else ""
            ),
            "reward_breakdown": (
                self._reward_coord.snapshot().get("last_signal", {})
                if self._reward_coord is not None
                else {}
            ),
            "proprio_abstracts": (
                self._proprio.snapshot().get("abstracts", {})
                if self._proprio is not None
                else {}
            ),
            "run_level3": self._should_run_level3(triggers),
            "llm_url": get_ollama_generate_url(),
            "llm_model": get_ollama_model(),
        }

        self._llm_loop_executor.submit(self._llm_bundle_worker, ctx)

    def _llm_bundle_worker(self, ctx: dict) -> None:
        try:
            from engine.llm_loop import consult_counterfactual_sync, structure_revision_sync

            valid = set(ctx.get("variable_ids") or [])
            l2 = consult_counterfactual_sync(
                ctx["llm_url"],
                ctx["llm_model"],
                ctx,
                valid,
            )
            l3 = None
            if ctx.get("run_level3") and l2.get("ok") and ctx.get("variable_ids"):
                l3 = structure_revision_sync(
                    ctx["llm_url"],
                    ctx["llm_model"],
                    list(ctx["variable_ids"]),
                )
                if not (isinstance(l3, dict) and l3.get("ok")):
                    l3 = None
            self._pending_llm_bundle = {"l2": l2, "l3": l3}
        except Exception as e:
            self._pending_llm_bundle = {"l2": {"ok": False, "error": str(e)}, "l3": None}
        finally:
            self._llm_level2_inflight = False

    @staticmethod
    def _unwrap_base_env(env):
        e = env
        while hasattr(e, "base_env"):
            e = e.base_env
        return e

    def _locomotion_cpg_enabled(self) -> bool:
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _cpg_decoupled_enabled(self) -> bool:
        return self._locomotion_cpg_enabled() and _cpg_loop_hz_from_env() > 0.0

    def _stop_cpg_background_loop(self) -> None:
        self._cpg_stop.set()
        th = self._cpg_loop_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.5)
        self._cpg_loop_thread = None
        self._cpg_stop.clear()
        self._drain_simple_queue(self._l1_motor_q)

    def _ensure_cpg_background_loop(self) -> None:
        if not self._cpg_decoupled_enabled():
            return
        if self.current_world != "humanoid" or self._fixed_root_active:
            return
        base = self._unwrap_base_env(self.agent.env)
        if not callable(getattr(base, "apply_cpg_leg_targets", None)):
            return
        if self._cpg_loop_thread is not None and self._cpg_loop_thread.is_alive():
            return
        self._cpg_stop.clear()
        self._cpg_loop_thread = threading.Thread(
            target=self._cpg_loop_worker,
            daemon=True,
            name="rkk-cpg-loop",
        )
        self._cpg_loop_thread.start()
        print(
            f"[Simulation] CPG low-level loop ~{_cpg_loop_hz_from_env():.0f} Hz "
            f"(decoupled from agent tick; RKK_CPG_LOOP_HZ)"
        )

    def _publish_cpg_node_snapshot(self) -> None:
        if not self._cpg_decoupled_enabled():
            return
        with self._cpg_snapshot_lock:
            self._cpg_node_snapshot = dict(self.agent.graph.nodes)

    def _cpg_loop_worker(self) -> None:
        hz = _cpg_loop_hz_from_env()
        dt = 1.0 / hz if hz > 0 else 0.05
        from engine.cpg_locomotion import LocomotionController

        while not self._cpg_stop.is_set():
            t0 = time.perf_counter()
            try:
                if not self._locomotion_cpg_enabled():
                    time.sleep(0.05)
                    continue
                if self.current_world != "humanoid" or self._fixed_root_active:
                    time.sleep(0.05)
                    continue
                base = self._unwrap_base_env(self.agent.env)
                fn = getattr(base, "apply_cpg_leg_targets", None)
                if not callable(fn):
                    time.sleep(0.05)
                    continue
                if self._locomotion_controller is None:
                    self._locomotion_controller = LocomotionController(self.device)
                with self._cpg_snapshot_lock:
                    nodes = dict(self._cpg_node_snapshot)
                if not nodes:
                    nodes = dict(self.agent.graph.nodes)
                # Актуальный CoM из физики — иначе отставание массы (com_lag) в trunk_sync ломается в фоне.
                try:
                    obs_live = self.agent.env.observe()
                    for _k in ("com_x", "com_y", "com_z"):
                        if _k in obs_live:
                            nodes[_k] = float(obs_live[_k])
                except Exception:
                    pass
                targets = self._locomotion_controller.get_joint_targets(nodes, dt=dt)
                cpg_sync = self._locomotion_controller.upper_body_cpg_sync()
                self._enqueue_l1_motor_command(
                    source="cpg",
                    joint_targets=targets,
                    intents=getattr(self._locomotion_controller, "_last_motor_state", None),
                    dt=dt,
                    cpg_sync=cpg_sync,
                )
            except Exception as ex:
                print(f"[Simulation] CPG loop: {ex}")
            elapsed = time.perf_counter() - t0
            wait = dt - elapsed
            if wait > 0:
                self._cpg_stop.wait(timeout=wait)

    def _maybe_apply_cpg_locomotion(self, fallen: bool) -> None:
        """Phase A+D: CPG + Motor Cortex blended locomotion."""
        dt = 0.05
        if self._cpg_decoupled_enabled():
            return
        if not self._locomotion_cpg_enabled():
            return
        if self.current_world != "humanoid" or self._fixed_root_active:
            return
        base = self._unwrap_base_env(self.agent.env)
        fn = getattr(base, "apply_cpg_leg_targets", None)
        if not callable(fn):
            return
        if self._locomotion_controller is None:
            from engine.cpg_locomotion import LocomotionController
            self._locomotion_controller = LocomotionController(self.device)

        try:
            nodes = dict(self.agent.graph.nodes)

            # CPG generates base targets
            cpg_targets = self._locomotion_controller.get_joint_targets(nodes, dt=dt)
            cpg_sync = self._locomotion_controller.upper_body_cpg_sync()

            # Phase D: Motor Cortex blending
            mc = self._ensure_motor_cortex()
            final_targets = dict(cpg_targets)
            if mc is not None and len(mc.programs) > 0:
                obs_now = dict(self.agent.env.observe())
                posture_now = float(obs_now.get(
                    "posture_stability", obs_now.get("phys_posture_stability", 0.5)
                ))
                com_z_now = float(obs_now.get("com_z", obs_now.get("phys_com_z", 0.5)))

                # Select active programs by situation
                if com_z_now < 0.35 or fallen:
                    active_progs = ["recovery"]
                elif posture_now < 0.58:
                    active_progs = ["balance", "recovery"]
                else:
                    active_progs = ["walk", "balance"]
                active_progs = [p for p in active_progs if p in mc.programs]

                if active_progs:
                    cortex_targets = mc.infer(nodes, active_progs)
                    final_targets = mc.blend_targets(cpg_targets, cortex_targets)
                    # Expose CPG weight to locomotion controller for diagnostics
                    self._locomotion_controller.cpg_weight = mc.cpg_weight

            obs_before_env = dict(self.agent.env.observe())
            fn(final_targets, cpg_sync=cpg_sync)
            obs = dict(self.agent.env.observe())

            self._sync_motor_state(obs, source="cpg+mc", tick=self.tick)
            self._log_motor_command(
                source="cpg+mc",
                joint_targets=final_targets,
                intents=getattr(self._locomotion_controller, "_last_motor_state", None),
                obs=self._motor_obs_payload(obs),
            )
            self._record_motor_burst_causal(
                obs_before_env=obs_before_env,
                obs_after_env=dict(obs),
                intents=dict(getattr(self._locomotion_controller, "_last_motor_state", {}) or {}),
            )

            # Extract metrics
            posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
            foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
            foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
            com_z = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
            com_x = float(obs.get("com_x", obs.get("phys_com_x", 0.5)))

            # Phase D: train motor cortex programs + anneal CPG
            if mc is not None:
                reward = (
                    posture * 2.0
                    + min(foot_l, foot_r) * 1.5
                    + com_z * 1.0
                    + max(0.0, com_x - 0.46) * 1.5   # forward bonus
                    - (3.0 if fallen else 0.0)
                )
                mc.push_and_train(nodes, cpg_targets, reward, posture, foot_l, foot_r)
                mc.anneal_step(posture, foot_l, foot_r, fallen, self.tick)

                # Inject abstract nodes once when first program spawned
                if not self._mc_abstract_nodes_injected and len(mc.programs) > 0:
                    added = mc.inject_abstract_nodes_into_graph(self.agent.graph)
                    if added > 0:
                        self._mc_abstract_nodes_injected = True
                        self._add_event(
                            f"🧠 MotorCortex: +{added} abstract nodes (mc_walk_drive, mc_balance_signal, …)",
                            "#ff88ff", "phase"
                        )
                mc.sync_abstract_nodes_to_graph(self.agent.graph)

            self._locomotion_controller.learn_from_reward(
                com_z, com_x, fallen, motor_obs=self._motor_obs_payload(obs)
            )
            # Track for embodied reward and motor cortex
            self._mc_posture_window.append(posture)
            self._mc_fallen_count_window.append(1 if fallen else 0)
        except Exception as ex:
            import traceback
            print(f"[Simulation] CPG+MC locomotion error: {ex}")
            if os.environ.get("RKK_DEBUG_CPG"):
                traceback.print_exc()

    def _rsi_full_enabled(self) -> bool:
        v = os.environ.get("RKK_RSI_FULL", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _ensure_motor_cortex(self):
        """Phase D: ленивая инициализация MotorCortexLibrary."""
        if not _MOTOR_CORTEX_AVAILABLE:
            return None
        env_flag = os.environ.get("RKK_MOTOR_CORTEX", "1").strip().lower()
        if env_flag in ("0", "false", "no", "off"):
            return None
        if self.current_world != "humanoid" or self._fixed_root_active:
            return None
        if self._motor_cortex is None:
            self._motor_cortex = _MotorCortexLibrary(self.device)
        return self._motor_cortex

    def _ensure_reward_coord(self) -> None:
        """Lazy-init RewardCoordinator; rebuild CuriosityICM when graph dimension changes."""
        if not _REWARD_COORD_AVAILABLE:
            return
        if not hasattr(self, "agent") or self.agent is None:
            return
        graph = getattr(self.agent, "graph", None)
        if graph is None:
            return
        nids = list(getattr(graph, "_node_ids", []) or [])
        d = len(nids) if nids else int(getattr(graph, "_d", 30) or 30)

        if self._reward_coord is not None:
            cur = getattr(self._reward_coord.curiosity, "d", -1)
            if cur != d:
                self._reward_coord = None
                self._reward_X_prev = []
                self._reward_a_prev = []

        if self._reward_coord is not None:
            return

        self._reward_coord = RewardCoordinator(d=d, device=self.device)
        print(f"[Simulation] RewardCoordinator init d={d}")

    def _locomotion_reward_ema(self) -> float:
        lc = self._locomotion_controller
        if lc is None or not lc._reward_history:
            return 0.0
        w = min(32, len(lc._reward_history))
        return float(np.mean(lc._reward_history[-w:]))

    def _motor_obs_payload(self, obs: dict) -> dict[str, float]:
        keys = (
            "gait_phase_l",
            "gait_phase_r",
            "foot_contact_l",
            "foot_contact_r",
            "support_bias",
            "motor_drive_l",
            "motor_drive_r",
            "posture_stability",
        )
        return {k: float(obs.get(k, 0.5)) for k in keys}

    def _sync_motor_state(self, obs: dict, *, source: str, tick: int | None = None) -> None:
        with self._motor_state_lock:
            self._motor_state.update_from_observation(obs, tick=tick if tick is not None else self.tick, source=source)

    def _log_motor_command(
        self,
        *,
        source: str,
        joint_targets: dict[str, float] | None = None,
        obs: dict | None = None,
        intents: dict[str, float] | None = None,
    ) -> None:
        with self._motor_state_lock:
            self._motor_state.update_from_command(
                tick=self.tick,
                source=source,
                intents=intents,
                joint_targets=joint_targets,
                obs=obs,
            )

    def _motor_state_snapshot(self) -> dict:
        with self._motor_state_lock:
            return self._motor_state.snapshot()

    def _enqueue_l1_motor_command(
        self,
        *,
        source: str,
        joint_targets: dict[str, float],
        intents: dict[str, float] | None = None,
        dt: float,
        cpg_sync: dict[str, float] | None = None,
    ) -> None:
        payload: dict = {
            "tick": int(self.tick),
            "source": str(source),
            "joint_targets": {k: float(v) for k, v in joint_targets.items()},
            "intents": {k: float(v) for k, v in (intents or {}).items()},
            "dt": float(dt),
        }
        if cpg_sync:
            payload["cpg_sync"] = {k: float(v) for k, v in cpg_sync.items()}
        self._l1_motor_q.put(payload)
        self._l1_last_cmd_tick = self.tick

    def _record_motor_burst_causal(
        self,
        *,
        obs_before_env: dict[str, float],
        obs_after_env: dict[str, float],
        intents: dict[str, float],
    ) -> None:
        """
        Low-level motor burst as record_intervention-like event in main writer.
        """
        self.agent.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.agent.graph.snapshot_vec_dict()
        self.agent.graph.apply_env_observation(obs_after_env)
        obs_after_full = self.agent.graph.snapshot_vec_dict()
        self.agent.graph.record_observation(obs_before_full)
        self.agent.graph.record_observation(obs_after_full)
        # Все значимые intent в burst (до 4 сильнейших по |v−0.5|).
        sig: list[tuple[float, str, float]] = []
        for k, v in (intents or {}).items():
            if k not in self.agent.graph.nodes:
                continue
            fv = float(v)
            d = abs(fv - 0.5)
            if d > 0.08:
                sig.append((d, k, fv))
        sig.sort(key=lambda t: -t[0])
        for _d, k, fv in sig[:4]:
            self.agent.graph.record_intervention(k, fv, obs_before_full, obs_after_full)

    def _drain_l1_motor_commands(self) -> None:
        base = self._unwrap_base_env(self.agent.env)
        fn = getattr(base, "apply_cpg_leg_targets", None)
        if not callable(fn):
            return
        latest = None
        while True:
            try:
                latest = self._l1_motor_q.get_nowait()
            except Exception:
                break
        if latest is None:
            return
        targets = dict(latest.get("joint_targets") or {})
        intents = dict(latest.get("intents") or {})
        cpg_sync = latest.get("cpg_sync")
        try:
            credit_every = int(os.environ.get("RKK_MOTOR_CREDIT_EVERY", "4"))
        except ValueError:
            credit_every = 4
        credit_every = max(1, min(credit_every, 64))
        strong_intent = any(abs(float(v) - 0.5) > 0.12 for v in intents.values())
        should_credit = strong_intent and ((self.tick - self._l1_last_credit_tick) >= credit_every)
        obs_before = dict(self.agent.env.observe()) if should_credit else None
        if cpg_sync:
            fn(targets, cpg_sync=dict(cpg_sync))
        else:
            fn(targets)
        obs_after = dict(self.agent.env.observe())
        self._sync_motor_state(obs_after, source=str(latest.get("source", "cpg")), tick=self.tick)
        self._log_motor_command(
            source=str(latest.get("source", "cpg")),
            joint_targets=targets,
            intents=intents or None,
            obs=self._motor_obs_payload(obs_after),
        )
        if should_credit and obs_before is not None:
            self._record_motor_burst_causal(
                obs_before_env=obs_before,
                obs_after_env=obs_after,
                intents=intents,
            )
            self._l1_last_credit_tick = self.tick
        lc = self._locomotion_controller
        if lc is not None:
            fallen = False
            is_fn = getattr(self.agent.env, "is_fallen", None)
            if callable(is_fn) and not self._fixed_root_active:
                fallen = bool(is_fn())
            lc.learn_from_reward(
                float(obs_after.get("com_z", 0.5)),
                float(obs_after.get("com_x", 0.5)),
                fallen,
                motor_obs=self._motor_obs_payload(obs_after),
            )
        self._l1_last_apply_tick = self.tick

    def _skill_library_enabled(self) -> bool:
        v = os.environ.get("RKK_SKILL_LIBRARY", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _skill_start_prob(self) -> float:
        try:
            p = float(os.environ.get("RKK_SKILL_LIBRARY_PROB", "0.1"))
        except ValueError:
            p = 0.1
        if self.current_world == "humanoid" and not self._fixed_root_active:
            obs = self.agent.env.observe()
            posture = float(
                obs.get(
                    "posture_stability",
                    obs.get("phys_posture_stability", 0.5),
                )
            )
            # Чем нестабильнее — тем выше доля скиллов (меньше сырого EIG).
            adaptive = 0.80 - posture * 0.30  # posture=0 → 0.80, posture=1 → 0.50
            p = max(p, adaptive)
        return float(np.clip(p, 0.0, 1.0))

    def _ensure_skill_library(self):
        if self._skill_library is None:
            from engine.skill_library import SkillLibrary

            self._skill_library = SkillLibrary()
        return self._skill_library

    @staticmethod
    def _skill_state_dict(obs: dict) -> dict:
        out = dict(obs)
        for k, v in list(obs.items()):
            if isinstance(k, str) and k.startswith("phys_"):
                out.setdefault(k[5:], v)
        return out

    def _skill_goal_hint(self, st: dict) -> str:
        cz = float(st.get("com_z", st.get("phys_com_z", 0.5)))
        posture = float(st.get("posture_stability", st.get("phys_posture_stability", 0.5)))
        foot_l = float(st.get("foot_contact_l", st.get("phys_foot_contact_l", 0.5)))
        foot_r = float(st.get("foot_contact_r", st.get("phys_foot_contact_r", 0.5)))
        if cz < 0.36:
            return "stand"
        if posture < 0.68 or min(foot_l, foot_r) < 0.54:
            return "stand"
        try:
            walk_min = int(os.environ.get("RKK_CURRICULUM_WALK_MIN_TICK", "2000"))
        except ValueError:
            walk_min = 2000
        if (
            walk_min > 0
            and self.current_world == "humanoid"
            and not self._fixed_root_active
            and self.tick < walk_min
        ):
            return "stand"
        g = os.environ.get("RKK_SKILL_GOAL", "walk").strip().lower()
        return g if g else "walk"

    def _sim_env_intervene(
        self, var: str, val: float, *, count_intervention: bool
    ) -> dict:
        from engine.graph_constants import is_read_only_macro_var

        if is_read_only_macro_var(var):
            return dict(self.agent.env.observe())
        env = self.agent.env
        fn = getattr(env, "intervene", None)
        if not callable(fn):
            return {}
        try:
            return fn(var, val, count_intervention=count_intervention)
        except TypeError:
            return fn(var, val)

    @staticmethod
    def _skill_step_to_pairs(step) -> list[tuple[str, float]]:
        if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str):
            return [(str(step[0]), float(step[1]))]
        if isinstance(step, list):
            return [(str(a), float(b)) for a, b in step]
        return []

    def _execute_skill_frame(self) -> dict:
        from engine.graph_constants import is_read_only_macro_var

        pack = self._skill_exec
        if pack is None:
            return self.agent.step(engine_tick=self.tick)
        skill = pack["skill"]
        idx: int = pack["index"]
        obs_before_init: dict = pack["obs_before"]
        step = skill.action_sequence[idx]
        pairs = [
            (v, x)
            for v, x in self._skill_step_to_pairs(step)
            if not is_read_only_macro_var(v)
        ]
        var0, val0 = (pairs[0] if pairs else ("", 0.5))

        obs_before_env = dict(self.agent.env.observe())
        self.agent.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.agent.graph.snapshot_vec_dict()

        if not pairs:
            idx += 1
            done = idx >= len(skill.action_sequence)
            if done:
                obs = dict(self.agent.env.observe())
                st = self._skill_state_dict(obs)
                cz_a = float(st.get("com_z", st.get("phys_com_z", 0.5)))
                cz_b = float(
                    obs_before_init.get(
                        "com_z", obs_before_init.get("phys_com_z", 0.5)
                    )
                )
                self._ensure_skill_library().record_outcome(
                    skill, st, cz_a - cz_b
                )
                self._skill_exec = None
            else:
                self._skill_exec = {
                    "skill": skill,
                    "index": idx,
                    "obs_before": obs_before_init,
                }
            return {
                "blocked": False,
                "skipped": True,
                "hierarchy": "skill",
                "skill": skill.name,
                "skill_step": idx,
                "skill_done": done,
                "variable": "",
                "value": 0.5,
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        burst = len(pairs) > 1

        if not burst:
            var, val = pairs[0]
            check = self.agent.value_layer.check_action(
                variable=var,
                value=float(val),
                current_nodes=dict(self.agent.graph.nodes),
                graph=self.agent.graph,
                temporal=self.agent.temporal,
                current_phi=self.agent.phi_approx(),
                other_agents_phi=self.agent.other_agents_phi,
                engine_tick=self.tick,
                imagination_horizon=0,
            )
            if not check.allowed:
                return {
                    "blocked": True,
                    "blocked_count": 1,
                    "reason": check.reason.value,
                    "variable": var,
                    "value": float(val),
                    "updated_edges": [],
                    "compression_delta": 0.0,
                    "prediction_error": 0.0,
                    "cf_predicted": {},
                    "cf_observed": {},
                    "goal_planned": False,
                    "hierarchy": "skill",
                    "skill": skill.name,
                    "skill_step": idx,
                    "skill_done": False,
                }
            obs_after = self._sim_env_intervene(var, val, count_intervention=True)
        else:
            burst_fn = getattr(self.agent.env, "intervene_burst", None)
            if callable(burst_fn):
                obs_after = dict(burst_fn(pairs, count_intervention=True))
            else:
                obs_after = {}
                for var, val in pairs:
                    obs_after = self._sim_env_intervene(
                        var, val, count_intervention=False
                    )
                if not obs_after:
                    obs_after = dict(self.agent.env.observe())

        if not obs_after:
            obs_after = dict(self.agent.env.observe())
        st_after = self._skill_state_dict(obs_after)
        self._sync_motor_state(obs_after, source="skill", tick=self.tick)
        intents_log = {
            v: float(x) for v, x in pairs if str(v).startswith("intent_")
        }
        self._log_motor_command(
            source="skill",
            intents=intents_log if intents_log else None,
            obs=self._motor_obs_payload(obs_after),
        )
        self.agent.graph.apply_env_observation(obs_after)
        obs_after_full = self.agent.graph.snapshot_vec_dict()
        self.agent.graph.record_observation(obs_before_full)
        self.agent.graph.record_observation(obs_after_full)
        for var, val in pairs:
            if var in self.agent.graph.nodes:
                self.agent.graph.record_intervention(
                    var, float(val), obs_before_full, obs_after_full
                )
        self.agent.temporal.step(obs_after)

        idx += 1
        done = idx >= len(skill.action_sequence)
        if done:
            cz_a = float(st_after.get("com_z", st_after.get("phys_com_z", 0.5)))
            cz_b = float(
                obs_before_init.get(
                    "com_z", obs_before_init.get("phys_com_z", 0.5)
                )
            )
            reward = cz_a - cz_b
            self._ensure_skill_library().record_outcome(skill, st_after, reward)
            self._skill_exec = None
        else:
            self._skill_exec = {
                "skill": skill,
                "index": idx,
                "obs_before": obs_before_init,
            }

        return {
            "blocked": False,
            "skipped": True,
            "hierarchy": "skill",
            "skill": skill.name,
            "skill_step": idx,
            "skill_done": done,
            "variable": var0,
            "value": float(val0),
            "updated_edges": [],
            "compression_delta": 0.0,
            "prediction_error": 0.0,
            "cf_predicted": {},
            "cf_observed": {},
            "goal_planned": False,
        }

    def _run_agent_or_skill_step(self, engine_tick: int) -> dict:
        """L3 внутри agent.step; L2 skill — один шаг последовательности за тик."""
        if (
            self.current_world == "humanoid"
            and not self._fixed_root_active
            and self._curriculum_stabilize_until > 0
            and engine_tick <= self._curriculum_stabilize_until
        ):
            if self._skill_exec is not None:
                return self._execute_skill_frame()
            if self._skill_library_enabled():
                lib = self._ensure_skill_library()
                obs = self.agent.env.observe()
                obs_st = self._skill_state_dict(obs)
                sk = lib.select_skill(obs_st, "stand")
                if sk is not None:
                    self._skill_exec = {
                        "skill": sk,
                        "index": 0,
                        "obs_before": dict(obs_st),
                    }
                    return self._execute_skill_frame()
            return self.agent.step(engine_tick=engine_tick, enable_l3=False)

        # Humanoid: при нестабильной позе не даём EIG выбирать сырые суставы — только скиллы / stand.
        if self.current_world == "humanoid" and not self._fixed_root_active:
            obs = self.agent.env.observe()
            posture = float(
                obs.get(
                    "posture_stability", obs.get("phys_posture_stability", 0.5)
                )
            )
            if posture < 0.65:
                if self._skill_exec is not None:
                    return self._execute_skill_frame()
                if self._skill_library_enabled():
                    lib = self._ensure_skill_library()
                    obs_st = self._skill_state_dict(obs)
                    sk = lib.select_skill(obs_st, "stand")
                    if sk is not None:
                        self._skill_exec = {
                            "skill": sk,
                            "index": 0,
                            "obs_before": dict(obs_st),
                        }
                        return self._execute_skill_frame()
        if (
            self._skill_library_enabled()
            and self.current_world == "humanoid"
            and not self._fixed_root_active
        ):
            if self._skill_exec is not None:
                return self._execute_skill_frame()
            if self._skill_start_prob() > 0.0 and np.random.random() < self._skill_start_prob():
                obs = self.agent.env.observe()
                st = self._skill_state_dict(obs)
                goal = self._skill_goal_hint(st)
                sk = self._ensure_skill_library().select_skill(st, goal)
                if sk is not None:
                    self._skill_exec = {
                        "skill": sk,
                        "index": 0,
                        "obs_before": dict(st),
                    }
                    return self._execute_skill_frame()
        return self.agent.step(
            engine_tick=engine_tick,
            enable_l3=self._l3_planning_due(),
        )

    def _skill_snapshot(self) -> dict | None:
        if not self._skill_library_enabled():
            return None
        lib = self._skill_library
        out: dict = {"enabled": True, "active": None}
        if lib is not None:
            out.update(lib.snapshot())
        else:
            out["n_skills"] = 0
            out["skills"] = []
            out["history_len"] = 0
        if self._skill_exec is not None:
            sk = self._skill_exec["skill"]
            out["active"] = {
                "name": sk.name,
                "step": self._skill_exec["index"],
                "total": len(sk.action_sequence),
            }
        return out

    def _pose_snapshot(self) -> "PoseSnapshot | None":
        """Level 1-A: Construct PoseSnapshot from current environment state."""
        if not _EMBODIED_REWARD_AVAILABLE:
            return None
        try:
            env = self.agent.env
            obs = dict(env.observe())
            nodes = dict(self.agent.graph.nodes)
            posture_window = (
                list(self._mc_posture_window)
                if hasattr(self, "_mc_posture_window")
                else []
            )
            fallen_window = (
                list(self._mc_fallen_count_window)
                if hasattr(self, "_mc_fallen_count_window")
                else []
            )
            recent_fall_rate = float(np.mean(fallen_window)) if fallen_window else 0.0
            mean_posture = float(np.mean(posture_window)) if posture_window else 0.5
            mc = self._ensure_motor_cortex() if hasattr(self, "_ensure_motor_cortex") else None
            cpg_w = mc.cpg_weight if mc is not None else 1.0
            mc_q = mc._quality_ema if mc is not None else 0.0
            return PoseSnapshot.from_obs_and_graph(
                obs=obs,
                graph_nodes=nodes,
                tick=self.tick,
                fallen=self._fall_count > 0,
                fall_count=self._fall_count,
                cpg_weight=cpg_w,
                mc_quality_ema=mc_q,
                recent_fall_rate=recent_fall_rate,
                mean_posture_recent=mean_posture,
            )
        except Exception as e:
            print(f"[Simulation] _pose_snapshot error: {e}")
            return None

    async def _run_embodied_reward_async(self) -> None:
        """Level 1-A: Run embodied LLM reward shaping (async, called as task)."""
        if not _EMBODIED_REWARD_AVAILABLE or self._embodied_reward_ctrl is None:
            return
        pose = self._pose_snapshot()
        if pose is None:
            return
        try:
            mc = self._ensure_motor_cortex() if hasattr(self, "_ensure_motor_cortex") else None
            result = await self._embodied_reward_ctrl.run(
                pose=pose,
                agent=self.agent,
                locomotion_ctrl=self._locomotion_controller,
                motor_cortex=mc,
                llm_url=get_ollama_generate_url(),
                llm_model=get_ollama_model(),
            )
            if result.ok and (result.verbal or result.priority_issue):
                issue_str = f" [{result.priority_issue}]" if result.priority_issue else ""
                self._add_event(
                    f"🧠 EmbodiedLLM: r={result.combined_reward:+.2f}"
                    f" pos={result.posture_score:.2f} gait={result.gait_quality:.2f}"
                    f"{issue_str} +{len(result.seeds)}seeds",
                    "#ff99ff",
                    "phase",
                )
        except Exception as e:
            print(f"[Simulation] embodied reward error: {e}")

    def _maybe_run_visual_grounding(self) -> None:
        """Level 1-B: Update visual-body grounding (slot → joint mapping)."""
        if not _VISUAL_GROUNDING_AVAILABLE or self._visual_grounding_ctrl is None:
            return
        if not self._visual_mode or self._visual_env is None:
            return
        if not self._visual_grounding_ctrl.should_run(self.tick):
            return

        # Get PyBullet state
        base_env = self._base_env_ref
        physics_client, robot_id, link_names = None, None, []
        if base_env is not None:
            physics_client, robot_id, link_names = get_pybullet_state_from_humanoid_env(
                base_env
            )

        result = self._visual_grounding_ctrl.update(
            tick=self.tick,
            visual_env=self._visual_env,
            agent_graph=self.agent.graph,
            physics_client=physics_client,
            robot_id=robot_id,
            link_names=link_names,
        )

        if result.get("ok") and result.get("edges_injected", 0) > 0:
            slot_map = result.get("slot_to_joint", {})
            if slot_map:
                mapping_str = ", ".join(
                    f"{k}→{v}" for k, v in list(slot_map.items())[:4]
                )
                self._add_event(
                    f"👁 Grounding: +{result['edges_injected']} edges [{mapping_str}]",
                    "#44ffcc",
                    "phase",
                )

    def _record_last_action(self, result: dict) -> None:
        """Level 2-D: Track last action for episodic memory."""
        if not result.get("blocked") and not result.get("skipped"):
            var = result.get("variable")
            val = result.get("value")
            if var is not None and val is not None:
                self._last_action_for_memory = (str(var), float(val))

    def _update_episodic_memory(
        self, tick: int, obs: dict, fallen: bool, posture: float
    ) -> None:
        """Level 2-D: Update episodic memory with current state."""
        if not _EPISODIC_MEMORY_AVAILABLE or self._episodic_memory is None:
            return
        if not episode_memory_enabled():
            return
        if self.current_world != "humanoid" or self._fixed_root_active:
            return

        self._episodic_memory.tick_update(
            tick=tick,
            obs=obs,
            last_action=self._last_action_for_memory,
            fallen=fallen,
            posture=posture,
        )

        if fallen and (tick - self._last_fall_memory_tick) > 5:
            env = self.agent.env
            intents = {}
            try:
                obs_now = dict(env.observe())
                intents = {
                    k: float(
                        obs_now.get(k, obs_now.get(f"phys_{k}", 0.5))
                    )
                    for k in [
                        "intent_stride",
                        "intent_torso_forward",
                        "intent_support_left",
                        "intent_support_right",
                        "intent_stop_recover",
                        "intent_gait_coupling",
                    ]
                }
            except Exception:
                pass
            ep = self._episodic_memory.on_fall(tick, obs, intents)
            if ep is not None:
                self._last_fall_memory_tick = tick
                seeds = self._episodic_memory.get_seeds_from_patterns(
                    set(self.agent.graph.nodes.keys())
                )
                if seeds:
                    self.agent.inject_text_priors(seeds)

    def _tick_curriculum(self, tick: int, obs: dict, fallen: bool) -> None:
        """Level 2-E: Update curriculum scheduler."""
        if not _CURRICULUM_AVAILABLE or self._curriculum is None:
            return
        if not curriculum_enabled():
            return
        if self.current_world != "humanoid" or self._fixed_root_active:
            return

        fall_rate = (
            float(np.mean(self._mc_fallen_count_window))
            if hasattr(self, "_mc_fallen_count_window")
            and self._mc_fallen_count_window
            else 0.0
        )

        stage, advanced = self._curriculum.tick(tick, obs, fallen, fall_rate)

        if advanced:
            injected = self._curriculum.inject_stage_seeds(self.agent)
            self._add_event(
                f"📚 Curriculum → '{stage.name}': {stage.description[:60]} "
                f"(+{injected} seeds)",
                "#aaffaa",
                "phase",
            )

        if tick % self._curriculum_apply_every == 0:
            self._curriculum.apply_stage_intents(self.agent.env)

        if (
            tick % 300 == 0
            and self._curriculum._current_idx >= len(self._curriculum._stages) - 2
        ):
            fall_summary = ""
            if self._episodic_memory is not None:
                fall_summary = self._episodic_memory.get_llm_context_block(
                    max_falls=3
                )
            skill_stats = self._skill_snapshot() or {}
            pose_metrics: dict = {}
            try:
                cur_obs = dict(self.agent.env.observe())
                pose_metrics = self._curriculum.compute_metrics(cur_obs)
            except Exception:
                pass
            valid_intent = [k for k in self.agent.graph.nodes if k.startswith("intent_")]
            valid_graph = list(self.agent.graph.nodes.keys())

            def _run_curriculum_llm() -> None:
                import asyncio

                asyncio.run(
                    self._curriculum.maybe_generate_next_stage_llm(
                        tick=tick,
                        skill_stats=skill_stats,
                        fall_summary=fall_summary,
                        pose_metrics=pose_metrics,
                        valid_intent_vars=valid_intent,
                        valid_graph_vars=valid_graph,
                        llm_url=get_ollama_generate_url(),
                        llm_model=get_ollama_model(),
                    )
                )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            try:
                if loop is not None:
                    try:
                        loop.create_task(
                            self._curriculum.maybe_generate_next_stage_llm(
                                tick=tick,
                                skill_stats=skill_stats,
                                fall_summary=fall_summary,
                                pose_metrics=pose_metrics,
                                valid_intent_vars=valid_intent,
                                valid_graph_vars=valid_graph,
                                llm_url=get_ollama_generate_url(),
                                llm_model=get_ollama_model(),
                            )
                        )
                    except RuntimeError:
                        self._llm_loop_executor.submit(_run_curriculum_llm)
                else:
                    self._llm_loop_executor.submit(_run_curriculum_llm)
            except Exception as e:
                print(f"[Simulation] curriculum LLM error: {e}")

    def _maybe_upgrade_rssm(self, tick: int) -> None:
        """Level 2-F: Upgrade GNN to RSSM after sufficient GNN training."""
        if not _RSSM_AVAILABLE or self._rssm_upgraded:
            return
        if not rssm_enabled():
            return
        if self.current_world != "humanoid":
            return
        try:
            min_tick = max(500, int(os.environ.get("RKK_WM_RSSM_UPGRADE_TICK", "500")))
        except ValueError:
            min_tick = 500
        if tick < min_tick:
            return
        upgraded, trainer = maybe_upgrade_graph_to_rssm(self.agent.graph, self.device)
        if upgraded and trainer is None:
            self._rssm_upgraded = True
            return
        if upgraded and trainer is not None:
            self._rssm_trainer = trainer
            self._rssm_imagination = RSSMImagination(
                self.agent.graph._core, self.device
            )
            self._rssm_upgraded = True
            self._rssm_upgrade_tick = tick
            try:
                h = int(os.environ.get("RKK_WM_RSSM_IMAGINATION", "12"))
            except ValueError:
                h = 12
            self.agent._imagination_horizon = h
            self._add_event(
                f"🔮 RSSM-lite activated at tick={tick} "
                f"(horizon={self.agent._imagination_horizon})",
                "#88aaff",
                "phase",
            )

    def _rssm_train_step(
        self,
        obs_before: dict[str, float],
        action_var: str,
        action_val: float,
        obs_after: dict[str, float],
    ) -> None:
        """Level 2-F: Push transition to RSSM trainer."""
        if self._rssm_trainer is None or not self._rssm_upgraded:
            return
        try:
            node_ids = list(self.agent.graph._node_ids)
            X_t = [float(obs_before.get(n, 0.0)) for n in node_ids]
            a_t = [float(action_val) if n == action_var else 0.0 for n in node_ids]
            X_tp1 = [float(obs_after.get(n, 0.0)) for n in node_ids]
            self._rssm_trainer.push(X_t, a_t, X_tp1)
            if self.tick % 8 == 0:
                self._rssm_trainer.maybe_train()
        except Exception:
            pass

    # ── Tick ──────────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        hz = _agent_loop_hz_from_env()
        if hz > 0.0:
            self._ensure_rkk_agent_loop_thread()
            with self._sim_step_lock:
                cached = self._agent_step_response
            if cached is not None:
                return copy.deepcopy(cached)
            return self.public_state()
        with self._sim_step_lock:
            return self._run_single_agent_timestep_inner()

    def advance_agent_steps(self, n: int) -> None:
        """Синхронно выполнить n логических тиков агента (bootstrap при RKK_AGENT_LOOP_HZ>0)."""
        n = max(0, int(n))
        if n == 0:
            return
        with self._sim_step_lock:
            for _ in range(n):
                self._agent_step_response = self._run_single_agent_timestep_inner()

    def _ensure_rkk_agent_loop_thread(self) -> None:
        if _agent_loop_hz_from_env() <= 0.0:
            return
        if self._agent_loop_thread is not None and self._agent_loop_thread.is_alive():
            return
        self._agent_stop.clear()
        self._agent_loop_thread = threading.Thread(
            target=self._rkk_agent_loop_worker,
            daemon=True,
            name="rkk-agent-loop",
        )
        self._agent_loop_thread.start()
        print(
            f"[Simulation] Agent high-level loop ~{_agent_loop_hz_from_env():.1f} Hz "
            f"(RKK_AGENT_LOOP_HZ; HTTP/WS tick_step → кэш)"
        )

    def _stop_rkk_agent_loop_thread(self) -> None:
        self._agent_stop.set()
        th = self._agent_loop_thread
        if th is not None and th.is_alive():
            th.join(timeout=2.5)
        self._agent_loop_thread = None
        self._agent_stop.clear()
        self._agent_step_response = None

    def _rkk_agent_loop_worker(self) -> None:
        hz = _agent_loop_hz_from_env()
        dt = 1.0 / hz if hz > 0 else 0.1
        while not self._agent_stop.is_set():
            t0 = time.perf_counter()
            try:
                with self._sim_step_lock:
                    self._agent_step_response = self._run_single_agent_timestep_inner()
            except Exception as e:
                print(f"[Simulation] Agent loop: {e}")
            elapsed = time.perf_counter() - t0
            self._agent_stop.wait(timeout=max(0.0, dt - elapsed))

    def _run_single_agent_timestep_inner(self) -> dict:
        self.tick += 1
        self._apply_pending_llm_bundle()
        self._ensure_phase2()

        # Humanoid curriculum: фаза 1 — fixed_root с тика 1; снятие после RKK_AUTO_FIXED_ROOT_TICKS.
        try:
            auto_fr_ticks = int(os.environ.get("RKK_AUTO_FIXED_ROOT_TICKS", "0"))
        except ValueError:
            auto_fr_ticks = 0
        if auto_fr_ticks > 0 and self.current_world == "humanoid":
            if self.tick == 1 and not self._fixed_root_active:
                self.enable_fixed_root()
                self._add_event(
                    "📌 Curriculum: fixed_root ON (phase 1, arms→cubes)",
                    "#66ccaa",
                    "phase",
                )
            if (
                self._fixed_root_active
                and self.tick >= auto_fr_ticks
                and not self._curriculum_auto_fr_released
            ):
                self._curriculum_auto_fr_released = True
                self.disable_fixed_root()
                try:
                    stab = int(os.environ.get("RKK_POST_FR_STABILIZE_TICKS", "80"))
                except ValueError:
                    stab = 80
                self._curriculum_stabilize_until = self.tick + max(0, stab)
                self._add_event(
                    f"📌 Auto fixed_root OFF at tick {self.tick}, stabilize until {self._curriculum_stabilize_until}",
                    "#66ccaa",
                    "phase",
                )

        # Fallen check + автосброс физики (иначе VL и block_rate залипают)
        fallen = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            fallen = is_fn()
            if self._fall_recovery_active and not fallen:
                self._clear_fall_recovery()
            if fallen:
                self._fall_count += 1
                obs_fall = dict(self.agent.env.observe())
                if self._maybe_recover_or_reset_after_fall(obs_fall):
                    obs = self.agent.env.observe()
                    self._sync_motor_state(obs, source="reset", tick=self.tick)
                    for nid in self.agent.graph._node_ids:
                        if nid in obs:
                            self.agent.graph.nodes[nid] = obs[nid]
                    self.agent.graph.record_observation(obs)
                    self.agent.temporal.step(obs)
                    fallen = is_fn()
                if self._fall_count % 20 == 1:
                    self._add_event(
                        f"💀 [FALLEN] Nova упал! (×{self._fall_count})",
                        "#ff2244", "value"
                    )

        # Фаза 12: передаём GNN prediction в visual env (не каждый тик — см. VISION_GNN_FEED_EVERY)
        if self._visual_mode and self._visual_env is not None:
            self._vision_ticks += 1
            if self._vision_ticks % VISION_GNN_FEED_EVERY == 0:
                self._feed_gnn_prediction_to_visual()

        # Фаза 3: annealing teacher_weight; VL-overlay только пока не истёк TTL и weight>0
        try:
            tmax = int(os.environ.get("RKK_TEACHER_T_MAX", "140"))
        except ValueError:
            tmax = 140
        tmax = max(1, tmax)
        tw = max(0.0, 1.0 - (self.agent._total_interventions / tmax))
        self.agent.set_teacher_state(self._phase3_teacher_rules, tw)
        ov = self._phase3_vl_overlay
        if ov is not None and self.tick <= ov.expires_at_tick and tw > 0:
            self.agent.value_layer.set_teacher_vl_overlay(ov)
        else:
            self.agent.value_layer.set_teacher_vl_overlay(None)

        # CPG runs BEFORE agent step so legs are stabilized before high-level exploration
        self._ensure_cpg_background_loop()
        self._drain_l1_motor_commands()
        fallen_pre = False
        is_fn_pre = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn_pre) and not self._fixed_root_active:
            fallen_pre = is_fn_pre()
        self._maybe_apply_cpg_locomotion(fallen_pre)
        self._publish_cpg_node_snapshot()
        self.agent.other_agents_phi = []
        self._maybe_step_hierarchical_l1()
        obs_pre_rssm = dict(self.agent.graph.snapshot_vec_dict())
        result = self._run_agent_or_skill_step(engine_tick=self.tick)

        # Track action for episodic memory
        self._record_last_action(result)

        _obs_for_d_e: dict = {}
        try:
            _obs_for_d_e = dict(self.agent.env.observe())
        except Exception:
            pass
        _posture_now = float(
            _obs_for_d_e.get(
                "posture_stability",
                _obs_for_d_e.get("phys_posture_stability", 0.5),
            )
        )

        # Level 3-I: Multi-scale time tick (first consumer of post-step obs)
        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            self._timescale.tick(self.tick, _obs_for_d_e)
            motor_intents = self._timescale.get_intents(LEVEL_MOTOR)
            for var, val in motor_intents.items():
                if var.startswith("intent_"):
                    try:
                        self.agent.env.intervene(var, float(val), count_intervention=False)
                    except Exception:
                        pass

        # Level 3-G: Proprioception update (after CPG + agent step; fresh obs)
        _proprio_anomaly = 0.0
        _proprio_emp_reward = 0.0
        if _PROPRIO_AVAILABLE and self._proprio is not None and self.current_world == "humanoid":
            self._proprio.update(
                tick=self.tick,
                obs=_obs_for_d_e,
                graph=self.agent.graph if hasattr(self.agent, "graph") else None,
                agent=self.agent,
            )
            _proprio_anomaly = self._proprio.anomaly_score
            _proprio_emp_reward = self._proprio.get_empowerment_reward()

            if _TIMESCALE_AVAILABLE and self._timescale is not None:
                if self._timescale.should_run(LEVEL_REFLEX, self.tick):
                    self._timescale.mark_ran(LEVEL_REFLEX, self.tick)

        # Level 3-H: Unified reward signal (humanoid)
        _reward_signal = None
        if _REWARD_COORD_AVAILABLE and self.current_world == "humanoid":
            self._ensure_reward_coord()
            if self._reward_coord is not None:
                node_ids = (
                    list(self.agent.graph._node_ids)
                    if hasattr(self.agent.graph, "_node_ids")
                    else []
                )
                X_now = [float(_obs_for_d_e.get(n, 0.0)) for n in node_ids]
                a_vec = [
                    float(self._reward_action_val) if n == self._reward_action_var else 0.0
                    for n in node_ids
                ]

                _task_r = 0.0
                _task_src = "heuristic"
                if _EMBODIED_REWARD_AVAILABLE and self._embodied_reward_ctrl is not None:
                    emb_sn = self._embodied_reward_ctrl.snapshot()
                    lr = emb_sn.get("last_result")
                    if isinstance(lr, dict):
                        _task_r = float(lr.get("combined_reward", 0.0))
                        err = str(lr.get("error") or "").strip()
                        _task_src = "llm" if lr.get("ok") and not err else "heuristic"

                if self._reward_action_var:
                    self._reward_coord.record_action(
                        self._reward_action_var, self._reward_action_val
                    )

                _h_W = 0.0
                try:
                    _h_W = float(self.agent.graph._core.dag_constraint().item())
                except Exception:
                    pass

                _reward_signal = self._reward_coord.compute(
                    tick=self.tick,
                    obs=_obs_for_d_e,
                    X_t=self._reward_X_prev if self._reward_X_prev else X_now,
                    a_t=self._reward_a_prev if self._reward_a_prev else a_vec,
                    X_tp1=X_now,
                    fallen=fallen,
                    anomaly=_proprio_anomaly,
                    empowerment_reward=_proprio_emp_reward,
                    task_reward=_task_r,
                    task_source=_task_src,
                    h_W=_h_W,
                    llm_url=get_ollama_generate_url(),
                    llm_model=get_ollama_model(),
                )

                self._reward_X_prev = X_now
                self._reward_a_prev = a_vec

                self._reward_coord.apply_to_learners(
                    signal=_reward_signal,
                    locomotion_ctrl=self._locomotion_controller,
                    motor_cortex=getattr(self, "_motor_cortex", None),
                    agent=self.agent,
                )

                if _reward_signal.constitution < 0.5 and _reward_signal.constitution_warning:
                    self._add_event(
                        f"⚠️ Constitution: {_reward_signal.constitution_warning[:60]}",
                        "#ffaa44",
                        "constitution",
                    )

                if _reward_signal.blocked and not self._was_blocked:
                    self._add_event(
                        f"🛑 Survival veto: {_reward_signal.survival_reason}",
                        "#ff4444",
                        "veto",
                    )
                self._was_blocked = bool(_reward_signal.blocked)

                if _TIMESCALE_AVAILABLE and self._timescale is not None:
                    if _reward_signal.curiosity > 0.6:
                        self._timescale.set_intent(LEVEL_COGNIT, "causal_eig", 0.8)

        if not result.get("blocked") and not result.get("skipped"):
            _var_now = result.get("variable", "")
            _val_now = result.get("value", 0.5)
            if _var_now:
                self._reward_action_var = str(_var_now)
                try:
                    self._reward_action_val = float(_val_now)
                except (TypeError, ValueError):
                    self._reward_action_val = 0.5

        if _TIMESCALE_AVAILABLE and self._timescale is not None:
            if self._timescale.should_run(LEVEL_MOTOR, self.tick):
                self._timescale.mark_ran(LEVEL_MOTOR, self.tick)
            if self._timescale.should_run(LEVEL_COGNIT, self.tick):
                self._timescale.mark_ran(LEVEL_COGNIT, self.tick)

        # Level 2-D: Episodic Memory
        self._update_episodic_memory(self.tick, _obs_for_d_e, fallen, _posture_now)

        # Level 2-E: Curriculum
        self._tick_curriculum(self.tick, _obs_for_d_e, fallen)

        # Level 2-F: RSSM upgrade + training
        self._maybe_upgrade_rssm(self.tick)
        if not result.get("blocked") and not result.get("skipped"):
            _var = str(result.get("variable", ""))
            _val = float(result.get("value", 0.5))
            obs_post = dict(self.agent.graph.snapshot_vec_dict())
            self._rssm_train_step(obs_pre_rssm, _var, _val, obs_post)

        # Фаза 2 ч.3: L4 concept mining (sync fallback или async worker + single-writer apply)
        if self._visual_env is not None and self.tick % self._concept_inject_every == 0:
            vis = self._visual_env.get_slot_visualization()
            slot_vecs = self._visual_env._last_slot_vecs
            if slot_vecs is not None:
                full_obs = dict(self._visual_env.observe())
                phys_obs = {
                    k: float(v)
                    for k, v in full_obs.items()
                    if not str(k).startswith("slot_")
                }
                if _l4_worker_enabled():
                    self._enqueue_l4_task(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                    )
                elif self._concept_store is not None:
                    new_concepts = self._concept_store.update(
                        slot_vecs=slot_vecs,
                        slot_values=vis.get("slot_values", []),
                        variability=vis.get("variability", []),
                        phys_obs=phys_obs,
                        tick=self.tick,
                        graph_node_ids=list(self.agent.graph._node_ids),
                    )
                    if new_concepts:
                        added = self._concept_store.inject_into_graph(self.agent.graph)
                        c0 = new_concepts[0]
                        self._add_event(
                            f"Concept formed: {c0.cid[:4]}, slot_{c0.slot_idx}, +{added} nodes",
                            "#EF9F27",
                            "phase",
                        )
        if _l4_worker_enabled():
            self._drain_l4_results()

        self._log_step(result, fallen)
        self._rolling_block_bits.append(1 if result.get("blocked") else 0)

        snap = self.agent.snapshot()
        snap["fallen"]     = fallen
        snap["fall_count"] = self._fall_count
        self._last_snapshot = snap

        if self._rsi_full_enabled():
            from engine.rsi_full import RSIController

            if self._rsi_full is None:
                sup = (
                    self._ensure_skill_library
                    if self._skill_library_enabled()
                    else None
                )
                self._rsi_full = RSIController(
                    self.agent,
                    self._locomotion_controller,
                    skill_library_supplier=sup,
                    motor_cortex_supplier=self._ensure_motor_cortex,
                )
            rsi_ev = self._rsi_full.tick(
                snap,
                self._locomotion_reward_ema(),
                tick=self.tick,
                locomotion_ctrl=self._locomotion_controller,
            )
            if rsi_ev is not None:
                t = rsi_ev.get("type", "?")
                self._add_event(f"🔧 RSI [{t}]", "#66ccaa", "phase")

        # Phase D: Motor Cortex RSI check (every 50 ticks)
        if self.tick % 50 == 0:
            mc = self._ensure_motor_cortex()
            if mc is not None:
                posture_mean = (
                    float(np.mean(self._mc_posture_window))
                    if self._mc_posture_window else 0.0
                )
                fallen_rate = (
                    float(np.mean(self._mc_fallen_count_window))
                    if self._mc_fallen_count_window else 0.0
                )
                loco_r = self._locomotion_reward_ema()
                new_progs = mc.rsi_check_and_spawn(
                    self.tick, posture_mean, loco_r, fallen_rate
                )
                for prog_name in new_progs:
                    self._add_event(
                        f"🧠 MC-RSI: spawned '{prog_name}' "
                        f"(posture={posture_mean:.2f}, cpg_w={mc.cpg_weight:.2f})",
                        "#ff88ff", "phase"
                    )

        dr = float(snap.get("discovery_rate", 0.0))
        self._tick_discovery_plateau(dr)
        if dr > self._best_discovery_rate + 1e-5:
            self._best_discovery_rate = dr
            self._last_dr_gain_tick = self.tick

        # Level 1-A: Embodied LLM Reward Shaping (async task)
        if (
            _EMBODIED_REWARD_AVAILABLE
            and self._embodied_reward_ctrl is not None
            and self.current_world == "humanoid"
            and not self._fixed_root_active
            and embodied_reward_enabled()
            and self._embodied_reward_ctrl.should_run(self.tick)
        ):
            import asyncio

            def _run_embodied_in_thread() -> None:
                asyncio.run(self._run_embodied_reward_async())

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            try:
                if loop is not None:
                    try:
                        loop.create_task(self._run_embodied_reward_async())
                    except RuntimeError:
                        self._llm_loop_executor.submit(_run_embodied_in_thread)
                else:
                    # rkk-agent-loop и др.: в потоке нет loop — не блокируем тик
                    self._llm_loop_executor.submit(_run_embodied_in_thread)
            except Exception as e:
                print(f"[Simulation] embodied reward schedule error: {e}")

        # Level 1-B: Visual Body Grounding
        self._maybe_run_visual_grounding()

        # Level 1-C: Standalone reconstruction training (warm up decoder early)
        if (
            self._visual_mode
            and self._visual_env is not None
            and self.tick % 5 == 0
            and hasattr(self._visual_env, "cortex")
        ):
            cortex = self._visual_env.cortex
            if (
                hasattr(cortex, "train_reconstruction_only")
                and hasattr(self._visual_env, "_last_frame")
                and self._visual_env._last_frame is not None
                and cortex.n_train == 0  # only during warmup phase
            ):
                try:
                    cortex.train_reconstruction_only(self._visual_env._last_frame)
                except Exception:
                    pass

        # Demon
        if self.demon._last_action is not None:
            pe = 0.0
            if not result.get("blocked") and not result.get("skipped"):
                pe = float(result.get("prediction_error", 0))
            self.demon.learn(pe, self.demon._last_action_complexity, [snap])
        self._step_demon(snap)

        smoothed = self._update_phase(snap)

        graph_deltas = {}
        cnt = len(self.agent.graph.edges)
        if cnt != self._prev_edge_count:
            graph_deltas[0] = [e.as_dict() for e in self.agent.graph.edges]
            self._prev_edge_count = cnt

        # Neurogenesis
        # Structural ASI: Neurogenesis
        if self.current_world == "humanoid" and not self._fixed_root_active:
            neuro_event = self.neuro_engine.scan_and_grow(self.agent, self.tick)
            if neuro_event is not None:
                self._add_event(
                    f"🧬 Neurogenesis: {neuro_event['new_node']} allocated", 
                    "#ff44cc", 
                    "phase"
                )
                
                # Обновляем TemporalBlankets под новую размерность (d = d + 1)
                from engine.temporal import TemporalBlankets
                new_d = self.agent.graph._d
                old_tb = self.agent.temporal
                new_tb = TemporalBlankets(d_input=new_d, device=self.device)
                # Копируем состояния TemporalBlankets (насколько возможно)
                # ... (миграция состояния SSM, аналогично resize_to в GNN)
                self.agent.temporal = new_tb

        # Scene
        scene_fn = getattr(self.agent.env, "get_full_scene", None)
        scene    = scene_fn() if callable(scene_fn) else {}

        # Vision state (кэш для /vision/slots endpoint)
        if self._visual_mode and self._visual_env is not None:
            try:
                self._last_vision_state = self._visual_env.get_slot_visualization()
            except Exception:
                pass

        self._maybe_schedule_llm_loop(result, snap)

        self._maybe_refresh_concepts_cache()
        self._maybe_autosave_memory()

        return self._build_snapshot(snap, graph_deltas, smoothed, scene)

    def _feed_gnn_prediction_to_visual(self):
        """Передаём текущий GNN-прогноз в visual env для predictive coding."""
        if self._visual_env is None or self.agent.graph._core is None:
            return
        try:
            current_obs = self._visual_env.observe()
            node_ids    = self.agent.graph._node_ids
            slot_ids    = [f"slot_{k}" for k in range(self._visual_env.n_slots)]
            values_list = [current_obs.get(sid, 0.5) for sid in slot_ids]
            current_t   = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            # Прогоняем текущие значения через GNN (опционально Neural ODE sub-steps)
            with torch.inference_mode():
                full_state = torch.tensor(
                    [self.agent.graph.nodes.get(n, 0.5) for n in node_ids],
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                z = torch.zeros_like(full_state)
                pred_full = integrate_world_model_step(
                    self.agent.graph._core, full_state, z
                ).squeeze(0)
            # Выбираем только slot_ переменные
            slot_pred = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            for i, sid in enumerate(slot_ids):
                if sid in node_ids:
                    idx = node_ids.index(sid)
                    if idx < len(pred_full):
                        p = float(pred_full[idx].item())
                        slot_pred[i] = min(0.95, max(0.05, p))
            self._visual_env.set_gnn_prediction(slot_pred)
        except Exception:
            pass

    def _step_demon(self, snap: dict):
        try:
            action = self.demon.step([snap], 1 - snap.get("peak_discovery_rate", 0))
        except RuntimeError as e:
            print(f"[Singleton] Demon step skipped: {e}")
            return
        if action is None:
            return
        corrupted = self.agent.demon_disrupt()
        self._add_event(
            f"⚠ Demon [{action.get('mode','?')}] → Nova: {corrupted}",
            "#ff2244", "demon"
        )

    def _update_phase(self, snap: dict) -> float:
        dr = snap.get("discovery_rate", 0)
        self._dr_window.append(dr)
        smoothed = float(np.mean(self._dr_window))

        potential = 1
        for i, t in enumerate(PHASE_THRESHOLDS):
            if smoothed >= t:
                potential = i + 1
        potential = min(potential, 5)

        if potential > self.max_phase:
            if potential == self._candidate_phase:
                self._phase_hold_counter += 1
            else:
                self._candidate_phase    = potential
                self._phase_hold_counter = 1
            if self._phase_hold_counter >= PHASE_HOLD_TICKS:
                self.max_phase = potential
                self.phase     = potential
                self._phase_hold_counter = 0
                self._add_event(f"⬆ Phase {potential}: {PHASE_NAMES[potential]}", "#ffcc00", "phase")
        else:
            self._candidate_phase    = self.max_phase
            self._phase_hold_counter = 0
        self.phase = self.max_phase
        return smoothed

    def _log_step(self, result: dict, fallen: bool):
        if result.get("blocked"):
            self._add_event(
                f"🛡 [BLOCKED] Nova: {result.get('reason','?')}",
                "#884400", "value"
            )
        elif result.get("hierarchy") == "skill":
            sk = result.get("skill", "?")
            var = result.get("variable", "?")
            val = result.get("value", 0)
            done = result.get("skill_done", False)
            self._add_event(
                f"🦿 Skill [{sk}] do({var}={float(val):.2f})"
                f"{' ✓' if done else ' …'}",
                "#66ccff",
                "skill",
            )
        elif result.get("updated_edges"):
            cg  = result.get("compression_delta", 0)
            var = result.get("variable", "?")
            val = result.get("value", 0)
            color = WORLDS.get(self.current_world, {}).get("color", "#cc44ff")
            if self._visual_mode:
                color = "#44ffcc"
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                val_f = 0.0
            try:
                cg_f = float(cg)
            except (TypeError, ValueError):
                cg_f = 0.0
            self._add_event(
                f"Nova: do({var}={val_f:.2f}) CG{'+' if cg_f >= 0 else ''}{cg_f:.3f}",
                color, "discovery"
            )

    def _add_event(self, text: str, color: str, type_: str):
        self.events.appendleft({"tick": self.tick, "text": text, "color": color, "type": type_})

    # ── Camera / Scene ────────────────────────────────────────────────────────
    def get_camera_frame(self, view: str | None = None) -> str | None:
        fn = getattr(self.agent.env, "get_frame_base64", None)
        return fn(view) if callable(fn) else None

    def get_vision_state(self) -> dict:
        """Данные для /vision/slots endpoint (свежий снимок, в т.ч. slot_labels Фазы 2)."""
        if not self._visual_mode or self._visual_env is None:
            return {"visual_mode": False}
        try:
            state = self._visual_env.get_slot_visualization()
        except Exception:
            state = dict(self._last_vision_state)
        state["visual_mode"] = True
        state["n_slots"] = self._visual_env.n_slots
        state["vision_ticks"] = self._vision_ticks
        state["cortex"] = self._visual_env.cortex.snapshot()
        return state

    async def vlm_label_slots(
        self,
        llm_url: str,
        llm_model: str,
        max_mask_images: int = 4,
        text_only: bool = False,
        inject_weak_edges: bool = False,
    ) -> dict:
        """
        Фаза 2: один вызов VLM (или текстовый fallback) → лексикон на EnvironmentVisual.
        Опционально слабые рёбра slot→phys при inject_weak_edges и confidence.

        Идея на будущее (без авто-повтора здесь): вызывать из tick_step при «запутался»
        — например серия fallen, высокий block_rate, длительная стагнация discovery;
        с дебаунсом и env RKK_VLM_ON_CONFUSION=1.
        """
        if not self._visual_mode or self._visual_env is None:
            return {"ok": False, "error": "visual mode off"}

        from engine.slot_lexicon import (
            run_slot_vlm_labeling,
            weak_slot_to_phys_edges,
        )

        vis = self._visual_env
        if vis._last_slots is None or vis._cached_frame_b64 is None:
            vis._refresh(run_encode=True)
            if vis._cached_frame_b64 is None:
                return {
                    "ok": False,
                    "error": "camera frame not available yet — retry after a few ticks",
                }

        snap = vis.get_slot_visualization()
        var_ids = list(self.agent.graph.nodes.keys())

        labels, mode, err = await run_slot_vlm_labeling(
            frame_b64=snap.get("frame"),
            masks_b64=list(snap.get("masks") or []),
            slot_values=list(snap.get("slot_values") or []),
            variability=list(snap.get("variability") or []),
            n_slots=vis.n_slots,
            variable_ids=var_ids,
            llm_url=llm_url,
            llm_model=llm_model,
            max_mask_images=max_mask_images,
            text_only=text_only,
        )

        if not labels:
            return {
                "ok": False,
                "mode": mode,
                "error": err or "empty labels",
            }

        vis.set_slot_lexicon(labels, self.tick, snap.get("frame"))

        injected = 0
        skipped: list[str] = []
        if inject_weak_edges:
            edges = weak_slot_to_phys_edges(labels)
            if edges:
                with self._sim_step_lock:
                    r = self.agent.inject_text_priors(edges)
                    injected = int(r.get("injected", 0))
                    skipped = list(r.get("skipped") or [])

        self._add_event(
            f"🔬 VLM slots: {mode}, {len(labels)} labels"
            + (f", +{injected} weak edges" if inject_weak_edges else ""),
            "#44ccff",
            "phase",
        )

        return {
            "ok": True,
            "mode": mode,
            "n_slots_labeled": len(labels),
            "warning": err,
            "slot_lexicon_tick": self.tick,
            "weak_edges_injected": injected,
            "weak_edges_skipped": skipped,
        }

    async def refresh_phase3_teacher_llm(self) -> dict:
        """
        Фаза 3: один вызов Ollama → правила IG-бонуса для System1 + TTL-дельты Value Layer.
        """
        from engine.phase3_teacher import (
            fetch_phase3_teacher_bundle,
            build_phase3_digest,
            top_uncertain_vars_from_agent,
            _slot_lexicon_summary,
        )

        llm_url = get_ollama_generate_url()
        model = get_ollama_model()
        agent = self.agent
        valid = set(agent.graph.nodes.keys())
        if not valid:
            return {"ok": False, "error": "no graph nodes"}

        fallen = False
        is_fn = getattr(agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            try:
                fallen = bool(is_fn())
            except Exception:
                fallen = False

        pn = (
            PHASE_NAMES[self.phase]
            if 0 <= self.phase < len(PHASE_NAMES)
            else ""
        )
        digest = build_phase3_digest(
            variable_ids=sorted(valid),
            nodes=dict(agent.graph.nodes),
            phase_idx=self.phase,
            phase_name=pn,
            fallen=fallen,
            block_rate=agent.value_layer.block_rate,
            total_interventions=agent._total_interventions,
            top_uncertain_vars=top_uncertain_vars_from_agent(agent),
            slot_lexicon=_slot_lexicon_summary(self._visual_env),
        )

        rules, ov, err = await fetch_phase3_teacher_bundle(
            llm_url=llm_url,
            llm_model=model,
            digest=digest,
            valid_vars=valid,
            current_tick=self.tick,
        )

        if err and not rules and ov is None:
            return {"ok": False, "error": err}

        try:
            tmax = int(os.environ.get("RKK_TEACHER_T_MAX", "140"))
        except ValueError:
            tmax = 140
        tw = max(0.0, 1.0 - (agent._total_interventions / max(1, tmax)))
        with self._sim_step_lock:
            self._phase3_teacher_rules = rules
            self._phase3_vl_overlay = ov
            agent.set_teacher_state(rules, tw)
            if ov is not None and self.tick <= ov.expires_at_tick and tw > 0:
                agent.value_layer.set_teacher_vl_overlay(ov)
            else:
                agent.value_layer.set_teacher_vl_overlay(None)

            msg = f"📚 Phase3 teacher: {len(rules)} rules"
            if ov is not None:
                msg += f", VL overlay ttl={ov.expires_at_tick - self.tick}t"
            self._add_event(msg, "#ddaa44", "phase")

        return {
            "ok": True,
            "n_rules": len(rules),
            "vl_overlay": ov is not None,
            "warning": err,
            "expires_at_tick": ov.expires_at_tick if ov is not None else None,
        }

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _build_snapshot(self, snap: dict, graph_deltas: dict,
                        smoothed: float, scene: dict) -> dict:
        winfo = WORLDS.get(self.current_world, {"color": "#cc44ff", "label": self.current_world})

        # Visual cortex summary
        vision_summary = None
        if self._visual_mode and self._visual_env is not None:
            vision_summary = self._visual_env.cortex.snapshot()

        return {
            "tick":          self.tick,
            "phase":         self.phase,
            "max_phase":     self.max_phase,
            "entropy":       round((1 - snap.get("peak_discovery_rate", 0)) * 100, 1),
            "smoothed_dr":   round(smoothed, 3),
            "agents":        [snap],
            "n_agents":      1,
            "demon":         self.demon.snapshot,
            "tom_links":     [],
            "events":        list(self.events),
            "graph_deltas":  graph_deltas,
            "value_layer":   {
                "total_blocked_all": snap.get("total_blocked", 0),
                "block_rates": [round(snap.get("value_layer", {}).get("block_rate", 0), 3)],
                "imagination_horizon": snap.get("value_layer", {}).get("imagination_horizon", 0),
                "imagination_checks": snap.get("value_layer", {}).get("imagination_checks", 0),
                "imagination_blocks": snap.get("value_layer", {}).get("imagination_blocks", 0),
            },
            "byzantine":     None,
            "motif":         None,
            "multiprocess":  False,
            "singleton":     True,
            "current_world": self.current_world,
            "world_label":   winfo["label"],
            "world_color":   winfo["color"],
            "worlds":        WORLDS,
            "switch_history":self.switcher.history[-5:],
            "gnn_d":         self.agent.graph._d,
            "fallen":        snap.get("fallen", False),
            "fall_count":    snap.get("fall_count", 0),
            "fall_recovery": {
                "active": bool(self._fall_recovery_active),
                "start_tick": int(self._fall_recovery_start_tick),
                "last_progress_tick": int(self._fall_recovery_last_progress_tick),
                "best_score": round(float(self._fall_recovery_best_score), 4),
            },
            "fixed_root":    self._fixed_root_active,
            "scene":         scene,
            # Фаза 12
            "visual_mode":   self._visual_mode,
            "vision_ticks":  self._vision_ticks,
            "vision":        vision_summary,
            "llm_loop":      {
                "enabled":           self._llm_loop_enabled(),
                "level2_inflight":   self._llm_level2_inflight,
                "pending_bundle":    self._pending_llm_bundle is not None,
                "last_schedule_tick": self._last_level2_schedule_tick,
                "last_dr_gain_tick": self._last_dr_gain_tick,
                "rolling_block_rate": round(self._rolling_block_rate(), 4),
                "stats":             dict(self._llm_loop_stats),
            },
            "agent_loop":    {
                "hz": round(_agent_loop_hz_from_env(), 1),
                "decoupled": _agent_loop_hz_from_env() > 0.0,
                "l3_hz": round(_l3_loop_hz_from_env(), 1),
                "l3_last_tick": int(self._l3_last_tick),
                "l4_worker": bool(_l4_worker_enabled()),
                "l4_pending": bool(self._l4_task_pending),
                "l4_last_submit_tick": int(self._l4_last_submit_tick),
                "l4_last_apply_tick": int(self._l4_last_apply_tick),
                "l1_last_cmd_tick": int(self._l1_last_cmd_tick),
                "l1_last_apply_tick": int(self._l1_last_apply_tick),
            },
            "locomotion":    (
                {
                    **self._locomotion_controller.snapshot(),
                    "decoupled_loop_hz": round(_cpg_loop_hz_from_env(), 1)
                    if self._cpg_decoupled_enabled()
                    else 0.0,
                }
                if self._locomotion_controller is not None
                else None
            ),
            "motor_state":   self._motor_state_snapshot(),
            "skills":        self._skill_snapshot(),
            "rsi_full":      self._rsi_full.snapshot()
            if self._rsi_full_enabled() and self._rsi_full is not None
            else None,
            "motor_cortex": (
                self._motor_cortex.snapshot()
                if self._motor_cortex is not None
                else None
            ),
            "concepts":      [
                {
                    "id": c["id"],
                    "pattern": c["pattern"],
                    "uses": c["uses"],
                    "alpha_mean": c["alpha_mean"],
                    "graph_node": c.get("graph_node"),
                }
                for c in self._concepts_cache
            ],
            "memory":        self._memory_snapshot_meta(),
            "embodied_reward": (
                self._embodied_reward_ctrl.snapshot()
                if self._embodied_reward_ctrl is not None
                else None
            ),
            "visual_grounding": (
                self._visual_grounding_ctrl.snapshot()
                if self._visual_grounding_ctrl is not None
                else None
            ),
            "episodic_memory": (
                self._episodic_memory.snapshot()
                if self._episodic_memory is not None
                else None
            ),
            "curriculum": (
                self._curriculum.snapshot()
                if self._curriculum is not None
                else None
            ),
            "rssm": (
                self._rssm_trainer.snapshot()
                if self._rssm_trainer is not None
                else {"enabled": rssm_enabled() if _RSSM_AVAILABLE else False}
            ),
            "proprioception": (
                self._proprio.snapshot()
                if self._proprio is not None
                else None
            ),
            "reward_coordinator": (
                self._reward_coord.snapshot()
                if self._reward_coord is not None
                else {"enabled": _REWARD_COORD_AVAILABLE}
            ),
            "timescale": (
                self._timescale.snapshot()
                if self._timescale is not None
                else (
                    {"enabled": timescale_enabled()}
                    if _TIMESCALE_AVAILABLE
                    else {"enabled": False}
                )
            ),
            "phase1":        self._phase1_snapshot_meta(),
            "phase2":        self._phase2_snapshot_meta(),
        }

    def public_state(self) -> dict:
        snap     = self._last_snapshot or self.agent.snapshot()
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        fn       = getattr(self.agent.env, "get_full_scene", None)
        scene    = fn() if callable(fn) else {}
        return self._build_snapshot(snap, {}, smoothed, scene)

    def shutdown(self):
        self._stop_rkk_agent_loop_thread()
        self._stop_cpg_background_loop()
        try:
            self._llm_loop_executor.shutdown(wait=False, cancel_futures=False)
        except TypeError:
            self._llm_loop_executor.shutdown(wait=False)
        except Exception:
            pass