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

Этап D (LLM в петле, RKK_LLM_LOOP=1):
  Уровень 1: каждый тик — GNN + System1 (как раньше).
  Уровень 2: фоновый Ollama по триггерам (стагнация discovery, block_rate, VLM unknown, surprise PE).
  Уровень 3: редко — перезапись гипотез (humanoid), в том же worker после L2 при run_level3.
"""
from __future__ import annotations

import os
import torch
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from engine.agent       import RKKAgent
from engine.demon       import AdversarialDemon
from engine.environment import Environment
from engine.ollama_env  import get_ollama_generate_url, get_ollama_model
from engine.value_layer import HomeostaticBounds
from engine.wm_neural_ode import integrate_world_model_step

PHASE_THRESHOLDS = [0.0, 0.15, 0.30, 0.50, 0.70, 0.88]
PHASE_HOLD_TICKS = 12
PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer",
                    "Social Sandbox", "Value Lock", "Open Reality"]

# Visual mode: полный GNN→cortex на каждом тике дорог; предсказание для PC обновляем реже
VISION_GNN_FEED_EVERY = 2


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
        warmup_ticks=1500,
        blend_ticks=500,
        phi_min_steady=0.04,
        env_entropy_max_delta_steady=0.60,
        h_slow_max_steady=11.0,
        predict_band_edge_steady=0.02,
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
        self._stand_ticks = 0
        self._last_fall_reset_tick: int = -999

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
        self._llm_loop_stats: dict = {
            "level2_runs": 0,
            "level3_runs": 0,
            "last_triggers": [],
            "last_level2_explanation": "",
        }

    # ── World switch ──────────────────────────────────────────────────────────
    def switch_world(self, new_world: str) -> dict:
        if new_world not in WORLDS:
            return {"error": f"unknown world: {new_world}"}

        # Если visual mode — сначала отключаем
        was_visual = self._visual_mode
        if was_visual:
            self._disable_visual_internal()

        result = self.switcher.switch(new_world)
        if result.get("switched"):
            self.current_world = new_world
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

        # Восстанавливаем visual mode если был
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

    def disable_visual(self) -> dict:
        """Отключаем Visual Cortex, возвращаемся к ручным переменным."""
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
            phi_min=0.01,
            h_slow_max=14.0,
            env_entropy_max_delta=0.96,
            warmup_ticks=1500,
            blend_ticks=500,
            phi_min_steady=0.04,
            env_entropy_max_delta_steady=0.60,
            h_slow_max_steady=11.0,
            predict_band_edge_steady=0.02,
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
            if edges:
                inj = self.agent.inject_text_priors(edges)
                n = inj.get("injected", 0)
                ex = (l2.get("explanation") or "").strip()
                self._add_event(f"🧠 LLM L2 (+{n} priors): {ex}", "#9bdcff", "phase")
            self._llm_loop_stats["level2_runs"] = int(self._llm_loop_stats.get("level2_runs", 0)) + 1
            self._llm_loop_stats["last_level2_explanation"] = (l2.get("explanation") or "").strip()
        elif l2.get("error"):
            self._add_event(f"LLM L2 error: {l2.get('error')}", "#cc6666", "phase")

        l3 = b.get("l3")
        if isinstance(l3, dict) and l3.get("ok"):
            edges3 = l3.get("edges") or []
            if edges3:
                inj3 = self.agent.inject_text_priors(edges3)
                self._add_event(
                    f"🧬 LLM L3 restructure +{inj3.get('injected', 0)} hypotheses",
                    "#66ddff",
                    "phase",
                )
            self._last_level3_tick = self.tick
            self._llm_loop_stats["level3_runs"] = int(self._llm_loop_stats.get("level3_runs", 0)) + 1

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
            cooldown = int(os.environ.get("RKK_LLM_LEVEL2_COOLDOWN", "240"))
        except ValueError:
            cooldown = 720
        if self.tick - self._last_level2_schedule_tick < cooldown:
            return

        try:
            stagnation_ticks = int(os.environ.get("RKK_LLM_STAGNATION_TICKS", "500"))
        except ValueError:
            stagnation_ticks = 500
        try:
            min_iv = int(os.environ.get("RKK_LLM_MIN_INTERVENTIONS", "36"))
        except ValueError:
            min_iv = 36

        triggers: list[str] = []
        if (
            self.agent._total_interventions >= min_iv
            and (self.tick - self._last_dr_gain_tick) >= stagnation_ticks
        ):
            triggers.append("discovery_stagnation")

        vl = self.agent.value_layer
        if vl.total_checked >= 48 and self._rolling_block_rate() > 0.4:
            triggers.append("block_rate")

        if self._vlm_unknown_slot_trigger():
            triggers.append("vlm_unknown_object")

        if self._prediction_surprise_trigger(result):
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

    # ── Tick ──────────────────────────────────────────────────────────────────
    def tick_step(self) -> dict:
        self.tick += 1
        self._apply_pending_llm_bundle()

        # Fallen check + автосброс физики (иначе VL и block_rate залипают)
        fallen = False
        is_fn  = getattr(self.agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            fallen = is_fn()
            if fallen:
                self._fall_count += 1
                if self._try_reset_pose_after_fall():
                    obs = self.agent.env.observe()
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

        self.agent.other_agents_phi = []
        result = self.agent.step(engine_tick=self.tick)
        self._log_step(result, fallen)
        self._rolling_block_bits.append(1 if result.get("blocked") else 0)

        snap = self.agent.snapshot()
        snap["fallen"]     = fallen
        snap["fall_count"] = self._fall_count
        self._last_snapshot = snap

        dr = float(snap.get("discovery_rate", 0.0))
        if dr > self._best_discovery_rate + 1e-5:
            self._best_discovery_rate = dr
            self._last_dr_gain_tick = self.tick

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
        elif result.get("updated_edges"):
            cg  = result.get("compression_delta", 0)
            var = result.get("variable", "?")
            val = result.get("value", 0)
            color = WORLDS.get(self.current_world, {}).get("color", "#cc44ff")
            if self._visual_mode:
                color = "#44ffcc"
            self._add_event(
                f"Nova: do({var}={val:.2f}) CG{'+' if cg>=0 else ''}{cg:.3f}",
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

        self._phase3_teacher_rules = rules
        self._phase3_vl_overlay = ov

        try:
            tmax = int(os.environ.get("RKK_TEACHER_T_MAX", "140"))
        except ValueError:
            tmax = 140
        tw = max(0.0, 1.0 - (agent._total_interventions / max(1, tmax)))
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
        }

    def public_state(self) -> dict:
        snap     = self._last_snapshot or self.agent.snapshot()
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        fn       = getattr(self.agent.env, "get_full_scene", None)
        scene    = fn() if callable(fn) else {}
        return self._build_snapshot(snap, {}, smoothed, scene)

    def shutdown(self):
        try:
            self._llm_loop_executor.shutdown(wait=False, cancel_futures=False)
        except TypeError:
            self._llm_loop_executor.shutdown(wait=False)
        except Exception:
            pass