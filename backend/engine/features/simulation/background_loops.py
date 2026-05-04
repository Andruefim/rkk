"""
Фоновые циклы: CPG low-level и high-level agent loop (daemon threads).
Инкапсулирует потоки, чтобы не раздувать Simulation.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from engine.core.constants import cpg_loop_hz_from_env as _cpg_loop_hz_from_env

if TYPE_CHECKING:
    from engine.simulation import Simulation


class BackgroundLoopService:
    """Управление rkk-cpg-loop и rkk-agent-loop."""

    __slots__ = ("_sim", "_cpg_loop_thread", "_cpg_stop", "_cpg_snapshot_lock", "_cpg_node_snapshot", "_agent_loop_thread", "_agent_stop")

    def __init__(self, sim: "Simulation") -> None:
        self._sim = sim
        self._cpg_loop_thread: threading.Thread | None = None
        self._cpg_stop = threading.Event()
        self._cpg_snapshot_lock = threading.Lock()
        self._cpg_node_snapshot: dict[str, float] = {}
        self._agent_loop_thread: threading.Thread | None = None
        self._agent_stop = threading.Event()

    def stop_cpg_loop(self) -> None:
        self._cpg_stop.set()
        th = self._cpg_loop_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.5)
        self._cpg_loop_thread = None
        self._cpg_stop.clear()
        self._sim._drain_simple_queue(self._sim._l1_motor_q)

    def ensure_cpg_background_loop(self) -> None:
        s = self._sim
        if not s._cpg_decoupled_enabled():
            return
        if s.current_world != "humanoid" or s._fixed_root_active:
            return
        base = s._unwrap_base_env(s.agent.env)
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

    def publish_cpg_node_snapshot(self) -> None:
        s = self._sim
        if not s._cpg_decoupled_enabled():
            return
        with self._cpg_snapshot_lock:
            self._cpg_node_snapshot = dict(s.agent.graph.nodes)

    def _cpg_loop_worker(self) -> None:
        hz = _cpg_loop_hz_from_env()
        dt = 1.0 / hz if hz > 0 else 0.05
        from engine.cpg_locomotion import LocomotionController

        s = self._sim
        while not self._cpg_stop.is_set():
            t0 = time.perf_counter()
            try:
                if not s._locomotion_cpg_enabled():
                    time.sleep(0.05)
                    continue
                if s.current_world != "humanoid" or s._fixed_root_active:
                    time.sleep(0.05)
                    continue
                base = s._unwrap_base_env(s.agent.env)
                fn = getattr(base, "apply_cpg_leg_targets", None)
                if not callable(fn):
                    time.sleep(0.05)
                    continue
                if s._locomotion_controller is None:
                    s._locomotion_controller = LocomotionController(s.device)
                with self._cpg_snapshot_lock:
                    nodes = dict(self._cpg_node_snapshot)
                if not nodes:
                    nodes = dict(s.agent.graph.nodes)
                obs_reflex = dict(nodes)
                try:
                    # Не брать _sim_step_lock: иначе CPG ждёт 1–2 с пока агент считает score/train.
                    # get_state() внутри observe() сериализуется через humanoid._physics_lock.
                    obs_live = s.agent.env.observe()
                    obs_reflex = {**nodes}
                    for k, v in obs_live.items():
                        try:
                            obs_reflex[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                    for _k in ("com_x", "com_y", "com_z"):
                        if _k in obs_live:
                            nodes[_k] = float(obs_live[_k])
                except Exception:
                    obs_reflex = dict(nodes)
                targets = s._locomotion_controller.get_joint_targets(nodes, dt=dt)
                cb = getattr(s, "_cerebellum", None)
                if cb is None:
                    try:
                        from engine.cerebellum import cerebellum_enabled

                        if cerebellum_enabled():
                            cb = s._ensure_cerebellum()
                    except Exception:
                        cb = None
                use_cb = cb is not None and cb.ready_for_control()
                if use_cb:
                    try:
                        from engine.features.humanoid.constants import LEG_VARS

                        leg = cb.get_joint_commands(obs_reflex)
                        for k in LEG_VARS:
                            if k in leg:
                                targets[k] = leg[k]
                    except Exception:
                        pass
                else:
                    rs = getattr(s, "_reflex_stabilizer", None)
                    if rs is None:
                        try:
                            from engine.reflex_stabilizer import reflex_stabilizer_enabled

                            if reflex_stabilizer_enabled():
                                rs = s._ensure_reflex_stabilizer()
                        except Exception:
                            rs = None
                    if rs is not None:
                        try:
                            targets = rs.step(obs_reflex, targets)
                        except Exception:
                            pass
                try:
                    targets = s._blend_causal_executor_targets(nodes, targets)
                except Exception:
                    pass
                cpg_sync = s._locomotion_controller.upper_body_cpg_sync()
                s._enqueue_l1_motor_command(
                    source="cpg",
                    joint_targets=targets,
                    intents=getattr(s._locomotion_controller, "_last_motor_state", None),
                    dt=dt,
                    cpg_sync=cpg_sync,
                )
            except Exception as ex:
                print(f"[Simulation] CPG loop: {ex}")
            elapsed = time.perf_counter() - t0
            wait = dt - elapsed
            if wait > 0:
                self._cpg_stop.wait(timeout=wait)

    def ensure_rkk_agent_loop(self) -> None:
        from engine.core.constants import agent_loop_hz_from_env as _agent_loop_hz_from_env

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
            f"(RKK_AGENT_LOOP_HZ; HTTP/WS tick_step -> cache)"
        )

    def stop_rkk_agent_loop(self) -> None:
        self._agent_stop.set()
        th = self._agent_loop_thread
        if th is not None and th.is_alive():
            th.join(timeout=2.5)
        self._agent_loop_thread = None
        self._agent_stop.clear()
        self._sim._agent_step_response = None

    def _rkk_agent_loop_worker(self) -> None:
        from engine.core.constants import agent_loop_hz_from_env as _agent_loop_hz_from_env

        hz = _agent_loop_hz_from_env()
        dt = 1.0 / hz if hz > 0 else 0.1
        s = self._sim
        while not self._agent_stop.is_set():
            t0 = time.perf_counter()
            try:
                result = None
                with s._sim_step_lock:
                    result = s._run_single_agent_timestep_inner()
                s._agent_step_response = result
            except Exception as e:
                print(f"[Simulation] Agent loop: {e}")
            elapsed = time.perf_counter() - t0
            self._agent_stop.wait(timeout=max(0.0, dt - elapsed))
