"""
agent_worker.py — Windows-safe multiprocessing worker.

Архитектура:
  Главный процесс (FastAPI):
    - Держит WebSocket, собирает снапшоты
    - Посылает AgentCommand в in_q каждого воркера
    - Читает AgentResult из out_q

  Воркер-процесс (AgentWorker):
    - Держит один RKKAgent на своём GPU-контексте
    - Блокируется на in_q.get()
    - Делает step() / inject() / snapshot()
    - Кладёт результат в out_q

Windows-специфика:
  - spawn (не fork): torch.multiprocessing работает через spawn
  - GPU: каждый процесс получает собственный HIP-контекст
  - Данные между процессами — только JSON-сериализуемые dict
    (тензоры НЕ передаём, только числа/строки)
  - if __name__ == "__main__" — обязательно в точке входа

Команды (dict):
  {"cmd": "step", "tick": int}
  {"cmd": "inject", "edges": [...]}
  {"cmd": "snapshot"}
  {"cmd": "graph_dict"}          → agent.graph.to_dict() для REST
  {"cmd": "consensus_data"}      → возвращает W как list[list[float]]
  {"cmd": "apply_alpha_decay", "from_": str, "to": str, "decay": float}
  {"cmd": "apply_s1_weights", "state_dict": dict}  → Motif Transfer
  {"cmd": "stop"}
"""
from __future__ import annotations

import torch
import multiprocessing as mp
import numpy as np
from typing import Any

# Важно: импорты тяжёлых модулей ВНУТРИ функции воркера,
# иначе spawn скопирует всё в дочерний процесс до инициализации.


def _agent_worker_fn(
    agent_id:    int,
    agent_name:  str,
    env_preset:  str,
    device_str:  str,
    bounds_dict: dict,
    in_q:        mp.Queue,
    out_q:       mp.Queue,
):
    """
    Точка входа дочернего процесса.
    Импортируем тяжёлые модули здесь — после spawn.
    """
    # ── Инициализация ──
    import torch
    from engine.causal_graph import CausalGraph
    from engine.environment  import Environment
    from engine.agent        import RKKAgent
    from engine.value_layer  import HomeostaticBounds

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    bounds = HomeostaticBounds(**bounds_dict)
    env    = Environment(env_preset, device)
    agent  = RKKAgent(agent_id, agent_name, env, device, bounds)

    print(f"[Worker {agent_name}] Started on {device}")
    out_q.put({"status": "ready", "agent_id": agent_id})

    # ── Основной цикл ──
    while True:
        try:
            cmd_dict: dict = in_q.get(timeout=5.0)
        except Exception:
            continue

        cmd = cmd_dict.get("cmd")

        if cmd == "stop":
            out_q.put({"status": "stopped", "agent_id": agent_id})
            break

        elif cmd == "step":
            tick   = cmd_dict.get("tick", 0)
            phi_others = cmd_dict.get("other_phi", [])
            agent.other_agents_phi = phi_others
            result = agent.step(engine_tick=tick)
            out_q.put({
                "status":   "step_done",
                "agent_id": agent_id,
                "result":   _safe_result(result),
                "snapshot": agent.snapshot(),
            })

        elif cmd == "snapshot":
            out_q.put({
                "status":   "snapshot",
                "agent_id": agent_id,
                "snapshot": agent.snapshot(),
            })

        elif cmd == "inject":
            edges  = cmd_dict.get("edges", [])
            result = agent.inject_text_priors(edges)
            out_q.put({
                "status":   "injected",
                "agent_id": agent_id,
                "result":   result,
            })

        elif cmd == "graph_dict":
            out_q.put({
                "status":   "graph_dict",
                "agent_id": agent_id,
                "data":     agent.graph.to_dict(),
            })

        elif cmd == "consensus_data":
            # Возвращаем W как list[list[float]] (JSON-safe)
            W_list   = None
            node_ids = []
            if agent.graph._core is not None:
                W_list   = agent.graph._core.W_masked().detach().cpu().tolist()
                node_ids = list(agent.graph._node_ids)
            out_q.put({
                "status":      "consensus_data",
                "agent_id":    agent_id,
                "phi":         agent.phi_approx(),
                "cg":          agent.compression_gain,
                "block_rate":  agent.value_layer.block_rate,
                "W":           W_list,
                "node_ids":    node_ids,
            })

        elif cmd == "apply_alpha_decay":
            from_ = cmd_dict.get("from_")
            to    = cmd_dict.get("to")
            decay = float(cmd_dict.get("decay", 0.12))
            for edge in agent.graph.edges:
                if edge.from_ == from_ and edge.to == to:
                    edge.alpha_trust = max(0.02, edge.alpha_trust - decay)
                    break
            agent.graph._invalidate_cache()
            out_q.put({"status": "alpha_decayed", "agent_id": agent_id})

        elif cmd == "apply_s1_weights":
            state_list = cmd_dict.get("state_dict", {})
            try:
                # Восстанавливаем тензоры из list
                state_dict = {
                    k: torch.tensor(v, device=device)
                    for k, v in state_list.items()
                }
                current = agent.system1.net.state_dict()
                new_state = {}
                ema = float(cmd_dict.get("ema", 0.15))
                for key in current:
                    if key in state_dict and current[key].shape == state_dict[key].shape:
                        new_state[key] = (1 - ema) * current[key].float() + ema * state_dict[key].float()
                    else:
                        new_state[key] = current[key]
                agent.system1.net.load_state_dict(new_state, strict=False)
                out_q.put({"status": "s1_updated", "agent_id": agent_id})
            except Exception as e:
                out_q.put({"status": "s1_update_failed", "agent_id": agent_id, "error": str(e)})

        elif cmd == "demon_disrupt":
            result = agent.demon_disrupt()
            out_q.put({"status": "disrupted", "agent_id": agent_id, "result": result})

        else:
            out_q.put({"status": "unknown_cmd", "agent_id": agent_id, "cmd": cmd})


def _safe_result(result: dict) -> dict:
    """Убираем нессериализуемые объекты."""
    safe = {}
    for k, v in result.items():
        if isinstance(v, (int, float, str, bool, list, type(None))):
            safe[k] = v
        elif isinstance(v, dict):
            safe[k] = _safe_result(v)
        else:
            safe[k] = str(v)
    return safe


# ─── AgentPool ───────────────────────────────────────────────────────────────
class AgentPool:
    """
    Пул из N воркеров-агентов.
    Управляет запуском, остановкой и синхронным обменом сообщениями.

    Использование:
      pool = AgentPool(...)
      pool.start()
      ...
      results = pool.step_all(tick=100)
      ...
      pool.stop()
    """

    AGENT_NAMES = ["Nova", "Aether", "Lyra"]
    ENV_PRESETS = ["physics", "chemistry", "logic"]

    def __init__(self, device_str: str, bounds_dict: dict):
        self.device_str  = device_str
        self.bounds_dict = bounds_dict
        self.n           = 3

        self.procs:   list[mp.Process] = []
        self.in_qs:   list[mp.Queue]   = []
        self.out_qs:  list[mp.Queue]   = []
        self._ready   = False

        # Последние снапшоты (обновляются после каждого step)
        self.snapshots: list[dict] = [{} for _ in range(self.n)]

    def start(self) -> bool:
        """Запускаем воркеры. Возвращает True если все стартовали."""
        ctx = mp.get_context("spawn")   # Windows-safe

        for i in range(self.n):
            in_q  = ctx.Queue(maxsize=4)
            out_q = ctx.Queue(maxsize=4)
            self.in_qs.append(in_q)
            self.out_qs.append(out_q)

            p = ctx.Process(
                target=_agent_worker_fn,
                args=(
                    i,
                    self.AGENT_NAMES[i],
                    self.ENV_PRESETS[i],
                    self.device_str,
                    self.bounds_dict,
                    in_q,
                    out_q,
                ),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        # Ждём "ready" от всех воркеров
        import time
        deadline = time.time() + 30.0
        ready_count = 0
        while ready_count < self.n and time.time() < deadline:
            for i, q in enumerate(self.out_qs):
                try:
                    msg = q.get(timeout=0.5)
                    if msg.get("status") == "ready":
                        ready_count += 1
                        print(f"[Pool] Worker {self.AGENT_NAMES[i]} ready")
                except Exception:
                    pass

        self._ready = ready_count == self.n
        if not self._ready:
            print(f"[Pool] Only {ready_count}/{self.n} workers ready!")
        return self._ready

    def step_all(self, tick: int) -> list[dict]:
        """Посылаем step всем воркерам, ждём результаты."""
        if not self._ready:
            return []

        # Собираем phi до шага (для ΔΦ≥0)
        phis = [s.get("phi", 0.1) for s in self.snapshots]

        # Рассылаем команды
        for i in range(self.n):
            other_phi = [p for j, p in enumerate(phis) if j != i]
            self.in_qs[i].put({
                "cmd":       "step",
                "tick":      tick,
                "other_phi": other_phi,
            })

        # Собираем результаты (с таймаутом)
        results = []
        for i in range(self.n):
            try:
                msg = self.out_qs[i].get(timeout=10.0)
                if msg.get("status") == "step_done":
                    self.snapshots[i] = msg.get("snapshot", {})
                results.append(msg)
            except Exception as e:
                results.append({
                    "status":   "timeout",
                    "agent_id": i,
                    "error":    str(e),
                })
        return results

    def get_consensus_data(self) -> list[dict]:
        """Запрашиваем W-матрицы для Byzantine консенсуса."""
        for i in range(self.n):
            self.in_qs[i].put({"cmd": "consensus_data"})

        data = []
        for i in range(self.n):
            try:
                msg = self.out_qs[i].get(timeout=5.0)
                if msg.get("status") == "consensus_data":
                    data.append(msg)
            except Exception:
                pass
        return data

    def apply_alpha_decay(self, agent_id: int, from_: str, to: str, decay: float = 0.12):
        self.in_qs[agent_id].put({
            "cmd":   "apply_alpha_decay",
            "from_": from_,
            "to":    to,
            "decay": decay,
        })
        try:
            self.out_qs[agent_id].get(timeout=3.0)
        except Exception:
            pass

    def apply_s1_weights(self, agent_id: int, state_dict_list: dict, ema: float = 0.15):
        self.in_qs[agent_id].put({
            "cmd":        "apply_s1_weights",
            "state_dict": state_dict_list,
            "ema":        ema,
        })
        try:
            self.out_qs[agent_id].get(timeout=5.0)
        except Exception:
            pass

    def inject_seeds(self, agent_id: int, edges: list[dict]) -> dict:
        self.in_qs[agent_id].put({"cmd": "inject", "edges": edges})
        try:
            msg = self.out_qs[agent_id].get(timeout=5.0)
            return msg.get("result", {})
        except Exception:
            return {"error": "timeout"}

    def demon_disrupt(self, agent_id: int) -> str:
        self.in_qs[agent_id].put({"cmd": "demon_disrupt"})
        try:
            msg = self.out_qs[agent_id].get(timeout=3.0)
            return msg.get("result", "")
        except Exception:
            return "timeout"

    def get_graph_dict(self, agent_id: int) -> dict:
        """Полный to_dict() графа для REST /graph/{id} (только multiprocess)."""
        self.in_qs[agent_id].put({"cmd": "graph_dict"})
        try:
            msg = self.out_qs[agent_id].get(timeout=5.0)
            if msg.get("status") == "graph_dict":
                return msg.get("data", {})
        except Exception:
            pass
        return {"error": "timeout"}

    def stop(self):
        for q in self.in_qs:
            try:
                q.put({"cmd": "stop"}, timeout=1.0)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        self._ready = False
        print("[Pool] All workers stopped")

    @property
    def is_ready(self) -> bool:
        return self._ready
