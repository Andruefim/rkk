"""
Онлайн-обучаемая инверсная динамика (forward + inverse), без привязки к конкретной задаче.

Вкл.: RKK_CEREBELLUM=1
См. также ReflexStabilizer — узкий корректор поверх CPG; мозжечок здесь общий маппинг
(state, desired_delta) → команды суставам (по умолчанию только ноги, как в apply_cpg_leg_targets).
"""
from __future__ import annotations

import os
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.features.humanoid.constants import LEG_VARS


def cerebellum_enabled() -> bool:
    return os.environ.get("RKK_CEREBELLUM", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Только каналы, которые стабильно есть в humanoid.observe() (нормализованные скаляры).
DEFAULT_BODY_STATE_KEYS: tuple[str, ...] = (
    "com_x",
    "com_y",
    "com_z",
    "torso_roll",
    "torso_pitch",
    "spine_yaw",
    "spine_pitch",
    "neck_yaw",
    "neck_pitch",
    "posture_stability",
    "foot_contact_l",
    "foot_contact_r",
    "lhip",
    "lknee",
    "lankle",
    "rhip",
    "rknee",
    "rankle",
    "lshoulder",
    "lelbow",
    "rshoulder",
    "relbow",
    "lfoot_z",
    "rfoot_z",
    "gait_phase_l",
    "gait_phase_r",
    "motor_drive_l",
    "motor_drive_r",
)


class Cerebellum:
    """
    Forward: (state, joint_cmd) → next_state
    Inverse: (state, state_delta) → joint_cmd (tanh → [0,1] для ног)
    """

    def __init__(
        self,
        *,
        body_state_keys: tuple[str, ...] | None = None,
        joint_keys: tuple[str, ...] | None = None,
        hidden: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        try:
            n_state_env = int(os.environ.get("RKK_CEREBELLUM_N_STATE", "26"))
        except ValueError:
            n_state_env = 26
        base_keys = tuple(body_state_keys) if body_state_keys else DEFAULT_BODY_STATE_KEYS
        n_state = max(4, min(n_state_env, len(base_keys)))
        self.body_state_keys: tuple[str, ...] = base_keys[:n_state]
        self.n_state = len(self.body_state_keys)

        jkeys = tuple(joint_keys) if joint_keys else tuple(LEG_VARS)
        self.joint_keys: tuple[str, ...] = jkeys
        self.n_joints = len(self.joint_keys)

        try:
            hidden = int(os.environ.get("RKK_CEREBELLUM_HIDDEN", str(hidden)))
        except ValueError:
            pass
        hidden = max(16, min(hidden, 512))

        d_fwd_in = self.n_state + self.n_joints
        d_inv_in = self.n_state * 2

        self.forward_model = nn.Sequential(
            nn.Linear(d_fwd_in, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.n_state),
        ).to(self.device)
        self.inverse_model = nn.Sequential(
            nn.Linear(d_inv_in, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.n_joints),
            nn.Tanh(),
        ).to(self.device)

        try:
            lr = float(os.environ.get("RKK_CEREBELLUM_LR", "1e-3"))
        except ValueError:
            lr = 1e-3
        self.opt_fwd = torch.optim.Adam(self.forward_model.parameters(), lr=lr)
        self.opt_inv = torch.optim.Adam(self.inverse_model.parameters(), lr=lr)

        try:
            buf_max = int(os.environ.get("RKK_CEREBELLUM_BUFFER", "512"))
        except ValueError:
            buf_max = 512
        buf_max = max(64, min(buf_max, 4096))
        self._buf: deque = deque(maxlen=buf_max)

        self._lock = threading.Lock()
        self._pending_delta = np.zeros(self.n_state, dtype=np.float32)
        self._n_trained = 0

    @property
    def n_trained(self) -> int:
        return int(self._n_trained)

    def ready_for_control(self) -> bool:
        try:
            min_steps = int(os.environ.get("RKK_CEREBELLUM_MIN_TRAIN_STEPS", "100"))
        except ValueError:
            min_steps = 100
        return self._n_trained >= max(0, min_steps)

    def _extract_state(self, obs: dict) -> np.ndarray:
        out = []
        for k in self.body_state_keys:
            v = obs.get(k, obs.get(f"phys_{k}", 0.5))
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.5)
        return np.asarray(out, dtype=np.float32)

    def _extract_joints(self, cmd: dict) -> np.ndarray:
        return np.array(
            [float(cmd.get(k, 0.5)) for k in self.joint_keys],
            dtype=np.float32,
        )

    def set_desired_from_graph(
        self,
        graph_nodes: dict,
        agent_intents: dict[str, float],
    ) -> None:
        """Efference copy: intent_* → желаемая дельта по первым координатам состояния."""
        key_to_idx = {k: i for i, k in enumerate(self.body_state_keys)}
        delta = np.zeros(self.n_state, dtype=np.float32)
        stride = float(agent_intents.get("intent_stride", graph_nodes.get("intent_stride", 0.5)))
        recover = float(
            agent_intents.get(
                "intent_stop_recover",
                graph_nodes.get("intent_stop_recover", 0.0),
            )
        )
        if "com_z" in key_to_idx:
            delta[key_to_idx["com_z"]] = float((stride - 0.5) * 0.1)
        if "posture_stability" in key_to_idx:
            delta[key_to_idx["posture_stability"]] = float(recover * 0.2)
        if "com_x" in key_to_idx:
            delta[key_to_idx["com_x"]] = float((stride - 0.5) * 0.06)

        with self._lock:
            self._pending_delta = delta

    def get_joint_commands(
        self,
        obs: dict,
        *,
        desired_delta: np.ndarray | None = None,
        desired_obs: dict | None = None,
    ) -> dict[str, float]:
        """(state, desired change) → цели суставов (ключи joint_keys)."""
        with self._lock:
            state = self._extract_state(obs)
            if desired_delta is not None:
                d = np.asarray(desired_delta, dtype=np.float32).reshape(-1)
                if d.shape[0] != self.n_state:
                    d = np.resize(d, (self.n_state,))
            elif desired_obs is not None:
                tgt = self._extract_state(desired_obs)
                d = tgt - state
            else:
                d = np.array(self._pending_delta, copy=True)

            x = np.concatenate([state, d]).astype(np.float32, copy=False)
            xt = torch.from_numpy(x).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                cmd = self.inverse_model(xt).squeeze(0).cpu().numpy()

        result: dict[str, float] = {}
        for i, k in enumerate(self.joint_keys):
            v = float(cmd[i]) * 0.5 + 0.5
            result[k] = float(np.clip(v, 0.05, 0.95))
        return result

    def record_transition(
        self,
        obs_before: dict,
        joint_cmd: dict[str, float],
        obs_after: dict,
    ) -> None:
        s0 = self._extract_state(obs_before)
        s1 = self._extract_state(obs_after)
        j = self._extract_joints(joint_cmd)
        with self._lock:
            self._buf.append((s0.copy(), j.copy(), s1.copy()))

    def train_step(self, batch_size: int = 32) -> dict[str, float]:
        with self._lock:
            if len(self._buf) < batch_size:
                return {}
            n = len(self._buf)
            idx = np.random.choice(n, size=batch_size, replace=(n < batch_size))
            batch = [self._buf[int(i)] for i in idx]
            s_b = torch.from_numpy(np.stack([t[0] for t in batch])).float().to(self.device)
            j_b = torch.from_numpy(np.stack([t[1] for t in batch])).float().to(self.device)
            s_n = torch.from_numpy(np.stack([t[2] for t in batch])).float().to(self.device)

            self.opt_fwd.zero_grad()
            pred_next = self.forward_model(torch.cat([s_b, j_b], dim=-1))
            fwd_loss = F.mse_loss(pred_next, s_n)
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
            self.opt_fwd.step()

            delta = s_n - s_b
            self.opt_inv.zero_grad()
            pred_cmd = self.inverse_model(torch.cat([s_b, delta], dim=-1))
            j_scaled = j_b * 2.0 - 1.0
            inv_loss = F.mse_loss(pred_cmd, j_scaled)
            inv_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.inverse_model.parameters(), 1.0)
            self.opt_inv.step()

            self._n_trained += 1
            return {
                "fwd_loss": float(fwd_loss.item()),
                "inv_loss": float(inv_loss.item()),
                "n_trained": float(self._n_trained),
                "buf_size": float(len(self._buf)),
            }
