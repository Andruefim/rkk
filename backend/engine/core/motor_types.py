"""Состояние моторики и лог команд (humanoid / CPG)."""
from __future__ import annotations

from dataclasses import dataclass, field


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
    intents: dict[str, float] = field(
        default_factory=lambda: {
            "intent_stride": 0.5,
            "intent_support_left": 0.5,
            "intent_support_right": 0.5,
            "intent_torso_forward": 0.5,
            "intent_gait_coupling": 0.88,
            "intent_arm_counterbalance": 0.5,
            "intent_stop_recover": 0.5,
            "intent_reach_right": 0.5,
            "intent_reach_left": 0.5,
            "intent_grasp": 0.5,
            "intent_look_at": 0.5,
            "intent_lean_forward": 0.5,
            "intent_wave": 0.5,
        }
    )
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

    def update_from_observation(
        self,
        obs: dict[str, float],
        *,
        tick: int | None = None,
        source: str | None = None,
    ) -> None:
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
