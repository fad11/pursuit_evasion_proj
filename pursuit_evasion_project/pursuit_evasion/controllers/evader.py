import numpy as np
from .base import BaseController
class PureEvasion(BaseController):
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); los = pos_self - pos_target; dist = np.linalg.norm(los)
        return self._velocity_to_acceleration(los / (dist + 1e-6), vel_self) if dist >= 1e-6 else np.zeros(self.dim)
class ProportionalEvasion(BaseController):
    def __init__(self, dodge_weight=0.5, **kwargs): super().__init__(**kwargs); self.dodge_weight = dodge_weight
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); los = pos_self - pos_target; dist = np.linalg.norm(los)
        if dist < 1e-6: return np.zeros(self.dim)
        flee = los / dist
        if self.dim == 2:
            dodge = np.array([-flee[1], flee[0]])
        else:
            axis = np.eye(3)[np.argmin(np.abs(flee))]
            dodge = np.cross(flee, axis)
            dodge /= np.linalg.norm(dodge) + 1e-6
        desired = flee + self.dodge_weight * dodge
        return self._velocity_to_acceleration(desired / (np.linalg.norm(desired) + 1e-6), vel_self)


class ManeuveringEvasion(BaseController):
    def __init__(
        self,
        maneuver_strength=0.5,
        maneuver_frequency=0.1,
        arena_limit=10.0,
        boundary_margin=3.0,
        boundary_strength=2.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if maneuver_frequency <= 0.0:
            raise ValueError("maneuver_frequency must be positive")
        if arena_limit <= 0.0:
            raise ValueError("arena_limit must be positive")
        if boundary_margin <= 0.0 or boundary_margin > arena_limit:
            raise ValueError(
                "boundary_margin must be positive and no greater than arena_limit"
            )
        self.maneuver_strength = maneuver_strength
        self.maneuver_frequency = maneuver_frequency
        self.arena_limit = arena_limit
        self.boundary_margin = boundary_margin
        self.boundary_strength = boundary_strength

    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance()
        los = np.asarray(pos_self, dtype=float) - np.asarray(pos_target, dtype=float)
        dist = np.linalg.norm(los)
        if dist < 1e-6:
            return np.zeros(self.dim)

        flee = los / dist
        relative_velocity = np.asarray(vel_target, dtype=float) - np.asarray(
            vel_self, dtype=float
        )
        lateral_velocity = relative_velocity - np.dot(relative_velocity, flee) * flee
        lateral_speed = np.linalg.norm(lateral_velocity)
        if lateral_speed < 1e-6:
            axis = np.eye(self.dim)[np.argmin(np.abs(flee))]
            lateral_direction = axis - np.dot(axis, flee) * flee
            lateral_direction /= np.linalg.norm(lateral_direction) + 1e-6
        else:
            lateral_direction = -lateral_velocity / lateral_speed

        phase = np.sin(
            2.0 * np.pi * self.maneuver_frequency * self._t * self.dt
        )
        desired = flee + self.maneuver_strength * phase * lateral_direction

        position = np.asarray(pos_self, dtype=float)
        room_to_wall = self.arena_limit - np.abs(position)
        boundary_risk = np.clip(
            (self.boundary_margin - room_to_wall) / self.boundary_margin,
            0.0,
            1.0,
        )
        desired -= (
            self.boundary_strength
            * boundary_risk
            * np.sign(position)
        )
        desired_norm = np.linalg.norm(desired)
        if desired_norm < 1e-6:
            desired = flee
        else:
            desired = desired / desired_norm
        return self._velocity_to_acceleration(desired, vel_self)
