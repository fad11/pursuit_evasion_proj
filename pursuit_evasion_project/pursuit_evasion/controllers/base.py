import numpy as np
from abc import ABC, abstractmethod

class BaseController(ABC):
    def __init__(self, dim=2, v_max=1.0, a_max=1.0, dt=0.1, **kwargs):
        self.dim, self.v_max, self.a_max, self.dt, self._t = dim, v_max, a_max, dt, 0
    def set_dim(self, dim): self.dim = dim
    def reset(self, initial_pos=None): self._t = 0
    @abstractmethod
    def compute(self, pos_self, vel_self, pos_target, vel_target): pass
    def _velocity_to_acceleration(self, desired_vel_dir, vel_self, kp=5.0):
        desired_vel = desired_vel_dir * self.v_max
        acc = kp * (desired_vel - vel_self)
        norm = np.linalg.norm(acc)
        if norm > self.a_max: acc = (acc / norm) * self.a_max
        return acc
    def _advance(self): self._t += 1

class Idle(BaseController):
    def compute(self, pos_self, vel_self, pos_target, vel_target): self._advance(); return np.zeros(self.dim)
class StraightLine(BaseController):
    def __init__(self, direction=None, **kwargs):
        super().__init__(**kwargs)
        self._default_direction = direction is None
        self._configured_direction = None if direction is None else np.asarray(direction, dtype=float)
        self.set_dim(self.dim)

    def set_dim(self, dim):
        super().set_dim(dim)
        direction = self._configured_direction
        if self._default_direction:
            direction = np.zeros(self.dim)
            direction[0] = 1.0
        self.direction = direction
        if self.direction.shape != (self.dim,):
            raise ValueError(f"direction must have shape ({self.dim},)")
        norm = np.linalg.norm(self.direction)
        if norm < 1e-6:
            raise ValueError("direction must be non-zero")
        self.direction = self.direction / norm
    def compute(self, pos_self, vel_self, pos_target, vel_target): self._advance(); return self._velocity_to_acceleration(self.direction, vel_self)
class Circular(BaseController):
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); radial = pos_self / (np.linalg.norm(pos_self) + 1e-6)
        if self.dim == 2:
            tangent = np.array([-radial[1], radial[0]])
        else:
            axis = np.eye(3)[np.argmin(np.abs(radial))]
            tangent = np.cross(axis, radial)
            tangent /= np.linalg.norm(tangent) + 1e-6
        return self._velocity_to_acceleration(tangent, vel_self)
class Zigzag(BaseController):
    def __init__(self, direction=None, amplitude=1.0, frequency=1.0, **kwargs):
        super().__init__(**kwargs)
        self._default_direction = direction is None
        self._configured_direction = None if direction is None else np.asarray(direction, dtype=float)
        self.set_dim(self.dim)
        self.amplitude, self.frequency = amplitude, frequency

    def set_dim(self, dim):
        super().set_dim(dim)
        direction = self._configured_direction
        if self._default_direction:
            direction = np.zeros(self.dim)
            direction[0] = 1.0
        self.base_dir = direction
        if self.base_dir.shape != (self.dim,):
            raise ValueError(f"direction must have shape ({self.dim},)")
        self.base_dir = self.base_dir / (np.linalg.norm(self.base_dir) + 1e-6)

    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); angle = self.amplitude * np.sin(2 * np.pi * self.frequency * self._t * self.dt)
        c, s = np.cos(angle), np.sin(angle)
        desired = self.base_dir.copy()
        desired[:2] = np.array([[c, -s], [s, c]]) @ self.base_dir[:2]
        return self._velocity_to_acceleration(desired, vel_self)
class Spiral(BaseController):
    def __init__(self, angular_speed=1.0, radial_speed=0.5, **kwargs):
        super().__init__(**kwargs); self.angular_speed, self.radial_speed, self.center = angular_speed, radial_speed, np.zeros(self.dim)
    def set_dim(self, dim):
        super().set_dim(dim)
        self.center = np.zeros(self.dim)
    def reset(self, initial_pos=None):
        super().reset(initial_pos); self.center = initial_pos[:self.dim].copy() if initial_pos is not None else np.zeros(self.dim)
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); rel_pos = pos_self[:2] - self.center[:2]; dist = np.linalg.norm(rel_pos)
        theta = 0.0 if dist < 1e-3 else np.arctan2(rel_pos[1], rel_pos[0])
        radial_dir = np.array([np.cos(theta), np.sin(theta)]); tangential_dir = np.array([-np.sin(theta), np.cos(theta)])
        v_r = self.radial_speed * np.sin(self._t * self.dt * self.angular_speed)
        desired_dir = np.zeros(self.dim)
        desired_dir[:2] = v_r * radial_dir + self.angular_speed * tangential_dir
        return self._velocity_to_acceleration(desired_dir / (np.linalg.norm(desired_dir) + 1e-6), vel_self)


class SquarePath(BaseController):
    def __init__(self, side_length=5.0, **kwargs):
        super().__init__(**kwargs)
        self.side_length = side_length
        self.waypoints = None
        self.current_wp = 0
    def reset(self, initial_pos=None):
        super().reset(initial_pos)
        self.current_wp = 0
        origin = initial_pos[:self.dim].copy() if initial_pos is not None else np.zeros(self.dim)
        L = self.side_length
        x_offset = np.zeros(self.dim); x_offset[0] = L
        y_offset = np.zeros(self.dim); y_offset[1] = L
        self.waypoints = [origin + x_offset, origin + x_offset + y_offset,
                          origin + y_offset, origin]
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance()
        target = self.waypoints[self.current_wp]
        if np.linalg.norm(pos_self[:self.dim] - target) < 0.2:
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            target = self.waypoints[self.current_wp]
        desired = target - pos_self[:self.dim]
        return self._velocity_to_acceleration(desired / (np.linalg.norm(desired) + 1e-6), vel_self)
