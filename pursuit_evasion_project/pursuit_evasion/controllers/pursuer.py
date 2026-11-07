import numpy as np
from .base import BaseController
class PurePursuit(BaseController):
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); los = pos_target - pos_self; dist = np.linalg.norm(los)
        return self._velocity_to_acceleration(los / (dist + 1e-6), vel_self) if dist >= 1e-6 else np.zeros(self.dim)
class ProportionalNavigation(BaseController):
    def __init__(self, N=3.0, **kwargs):
        super().__init__(**kwargs); self.N, self.prev_angle = N, None
    def reset(self, initial_pos=None): super().reset(initial_pos); self.prev_angle = None
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        los = pos_target - pos_self; rel_vel = vel_target - vel_self
        if self.dim == 2:
            angle = np.arctan2(los[1], los[0])
            rate = 0.0 if self.prev_angle is None else ((angle - self.prev_angle + np.pi) % (2 * np.pi) - np.pi) / self.dt
            self.prev_angle = angle; self._advance()
            lead = angle + self.N * rate * self.dt
            return self._velocity_to_acceleration(np.array([np.cos(lead), np.sin(lead)]), vel_self)
        los_norm = np.linalg.norm(los) + 1e-6
        los_rate = np.cross(los, rel_vel) / (np.dot(los, los) + 1e-6)
        lead_dir = los / los_norm + self.N * self.dt * los_rate
        lead_dir = lead_dir / (np.linalg.norm(lead_dir) + 1e-6)
        self._advance()
        return self._velocity_to_acceleration(lead_dir, vel_self)
