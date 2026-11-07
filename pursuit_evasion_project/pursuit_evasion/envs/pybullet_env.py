import os
import pybullet as p
import pybullet_data
import numpy as np
from .base_env import BasePursuitEvasionEnv
from ._assets import resolve_asset

class PyBulletPursuitEvasionEnv(BasePursuitEvasionEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_physics_backend = True
        self.mass = 1.0
        self.client = p.connect(p.DIRECT if self.render_mode != "human" else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -self.gravity, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.p_body = p.loadURDF(resolve_asset("pursuer.urdf"), physicsClientId=self.client)
        self.e_body = p.loadURDF(resolve_asset("evader.urdf"), physicsClientId=self.client)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        p.resetBasePositionAndOrientation(self.p_body, np.pad(self.p_pos, (0, 3 - self.dim)), [0,0,0,1], physicsClientId=self.client)
        p.resetBaseVelocity(self.p_body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.e_body, np.pad(self.e_pos, (0, 3 - self.dim)), [0,0,0,1], physicsClientId=self.client)
        p.resetBaseVelocity(self.e_body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        return obs, info

    def _step_physics(self, a_p, a_e):
        p.applyExternalForce(self.p_body, -1, np.pad(a_p * self.mass, (0, 3 - self.dim)), [0,0,0], p.WORLD_FRAME, physicsClientId=self.client)
        p.applyExternalForce(self.e_body, -1, np.pad(a_e * self.mass, (0, 3 - self.dim)), [0,0,0], p.WORLD_FRAME, physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        p_pos, _ = p.getBasePositionAndOrientation(self.p_body, physicsClientId=self.client)
        p_vel, _ = p.getBaseVelocity(self.p_body, physicsClientId=self.client)
        e_pos, _ = p.getBasePositionAndOrientation(self.e_body, physicsClientId=self.client)
        e_vel, _ = p.getBaseVelocity(self.e_body, physicsClientId=self.client)
        self.p_pos, self.p_vel = np.array(p_pos[:self.dim]), np.array(p_vel[:self.dim])
        self.e_pos, self.e_vel = np.array(e_pos[:self.dim]), np.array(e_vel[:self.dim])

    def _set_gravity(self, gravity):
        if hasattr(self, "client"):
            p.setGravity(0, 0, -gravity, physicsClientId=self.client)

    def _sync_physics_state(self):
        p.resetBasePositionAndOrientation(
            self.p_body, np.pad(self.p_pos, (0, 3 - self.dim)), [0, 0, 0, 1],
            physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.p_body, np.pad(self.p_vel, (0, 3 - self.dim)),
            [0, 0, 0], physicsClientId=self.client
        )
        p.resetBasePositionAndOrientation(
            self.e_body, np.pad(self.e_pos, (0, 3 - self.dim)), [0, 0, 0, 1],
            physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.e_body, np.pad(self.e_vel, (0, 3 - self.dim)),
            [0, 0, 0], physicsClientId=self.client
        )

    def spawn_agents(self, center=None, distance=None):
        if center is None:
            center = np.zeros(self.dim)
        center = np.asarray(center, dtype=float)
        if distance is None:
            distance = float(np.random.uniform(self.reset_range[0], self.reset_range[1]))
        direction = np.random.normal(size=self.dim)
        direction /= np.linalg.norm(direction) + 1e-6
        self.p_pos = center.copy()
        self.e_pos = center + direction * distance
        for i in range(self.dim):
            if self.e_pos[i] > self.arena_size / 2:
                self.e_pos[i] = self.arena_size / 2
            elif self.e_pos[i] < -self.arena_size / 2:
                self.e_pos[i] = -self.arena_size / 2
        self.p_vel = np.zeros(self.dim)
        self.e_vel = np.zeros(self.dim)
        self._sync_physics_state()

    def apply_level(self, gravity=None, reset_range=None):
        if gravity is not None:
            self.gravity = gravity
            p.setGravity(0, 0, -gravity, physicsClientId=self.client)
        if reset_range is not None:
            self.reset_range = reset_range

    def render(self):
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrix([0, 0, 10], [0, 0, 0], [0, 1, 0], physicsClientId=self.client)
            proj_matrix = p.computeProjectionMatrixFOV(60, 640/480, 0.1, 100.0, physicsClientId=self.client)
            _, _, rgb, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix, physicsClientId=self.client)
            return np.reshape(rgb, (480, 640, 4))[:, :, :3]

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
