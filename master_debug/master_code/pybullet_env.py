import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from control import BaseEvaderController, RandomEvaderController


class PursuitEvasionEnv(gym.Env):
    def __init__(
        self,
        render_mode: str = "DIRECT",
        gravity: float = 9.81,
        x_range: tuple[float, float] = (-5.0, 5.0),
        z_range: tuple[float, float] = (0.0, 5.0),
        max_steps: int = 1000,
        evader_controller: BaseEvaderController | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.gravity = gravity
        self.x_range = x_range
        self.z_range = z_range
        self.max_steps = max_steps
        self.max_accel = 10.0
        self.evader_controller = (
            evader_controller
            if evader_controller is not None
            else RandomEvaderController()
        )

        connection_mode = getattr(p, render_mode, None)
        if connection_mode not in (p.DIRECT, p.GUI):
            raise ValueError("render_mode must be 'DIRECT' or 'GUI'")
        self.physics_client_id = p.connect(connection_mode)
        if self.physics_client_id < 0:
            raise RuntimeError(f"Could not connect to PyBullet in {render_mode} mode")

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.physics_client_id,
        )
        p.setGravity(0.0, 0.0, -self.gravity, physicsClientId=self.physics_client_id)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.evader_controller.reset()
        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setGravity(
            0.0,
            0.0,
            -self.gravity,
            physicsClientId=self.physics_client_id,
        )
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client_id)

        start_x, start_z, evader_x, evader_z = self._sample_spawn_positions()

        self.pursuer_id = p.loadURDF(
            "assets/drone.urdf",
            [float(start_x), 0.0, float(start_z)],
            physicsClientId=self.physics_client_id,
        )
        self.evader_id = p.loadURDF(
            "assets/drone.urdf",
            [float(evader_x), 0.0, float(evader_z)],
            physicsClientId=self.physics_client_id,
        )
        p.changeVisualShape(
            self.pursuer_id,
            -1,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            physicsClientId=self.physics_client_id,
        )
        p.changeVisualShape(
            self.evader_id,
            -1,
            rgbaColor=[0.0, 1.0, 0.0, 1.0],
            physicsClientId=self.physics_client_id,
        )
        self._enforce_2d_constraints()
        return self._get_obs(), {}

    def _sample_spawn_positions(self) -> tuple[float, float, float, float]:
        start_x = self.np_random.uniform(*self.x_range)
        start_z = self.np_random.uniform(*self.z_range)
        evader_x = self.np_random.uniform(*self.x_range)
        evader_z = self.np_random.uniform(*self.z_range)
        while np.hypot(evader_x - start_x, evader_z - start_z) <= 3.0:
            evader_x = self.np_random.uniform(*self.x_range)
            evader_z = self.np_random.uniform(*self.z_range)
        return start_x, start_z, evader_x, evader_z

    def _enforce_2d_constraints(self) -> None:
        for body_id in (self.pursuer_id, self.evader_id):
            position, _ = p.getBasePositionAndOrientation(
                body_id,
                physicsClientId=self.physics_client_id,
            )
            linear_velocity, _ = p.getBaseVelocity(
                body_id,
                physicsClientId=self.physics_client_id,
            )
            p.resetBasePositionAndOrientation(
                body_id,
                [position[0], 0.0, position[2]],
                p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                physicsClientId=self.physics_client_id,
            )
            p.resetBaseVelocity(
                body_id,
                linearVelocity=[linear_velocity[0], 0.0, linear_velocity[2]],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.physics_client_id,
            )

    def _get_obs(self) -> np.ndarray:
        return self._get_relative_obs(self.pursuer_id, self.evader_id)

    def _get_evader_obs(self) -> np.ndarray:
        return self._get_relative_obs(self.evader_id, self.pursuer_id)

    def _get_relative_obs(self, observer_id: int, target_id: int) -> np.ndarray:
        observer_position, _ = p.getBasePositionAndOrientation(
            observer_id,
            physicsClientId=self.physics_client_id,
        )
        target_position, _ = p.getBasePositionAndOrientation(
            target_id,
            physicsClientId=self.physics_client_id,
        )
        observer_velocity, _ = p.getBaseVelocity(
            observer_id,
            physicsClientId=self.physics_client_id,
        )
        target_velocity, _ = p.getBaseVelocity(
            target_id,
            physicsClientId=self.physics_client_id,
        )

        relative_x = target_position[0] - observer_position[0]
        relative_z = target_position[2] - observer_position[2]
        relative_vx = target_velocity[0] - observer_velocity[0]
        relative_vz = target_velocity[2] - observer_velocity[2]
        range_to_evader = float(np.hypot(relative_x, relative_z))
        range_squared = relative_x**2 + relative_z**2
        observer_vx, observer_vz = observer_velocity[0], observer_velocity[2]

        return np.array(
            [
                range_to_evader,
                np.arctan2(relative_x, relative_z),
                -(relative_x * relative_vx + relative_z * relative_vz)
                / (range_to_evader + 1e-5),
                (relative_z * relative_vx - relative_x * relative_vz)
                / (range_squared + 1e-5),
                observer_vx,
                observer_vz,
                np.hypot(observer_vx, observer_vz),
            ],
            dtype=np.float32,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        pursuer_force = [
            float(action[0]) * self.max_accel,
            0.0,
            float(action[1]) * self.max_accel,
        ]
        evader_action = self.evader_controller.get_action(self._get_evader_obs())
        evader_force = [
            float(evader_action[0]) * self.max_accel,
            0.0,
            float(evader_action[1]) * self.max_accel,
        ]
        p.applyExternalForce(
            self.pursuer_id,
            -1,
            pursuer_force,
            [0.0, 0.0, 0.0],
            p.WORLD_FRAME,
            physicsClientId=self.physics_client_id,
        )
        p.applyExternalForce(
            self.evader_id,
            -1,
            evader_force,
            [0.0, 0.0, 0.0],
            p.WORLD_FRAME,
            physicsClientId=self.physics_client_id,
        )
        p.stepSimulation(physicsClientId=self.physics_client_id)
        self._enforce_2d_constraints()

        obs = self._get_obs()
        range_to_evader = float(obs[0])
        terminated = range_to_evader < 0.5
        reward = 100.0 if terminated else -range_to_evader

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        for body_id in (self.pursuer_id, self.evader_id):
            position, _ = p.getBasePositionAndOrientation(
                body_id,
                physicsClientId=self.physics_client_id,
            )
            if not (
                self.x_range[0] <= position[0] <= self.x_range[1]
                and self.z_range[0] <= position[2] <= self.z_range[1]
            ):
                truncated = True
                break

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        if p.isConnected(self.physics_client_id):
            p.disconnect(physicsClientId=self.physics_client_id)
