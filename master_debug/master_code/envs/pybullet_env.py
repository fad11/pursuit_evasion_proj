import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from control import BaseEvaderController, RandomEvaderController


class PursuitEvasionEnv(gym.Env):
    CATCH_RADIUS = 0.5
    CATCH_REWARD = 10.0
    TIME_PENALTY = 0.01

    def __init__(
        self,
        render_mode: str = "DIRECT",
        gravity: float = 9.81,
        max_steps: int = 400,
        evader_controller: BaseEvaderController | None = None,
        frame_skip: int = 8,
        evader_accel_scale: float = 1.0,
        linear_damping: float = 0.0,
        evader_controllers: list[BaseEvaderController] | None = None,
    ) -> None:
        super().__init__()
        if frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")
        if evader_controller is not None and evader_controllers is not None:
            raise ValueError(
                "Provide either evader_controller or evader_controllers, not both"
            )
        if evader_controllers is not None and not evader_controllers:
            raise ValueError("evader_controllers must contain at least one controller")

        self.render_mode = render_mode
        self.gravity = gravity
        self.max_steps = max_steps
        self.max_accel = 10.0
        self.frame_skip = frame_skip
        self.evader_accel_scale = evader_accel_scale
        self.linear_damping = linear_damping
        if evader_controllers is not None:
            self.evader_controllers = list(evader_controllers)
        elif evader_controller is not None:
            self.evader_controllers = [evader_controller]
        else:
            self.evader_controllers = [RandomEvaderController()]
        self.evader_controller = self.evader_controllers[0]

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
            shape=(12,),
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
        controller_index = int(self.np_random.integers(len(self.evader_controllers)))
        self.evader_controller = self.evader_controllers[controller_index]
        controller_seed = int(self.np_random.integers(0, 2**32, dtype=np.uint32))
        if isinstance(self.evader_controller, RandomEvaderController):
            self.evader_controller.reset(seed=controller_seed)
        else:
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
        for body_id in (self.pursuer_id, self.evader_id):
            p.changeDynamics(
                body_id,
                -1,
                linearDamping=self.linear_damping,
                angularDamping=0.0,
                physicsClientId=self.physics_client_id,
            )
        p.setCollisionFilterPair(
            self.pursuer_id,
            self.evader_id,
            -1,
            -1,
            0,
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
        start_x = self.np_random.normal(0.0, 5.0)
        start_z = 0.5 + self.np_random.exponential(2.5)
        evader_x = self.np_random.normal(0.0, 5.0)
        evader_z = 0.5 + self.np_random.exponential(2.5)
        while np.hypot(evader_x - start_x, evader_z - start_z) <= 3.0:
            evader_x = self.np_random.normal(0.0, 5.0)
            evader_z = 0.5 + self.np_random.exponential(2.5)
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
        los = np.arctan2(relative_x, relative_z)
        closing_vel = -(relative_x * relative_vx + relative_z * relative_vz) / (
            range_to_evader + 1e-5
        )
        los_rate = (relative_z * relative_vx - relative_x * relative_vz) / (
            range_squared + 1e-5
        )
        observer_vx, observer_vz = observer_velocity[0], observer_velocity[2]

        return np.array(
            [
                range_to_evader,
                np.sin(los),
                np.cos(los),
                closing_vel,
                np.clip(los_rate, -10.0, 10.0),
                observer_vx,
                observer_vz,
                np.hypot(observer_vx, observer_vz),
                relative_x,
                relative_z,
                relative_vx,
                relative_vz,
            ],
            dtype=np.float32,
        )

    def _apply_accel(self, body_id: int, accel: np.ndarray) -> None:
        force = [
            float(accel[0]) * self.max_accel,
            0.0,
            float(accel[1]) * self.max_accel,
        ]
        p.applyExternalForce(
            body_id,
            -1,
            force,
            [0.0, 0.0, 0.0],
            p.LINK_FRAME,
            physicsClientId=self.physics_client_id,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        previous_range = float(self._get_obs()[0])
        evader_action = self.evader_controller.get_action(self._get_evader_obs())
        terminated = False
        for _ in range(self.frame_skip):
            self._apply_accel(self.pursuer_id, action)
            self._apply_accel(
                self.evader_id,
                evader_action * self.evader_accel_scale,
            )
            p.stepSimulation(physicsClientId=self.physics_client_id)
            self._enforce_2d_constraints()
            obs = self._get_obs()
            range_to_evader = float(obs[0])
            terminated = range_to_evader < self.CATCH_RADIUS
            if terminated:
                break

        reward = previous_range - range_to_evader - self.TIME_PENALTY
        if terminated:
            reward += self.CATCH_REWARD

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, {"is_success": terminated}

    def close(self) -> None:
        if p.isConnected(self.physics_client_id):
            p.disconnect(physicsClientId=self.physics_client_id)
