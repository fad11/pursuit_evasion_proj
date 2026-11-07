import os
from pathlib import Path

import imageio.v2 as imageio
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from ._assets import resolve_asset


class PursuitEvasionBullet(gym.Env):
    """2D x/z PyBullet environment with acceleration or body-axis thrust actions.

    In thrust mode, RL actions are [thrust, body_rate]; torque-based rate
    control is enabled by use_torque. In multi mode, pursuer_rl and evader_rl
    choose which agents are controlled by input actions; other agents use their
    scripted controller or a seeded random acceleration. GUI recording writes
    MP4; DIRECT recording writes PNG frames.
    """

    _control_fps = 10
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": _control_fps}
    _frame_size = (640, 480)
    _body_rate_gain = 20.0

    def __init__(
        self,
        mode="pursuer",
        obs_type="relative",
        dim=2,
        arena_size=20.0,
        max_steps=300,
        v_max=1.0,
        a_max=1.0,
        thrust_max=None,
        body_rate_max=None,
        capture_radius=0.3,
        gravity=0.0,
        reset_range=(1.0, 5.0),
        action_type="acceleration",
        use_torque=True,
        pursuer_rl=False,
        evader_rl=False,
        pursuer_controller=None,
        evader_controller=None,
        record=False,
        gui=False,
        render_mode=None,
        video_path="output.mp4",
        frame_dir="frames",
    ):
        super().__init__()
        if dim != 2:
            raise ValueError("This env is 2D (x,z) only")
        if mode not in ("pursuer", "evader", "multi"):
            raise ValueError("mode must be 'pursuer', 'evader', or 'multi'")
        if obs_type not in ("relative", "los"):
            raise ValueError("obs_type must be 'relative' or 'los'")
        if action_type not in ("acceleration", "thrust"):
            raise ValueError("action_type must be 'acceleration' or 'thrust'")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if arena_size <= 0 or v_max <= 0 or a_max <= 0 or max_steps <= 0:
            raise ValueError("arena_size, v_max, a_max, and max_steps must be positive")

        self.mode = mode
        self.obs_type = obs_type
        self.dim = dim
        self.arena_size = float(arena_size)
        self.max_steps = int(max_steps)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.thrust_max = float(a_max if thrust_max is None else thrust_max)
        self.body_rate_max = float(a_max if body_rate_max is None else body_rate_max)
        if self.thrust_max <= 0 or self.body_rate_max <= 0:
            raise ValueError("thrust_max and body_rate_max must be positive")
        self.capture_radius = float(capture_radius)
        self.gravity = float(gravity)
        self.reset_range = tuple(reset_range)
        self.action_type = action_type
        self.use_torque = bool(use_torque)
        self.pursuer_rl = bool(pursuer_rl)
        self.evader_rl = bool(evader_rl)
        self.pursuer_controller = pursuer_controller
        self.evader_controller = evader_controller
        self.record = bool(record)
        self.render_mode = render_mode
        self.gui = bool(gui or render_mode == "human")
        self.video_path = video_path
        self.frame_dir = frame_dir
        self.dt = 1.0 / self.metadata["render_fps"]
        self._substeps = 10
        self._physics_dt = self.dt / self._substeps
        self.step_count = 0
        self.frame_idx = 0
        self.video_writer = None

        self.acceleration_space = spaces.Box(
            low=-self.a_max,
            high=self.a_max,
            shape=(2,),
            dtype=np.float32,
        )
        self.thrust_space = spaces.Box(
            low=np.array([-self.thrust_max, -self.body_rate_max], dtype=np.float32),
            high=np.array([self.thrust_max, self.body_rate_max], dtype=np.float32),
            dtype=np.float32,
        )
        if mode == "multi":
            self.action_space = spaces.Dict({
                role: self._role_action_space(role)
                for role in ("pursuer", "evader")
                if self._role_is_rl(role)
            })
        else:
            self.action_space = self._role_action_space(mode)

        obs_dim = 6 if obs_type == "relative" else 4
        role_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        if mode == "multi":
            self.observation_space = spaces.Dict({
                role: role_observation_space
                for role in ("pursuer", "evader")
            })
        else:
            self.observation_space = role_observation_space

        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to the PyBullet physics server")
        try:
            p.setAdditionalSearchPath(
                pybullet_data.getDataPath(),
                physicsClientId=self.client,
            )
            p.setGravity(0, 0, -self.gravity, physicsClientId=self.client)
            p.setTimeStep(self._physics_dt, physicsClientId=self.client)
            self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
            self.p_body = p.loadURDF(
                resolve_asset("pursuer.urdf"),
                physicsClientId=self.client,
            )
            self.e_body = p.loadURDF(
                resolve_asset("evader.urdf"),
                physicsClientId=self.client,
            )
            self.p_mass = p.getDynamicsInfo(
                self.p_body, -1, physicsClientId=self.client
            )[0]
            self.e_mass = p.getDynamicsInfo(
                self.e_body, -1, physicsClientId=self.client
            )[0]
            self.p_inertia_y = p.getDynamicsInfo(
                self.p_body, -1, physicsClientId=self.client
            )[2][1]
            self.e_inertia_y = p.getDynamicsInfo(
                self.e_body, -1, physicsClientId=self.client
            )[2][1]
            if self.record:
                self._start_recording()
        except Exception:
            p.disconnect(self.client)
            raise

        self.p_pos = np.zeros(2, dtype=float)
        self.e_pos = np.zeros(2, dtype=float)
        self.p_vel = np.zeros(2, dtype=float)
        self.e_vel = np.zeros(2, dtype=float)
        self.current_los_rate = 0.0

    def _role_action_space(self, role):
        if self.action_type == "acceleration" or not self._role_uses_thrust(role):
            return self.acceleration_space
        return self.thrust_space

    def _role_uses_thrust(self, role):
        if self.action_type != "thrust":
            return False
        if self.mode == "multi":
            return self._role_is_rl(role)
        return role == self.mode

    def _role_is_rl(self, role):
        if self.mode == "multi":
            return self.pursuer_rl if role == "pursuer" else self.evader_rl
        return role == self.mode

    def _start_recording(self):
        if self.gui:
            video_path = os.path.abspath(self.video_path)
            parent = os.path.dirname(video_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self.video_writer = imageio.get_writer(
                video_path,
                format="FFMPEG",
                mode="I",
                fps=self.metadata["render_fps"],
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        else:
            os.makedirs(self.frame_dir, exist_ok=True)
            existing_indices = (
                int(path.stem.removeprefix("frame_"))
                for path in Path(self.frame_dir).glob("frame_*.png")
                if path.stem.removeprefix("frame_").isdigit()
            )
            self.frame_idx = max(existing_indices, default=-1) + 1

    def _camera_matrices(self):
        distance = (
            (self.arena_size / 2.0) / np.tan(np.deg2rad(30.0)) * 1.1
        )
        view = p.computeViewMatrix(
            [0.0, -distance, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )
        width, height = self._frame_size
        projection = p.computeProjectionMatrixFOV(
            60.0,
            width / height,
            0.1,
            100.0,
            physicsClientId=self.client,
        )
        return view, projection

    def _capture_rgb(self):
        view, projection = self._camera_matrices()
        width, height = self._frame_size
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            view,
            projection,
            physicsClientId=self.client,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(
            height, width, 4
        )[:, :, :3]

    def _record_frame(self):
        if not self.record:
            return
        if self.gui:
            self.video_writer.append_data(self._capture_rgb())
        else:
            from PIL import Image

            filename = os.path.join(
                self.frame_dir,
                f"frame_{self.frame_idx:05d}.png",
            )
            Image.fromarray(self._capture_rgb()).save(filename)
            self.frame_idx += 1

    def _sync(self):
        for body, position, velocity in (
            (self.p_body, self.p_pos, self.p_vel),
            (self.e_body, self.e_pos, self.e_vel),
        ):
            p.resetBasePositionAndOrientation(
                body,
                [position[0], 0.0, position[1]],
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
            p.resetBaseVelocity(
                body,
                [velocity[0], 0.0, velocity[1]],
                [0.0, 0.0, 0.0],
                physicsClientId=self.client,
            )

    def _spawn(self, center=None, distance=None):
        center = (
            np.zeros(2)
            if center is None
            else np.asarray(center, dtype=float)
        )
        if center.shape != (2,):
            raise ValueError("center must contain exactly two coordinates (x, z)")
        if np.any(np.abs(center) > self.arena_size / 2):
            raise ValueError("center must be inside the arena")
        if distance is None:
            distance = float(self.np_random.uniform(*self.reset_range))
        if distance < 0:
            raise ValueError("distance must be non-negative")
        direction = self.np_random.normal(size=2)
        direction /= np.linalg.norm(direction) + 1e-6
        self.p_pos = center.copy()
        self.e_pos = center + direction * distance
        self.e_pos = np.clip(
            self.e_pos,
            -self.arena_size / 2,
            self.arena_size / 2,
        )
        self.p_vel = np.zeros(2)
        self.e_vel = np.zeros(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.step_count = 0
        self.current_los_rate = 0.0
        self._spawn(
            center=options.get("center"),
            distance=options.get("distance"),
        )
        if self.pursuer_controller is not None:
            self.pursuer_controller.reset(self.p_pos.copy())
        if self.evader_controller is not None:
            self.evader_controller.reset(self.e_pos.copy())
        self._sync()
        self._record_frame()
        if self.mode == "multi":
            return {
                "pursuer": self._get_obs("pursuer"),
                "evader": self._get_obs("evader"),
            }, {}
        return self._get_obs(self.mode), {}

    def _get_obs(self, role):
        if role == "pursuer":
            pos_self, vel_self = self.p_pos, self.p_vel
            pos_target, vel_target = self.e_pos, self.e_vel
        else:
            pos_self, vel_self = self.e_pos, self.e_vel
            pos_target, vel_target = self.p_pos, self.p_vel
        rel = pos_target - pos_self
        rel_vel = vel_target - vel_self
        if self.obs_type == "relative":
            return np.concatenate([rel, rel_vel, vel_self]).astype(np.float32)
        distance = np.linalg.norm(rel) + 1e-6
        los = np.arctan2(rel[1], rel[0])
        closing = np.dot(rel, rel_vel) / distance
        return np.array(
            [distance, los, self.current_los_rate, closing],
            dtype=np.float32,
        )

    def _clip_action(self, action, role):
        action = np.asarray(action, dtype=np.float32)
        action_space = self._role_action_space(role)
        if action.shape != action_space.shape:
            raise ValueError(
                f"{role} action must have shape {action_space.shape}, "
                f"got {action.shape}"
            )
        return np.clip(action, action_space.low, action_space.high)

    def _sample_acceleration(self):
        return self.np_random.uniform(
            low=self.acceleration_space.low,
            high=self.acceleration_space.high,
        ).astype(np.float32)

    def _choose_actions(self, action):
        if self.mode == "multi":
            rl_roles = {
                role for role in ("pursuer", "evader") if self._role_is_rl(role)
            }
            if not isinstance(action, dict) or set(action) != rl_roles:
                raise ValueError(
                    f"multi mode action keys must be {sorted(rl_roles)}"
                )
            actions = {}
            for role in ("pursuer", "evader"):
                if self._role_is_rl(role):
                    actions[role] = (action[role], self._role_uses_thrust(role))
                    continue
                controller = (
                    self.pursuer_controller
                    if role == "pursuer"
                    else self.evader_controller
                )
                if controller is None:
                    actions[role] = (self._sample_acceleration(), False)
                elif role == "pursuer":
                    actions[role] = (
                        controller.compute(
                            self.p_pos,
                            self.p_vel,
                            self.e_pos,
                            self.e_vel,
                        ),
                        False,
                    )
                else:
                    actions[role] = (
                        controller.compute(
                            self.e_pos,
                            self.e_vel,
                            self.p_pos,
                            self.p_vel,
                        ),
                        False,
                    )
            return actions

        active_role = self.mode
        opponent_role = "evader" if active_role == "pursuer" else "pursuer"
        actions = {
            active_role: (action, self._role_uses_thrust(active_role))
        }
        controller = (
            self.evader_controller
            if opponent_role == "evader"
            else self.pursuer_controller
        )
        if controller is not None:
            if opponent_role == "evader":
                raw_action = controller.compute(
                    self.e_pos,
                    self.e_vel,
                    self.p_pos,
                    self.p_vel,
                )
            else:
                raw_action = controller.compute(
                    self.p_pos,
                    self.p_vel,
                    self.e_pos,
                    self.e_vel,
                )
            actions[opponent_role] = (raw_action, False)
        else:
            actions[opponent_role] = (
                self._sample_acceleration(),
                False,
            )
        return actions

    def _apply_agent_action(self, role, action, use_thrust):
        body = self.p_body if role == "pursuer" else self.e_body
        mass = self.p_mass if role == "pursuer" else self.e_mass
        inertia_y = (
            self.p_inertia_y if role == "pursuer" else self.e_inertia_y
        )
        position, _ = p.getBasePositionAndOrientation(
            body,
            physicsClientId=self.client,
        )
        if not use_thrust:
            force = [
                float(action[0]) * mass,
                0.0,
                float(action[1]) * mass,
            ]
            p.applyExternalForce(
                body,
                -1,
                force,
                position,
                p.WORLD_FRAME,
                physicsClientId=self.client,
            )
            return

        thrust, target_rate = map(float, action)
        p.applyExternalForce(
            body,
            -1,
            [0.0, 0.0, thrust],
            [0.0, 0.0, 0.0],
            p.LINK_FRAME,
            physicsClientId=self.client,
        )
        if self.use_torque and inertia_y > 0.0:
            _, angular_velocity = p.getBaseVelocity(
                body,
                physicsClientId=self.client,
            )
            torque_y = inertia_y * self._body_rate_gain * (
                target_rate - angular_velocity[1]
            )
            p.applyExternalTorque(
                body,
                -1,
                [0.0, torque_y, 0.0],
                p.WORLD_FRAME,
                physicsClientId=self.client,
            )

    def _read_and_limit_state(self):
        for role in ("pursuer", "evader"):
            body = self.p_body if role == "pursuer" else self.e_body
            position, orientation = p.getBasePositionAndOrientation(
                body,
                physicsClientId=self.client,
            )
            linear_velocity, _ = p.getBaseVelocity(
                body,
                physicsClientId=self.client,
            )
            pos = np.array([position[0], position[2]], dtype=float)
            vel = np.array([linear_velocity[0], linear_velocity[2]], dtype=float)
            speed = np.linalg.norm(vel)
            if speed > self.v_max:
                vel *= self.v_max / speed
            wall = self.arena_size / 2
            for axis in range(2):
                if pos[axis] > wall:
                    pos[axis], vel[axis] = wall, -0.5 * vel[axis]
                elif pos[axis] < -wall:
                    pos[axis], vel[axis] = -wall, -0.5 * vel[axis]
            if role == "pursuer":
                self.p_pos, self.p_vel = pos, vel
            else:
                self.e_pos, self.e_vel = pos, vel
            p.resetBasePositionAndOrientation(
                body,
                [pos[0], 0.0, pos[1]],
                orientation,
                physicsClientId=self.client,
            )
            p.resetBaseVelocity(
                body,
                [vel[0], 0.0, vel[1]],
                [0.0, 0.0, 0.0],
                physicsClientId=self.client,
            )

    def step(self, action):
        actions = self._choose_actions(action)
        clipped_actions = {
            role: (self._clip_action(values, role), use_thrust)
            for role, (values, use_thrust) in actions.items()
        }

        for _ in range(self._substeps):
            for role, (agent_action, use_thrust) in clipped_actions.items():
                self._apply_agent_action(role, agent_action, use_thrust)
            p.stepSimulation(physicsClientId=self.client)

        self._read_and_limit_state()
        self.step_count += 1
        if self.obs_type == "los":
            rel = self.e_pos - self.p_pos
            rel_vel = self.e_vel - self.p_vel
            cross = rel[0] * rel_vel[1] - rel[1] * rel_vel[0]
            self.current_los_rate = cross / (np.dot(rel, rel) + 1e-6)

        distance = np.linalg.norm(self.p_pos - self.e_pos)
        captured = distance < self.capture_radius
        truncated = self.step_count >= self.max_steps
        if captured:
            pursuer_reward, evader_reward = 10.0, -10.0
        else:
            pursuer_reward = -0.1 - distance / self.arena_size
            evader_reward = 0.1 + min(0.5, distance / self.arena_size)

        self._record_frame()
        if self.mode == "multi":
            return (
                {
                    "pursuer": self._get_obs("pursuer"),
                    "evader": self._get_obs("evader"),
                },
                {"pursuer": pursuer_reward, "evader": evader_reward},
                {"pursuer": captured, "evader": captured},
                {"pursuer": truncated, "evader": truncated},
                {
                    "distance": float(distance),
                    "captured": captured,
                    "is_success": {
                        "pursuer": captured,
                        "evader": not captured,
                    },
                },
            )
        reward = pursuer_reward if self.mode == "pursuer" else evader_reward
        success = captured if self.mode == "pursuer" else not captured
        return (
            self._get_obs(self.mode),
            reward,
            captured,
            truncated,
            {
                "distance": float(distance),
                "captured": captured,
                "is_success": success,
            },
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._capture_rgb()
        return None

    def housekeeping(self, center=None, distance=None):
        self._spawn(center=center, distance=distance)
        self._sync()

    def close(self):
        try:
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None
        finally:
            if p.isConnected(self.client):
                p.disconnect(self.client)
