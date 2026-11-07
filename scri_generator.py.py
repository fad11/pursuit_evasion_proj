import os

FILES = {
    # --- PACKAGING ---
    "setup.py": """from setuptools import setup, find_packages
setup(
    name="pursuit_evasion",
    version="0.3.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"pursuit_evasion": ["../assets/*.xml", "../assets/*.urdf"]},
    install_requires=["gymnasium>=1.0.0", "stable-baselines3>=2.9.0", "sb3-contrib>=2.0.0", "mujoco>=3.15.0", "pybullet>=3.2.6", "numpy", "tensorboard"],
)""",
    
    "requirements.txt": "gymnasium>=1.0.0\nstable-baselines3>=2.9.0\nsb3-contrib>=2.0.0\nmujoco>=3.15.0\npybullet>=3.2.6\nnumpy\ntensorboard\n",

    # --- ASSETS ---
    "assets/pursuer.urdf": """<?xml version="1.0" ?>
<robot name="pursuer"><link name="base_link"><inertial><mass value="1.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial><visual><geometry><sphere radius="0.15"/></geometry><material name="red"><color rgba="1 0 0 1"/></material></visual><collision><geometry><sphere radius="0.15"/></geometry></collision></link></robot>""",

    "assets/evader.urdf": """<?xml version="1.0" ?>
<robot name="evader"><link name="base_link"><inertial><mass value="1.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial><visual><geometry><sphere radius="0.15"/></geometry><material name="green"><color rgba="0 1 0 1"/></material></visual><collision><geometry><sphere radius="0.15"/></geometry></collision></link></robot>""",

    "assets/world.xml": """<mujoco><worldbody><geom name="floor" type="plane" size="20 20 0.1" rgba="0.8 0.8 0.8 1"/><camera name="top_down" pos="0 0 20" euler="0 0 0" /><body name="pursuer" pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.15" rgba="1 0 0 1" mass="1.0"/></body><body name="evader" pos="1 1 0.1"><freejoint/><geom type="sphere" size="0.15" rgba="0 1 0 1" mass="1.0"/></body></worldbody></mujoco>""",

    # --- CONTROLLERS ---
    "pursuit_evasion/controllers/__init__.py": "",
    
    "pursuit_evasion/controllers/base.py": """import numpy as np
from abc import ABC, abstractmethod

class BaseController(ABC):
    def __init__(self, dim=2, v_max=1.0, a_max=1.0, dt=0.1, **kwargs):
        self.dim, self.v_max, self.a_max, self.dt, self._t = dim, v_max, a_max, dt, 0
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
        self.direction = direction if direction is not None else np.array([1.0, 0.0])
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-6)
    def compute(self, pos_self, vel_self, pos_target, vel_target): self._advance(); return self._velocity_to_acceleration(self.direction, vel_self)
class Circular(BaseController):
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); radial = pos_self / (np.linalg.norm(pos_self) + 1e-6)
        tangent = np.array([-radial[1], radial[0]])
        return self._velocity_to_acceleration(tangent, vel_self)
class Zigzag(BaseController):
    def __init__(self, direction=None, amplitude=1.0, frequency=1.0, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = direction if direction is not None else np.array([1.0, 0.0])
        self.base_dir = self.base_dir / (np.linalg.norm(self.base_dir) + 1e-6)
        self.amplitude, self.frequency = amplitude, frequency
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); angle = self.amplitude * np.sin(2 * np.pi * self.frequency * self._t * self.dt)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return self._velocity_to_acceleration(rot @ self.base_dir, vel_self)
class Spiral(BaseController):
    def __init__(self, angular_speed=1.0, radial_speed=0.5, **kwargs):
        super().__init__(**kwargs); self.angular_speed, self.radial_speed, self.center = angular_speed, radial_speed, np.zeros(2)
    def reset(self, initial_pos=None):
        super().reset(initial_pos); self.center = initial_pos[:2].copy() if initial_pos is not None else np.zeros(2)
    def compute(self, pos_self, vel_self, pos_target, vel_target):
        self._advance(); rel_pos = pos_self[:2] - self.center; dist = np.linalg.norm(rel_pos)
        theta = 0.0 if dist < 1e-3 else np.arctan2(rel_pos[1], rel_pos[0])
        radial_dir = np.array([np.cos(theta), np.sin(theta)]); tangential_dir = np.array([-np.sin(theta), np.cos(theta)])
        v_r = self.radial_speed * np.sin(self._t * self.dt * self.angular_speed)
        desired_dir = v_r * radial_dir + self.angular_speed * tangential_dir
        return self._velocity_to_acceleration(desired_dir / (np.linalg.norm(desired_dir) + 1e-6), vel_self)
""",
    "pursuit_evasion/controllers/pursuer.py": """import numpy as np
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
        los = pos_target - pos_self; angle = np.arctan2(los[1], los[0])
        rate = 0.0 if self.prev_angle is None else ((angle - self.prev_angle + np.pi) % (2 * np.pi) - np.pi) / self.dt
        self.prev_angle = angle; self._advance()
        lead = angle + self.N * rate * self.dt
        return self._velocity_to_acceleration(np.array([np.cos(lead), np.sin(lead)]), vel_self)
""",
    "pursuit_evasion/controllers/evader.py": """import numpy as np
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
        flee = los / dist; dodge = np.array([-flee[1], flee[0]])
        desired = flee + self.dodge_weight * dodge
        return self._velocity_to_acceleration(desired / (np.linalg.norm(desired) + 1e-6), vel_self)
""",

    # --- ENVIRONMENTS ---
    "pursuit_evasion/envs/__init__.py": "",

    "pursuit_evasion/envs/base_env.py": """import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BasePursuitEvasionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, mode="pursuer", obs_type="relative", dim=2, arena_size=20.0, 
                 max_steps=300, v_max=1.0, a_max=1.0, capture_radius=0.3,
                 gravity=0.0, reset_range=(1.0, 5.0),
                 pursuer_controller=None, evader_controller=None, render_mode=None):
        super().__init__()
        self.mode, self.obs_type, self.dim, self.arena_size = mode, obs_type, dim, arena_size
        self.max_steps, self.v_max, self.a_max, self.capture_radius = max_steps, v_max, a_max, capture_radius
        self.gravity, self.reset_range = gravity, reset_range
        self.render_mode, self.pursuer_controller, self.evader_controller = render_mode, pursuer_controller, evader_controller
        self.dt, self.use_physics_backend = 0.1, False

        if self.pursuer_controller:
            self.pursuer_controller.v_max, self.pursuer_controller.a_max = self.v_max, self.a_max
            self.pursuer_controller.dt, self.pursuer_controller.dim = self.dt, self.dim
        if self.evader_controller:
            self.evader_controller.v_max, self.evader_controller.a_max = self.v_max, self.a_max
            self.evader_controller.dt, self.evader_controller.dim = self.dt, self.dim

        self.action_space = spaces.Box(low=-a_max, high=a_max, shape=(dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((3*dim,) if self.obs_type=="relative" else (4,)), dtype=np.float32)

        self.step_count, self.prev_los_angle = 0, None
        self.current_los_rate = 0.0

    def _get_obs(self, role):
        if role == "pursuer": pos_self, vel_self, pos_target, vel_target = self.p_pos, self.p_vel, self.e_pos, self.e_vel
        else: pos_self, vel_self, pos_target, vel_target = self.e_pos, self.e_vel, self.p_pos, self.p_vel
        rel_pos, rel_vel = pos_target - pos_self, vel_target - vel_self
        if self.obs_type == "relative": return np.concatenate([rel_pos, rel_vel, vel_self]).astype(np.float32)
        elif self.obs_type == "los":
            rng = np.linalg.norm(rel_pos); closing = np.dot(rel_pos, rel_vel) / (rng + 1e-6)
            los = np.arctan2(rel_pos[1], rel_pos[0])
            return np.array([rng, los, self.current_los_rate, closing], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count, self.prev_los_angle, self.current_los_rate = 0, None, 0.0
        self.p_pos = self.np_random.uniform(-self.arena_size/2, self.arena_size/2, self.dim)
        
        while True:
            self.e_pos = self.np_random.uniform(-self.arena_size/2, self.arena_size/2, self.dim)
            dist = np.linalg.norm(self.p_pos - self.e_pos)
            if self.reset_range[0] <= dist <= self.reset_range[1]: break

        self.p_vel, self.e_vel = np.zeros(self.dim), np.zeros(self.dim)
        if self.pursuer_controller: self.pursuer_controller.reset(self.p_pos)
        if self.evader_controller: self.evader_controller.reset(self.e_pos)
        
        if self.mode == "multi": return {"pursuer": self._get_obs("pursuer"), "evader": self._get_obs("evader")}, {}
        return self._get_obs(self.mode), {}

    def step(self, action):
        self.step_count += 1
        if self.mode == "multi": a_p, a_e = action["pursuer"], action["evader"]
        elif self.mode == "pursuer":
            a_p = action; a_e = self.evader_controller.compute(self.e_pos, self.e_vel, self.p_pos, self.p_vel)
        elif self.mode == "evader":
            a_e = action; a_p = self.pursuer_controller.compute(self.p_pos, self.p_vel, self.e_pos, self.e_vel)

        a_p, a_e = np.clip(a_p, -self.a_max, self.a_max), np.clip(a_e, -self.a_max, self.a_max)

        if not self.use_physics_backend:
            self.p_vel += a_p * self.dt; self.e_vel += a_e * self.dt
            if np.linalg.norm(self.p_vel) > self.v_max: self.p_vel = (self.p_vel / np.linalg.norm(self.p_vel)) * self.v_max
            if np.linalg.norm(self.e_vel) > self.v_max: self.e_vel = (self.e_vel / np.linalg.norm(self.e_vel)) * self.v_max
            self.p_pos += self.p_vel * self.dt; self.e_pos += self.e_vel * self.dt
            self.p_pos, self.p_vel = self._apply_walls(self.p_pos, self.p_vel)
            self.e_pos, self.e_vel = self._apply_walls(self.e_pos, self.e_vel)
        else: self._step_physics(a_p, a_e)

        if self.obs_type == "los":
            rel_pos = self.e_pos - self.p_pos; rng = np.linalg.norm(rel_pos) + 1e-6
            los = np.arctan2(rel_pos[1], rel_pos[0])
            if self.prev_los_angle is not None:
                d_los = (los - self.prev_los_angle + np.pi) % (2 * np.pi) - np.pi
                self.current_los_rate = d_los / self.dt
            else: self.current_los_rate = 0.0
            self.prev_los_angle = los

        dist = np.linalg.norm(self.p_pos - self.e_pos)
        captured, truncated = dist < self.capture_radius, self.step_count >= self.max_steps
        reward_p = 10.0 if captured else -0.1 - (dist / self.arena_size)
        reward_e = -10.0 if captured else -0.1 + (dist / self.arena_size)

        if self.mode == "multi": return {"pursuer": self._get_obs("pursuer"), "evader": self._get_obs("evader")}, {"pursuer": reward_p, "evader": reward_e}, {"pursuer": captured, "evader": captured}, {"pursuer": truncated, "evader": truncated}, {}
        return self._get_obs(self.mode), (reward_p if self.mode == "pursuer" else reward_e), captured, truncated, {}

    def _apply_walls(self, pos, vel):
        for i in range(self.dim):
            if pos[i] > self.arena_size/2: pos[i], vel[i] = self.arena_size/2, vel[i] * -0.5
            elif pos[i] < -self.arena_size/2: pos[i], vel[i] = -self.arena_size/2, vel[i] * -0.5
        return pos, vel
""",

    "pursuit_evasion/envs/mujoco_env.py": """import os, mujoco, numpy as np
from .base_env import BasePursuitEvasionEnv
class MujocoPursuitEvasionEnv(BasePursuitEvasionEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.use_physics_backend, self.mass = True, 1.0
        xml_path = os.path.join(os.path.dirname(__file__), "..", "assets", "world.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path); self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[2] = -self.gravity
        self.viewer, self.renderer = None, None
        self.p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pursuer")
        self.e_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "evader")
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs); mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = self.p_pos; self.data.qpos[7:9] = self.e_pos; mujoco.mj_forward(self.model, self.data)
        return obs, info
    def _step_physics(self, a_p, a_e):
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.p_id, 0:2], self.data.xfrc_applied[self.e_id, 0:2] = a_p * self.mass, a_e * self.mass
        mujoco.mj_step(self.model, self.data)
        self.p_pos, self.p_vel = self.data.qpos[0:2].copy(), self.data.qvel[0:2].copy()
        self.e_pos, self.e_vel = self.data.qpos[7:9].copy(), self.data.qvel[7:9].copy()
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer; self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None: self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="top_down"); return self.renderer.render()
    def close(self):
        if self.viewer: self.viewer.close()
""",

    "pursuit_evasion/envs/pybullet_env.py": """import os
import pybullet as p
import pybullet_data
import numpy as np
from .base_env import BasePursuitEvasionEnv

class PyBulletPursuitEvasionEnv(BasePursuitEvasionEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_physics_backend = True
        self.mass = 1.0
        self.client = p.connect(p.DIRECT if self.render_mode != "human" else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -self.gravity, physicsClientId=self.client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        asset_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        self.p_body = p.loadURDF(os.path.join(asset_dir, "pursuer.urdf"), physicsClientId=self.client)
        self.e_body = p.loadURDF(os.path.join(asset_dir, "evader.urdf"), physicsClientId=self.client)
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        p.resetBasePositionAndOrientation(self.p_body, np.pad(self.p_pos, (0, 1)), [0,0,0,1], physicsClientId=self.client)
        p.resetBaseVelocity(self.p_body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.e_body, np.pad(self.e_pos, (0, 1)), [0,0,0,1], physicsClientId=self.client)
        p.resetBaseVelocity(self.e_body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        return obs, info
    def _step_physics(self, a_p, a_e):
        p.applyExternalForce(self.p_body, -1, np.pad(a_p * self.mass, (0, 1)), [0,0,0], p.WORLD_FRAME, physicsClientId=self.client)
        p.applyExternalForce(self.e_body, -1, np.pad(a_e * self.mass, (0, 1)), [0,0,0], p.WORLD_FRAME, physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        p_pos, _ = p.getBasePositionAndOrientation(self.p_body, physicsClientId=self.client)
        p_vel, _ = p.getBaseVelocity(self.p_body, physicsClientId=self.client)
        e_pos, _ = p.getBasePositionAndOrientation(self.e_body, physicsClientId=self.client)
        e_vel, _ = p.getBaseVelocity(self.e_body, physicsClientId=self.client)
        self.p_pos, self.p_vel = np.array(p_pos[:self.dim]), np.array(p_vel[:self.dim])
        self.e_pos, self.e_vel = np.array(e_pos[:self.dim]), np.array(e_vel[:self.dim])
    def render(self):
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrix([0, 0, 10], [0, 0, 0], [0, 1, 0], physicsClientId=self.client)
            proj_matrix = p.computeProjectionMatrixFOV(60, 640/480, 0.1, 100.0, physicsClientId=self.client)
            _, _, rgb, _, _ = p.getCameraImage(640, 480, view_matrix, proj_matrix, physicsClientId=self.client)
            return np.reshape(rgb, (480, 640, 4))[:, :, :3]
    def close(self):
        p.disconnect(self.client)
""",

    # --- WRAPPERS & TRAINING ---
    "pursuit_evasion/wrappers.py": """import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import OrderedDict

class GoalConditionedWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(env.dim,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(env.dim,), dtype=np.float32)
        })
        self.dim = env.dim
    def observation(self, obs):
        unwrapped = self.env.unwrapped
        if self.env.mode == "pursuer":
            achieved_goal, desired_goal = unwrapped.p_pos.copy(), unwrapped.e_pos.copy()
        else:
            achieved_goal, desired_goal = unwrapped.e_pos.copy(), unwrapped.p_pos.copy()
        return OrderedDict([("observation", obs), ("achieved_goal", achieved_goal.astype(np.float32)), ("desired_goal", desired_goal.astype(np.float32))])
""",

    "train.py": """import argparse
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her import HerReplayBuffer
from pursuit_evasion.envs.mujoco_env import MujocoPursuitEvasionEnv
from pursuit_evasion.wrappers import GoalConditionedWrapper
from pursuit_evasion.controllers.base import Idle, StraightLine, Spiral
from pursuit_evasion.controllers.pursuer import PurePursuit, ProportionalNavigation
from pursuit_evasion.controllers.evader import PureEvasion, ProportionalEvasion

class CurriculumCallback(BaseCallback):
    def __init__(self, role, verbose=0):
        super().__init__(verbose); self.role = role
        self.levels = [
            {"step": 0,       "gravity": 0.0,  "range": (1, 5),  "opponent": Idle},
            {"step": 20_000,  "gravity": 0.0,  "range": (1, 5),  "opponent": Idle},
            {"step": 40_000,  "gravity": 9.81, "range": (1, 5),  "opponent": Idle},
            {"step": 60_000,  "gravity": 9.81, "range": (1, 5) if role=="evader" else (5, 20), "opponent": StraightLine},
            {"step": 80_000,  "gravity": 9.81, "range": (5, 20), "opponent": Spiral},
            {"step": 100_000, "gravity": 9.81, "range": (5, 20), "opponent": PurePursuit if role=="evader" else PureEvasion},
            {"step": 120_000, "gravity": 9.81, "range": (5, 20), "opponent": ProportionalNavigation if role=="evader" else ProportionalEvasion},
        ]
    def _on_step(self):
        current_level = self.levels[0]
        for lvl in self.levels:
            if self.num_timesteps >= lvl["step"]: current_level = lvl
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            print(f"[{self.num_timesteps} steps] Level: Gravity={current_level['gravity']}, Opponent={current_level['opponent'].__name__}")
        for env in self.training_env.envs:
            unwrapped = env.unwrapped
            unwrapped.gravity = current_level["gravity"]; unwrapped.reset_range = current_level["range"]
            new_opponent = current_level["opponent"]()
            if self.role == "pursuer": unwrapped.evader_controller = new_opponent
            else: unwrapped.pursuer_controller = new_opponent
        return True

def make_env(role, use_her, rank):
    def _init():
        env = MujocoPursuitEvasionEnv(mode=role, obs_type="relative", gravity=0.0, reset_range=(1, 5))
        if use_her:
            env = GoalConditionedWrapper(env); env.mode = role
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "td3"])
    parser.add_argument("--role", type=str, default="pursuer", choices=["pursuer", "evader"])
    parser.add_argument("--her", action="store_true")
    parser.add_argument("--n_envs", type=int, default=4)
    args = parser.parse_args()

    if args.her and args.algo == "ppo": raise ValueError("HER only works with SAC or TD3.")

    env = SubprocVecEnv([make_env(args.role, args.her, i) for i in range(args.n_envs)])
    print(f"Starting {args.algo.upper()} for {args.role.upper()} with {args.n_envs} envs. HER={args.her}")
    
    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=f"./logs/{args.algo}_{args.role}")
    else:
        rb_class = HerReplayBuffer if args.her else None
        rb_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future") if args.her else None
        AlgoClass = SAC if args.algo == "sac" else TD3
        model = AlgoClass("MlpPolicy", env, verbose=1, device="cpu", replay_buffer_class=rb_class, replay_buffer_kwargs=rb_kwargs, tensorboard_log=f"./logs/{args.algo}_{args.role}")

    model.learn(total_timesteps=150_000, callback=CurriculumCallback(role=args.role, verbose=1), tb_log_name="run_1")
    model.save(f"{args.algo}_{args.role}_final")
""",

    "evaluate.py": """import argparse
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from pursuit_evasion.envs.mujoco_env import MujocoPursuitEvasionEnv
from pursuit_evasion.controllers.pursuer import ProportionalNavigation
from pursuit_evasion.controllers.evader import ProportionalEvasion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "td3"], required=True)
    parser.add_argument("--role", type=str, choices=["pursuer", "evader"], required=True)
    args = parser.parse_args()

    env = MujocoPursuitEvasionEnv(mode=args.role, render_mode="human", gravity=9.81, reset_range=(5, 20), max_steps=300)
    if args.role == "pursuer": env.evader_controller = ProportionalEvasion()
    else: env.pursuer_controller = ProportionalNavigation()

    model = PPO.load(args.model, env=env) if args.algo == "ppo" else (SAC.load(args.model, env=env) if args.algo == "sac" else TD3.load(args.model, env=env))

    wins, rewards = 0, []
    for _ in range(20):
        obs, _ = env.reset(); done, ep_r = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action); ep_r += r; done = term or trunc
        rewards.append(ep_r)
        if ep_r > 0: wins += 1
    print(f"Win Rate: {wins/20*100:.1f}% | Mean Reward: {np.mean(rewards):.2f}")
    env.close()
"""
}

def create_project():
    base_dir = "pursuit_evasion_project"
    print(f"Creating project structure in: {os.path.abspath(base_dir)}")
    for filepath, content in FILES.items():
        full_path = os.path.join(base_dir, filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True) if os.path.dirname(full_path) else None
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [Created] {filepath}")
        
    print("\n✅ Project generated successfully!")

if __name__ == "__main__":
    create_project()