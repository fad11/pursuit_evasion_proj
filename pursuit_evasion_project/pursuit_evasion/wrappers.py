import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import OrderedDict

class GoalConditionedWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if env.mode != "pursuer":
            raise ValueError("HER goal relabeling is only supported for the pursuer role.")
        self.observation_space = spaces.Dict({
            "observation": env.observation_space,
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(env.dim,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(env.dim,), dtype=np.float32)
        })
        self.dim = env.dim
        self.capture_radius = env.capture_radius

    def observation(self, obs):
        unwrapped = self.env.unwrapped
        achieved_goal, desired_goal = unwrapped.p_pos.copy(), unwrapped.e_pos.copy()
        return OrderedDict([("observation", obs), ("achieved_goal", achieved_goal.astype(np.float32)), ("desired_goal", desired_goal.astype(np.float32))])

    def configure_curriculum(self, gravity, reset_range, role, opponent_class):
        self.env.unwrapped.configure_curriculum(
            gravity, reset_range, role, opponent_class
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(
            np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1
        )
        reward = np.where(distance < self.capture_radius, 0.0, -1.0)
        return reward.astype(np.float32)
