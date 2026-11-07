import argparse
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
        self.current_level_idx = None
        self.levels = [
            {"step": 0,       "gravity": 0.0,  "range": (1, 5),  "opponent": Idle},
            {"step": 20_000,  "gravity": 0.0,  "range": (1, 5),  "opponent": Idle},
            {"step": 40_000,  "gravity": 9.81, "range": (1, 5),  "opponent": Idle},
            {"step": 60_000,  "gravity": 9.81, "range": (1, 5) if role=="evader" else (5, 20), "opponent": StraightLine},
            {"step": 80_000,  "gravity": 9.81, "range": (5, 20), "opponent": Spiral},
            {"step": 100_000, "gravity": 9.81, "range": (5, 20), "opponent": PurePursuit if role=="evader" else PureEvasion},
            {"step": 120_000, "gravity": 9.81, "range": (5, 20), "opponent": ProportionalNavigation if role=="evader" else ProportionalEvasion},
        ]

    def _apply_level(self, level_index):
        level = self.levels[level_index]
        self.training_env.env_method(
            "configure_curriculum",
            level["gravity"], level["range"], self.role, level["opponent"]
        )
        self.current_level_idx = level_index

    def _on_training_start(self):
        self._apply_level(0)

    def _on_step(self):
        level_index = max(
            index for index, level in enumerate(self.levels)
            if self.num_timesteps >= level["step"]
        )
        if level_index != self.current_level_idx:
            self._apply_level(level_index)
            level = self.levels[level_index]
            if self.verbose > 0:
                print(
                    f"[{self.num_timesteps} steps] Level {level_index + 1}: "
                    f"Gravity={level['gravity']}, Opponent={level['opponent'].__name__}"
                )
        return True

def make_env(role, use_her, rank):
    def _init():
        env = MujocoPursuitEvasionEnv(mode=role, obs_type="relative", gravity=0.0, reset_range=(1, 5))
        if use_her:
            env = GoalConditionedWrapper(env)
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
    if args.her and args.role != "pursuer":
        raise ValueError("HER is only supported for the pursuer role.")

    env = SubprocVecEnv(
        [make_env(args.role, args.her, i) for i in range(args.n_envs)],
        start_method="spawn"
    )
    print(f"Starting {args.algo.upper()} for {args.role.upper()} with {args.n_envs} envs. HER={args.her}")
    
    if args.algo == "ppo":
        model = PPO("MultiInputPolicy" if args.her else "MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=f"./logs/{args.algo}_{args.role}")
    else:
        rb_class = HerReplayBuffer if args.her else None
        rb_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future") if args.her else None
        AlgoClass = SAC if args.algo == "sac" else TD3
        model_kwargs = {"learning_starts": 301} if args.her else {}
        model = AlgoClass(
            "MultiInputPolicy" if args.her else "MlpPolicy", env,
            verbose=1, device="cpu", replay_buffer_class=rb_class,
            replay_buffer_kwargs=rb_kwargs,
            tensorboard_log=f"./logs/{args.algo}_{args.role}", **model_kwargs
        )

    try:
        model.learn(total_timesteps=150_000, callback=CurriculumCallback(role=args.role, verbose=1), tb_log_name="run_1")
        model.save(f"{args.algo}_{args.role}_final")
    finally:
        env.close()
