import argparse
import os
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her import HerReplayBuffer
from pursuit_evasion.envs.mujoco_env import MujocoPursuitEvasionEnv
from pursuit_evasion.wrappers import GoalConditionedWrapper
from pursuit_evasion.controllers.base import Idle, StraightLine, Spiral
from pursuit_evasion.controllers.pursuer import PurePursuit, ProportionalNavigation
from pursuit_evasion.controllers.evader import PureEvasion, ProportionalEvasion

class CurriculumEvalCallback(BaseCallback):
    def __init__(self, role, eval_env, verbose=1):
        super().__init__(verbose)
        self.role = role
        self.eval_env = eval_env
        self.current_level_idx = 0
        self.next_eval_timestep = 10_000
        self.levels = [
            {"name": "L1_Idle_NoGrav", "gravity": 0.0,  "range": (1, 5),  "opponent": Idle},
            {"name": "L2_Idle_Grav",   "gravity": 9.81, "range": (1, 5),  "opponent": Idle},
            {"name": "L3_Range_5_20",  "gravity": 9.81, "range": (5, 20), "opponent": Idle},
            {"name": "L4_Straight",    "gravity": 9.81, "range": (5, 20), "opponent": StraightLine},
            {"name": "L5_Spiral",      "gravity": 9.81, "range": (5, 20), "opponent": Spiral},
            {"name": "L6_Pure",        "gravity": 9.81, "range": (5, 20), "opponent": PurePursuit if role=="evader" else PureEvasion},
            {"name": "L7_Prop",        "gravity": 9.81, "range": (5, 20), "opponent": ProportionalNavigation if role=="evader" else ProportionalEvasion},
        ]

    def _update_envs(self):
        lvl = self.levels[self.current_level_idx]
        self.training_env.env_method(
            "configure_curriculum",
            lvl["gravity"], lvl["range"], self.role, lvl["opponent"]
        )
        self.eval_env.configure_curriculum(
            lvl["gravity"], lvl["range"], self.role, lvl["opponent"]
        )

    def _on_training_start(self):
        self._update_envs()

    def _on_step(self):
        if self.num_timesteps >= self.next_eval_timestep:
            while self.next_eval_timestep <= self.num_timesteps:
                self.next_eval_timestep += 10_000
            wins = 0
            episodes = 20
            for _ in range(episodes):
                obs, _ = self.eval_env.reset()
                done, is_success = False, False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = self.eval_env.step(action)
                    done = term or trunc
                is_success = info["is_success"]
                if is_success:
                    wins += 1
            win_rate = wins / episodes
            lvl = self.levels[self.current_level_idx]
            print(f"[{self.num_timesteps} steps] Evaluated on {lvl['name']}: Win Rate = {win_rate*100:.1f}%")
            if win_rate >= 0.9 and self.current_level_idx < len(self.levels) - 1:
                self.current_level_idx += 1
                new_lvl = self.levels[self.current_level_idx]
                print(f"GREEN PASSED! Advancing to Level {self.current_level_idx + 1}: {new_lvl['name']}")
                self._update_envs()
        return True

def make_train_env(role, use_her):
    def _init():
        env = MujocoPursuitEvasionEnv(mode=role, obs_type="relative", gravity=0.0, reset_range=(1, 5))
        if use_her:
            env = GoalConditionedWrapper(env)
        return env
    return _init

def make_eval_env(role, use_her):
    env = MujocoPursuitEvasionEnv(mode=role, obs_type="relative", gravity=0.0, reset_range=(1, 5))
    if use_her:
        env = GoalConditionedWrapper(env)
    return env

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
    if args.n_envs < 1:
        parser.error("--n_envs must be at least 1")

    run_name = f"{args.algo}_{args.role}_{'HER' if args.her else 'NoHER'}"
    print(f"Starting {run_name} with {args.n_envs} parallel envs...")

    train_env = SubprocVecEnv(
        [make_train_env(args.role, args.her) for _ in range(args.n_envs)],
        start_method="spawn"
    )
    eval_env = make_eval_env(args.role, args.her)
    policy = "MultiInputPolicy" if args.her else "MlpPolicy"

    if args.algo == "ppo":
        model = PPO(policy, train_env, verbose=0, device="cpu", tensorboard_log=f"./logs/{run_name}")
    else:
        rb_class = HerReplayBuffer if args.her else None
        rb_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future") if args.her else None
        AlgoClass = SAC if args.algo == "sac" else TD3
        model_kwargs = {}
        if args.her:
            model_kwargs["learning_starts"] = 301
        model = AlgoClass(
            policy, train_env, verbose=0, device="cpu",
            replay_buffer_class=rb_class, replay_buffer_kwargs=rb_kwargs,
            tensorboard_log=f"./logs/{run_name}", **model_kwargs
        )

    callback = CurriculumEvalCallback(role=args.role, eval_env=eval_env, verbose=1)

    try:
        model.learn(total_timesteps=300_000, callback=callback, tb_log_name="run_1")
        os.makedirs("models", exist_ok=True)
        model.save(f"models/{run_name}_final")
        print(f"Training Complete for {run_name}. Saved to models/{run_name}_final.zip")
    finally:
        train_env.close()
        eval_env.close()
