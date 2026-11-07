import os
import time

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from callbacks import VideoRecordCallback
from control import BaseEvaderController, RandomEvaderController
from pybullet_env import PursuitEvasionEnv


def make_env() -> TimeLimit:
    evader_controller: BaseEvaderController = RandomEvaderController()
    return TimeLimit(
        PursuitEvasionEnv(
            render_mode="DIRECT",
            gravity=0.0,
            evader_controller=evader_controller,
        ),
        max_episode_steps=1000,
    )


def make_gui_env() -> TimeLimit:
    evader_controller: BaseEvaderController = RandomEvaderController()
    return TimeLimit(
        PursuitEvasionEnv(
            render_mode="GUI",
            gravity=0.0,
            evader_controller=evader_controller,
        ),
        max_episode_steps=1000,
    )


def main() -> None:
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./videos/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    validation_env = make_env()
    try:
        check_env(validation_env, warn=True)
    finally:
        validation_env.close()

    env = VecNormalize(
        make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    eval_env: VecNormalize | None = None
    try:
        eval_env = VecNormalize(
            make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv),
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
        )
        video_callback = VideoRecordCallback(
            video_folder="./videos/",
            record_freq=10000,
        )
        model.learn(
            total_timesteps=200_000,
            callback=[eval_callback, video_callback],
        )
        model.save("./models/ppo_pursuit_final.zip")
        env.save("vec_normalize.pkl")
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()

    gui_vec_env = make_vec_env(make_gui_env, n_envs=1)
    eval_env = VecNormalize.load("vec_normalize.pkl", gui_vec_env)
    eval_env.training = False
    eval_env.norm_reward = False
    try:
        model = PPO.load("./models/ppo_pursuit_final.zip", env=eval_env)
        obs = eval_env.reset()
        cumulative_reward = 0.0
        for step in range(1500):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = eval_env.step(action)
            cumulative_reward += float(rewards[0])
            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}: cumulative reward = {cumulative_reward:.2f}")
            time.sleep(1 / 240)
            if dones[0]:
                break
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()
