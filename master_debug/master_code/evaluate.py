import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from control import RandomEvaderController
from pybullet_env import PursuitEvasionEnv


def make_env() -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        render_mode="GUI",
        gravity=0.0,
        evader_controller=RandomEvaderController(),
    )


def main() -> None:
    base_env = DummyVecEnv([make_env])
    env = base_env
    try:
        env = VecNormalize.load("vec_normalize.pkl", base_env)
        env.training = False
        env.norm_reward = False

        model_path = "models/best_model.zip"
        if not os.path.exists(model_path):
            model_path = "models/ppo_pursuit_final.zip"
        model = PPO.load(model_path)

        for trial in range(1, 13):
            obs = env.reset()
            done = False
            step = 0
            while not done and step < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, _ = env.step(action)
                time.sleep(1 / 60)
                done = bool(np.asarray(dones).reshape(-1)[0])
                step += 1

            print(f"Trial {trial}/12 finished in {step} steps.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
