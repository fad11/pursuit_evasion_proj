import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from control.evader_control import RandomEvaderController
from envs.pybullet_env import PursuitEvasionEnv


N_EPISODES = 50
SEED = 0
LINEAR_DAMPING = 0.1


def main() -> None:
    env = PursuitEvasionEnv(
        render_mode="DIRECT",
        gravity=0.0,
        linear_damping=LINEAR_DAMPING,
        evader_controller=RandomEvaderController(),
    )
    successes = 0
    episode_lengths: list[int] = []
    try:
        for episode in range(N_EPISODES):
            obs, _ = env.reset(seed=SEED + episode)
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                action = np.asarray([obs[1], obs[2]], dtype=np.float32)
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
            successes += int(terminated)
            episode_lengths.append(steps)
    finally:
        env.close()

    success_rate = successes / N_EPISODES
    mean_episode_length = float(np.mean(episode_lengths))
    print(f"Linear damping: {LINEAR_DAMPING}")
    print(f"Success rate: {successes}/{N_EPISODES} ({success_rate:.1%})")
    print(f"Mean episode length: {mean_episode_length:.1f} steps")


if __name__ == "__main__":
    main()
