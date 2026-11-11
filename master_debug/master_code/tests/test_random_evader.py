import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import numpy as np

from envs.pybullet_env import PursuitEvasionEnv


def main() -> None:
    env = PursuitEvasionEnv(render_mode="GUI", gravity=0.0)
    try:
        env.reset()
        action = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(300):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            time.sleep(1 /240)
    finally:
        env.close()


if __name__ == "__main__":
    main()
