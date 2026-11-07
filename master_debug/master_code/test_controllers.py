import time

import numpy as np

from control.evader_control import CircularEvaderController
from pybullet_env import PursuitEvasionEnv


def main() -> None:
    env = PursuitEvasionEnv(
        render_mode="GUI",
        gravity=0.0,
        evader_controller=CircularEvaderController(),
    )
    try:
        env.reset()
        action = np.array([1.0, 1.0])
        for _ in range(300):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
            time.sleep(1 / 240)
    finally:
        env.close()


if __name__ == "__main__":
    main()
