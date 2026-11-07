import os

import numpy as np
import pybullet as p
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from pybullet_env import PursuitEvasionEnv


class VideoRecordCallback(BaseCallback):
    def __init__(self, video_folder: str, record_freq: int = 10000) -> None:
        super().__init__()
        if record_freq <= 0:
            raise ValueError("record_freq must be a positive integer")
        self.video_folder = video_folder
        self.record_freq = record_freq
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq == 0:
            env = PursuitEvasionEnv(render_mode="GUI", gravity=0.0)
            logging_id: int | None = None
            try:
                obs, _ = env.reset()
                video_path = os.path.join(
                    self.video_folder,
                    f"record_{self.n_calls}.mp4",
                )
                logging_id = p.startStateLogging(
                    p.STATE_LOGGING_VIDEO_MP4,
                    video_path,
                    physicsClientId=env.physics_client_id,
                )
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    prediction_obs = obs
                    if isinstance(self.training_env, VecNormalize):
                        prediction_obs = self.training_env.normalize_obs(
                            obs[np.newaxis, :]
                        )[0]
                    action, _ = self.model.predict(
                        prediction_obs,
                        deterministic=True,
                    )
                    obs, _, terminated, truncated, _ = env.step(
                        np.asarray(action).reshape(-1)
                    )
            finally:
                if logging_id is not None:
                    p.stopStateLogging(
                        logging_id,
                        physicsClientId=env.physics_client_id,
                    )
                env.close()
        return True
