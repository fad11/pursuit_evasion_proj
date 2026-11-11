import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from callbacks import VideoRecordCallback
from control.evader_control import RandomEvaderController
from envs.pybullet_env import PursuitEvasionEnv


N_ENVS = 16
N_STEPS = 512
BATCH_SIZE = 2048
N_EPOCHS = 5
GAMMA = 0.995
GAE_LAMBDA = 0.95
ENT_COEF = 0.0
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 2_000_000
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 10
CHECKPOINT_INTERVAL = 100_000
VIDEO_INTERVAL = 10_000

LOG_ROOT = Path("./logs")
MODEL_ROOT = Path("./models")


class SaveBestVecNormalizeCallback(BaseCallback):
    def __init__(self, normalizer: VecNormalize, save_path: Path) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.normalizer.save(str(self.save_path))
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pursuit policy.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--run-name",
        default=datetime.now().strftime("ppo_%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be a positive integer")
    if not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a non-empty directory name")
    return args


def make_env() -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        render_mode="DIRECT",
        gravity=0.0,
        evader_controller=RandomEvaderController(),
    )


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_run_config(args: argparse.Namespace, git_commit: str) -> dict[str, Any]:
    return {
        "run_name": args.run_name,
        "seed": args.seed,
        "total_timesteps": args.timesteps,
        "model_dir": str(args.model_dir),
        "n_envs": N_ENVS,
        "ppo": {
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "ent_coef": ENT_COEF,
            "learning_rate": LEARNING_RATE,
        },
        "eval_freq_calls": max(EVAL_INTERVAL // N_ENVS, 1),
        "eval_episodes": EVAL_EPISODES,
        "checkpoint_freq_calls": max(CHECKPOINT_INTERVAL // N_ENVS, 1),
        "git_commit": git_commit,
    }


def main() -> None:
    args = parse_args()
    run_log_dir = LOG_ROOT / args.run_name
    run_log_dir.mkdir(parents=True, exist_ok=False)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    git_commit = get_git_commit()
    config = build_run_config(args, git_commit)
    config_path = run_log_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)

    validation_env = make_env()
    try:
        check_env(validation_env, warn=True)
    finally:
        validation_env.close()

    env = VecNormalize(
        make_vec_env(
            make_env,
            n_envs=N_ENVS,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv,
            monitor_kwargs={"info_keywords": ("is_success",)},
        ),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    eval_env: VecNormalize | None = None
    try:
        eval_env = VecNormalize(
            make_vec_env(
                make_env,
                n_envs=1,
                seed=args.seed + N_ENVS,
                vec_env_cls=SubprocVecEnv,
                monitor_kwargs={"info_keywords": ("is_success",)},
            ),
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            learning_rate=LEARNING_RATE,
            verbose=1,
            tensorboard_log=str(run_log_dir),
            seed=args.seed,
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(args.model_dir),
            log_path=str(run_log_dir),
            eval_freq=max(EVAL_INTERVAL // N_ENVS, 1),
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            callback_on_new_best=SaveBestVecNormalizeCallback(
                env,
                args.model_dir / "best_vecnormalize.pkl",
            ),
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=max(CHECKPOINT_INTERVAL // N_ENVS, 1),
            save_path=str(args.model_dir),
            name_prefix=f"{args.run_name}_checkpoint",
            save_vecnormalize=True,
        )
        video_callback = VideoRecordCallback(
            video_folder="./videos/",
            record_freq=VIDEO_INTERVAL,
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(
                [eval_callback, checkpoint_callback, video_callback]
            ),
        )
        model.save(str(args.model_dir / "ppo_pursuit_final.zip"))
        env.save("vec_normalize.pkl")
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
