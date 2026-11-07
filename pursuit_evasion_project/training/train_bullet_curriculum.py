from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.her import HerReplayBuffer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pursuit_evasion.controllers.base import (  # noqa: E402
    Circular,
    Idle,
    Spiral,
    SquarePath,
    StraightLine,
    Zigzag,
)
from pursuit_evasion.controllers.evader import (  # noqa: E402
    ManeuveringEvasion,
    ProportionalEvasion,
    PureEvasion,
)
from pursuit_evasion.controllers.pursuer import (  # noqa: E402
    ProportionalNavigation,
    PurePursuit,
)
from pursuit_evasion.envs.pursuit_evasion_bullet import (  # noqa: E402
    PursuitEvasionBullet,
)
from pursuit_evasion.wrappers import GoalConditionedWrapper  # noqa: E402


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    gravity: float
    reset_range: tuple[float, float]
    opponent_classes: tuple[type, ...]


def build_curriculum(role: str) -> tuple[CurriculumStage, ...]:
    if role not in ("pursuer", "evader"):
        raise ValueError("role must be 'pursuer' or 'evader'")

    stages = [
        CurriculumStage("stationary_no_gravity", 0.0, (1.0, 5.0), (Idle,)),
        CurriculumStage("stationary_with_gravity", 9.81, (3.0, 10.0), (Idle,)),
        CurriculumStage("straight_line", 9.81, (3.0, 10.0), (StraightLine,)),
        CurriculumStage(
            "random_maneuvers",
            9.81,
            (5.0, 15.0),
            (Zigzag, Circular, SquarePath),
        ),
        CurriculumStage("spiral", 9.81, (5.0, 15.0), (Spiral,)),
    ]
    if role == "pursuer":
        stages.extend(
            (
                CurriculumStage("pure_evasion", 9.81, (5.0, 20.0), (PureEvasion,)),
                CurriculumStage(
                    "proportional_evasion",
                    9.81,
                    (5.0, 20.0),
                    (ProportionalEvasion,),
                ),
                CurriculumStage(
                    "maneuvering_evasion",
                    9.81,
                    (5.0, 20.0),
                    (ManeuveringEvasion,),
                ),
            )
        )
    else:
        stages.extend(
            (
                CurriculumStage("pure_pursuit", 9.81, (5.0, 20.0), (PurePursuit,)),
                CurriculumStage(
                    "proportional_navigation",
                    9.81,
                    (5.0, 20.0),
                    (ProportionalNavigation,),
                ),
            )
        )
    return tuple(stages)


class BulletCurriculumWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, role: str, stages: tuple[CurriculumStage, ...]):
        super().__init__(env)
        self.role = role
        self.stages = stages
        self.level_index = 0
        self._opponent_index = 0
        self.set_curriculum_level(0)

    def _opponent_attribute(self) -> str:
        return "evader_controller" if self.role == "pursuer" else "pursuer_controller"

    def _new_controller(self, controller_class: type):
        bullet_env = self.unwrapped
        return controller_class(
            dim=bullet_env.dim,
            v_max=bullet_env.v_max,
            a_max=bullet_env.a_max,
            dt=bullet_env.dt,
        )

    def _set_opponent_controller(self, controller_class: type) -> None:
        bullet_env = self.unwrapped
        controller = self._new_controller(controller_class)
        setattr(bullet_env, self._opponent_attribute(), controller)
        initial_position = (
            bullet_env.e_pos if self.role == "pursuer" else bullet_env.p_pos
        )
        controller.reset(initial_position.copy())

    def set_curriculum_level(self, level_index: int) -> None:
        if not 0 <= level_index < len(self.stages):
            raise ValueError(f"invalid curriculum level index: {level_index}")
        self.level_index = level_index
        stage = self.stages[level_index]
        bullet_env = self.unwrapped
        bullet_env.gravity = stage.gravity
        bullet_env.reset_range = stage.reset_range
        import pybullet as pybullet

        pybullet.setGravity(
            0.0,
            0.0,
            -stage.gravity,
            physicsClientId=bullet_env.client,
        )
        self._opponent_index = 0
        self._set_opponent_controller(stage.opponent_classes[0])

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        stage = self.stages[self.level_index]
        if len(stage.opponent_classes) > 1:
            bullet_env = self.unwrapped
            self._opponent_index = int(
                bullet_env.np_random.integers(len(stage.opponent_classes))
            )
            controller_class = stage.opponent_classes[self._opponent_index]
            self._set_opponent_controller(controller_class)
        return observation, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        compute_reward = self.env.get_wrapper_attr("compute_reward")
        return compute_reward(achieved_goal, desired_goal, info)


def make_env(role: str, use_her: bool, seed: int):
    stages = build_curriculum(role)

    def _initialize():
        env = PursuitEvasionBullet(
            mode=role,
            obs_type="relative",
            gravity=stages[0].gravity,
            reset_range=stages[0].reset_range,
            pursuer_controller=Idle(dim=2) if role == "evader" else None,
            evader_controller=Idle(dim=2) if role == "pursuer" else None,
        )
        env.reset(seed=seed)
        if use_her:
            env = GoalConditionedWrapper(env)
        return BulletCurriculumWrapper(env, role, stages)

    return _initialize


def make_eval_env(role: str, use_her: bool, seed: int):
    return make_env(role, use_her, seed)()


class CurriculumEvalCallback(BaseCallback):
    def __init__(
        self,
        role: str,
        eval_env: BulletCurriculumWrapper,
        eval_freq: int,
        eval_episodes: int,
        success_threshold: float,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.role = role
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.success_threshold = success_threshold
        self.next_eval_timestep = eval_freq
        self.current_level_idx = 0
        self.stages = build_curriculum(role)

    def _set_level(self, level_index: int) -> None:
        self.current_level_idx = level_index
        self.training_env.env_method("set_curriculum_level", level_index)
        self.eval_env.set_curriculum_level(level_index)
        stage = self.stages[level_index]
        if self.verbose:
            print(
                f"Curriculum level {level_index + 1}/{len(self.stages)}: "
                f"{stage.name}; gravity={stage.gravity}, "
                f"spawn_range={stage.reset_range}"
            )

    def _on_training_start(self) -> None:
        self._set_level(0)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval_timestep:
            return True
        while self.next_eval_timestep <= self.num_timesteps:
            self.next_eval_timestep += self.eval_freq

        successes = 0
        for episode in range(self.eval_episodes):
            observation, _ = self.eval_env.reset(seed=episode)
            done = False
            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
            successes += int(bool(info.get("is_success", False)))

        success_rate = successes / self.eval_episodes
        stage = self.stages[self.current_level_idx]
        print(
            f"[{self.num_timesteps} steps] {stage.name}: "
            f"success rate {success_rate:.0%}"
        )
        if (
            success_rate >= self.success_threshold
            and self.current_level_idx < len(self.stages) - 1
        ):
            self._set_level(self.current_level_idx + 1)
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "a2c": A2C,
}


def build_model(
    algorithm: str,
    env: SubprocVecEnv,
    use_her: bool,
    log_dir: Path,
):
    algorithm_class = ALGORITHMS[algorithm]
    if use_her:
        return algorithm_class(
            "MultiInputPolicy",
            env,
            verbose=1,
            device="cpu",
            tensorboard_log=str(log_dir),
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": 4,
                "goal_selection_strategy": "future",
            },
            learning_starts=301,
        )
    return algorithm_class(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        tensorboard_log=str(log_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyBullet pursuer or evader using a staged curriculum."
    )
    parser.add_argument("--algo", choices=tuple(ALGORITHMS), default="ppo")
    parser.add_argument("--role", choices=("pursuer", "evader"), default="pursuer")
    parser.add_argument("--her", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--success-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.her and args.algo not in ("sac", "td3"):
        raise SystemExit("HER is supported here only with SAC or TD3.")
    if args.her and args.role != "pursuer":
        raise SystemExit("HER is only supported for pursuer goal-reaching training.")
    if args.n_envs < 1 or args.timesteps < 1:
        raise SystemExit("--n-envs and --timesteps must be positive.")
    if args.eval_freq < 1 or args.eval_episodes < 1:
        raise SystemExit("--eval-freq and --eval-episodes must be positive.")
    if not 0.0 <= args.success_threshold <= 1.0:
        raise SystemExit("--success-threshold must be between 0 and 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    run_name = args.run_name or (
        f"bullet_{args.algo}_{args.role}_{'HER' if args.her else 'NoHER'}"
    )
    log_dir = PROJECT_ROOT / "logs" / "bullet_curriculum" / run_name
    model_dir = PROJECT_ROOT / "models" / "bullet_curriculum"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting {run_name}: {args.n_envs} parallel environments, "
        f"{args.timesteps} timesteps"
    )

    train_env = VecMonitor(
        SubprocVecEnv(
            [
                make_env(args.role, args.her, args.seed + index)
                for index in range(args.n_envs)
            ],
            start_method="spawn",
        )
    )
    eval_env = make_eval_env(args.role, args.her, args.seed + 10_000)
    model = None
    try:
        model = build_model(args.algo, train_env, args.her, log_dir)
        callback = CurriculumEvalCallback(
            role=args.role,
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            success_threshold=args.success_threshold,
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=run_name,
        )
        save_path = model_dir / run_name
        model.save(str(save_path))
        print(f"Saved model: {save_path}.zip")
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
