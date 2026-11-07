from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = Path(__file__).with_name("train_bullet_curriculum.py")
EXPERIMENTS = [
    (algorithm, role, False)
    for role in ("pursuer", "evader")
    for algorithm in ("ppo", "sac", "td3", "a2c")
] + [
    ("sac", "pursuer", True),
    ("td3", "pursuer", True),
]


def build_command(
    algorithm: str,
    role: str,
    use_her: bool,
    n_envs: int,
    timesteps: int,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--algo",
        algorithm,
        "--role",
        role,
        "--n-envs",
        str(n_envs),
        "--timesteps",
        str(timesteps),
        "--seed",
        str(seed),
    ]
    if use_her:
        command.append("--her")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the supported PyBullet curriculum experiments in separate "
            "terminals with a bounded number active at once."
        )
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    tensorboard_group = parser.add_mutually_exclusive_group()
    tensorboard_group.add_argument(
        "--tensorboard",
        dest="tensorboard",
        action="store_true",
        help="Start TensorBoard alongside the training runs (default).",
    )
    tensorboard_group.add_argument(
        "--no-tensorboard",
        dest="tensorboard",
        action="store_false",
        help="Do not start TensorBoard.",
    )
    parser.set_defaults(tensorboard=True)
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List planned runs without launching them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_workers < 1 or args.n_envs < 1 or args.timesteps < 1:
        raise SystemExit("--max-workers, --n-envs, and --timesteps must be positive.")

    if args.list:
        for index, (algorithm, role, use_her) in enumerate(EXPERIMENTS, start=1):
            suffix = "HER" if use_her else "NoHER"
            print(f"{index:02d}: {algorithm.upper()} {role} {suffix}")
        print(f"{len(EXPERIMENTS)} total runs; maximum concurrent: {args.max_workers}")
        return

    windows = os.name == "nt"
    creation_flags = subprocess.CREATE_NEW_CONSOLE if windows else 0
    active: list[tuple[subprocess.Popen, str]] = []
    pending = list(enumerate(EXPERIMENTS))
    tensorboard_process = None
    print(
        f"Launching {len(pending)} experiments; up to {args.max_workers} "
        "training terminals will run at once."
    )
    try:
        if args.tensorboard:
            log_dir = PROJECT_ROOT / "logs" / "bullet_curriculum"
            log_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(log_dir),
                    "--port",
                    str(args.tensorboard_port),
                ],
                cwd=PROJECT_ROOT,
                creationflags=creation_flags,
                start_new_session=not windows,
            )
            print(
                f"TensorBoard started (PID {tensorboard_process.pid}): "
                f"http://localhost:{args.tensorboard_port}"
            )

        while pending or active:
            while pending and len(active) < args.max_workers:
                index, (algorithm, role, use_her) = pending.pop(0)
                run_name = (
                    f"bullet_{algorithm}_{role}_{'HER' if use_her else 'NoHER'}"
                )
                command = build_command(
                    algorithm,
                    role,
                    use_her,
                    args.n_envs,
                    args.timesteps,
                    args.seed + index,
                )
                command.extend(["--run-name", run_name])
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    creationflags=creation_flags,
                    start_new_session=not windows,
                )
                active.append((process, run_name))
                print(f"Started {run_name} (PID {process.pid})")

            for process, run_name in active[:]:
                exit_code = process.poll()
                if exit_code is not None:
                    active.remove((process, run_name))
                    print(f"Finished {run_name} with exit code {exit_code}")
            if active:
                time.sleep(2.0)
    except KeyboardInterrupt:
        print("Stopping active training processes...")
        for process, _ in active:
            process.terminate()
        for process, run_name in active:
            process.wait()
            print(f"Stopped {run_name}")
        if tensorboard_process is not None:
            tensorboard_process.terminate()
            tensorboard_process.wait()
            print("Stopped TensorBoard.")


if __name__ == "__main__":
    main()
