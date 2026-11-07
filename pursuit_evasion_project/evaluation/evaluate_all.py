import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from pursuit_evasion.envs.mujoco_env import MujocoPursuitEvasionEnv
from pursuit_evasion.wrappers import GoalConditionedWrapper
from pursuit_evasion.controllers.base import Idle, StraightLine, Spiral, SquarePath
from pursuit_evasion.controllers.pursuer import PurePursuit, ProportionalNavigation
from pursuit_evasion.controllers.evader import PureEvasion, ProportionalEvasion

LEVELS = [
    {"name": "L1_Idle_NoGrav", "gravity": 0.0,  "range": (1, 5),  "opponent": Idle},
    {"name": "L2_Idle_Grav",   "gravity": 9.81, "range": (1, 5),  "opponent": Idle},
    {"name": "L3_Range_5_20",  "gravity": 9.81, "range": (5, 20), "opponent": Idle},
    {"name": "L4_Straight",    "gravity": 9.81, "range": (5, 20), "opponent": StraightLine},
    {"name": "L5_Spiral",      "gravity": 9.81, "range": (5, 20), "opponent": Spiral},
    {"name": "L6_Pure",        "gravity": 9.81, "range": (5, 20), "opponent": PurePursuit},
    {"name": "L7_Prop",        "gravity": 9.81, "range": (5, 20), "opponent": ProportionalNavigation},
]

def build_env(role, level, use_her=False, render_mode=None):
    opp = level["opponent"]()
    env = MujocoPursuitEvasionEnv(
        mode=role, obs_type="relative",
        gravity=level["gravity"], reset_range=level["range"],
        render_mode=render_mode
    )
    if role == "pursuer": env.evader_controller = opp
    else: env.pursuer_controller = opp
    if use_her:
        env = GoalConditionedWrapper(env)
    return env

def evaluate(model, role, level, use_her=False, episodes=20):
    env = build_env(role, level, use_her=use_her, render_mode=None)
    wins, rewards = 0, []
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            done, ep_r = False, 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                ep_r += reward
                done = term or trunc
            rewards.append(ep_r)
            if info["is_success"]:
                wins += 1
    finally:
        env.close()
    return wins / episodes, np.mean(rewards), np.std(rewards)

def plot_results(all_results):
    runs = list(all_results.keys())
    levels = [name for name, _, _ in next(iter(all_results.values()))]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(levels))
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        result_by_level = {name: result for name, *result in all_results[run]}
        win_rates = [result_by_level[level][0] for level in levels]
        ax.bar(x + i * width - (len(runs) - 1) * width / 2, win_rates,
               width, label=run)
    ax.set_ylabel("Win Rate (%)")
    ax.set_xlabel("Curriculum Level")
    ax.set_title("Pursuit-Evasion RL Evaluation: Win Rate per Curriculum Level")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=20, ha='right')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label="90% (Curriculum Advance)")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/win_rates.png", dpi=150)
    print("Saved: plots/win_rates.png")
    plt.close()
    fig, ax = plt.subplots(figsize=(14, 6))
    for run in runs:
        result_by_level = {name: result for name, *result in all_results[run]}
        rewards = [result_by_level[level][1] for level in levels]
        ax.plot(levels, rewards, marker='o', label=run)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_xlabel("Curriculum Level")
    ax.set_title("Mean Reward per Curriculum Level")
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/rewards.png", dpi=150)
    print("Saved: plots/rewards.png")
    plt.close()

def save_video(run_name, model, role, level, episodes=3):
    os.makedirs("videos", exist_ok=True)
    for episode in range(episodes):
        env = DummyVecEnv([
            lambda: build_env(role, level, use_her=False, render_mode="rgb_array")
        ])
        env = VecVideoRecorder(
            env, video_folder="videos",
            record_video_trigger=lambda step: step == 0,
            video_length=300,
            name_prefix=f"{run_name}_{level['name']}_episode{episode + 1}"
        )
        try:
            obs = env.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, _ = env.step(action)
                if dones[0]:
                    break
        finally:
            env.close()
    print(f"Saved {episodes} videos for {run_name} at level {level['name']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--record", action="store_true", help="Also save video clips")
    parser.add_argument("--level", type=str, default=None, help="Evaluate only one level (e.g. L7_Prop)")
    args = parser.parse_args()

    if not os.path.exists(args.models_dir):
        print(f"{args.models_dir} folder not found. Train some models first.")
        exit(1)

    model_files = glob.glob(os.path.join(args.models_dir, "*_final.zip"))
    if not model_files:
        print(f"No .zip models found in {args.models_dir}/")
        exit(1)

    print(f"Found {len(model_files)} trained models:")
    for f in model_files:
        print(f"   - {os.path.basename(f)}")
    print()

    all_results = {}

    for model_path in model_files:
        basename = os.path.basename(model_path).replace("_final.zip", "")
        parts = basename.split("_")
        if len(parts) < 3:
            print(f"Skipping unrecognized model: {basename}")
            continue
        algo = parts[0]
        role = parts[1]
        use_her = "HER" in parts[2] and "NoHER" not in parts[2]
        if use_her and role != "pursuer":
            print(f"Skipping unsupported evader HER model: {basename}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {basename}")
        print(f"   algo={algo}, role={role}, HER={use_her}")
        print('='*60)

        env_for_load = DummyVecEnv([lambda: build_env(role, LEVELS[0], use_her)])
        AlgoClass = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo]
        model = AlgoClass.load(model_path, env=env_for_load)
        raw_model = model

        results = []

        for level in LEVELS:
            if args.level and level["name"] != args.level:
                continue
            level_eval = level.copy()
            if role == "evader" and level["opponent"] not in [Idle, StraightLine, Spiral]:
                level_eval["opponent"] = PurePursuit if level["opponent"] == PurePursuit else ProportionalNavigation
            win_rate, mean_r, std_r = evaluate(raw_model, role, level_eval, use_her=use_her, episodes=args.episodes)
            results.append((level["name"], win_rate, mean_r))
            status = "GREEN" if win_rate >= 0.9 else ("YELLOW" if win_rate >= 0.5 else "RED")
            print(f"  {status} {level['name']:20s}  Win: {win_rate*100:5.1f}%  Reward: {mean_r:+.2f} +- {std_r:.2f}")

        all_results[basename] = results

        if args.record:
            print(f"\n  Recording videos for {basename}...")
            for level in LEVELS:
                if level["name"] in [lv["name"] for lv in LEVELS[-2:]]:
                    level_eval = level.copy()
                    if role == "evader" and level["opponent"] not in [Idle, StraightLine, Spiral]:
                        level_eval["opponent"] = PurePursuit if level["opponent"] == PurePursuit else ProportionalNavigation
                    save_video(basename, raw_model, role, level_eval, episodes=2)

        env_for_load.close()

    if all_results:
        print(f"\n{'='*60}")
        print("Generating comparison plots...")
        plot_results(all_results)
        print(f"\nAll evaluations complete!")
        print(f"   Plots saved to: plots/")
        if args.record:
            print(f"   Videos saved to: videos/")
