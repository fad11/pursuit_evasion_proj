import argparse
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from pursuit_evasion.envs.mujoco_env import MujocoPursuitEvasionEnv
from pursuit_evasion.controllers.pursuer import ProportionalNavigation
from pursuit_evasion.controllers.evader import ProportionalEvasion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "td3"], required=True)
    parser.add_argument("--role", type=str, choices=["pursuer", "evader"], required=True)
    args = parser.parse_args()

    env = MujocoPursuitEvasionEnv(mode=args.role, render_mode="human", gravity=9.81, reset_range=(5, 20), max_steps=300)
    if args.role == "pursuer": env.evader_controller = ProportionalEvasion()
    else: env.pursuer_controller = ProportionalNavigation()

    model = PPO.load(args.model, env=env) if args.algo == "ppo" else (SAC.load(args.model, env=env) if args.algo == "sac" else TD3.load(args.model, env=env))

    wins, rewards = 0, []
    for _ in range(20):
        obs, _ = env.reset(); done, ep_r = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action); ep_r += r; done = term or trunc
        rewards.append(ep_r)
        if info["is_success"]: wins += 1
    print(f"Win Rate: {wins/20*100:.1f}% | Mean Reward: {np.mean(rewards):.2f}")
    env.close()
