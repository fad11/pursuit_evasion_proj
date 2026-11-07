import argparse
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from pursuit_evasion.envs.mujoco_env import MujocoPursuitEvasionEnv
from pursuit_evasion.wrappers import GoalConditionedWrapper
from pursuit_evasion.controllers.base import Idle, StraightLine, Spiral
from pursuit_evasion.controllers.pursuer import PurePursuit, ProportionalNavigation
from pursuit_evasion.controllers.evader import PureEvasion, ProportionalEvasion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .zip (e.g. models/ppo_pursuer_NoHER_final)")
    parser.add_argument("--role", type=str, required=True, choices=["pursuer", "evader"])
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--opponent", type=str, default="hardest",
                        choices=["idle", "straight", "spiral", "pure", "prop", "hardest"])
    parser.add_argument("--her", action="store_true")
    args = parser.parse_args()

    if args.her and args.role != "pursuer":
        parser.error("HER goal relabeling is only supported for the pursuer role.")
    if args.her and args.algo == "ppo":
        parser.error("HER is only supported with SAC or TD3.")

    opponents = {
        "idle": Idle, "straight": StraightLine, "spiral": Spiral,
        "pure": PurePursuit if args.role == "pursuer" else PureEvasion,
        "prop": ProportionalNavigation if args.role == "pursuer" else ProportionalEvasion,
        "hardest": ProportionalNavigation if args.role == "evader" else ProportionalEvasion,
    }
    opp_class = opponents[args.opponent]

    env = MujocoPursuitEvasionEnv(
        mode=args.role, obs_type="relative",
        gravity=9.81, reset_range=(5, 20),
        render_mode="human"
    )
    if args.role == "pursuer": env.evader_controller = opp_class()
    else: env.pursuer_controller = opp_class()
    if args.her:
        env = GoalConditionedWrapper(env)

    AlgoClass = {"ppo": PPO, "sac": SAC, "td3": TD3}[args.algo]
    vec_env = DummyVecEnv([lambda: env])
    model = AlgoClass.load(args.model, env=vec_env)

    print(f"Watching {args.model}")
    print(f"   Role: {args.role}, Opponent: {opp_class.__name__}")
    print(f"   Episodes: {args.episodes}")
    print("   Close the MuJoCo window to stop.\n")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            env.render()
            ep_reward += reward
            steps += 1
            done = term or trunc

        result = "WON" if info["is_success"] else "LOST"
        print(f"Episode {ep+1}/{args.episodes}: {result}  |  Reward: {ep_reward:+.2f}  |  Steps: {steps}")

    vec_env.close()
    print("\nDone.")
