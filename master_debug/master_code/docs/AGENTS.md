# Project Coding Rules: PyBullet RL

## General Guidelines

- Read `PLAN.md` for the game design and physics definitions.
- Use Python 3.10+, PyBullet, Stable Baselines3, and Gymnasium.
- Always use type hints.

## PyBullet Constraints

- **Ground Plane**: Must import `pybullet_data`, set the search path using `p.setAdditionalSearchPath(pybullet_data.getDataPath())`, and load `p.loadURDF("plane.urdf")` inside `reset()` right after `p.resetSimulation()`.
- **Drift Prevention**: After every `p.stepSimulation()`, forcibly reset the Y position, Y velocity, Roll, and Yaw to exactly `0` for all bodies.
- **URDF**: Load `assets/drone.urdf` for both agents.
- **Testing**: Tests must use `p.connect(p.DIRECT)`. Use `1e-5` in denominators to prevent division-by-zero.
- **Arena**: The plane is infinite, with no arena bounds, walls, out-of-bounds penalties, or position clamping. Keep spawn positions at least 3.0 meters apart.

## RL Environment Rules (`envs/pybullet_env.py`)

- Must inherit from `gymnasium.Env`.
- Accept `render_mode` and `gravity` in `__init__`.
- Do not implement complex physics if a simple kinematic step will do (we are using direct acceleration).

## Gymnasium API

- The `reset()` method MUST return a tuple: `(observation, info)`.

## Control Architecture (`control/` directory)

- All agent control logic must be separated from the environment.
- Create a `control/` folder with an `__init__.py`.
- The base controller interface must have a `get_action(obs: np.ndarray) -> np.ndarray` method.
- The environment must instantiate the Evader's controller in `__init__` and call `get_action()` in `step()`.

## Evaluation & Recording Rules

- Use Stable Baselines3's `EvalCallback` for periodic evaluation (default every 5000 steps, 5 eval episodes).
- Write a custom `BaseCallback` called `VideoRecordCallback` that records one episode to MP4 every 10000 steps.
- The video callback MUST create a fresh `PursuitEvasionEnv(render_mode="GUI", gravity=0.0)`, use `p.startStateLogging` / `p.stopStateLogging`, then call `env.close()` to prevent PyBullet memory leaks.
- Both callbacks must be passed to `model.learn(callback=[...])`.

## Rules Updates

- `max_steps` is `1000`; the observation space shape is `(8,)`.
- The minimum spawn distance in `_sample_spawn_positions` must be `3.0`.
