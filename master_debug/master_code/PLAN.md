# Environment Plan: Pursuit-Evasion RL

## 1. Game Overview

- Zero-sum game between two agents: a **Pursuer** and an **Evader**.
- **Objective**: Pursuer aims to minimize distance (catch the Evader). Evader aims to maximize distance (survive).
- Arena: Infinite flat 2D plane.

## 2. Physics & Kinematics

- **Axes**: X is Left/Right. Z is Up/Down (vertical). Y is Depth (locked to 0).
- **Gravity**: Variable (passed as a parameter, defaults to `9.81`). Pulls along the negative Z-axis.
- **Dynamics**: Direct acceleration control. Mass = 1.0 kg for both agents.

## 3. Action Space (2D Continuous)

- Box `[-1.0, 1.0]` for `[accel_x, accel_z]`.
- Represents the desired acceleration command in the World Frame.
- Max acceleration scaling: `10.0` m/s².

## 4. Observation Space (7D Relative Kinematics)

Instead of absolute positions, agents observe relative kinematics to learn faster:

1. `Range`: Euclidean distance to the opponent.
2. `LOS_Angle`: Line of Sight angle (opponent relative to agent's forward vector).
3. `Closing_Velocity`: Rate at which Range is decreasing.
4. `LOS_Rate`: Rate of change of LOS_Angle.
5. `Pursuer_Vx`: Agent's current X velocity.
6. `Pursuer_Vz`: Agent's current Z velocity.
7. `Pursuer_Speed`: Magnitude of agent's velocity.

## 5. Reward Structure (Zero-Sum)

- **Pursuer**: `+100` if Range < `catch_radius` (0.5m). Otherwise, `-Range` per step (incentivizes getting closer).
- **Evader**: Exact negative of the Pursuer's reward.

## 6. Success Criteria

- [ ] Agents stay strictly on the Y=0 plane.
- [ ] Observation space shape is exactly `(7,)`.
- [ ] Action space shape is exactly `(2,)`.
- [ ] Training completes without NaN/Infinity errors.

## 7. Arena Bounds (Controlled Parameters)

- The environment must accept configurable bounds in `__init__`:
- `x_range`: Tuple `(x_min, x_max)`. Defaults to `(-5.0, 5.0)`.
- `z_range`: Tuple `(z_min, z_max)`. Defaults to `(0.0, 5.0)` (Z cannot be negative).
- Spawning distance: Agents must spawn at least `3.0` meters apart.
- **Ground Plane**: A static visual and collision plane MUST be loaded at Z=0. This provides a spatial reference for the agents and acts as the floor.
- Agents must spawn randomly within these bounds.

## 8. Episode Limits

- The environment must have a `max_steps` parameter
- **Max Steps**: `1000` (Handled externally via `TimeLimit` wrapper).
  **Terminated**: ONLY when Pursuer catches Evader (range < 0.5). Reward +100.
- **Boundary Handling**: Clamp position, reflect velocity (0.8 damping), apply `-5.0` penalty.

## 9. Evader Control Strategy

- The Evader must use a separate control policy to generate its actions.
- The control logic must be decoupled from the environment and stored in the `control/` directory.
- **Available Heuristic Controllers**:
  - `RandomEvaderController`: Uniform random noise.
  - `PureEvaderController`: Flees directly away based on Line of Sight.
  - `ManeuveringEvaderController`: Flees with a sinusoidal lateral component.
  - `StraightLineEvaderController`: Constant +X velocity.
  - `CircularEvaderController`: Moves in a circle.
  - `SpiralEvaderController`: Moves in a growing spiral.
  - `ZigZagEvaderController`: Moves forward with alternating lateral components.
- **Phase 3 (Future)**: Self-Play RL policy.

## 10. Evaluation & Recording

- **Evaluation**: Every `N` timesteps, run the current policy for 5 episodes in a separate DIRECT env. Log mean reward and success rate to TensorBoard. Save the best model as `best_model.zip`.
- **Video Recording**: Every `M` timesteps, spin up a fresh GUI env, run one episode with the current policy, and record it to `videos/record_{timestep}.mp4` using `p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4)`.
- **Directories**: `./logs/` (TensorBoard), `./videos/` (MP4s), `./models/` (checkpoints).
  📝 Step 2: Update AGENTS.md
