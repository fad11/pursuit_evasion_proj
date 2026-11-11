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

## 4. Observation Space (8D Relative Kinematics)

Instead of absolute positions, agents observe relative kinematics to learn faster:

1. `Range`: Euclidean distance to the opponent.
2. `LOS_Sin`: Sine of the Line of Sight angle.
3. `LOS_Cos`: Cosine of the Line of Sight angle.
4. `Closing_Velocity`: Rate at which Range is decreasing.
5. `LOS_Rate`: Rate of change of the Line of Sight angle, clipped to `[-10.0, 10.0]`.
6. `Observer_Vx`: Agent's current X velocity.
7. `Observer_Vz`: Agent's current Z velocity.
8. `Observer_Speed`: Magnitude of agent's velocity.

## 5. Reward Structure (Zero-Sum)

- **Pursuer**: `+100` if Range < `catch_radius` (0.5m). Otherwise, `-(Range / 10.0)` per step (incentivizes staying close).
- **Evader**: Exact negative of the Pursuer's reward.

## 6. Success Criteria

- [ ] Agents stay strictly on the Y=0 plane.
- [ ] Observation space shape is exactly `(8,)`.
- [ ] Action space shape is exactly `(2,)`.
- [ ] Training completes without NaN/Infinity errors.

## 7. Arena and Spawning

- The environment operates on an infinite 2D plane; there are no arena walls or arena bounds.
- Do not clamp agent positions or apply out-of-bounds penalties or termination.
- Agents must spawn at least `3.0` meters apart.
- **Ground Plane**: A static visual and collision plane MUST be loaded at Z=0.

## 8. Episode Limits

- The environment must have a `max_steps` parameter
- **Max Steps**: `1000` (Handled externally via `TimeLimit` wrapper).
  **Terminated**: ONLY when Pursuer catches Evader (range < 0.5). Reward +100.

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

## Implementation Roadmap

- [x] **Phase 0 — Audit:** Inspect the environment, controllers, scripts, tests, and drone URDF; report the findings and verify whether tests run.
- [x] **Phase 1 — Scripted baseline:** Add `scripts/baseline.py`; run a pursuer using `[obs[1], obs[2]]` against `RandomEvaderController` for 50 seeded episodes. Acceptance passed: 50/50 catches (100% success), with a mean episode length of 294.6 steps.
- [x] **Phase 2 — Environment fixes:** In `envs/pybullet_env.py`, add `frame_skip=8`, torque-free acceleration application, progress reward with a time penalty and catch bonus, one 400-decision episode limit, configurable evader acceleration, safe spawn altitude, explicit damping, disabled inter-agent collision, and four appended Cartesian observation values (12 total). Support per-episode controller sampling and seed the random controller from the environment. Update `control/evader_control.py` to use an environment-seeded RNG. Acceptance passed: `pytest -q` (5 passed); baseline 50/50 (100%, mean 65.6 decisions) using baseline-only `linear_damping=0.1` while the environment default remains `0.0` (at default damping, baseline achieved 44/50); observation shape `(12,)`; 19/20 sampled pursuit rewards positive; minimum spawn z was `0.501` over 100 resets.
- [ ] **Phase 3 — Training script:** Configure 16 DIRECT environments, success monitoring, the specified PPO hyperparameters and CLI overrides, normalized training/evaluation, correctly scaled callback frequency, best-model normalization stats, checkpoints, and saved run configuration/git hash. Remove the GUI playback and duplicate `TimeLimit`. Acceptance: 50k-step smoke run, expected TensorBoard metrics, and best-model normalization stats.
- [ ] **Phase 4 — Callbacks:** Trigger video recording at the requested timestep interval adjusted for `n_envs`; normalize policy input with training statistics; pass the PyBullet client ID to calls; and close each recording environment. Prefer DIRECT-mode recording. Acceptance: 20k-step smoke run produces a moving-agent MP4 and confirms connections close.
- [ ] **Phase 5 — Evaluation script:** Load model and normalization stats; report success rate, mean steps to catch, and final range for each of the seven controller classes, with configurable evader acceleration and optional GUI playback. Acceptance: evaluate the best model and print the table.
- [ ] **Phase 6 — Curriculum:** Start with Random and require success rate above 0.9; then train on all controllers with the specified acceleration scales, increasing scales only after success rate exceeds 0.8. Expose an explicit curriculum-stage option and do not auto-advance. Acceptance: report per-controller evaluation results for each stage.
- [ ] **Phase 7 — Documentation:** Reconcile `AGENTS.md` and `PLAN.md` with the implemented behavior and add `docs/HANDOVER.md`. Remove stale observation/reward/episode-limit/controller/video descriptions. Acceptance: search finds no stale `(7,)`, `(8,)`, `+100`, `TimeLimit`, or `2.0 meters` requirements.
