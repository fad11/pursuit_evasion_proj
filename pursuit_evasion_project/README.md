# Pursuit-Evasion Differential Game (Reinforcement Learning)

## Overview
This project is a physics-based Reinforcement Learning (RL) environment for a **Pursuit-Evasion Differential Game**. Agents can use either **pure math (fast)**, **MuJoCo**, or **PyBullet** as the physics backend.

The training system uses a **7-level auto-advancing curriculum**, supports parallel training via `SubprocVecEnv`, integrates pursuer-side Hindsight Experience Replay (HER), and includes evaluation, plotting, video recording, and a live GUI viewer.

---

## Project Structure

```text
pursuit_evasion_project/
├── assets/
│   ├── pursuer.urdf
│   ├── evader.urdf
│   └── world.xml
├── pursuit_evasion/
│   ├── controllers/
│   │   ├── base.py
│   │   ├── pursuer.py
│   │   └── evader.py
│   ├── envs/
│   │   ├── base_env.py
│   │   ├── mujoco_env.py
│   │   ├── pybullet_env.py
│   │   └── _assets.py
│   └── wrappers.py
├── training/
│   ├── train_curriculum.py
│   └── run_experiments.py
├── evaluation/
│   ├── evaluate_all.py
│   └── watch_agent.py
├── tests/
│   └── motion_check.py
├── reolve_these/
│   ├── evaluate.py
│   └── train.py
├── setup.py
├── requirements.txt
└── README.md
```

---

## 1. Environment Architecture

### Agents
- Two point-mass agents (Pursuer, Evader).
- Controlled via acceleration or thrust + body-rate.
- Speed capped at v_max=1.0, acceleration at a_max=1.0.
- Wall collisions: position clamped, velocity reversed with 0.5 damping.

### Physics Backends
- BasePursuitEvasionEnv: pure Python math (semi-implicit Euler).
- MujocoPursuitEvasionEnv: MuJoCo backend.
- PyBulletPursuitEvasionEnv: PyBullet backend.
- Both physics engines use the environment time step and apply the configured gravity.
- All backends enforce the arena bounds and speed limit.

### Observation Modes
1. relative: [rel_position, rel_velocity, self_velocity] -> (3*dim,).
2. los: [range, LOS angle, LOS rate, closing velocity] -> (4,) in 2D, (5,) in 3D.

### Action Modes
1. acceleration: [ax, ay] (or [ax, ay, az]).
2. thrust: [thrust, body_rate] in 2D, [thrust, wx, wy, wz] in 3D.

### Rewards
- Captured (dist < 0.3): Pursuer +10, Evader -10.
- Not captured: Pursuer -0.1 - dist/arena, Evader +0.1 + min(0.5, dist/arena).

---

## 2. Scripted Controllers

All controllers output acceleration vectors using a PD loop.

Open-Loop (Base): Idle, StraightLine, Circular, Zigzag, SquarePath, Spiral.
Closed-Loop (Tactical): PurePursuit, ProportionalNavigation, PureEvasion, ProportionalEvasion.

---

## 3. Curriculum Training (7 Levels)

| Level | Gravity | Spawn Range | Pursuer vs | Evader vs |
|---|---|---|---|---|
| L1 | 0.0 | (1, 5) | Idle | Idle |
| L2 | 9.81 | (1, 5) | Idle | Idle |
| L3 | 9.81 | (5, 20) | Idle | Idle |
| L4 | 9.81 | (5, 20) | StraightLine | StraightLine |
| L5 | 9.81 | (5, 20) | Spiral | Spiral |
| L6 | 9.81 | (5, 20) | PureEvasion | PurePursuit |
| L7 | 9.81 | (5, 20) | ProportionalEvasion | ProportionalNavigation |

---

## 4. Scripts & Usage

### Installation
```bash
pip install -e .
```

### Training
```bash
python training/train_curriculum.py --algo ppo --role pursuer --n_envs 4
```
The curriculum uses subprocess vector environments. HER is supported with SAC or TD3 for the pursuer role; it is not enabled for evader training because evasion is not a goal-reaching objective.

### Parallel Training
```bash
python training/run_experiments.py
```

### Evaluation
```bash
python evaluation/evaluate_all.py
```

### Live Viewer
```bash
python evaluation/watch_agent.py --model models/ppo_pursuer_NoHER_final --role pursuer --algo ppo --opponent hardest
```

### Tests
```bash
python tests/motion_check.py
python tests/vs.py
python tests/maneuver_vs.py
```
The motion check simulates every scripted controller in 2D. It also compares
`PureEvasion` and `ProportionalEvasion` against the same seeded, randomly
moving pursuer. The 1v1 script pits `PurePursuit` against `PureEvasion` and
`ProportionalNavigation` against `ProportionalEvasion` in separate matches.
`maneuver_vs.py` compares those original evaders with `ManeuveringEvasion`
against the same pursuers. The scripts print results and save/display their
path visualizations.

---

## 5. Test Coverage

The motion check covers controller calls, finite accelerations and trajectories,
speed and acceleration limits, and waypoint following. The 1v1 scripts compare
the original evasion controllers with `ManeuveringEvasion` against scripted
pursuers. These scripts do not run the MuJoCo or PyBullet environments.

---

## 6. Scientific Significance

This project explores whether RL can learn proportional navigation purely from reward signal, and compares PPO vs SAC/TD3 with HER.
