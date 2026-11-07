from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pursuit_evasion.controllers.base import (  # noqa: E402
    BaseController,
    Circular,
    Idle,
    Spiral,
    SquarePath,
    StraightLine,
    Zigzag,
)
from pursuit_evasion.controllers.evader import (  # noqa: E402
    ProportionalEvasion,
    PureEvasion,
)
from pursuit_evasion.controllers.pursuer import (  # noqa: E402
    ProportionalNavigation,
    PurePursuit,
)

DT = 0.1
MAX_SPEED = 1.0
MAX_ACCELERATION = 1.0
STEPS = 180
WAYPOINT_TOLERANCE = 0.2
PURSUER_SPEED = 0.7
PURSUER_TURN_STEPS = 20
ARENA_LIMIT = 8.0
RANDOM_SEED = 42
CAPTURE_RADIUS = 0.3

SCENARIOS: tuple[
    tuple[type[BaseController], tuple[float, float], tuple[float, float]], ...
] = (
    (Idle, (0.0, 0.0), (4.0, 3.0)),
    (StraightLine, (0.0, 0.0), (4.0, 3.0)),
    (Circular, (2.0, 0.0), (4.0, 3.0)),
    (Zigzag, (0.0, 0.0), (4.0, 3.0)),
    (Spiral, (0.0, 0.0), (4.0, 3.0)),
    (SquarePath, (0.0, 0.0), (4.0, 3.0)),
    (PurePursuit, (0.0, 0.0), (4.0, 3.0)),
    (ProportionalNavigation, (0.0, 0.0), (4.0, 3.0)),
    (PureEvasion, (0.0, 0.0), (4.0, 3.0)),
    (ProportionalEvasion, (0.0, 0.0), (4.0, 3.0)),
)


def simulate(
    controller: BaseController,
    initial_position: tuple[float, ...],
    target_position: tuple[float, ...],
    steps: int,
) -> tuple[np.ndarray, float, float]:
    position = np.asarray(initial_position, dtype=float)
    target = np.asarray(target_position, dtype=float)
    velocity = np.zeros_like(position)
    controller.reset(position.copy())
    positions = [position.copy()]
    max_acceleration = 0.0
    max_speed = 0.0

    for step in range(steps):
        acceleration = np.asarray(
            controller.compute(position, velocity, target, np.zeros_like(velocity)),
            dtype=float,
        )
        assert acceleration.shape == position.shape, (
            f"{type(controller).__name__} returned {acceleration.shape}; "
            f"expected {position.shape}"
        )
        assert np.isfinite(acceleration).all(), (
            f"{type(controller).__name__} returned a non-finite acceleration "
            f"at step {step + 1}"
        )

        acceleration_magnitude = float(np.linalg.norm(acceleration))
        max_acceleration = max(max_acceleration, acceleration_magnitude)
        assert acceleration_magnitude <= controller.a_max + 1e-9, (
            f"{type(controller).__name__} exceeded its acceleration limit"
        )

        velocity = velocity + acceleration * controller.dt
        speed = float(np.linalg.norm(velocity))
        if speed > controller.v_max:
            velocity = velocity * (controller.v_max / speed)
        max_speed = max(max_speed, float(np.linalg.norm(velocity)))
        position = position + velocity * controller.dt
        positions.append(position.copy())

    assert controller._t == steps, (
        f"{type(controller).__name__}.compute was expected to advance "
        f"the controller clock {steps} times, got {controller._t}"
    )
    trajectory = np.asarray(positions)
    assert np.isfinite(trajectory).all(), (
        f"{type(controller).__name__} produced a non-finite path"
    )
    assert max_speed <= controller.v_max + 1e-9, (
        f"{type(controller).__name__} exceeded its speed limit"
    )
    return trajectory, max_acceleration, max_speed


def simulate_random_pursuer(
    initial_position: tuple[float, float],
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    position = np.asarray(initial_position, dtype=float)
    positions = [position.copy()]
    velocities = []
    direction = np.array([1.0, 0.0])

    for step in range(steps):
        if step % PURSUER_TURN_STEPS == 0:
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)

        velocity = direction * PURSUER_SPEED
        next_position = position + velocity * DT
        for axis in range(2):
            if abs(next_position[axis]) > ARENA_LIMIT:
                direction[axis] *= -1.0
                velocity[axis] = direction[axis] * PURSUER_SPEED
                next_position[axis] = position[axis] + velocity[axis] * DT

        velocities.append(velocity.copy())
        position = next_position
        positions.append(position.copy())

    pursuer_path = np.asarray(positions)
    pursuer_velocities = np.vstack((velocities, np.zeros(2)))
    assert np.isfinite(pursuer_path).all(), "Random pursuer path is not finite"
    assert np.isfinite(pursuer_velocities).all(), (
        "Random pursuer velocities are not finite"
    )
    assert np.all(np.abs(pursuer_path) <= ARENA_LIMIT), (
        "Random pursuer left the arena"
    )
    assert np.all(np.linalg.norm(pursuer_velocities[:-1], axis=1) <= PURSUER_SPEED + 1e-9), (
        "Random pursuer exceeded its speed limit"
    )
    return pursuer_path, pursuer_velocities


def simulate_evader_against_pursuer(
    controller_type: type[BaseController],
    pursuer_path: np.ndarray,
    pursuer_velocities: np.ndarray,
    initial_position: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    controller = controller_type(
        dim=2,
        dt=DT,
        v_max=MAX_SPEED,
        a_max=MAX_ACCELERATION,
    )
    position = np.asarray(initial_position, dtype=float)
    velocity = np.zeros(2)
    controller.reset(position.copy())
    evader_path = [position.copy()]
    separations = [
        float(np.linalg.norm(position - pursuer_path[0]))
    ]

    for step in range(len(pursuer_path) - 1):
        acceleration = np.asarray(
            controller.compute(
                position,
                velocity,
                pursuer_path[step],
                pursuer_velocities[step],
            ),
            dtype=float,
        )
        assert acceleration.shape == (2,), (
            f"{controller_type.__name__} returned {acceleration.shape}"
        )
        assert np.isfinite(acceleration).all(), (
            f"{controller_type.__name__} returned a non-finite acceleration "
            f"at step {step + 1}"
        )
        assert np.linalg.norm(acceleration) <= controller.a_max + 1e-9, (
            f"{controller_type.__name__} exceeded its acceleration limit"
        )

        velocity = velocity + acceleration * DT
        speed = float(np.linalg.norm(velocity))
        if speed > controller.v_max:
            velocity = velocity * (controller.v_max / speed)
        position = position + velocity * DT
        assert np.isfinite(position).all(), (
            f"{controller_type.__name__} produced a non-finite position"
        )
        evader_path.append(position.copy())
        separations.append(
            float(np.linalg.norm(position - pursuer_path[step + 1]))
        )

    assert controller._t == len(pursuer_path) - 1, (
        f"{controller_type.__name__} was not called at every simulation step"
    )
    return np.asarray(evader_path), np.asarray(separations)


def check_three_dimensional_calls() -> None:
    for controller_type, _, _ in SCENARIOS:
        controller = controller_type(
            dim=3,
            dt=DT,
            v_max=MAX_SPEED,
            a_max=MAX_ACCELERATION,
        )
        position = np.array([1.0, 2.0, 3.0])
        controller.reset(position.copy())
        acceleration = np.asarray(
            controller.compute(
                position,
                np.array([0.1, 0.2, 0.3]),
                np.array([4.0, 5.0, 6.0]),
                np.array([-0.1, 0.0, 0.1]),
            ),
            dtype=float,
        )
        assert acceleration.shape == (3,), (
            f"{controller_type.__name__} returned {acceleration.shape} in 3D"
        )
        assert np.isfinite(acceleration).all(), (
            f"{controller_type.__name__} returned non-finite 3D acceleration"
        )
        assert np.linalg.norm(acceleration) <= controller.a_max + 1e-9, (
            f"{controller_type.__name__} exceeded its 3D acceleration limit"
        )
        assert controller._t == 1, (
            f"{controller_type.__name__}.compute did not advance its 3D clock"
        )
        print(f"[PASS] {controller_type.__name__}: 3D call and limits")


def check_square_path() -> tuple[np.ndarray, np.ndarray]:
    controller = SquarePath(
        side_length=3.0,
        dim=2,
        dt=DT,
        v_max=MAX_SPEED,
        a_max=MAX_ACCELERATION,
    )
    position = np.zeros(2)
    velocity = np.zeros(2)
    controller.reset(position.copy())
    waypoints = np.asarray(controller.waypoints)
    trajectory = [position.copy()]
    reached: dict[int, float] = {}
    reached_in_order: list[int] = []
    max_speed = 0.0

    for _ in range(3000):
        active_waypoint = controller.current_wp
        acceleration = np.asarray(
            controller.compute(
                position, velocity, np.zeros(2), np.zeros(2)
            ),
            dtype=float,
        )
        assert np.isfinite(acceleration).all(), (
            "SquarePath returned a non-finite acceleration"
        )
        assert np.linalg.norm(acceleration) <= controller.a_max + 1e-9, (
            "SquarePath exceeded its acceleration limit"
        )

        if controller.current_wp != active_waypoint:
            error = float(
                np.linalg.norm(position - waypoints[active_waypoint])
            )
            reached.setdefault(active_waypoint, error)
            reached_in_order.append(active_waypoint)

        velocity = velocity + acceleration * controller.dt
        speed = float(np.linalg.norm(velocity))
        if speed > controller.v_max:
            velocity = velocity * (controller.v_max / speed)
        max_speed = max(max_speed, float(np.linalg.norm(velocity)))
        position = position + velocity * controller.dt
        trajectory.append(position.copy())

        if len(reached) == len(waypoints):
            break

    assert len(reached) == len(waypoints), (
        f"SquarePath reached {len(reached)} of {len(waypoints)} waypoints "
        "within 3000 steps"
    )
    assert reached_in_order[: len(waypoints)] == list(range(len(waypoints))), (
        f"SquarePath visited waypoints out of order: {reached_in_order}"
    )
    for index, error in reached.items():
        assert error < WAYPOINT_TOLERANCE, (
            f"SquarePath missed waypoint {index} by {error:.3f}; "
            f"tolerance is {WAYPOINT_TOLERANCE:.3f}"
        )
        print(
            f"[PASS] SquarePath waypoint {index + 1}: "
            f"miss distance {error:.3f}"
        )

    assert np.isfinite(trajectory).all(), "SquarePath produced a non-finite path"
    assert max_speed <= controller.v_max + 1e-9, (
        "SquarePath exceeded its speed limit"
    )
    return np.asarray(trajectory), waypoints


def save_and_show_plot(
    trajectories: dict[str, np.ndarray],
    square_trajectory: np.ndarray,
    square_waypoints: np.ndarray,
    random_pursuer_path: np.ndarray,
    evader_encounters: dict[str, tuple[np.ndarray, np.ndarray]],
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    scenario_by_name = {
        controller_type.__name__: (initial_position, target_position)
        for controller_type, initial_position, target_position in SCENARIOS
    }
    groups = (
        (
            axes[0, 0],
            "Open-loop controllers",
            ("Idle", "StraightLine", "Circular", "Zigzag", "Spiral"),
            False,
        ),
        (
            axes[0, 1],
            "Pursuer alone (fixed target)",
            ("PurePursuit", "ProportionalNavigation"),
            True,
        ),
        (
            axes[1, 0],
            "Evaders vs random-moving pursuer",
            (),
            False,
        ),
    )

    for axis, title, names, show_reference in groups:
        for name in names:
            path = trajectories[name]
            axis.plot(path[:, 0], path[:, 1], label=name)
            axis.scatter(path[0, 0], path[0, 1], marker="o", s=18)
        if show_reference:
            _, reference = scenario_by_name[names[0]]
            axis.scatter(
                reference[0],
                reference[1],
                marker="x",
                s=70,
                color="black",
                label="Fixed target/reference",
            )
        if title == "Evaders vs random-moving pursuer":
            axis.plot(
                random_pursuer_path[:, 0],
                random_pursuer_path[:, 1],
                color="black",
                linestyle="--",
                label="Random-moving pursuer",
            )
            for name, (path, _) in evader_encounters.items():
                axis.plot(path[:, 0], path[:, 1], label=f"{name} evader")
            axis.scatter(
                random_pursuer_path[0, 0],
                random_pursuer_path[0, 1],
                color="black",
                marker="x",
                s=70,
                label="Pursuer start",
            )
            axis.scatter(
                evader_encounters["PureEvasion"][0][0, 0],
                evader_encounters["PureEvasion"][0][0, 1],
                color="green",
                marker="o",
                s=36,
                label="Evader start",
            )
        axis.set_title(title)
        axis.set_xlabel("x position")
        axis.set_ylabel("y position")
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="datalim")
        axis.legend(fontsize=8)

    square_axis = axes[1, 1]
    square_axis.plot(
        square_trajectory[:, 0],
        square_trajectory[:, 1],
        label="Simulated path",
    )
    route = np.vstack((np.zeros(2), square_waypoints, np.zeros(2)))
    square_axis.plot(
        route[:, 0],
        route[:, 1],
        "--",
        marker="x",
        label="Generated waypoint route",
    )
    square_axis.scatter(
        square_waypoints[:, 0],
        square_waypoints[:, 1],
        marker="D",
        s=32,
        label="Waypoints",
    )
    square_axis.set_title("SquarePath waypoint following")
    square_axis.set_xlabel("x position")
    square_axis.set_ylabel("y position")
    square_axis.grid(True, alpha=0.3)
    square_axis.set_aspect("equal", adjustable="datalim")
    square_axis.legend(fontsize=8)

    figure.tight_layout()
    output_path = Path(__file__).with_name("motion_check.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
    if plt.get_backend().lower() in {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }:
        print("No interactive plot window is available; open the saved image to view it.")
    else:
        plt.show()
    plt.close(figure)
    return output_path


def main() -> None:
    print("Motion check: running controller smoke tests")
    trajectories: dict[str, np.ndarray] = {}
    for controller_type, initial_position, target_position in SCENARIOS:
        controller = controller_type(
            dim=2,
            dt=DT,
            v_max=MAX_SPEED,
            a_max=MAX_ACCELERATION,
        )
        path, max_acceleration, max_speed = simulate(
            controller, initial_position, target_position, STEPS
        )
        name = controller_type.__name__
        trajectories[name] = path
        displacement = float(np.linalg.norm(path[-1] - path[0]))
        if controller_type is Idle:
            assert displacement < 1e-9, "Idle unexpectedly moved"
        else:
            assert displacement > 0.1, f"{name} did not produce movement"
        print(
            f"[PASS] {name}: {STEPS} calls, finite path, "
            f"max acceleration {max_acceleration:.3f}, "
            f"max speed {max_speed:.3f}, displacement {displacement:.3f}"
        )

    random_pursuer_path, random_pursuer_velocities = simulate_random_pursuer(
        (4.0, 3.0), STEPS, RANDOM_SEED
    )
    print(
        "\nMoving-pursuer simulation: one seeded random walker vs each evader"
    )
    evader_encounters: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for controller_type in (PureEvasion, ProportionalEvasion):
        name = controller_type.__name__
        path, separations = simulate_evader_against_pursuer(
            controller_type,
            random_pursuer_path,
            random_pursuer_velocities,
            (0.0, 0.0),
        )
        assert float(np.min(separations)) > CAPTURE_RADIUS, (
            f"{name} was caught by the random-moving pursuer"
        )
        assert separations[-1] > separations[0], (
            f"{name} did not increase its distance from the pursuer "
            "in this seeded simulation"
        )
        evader_encounters[name] = (path, separations)
        print(
            f"[PASS] {name} escaped: "
            f"start distance {separations[0]:.3f}, "
            f"minimum distance {np.min(separations):.3f}, "
            f"final distance {separations[-1]:.3f}"
        )

    check_three_dimensional_calls()
    square_trajectory, square_waypoints = check_square_path()
    save_and_show_plot(
        trajectories,
        square_trajectory,
        square_waypoints,
        random_pursuer_path,
        evader_encounters,
    )
    print("\nAll motion checks passed.")


if __name__ == "__main__":
    main()
