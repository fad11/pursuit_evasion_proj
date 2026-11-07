from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pursuit_evasion.controllers.base import BaseController  # noqa: E402
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
ARENA_SIZE = 20.0
CAPTURE_RADIUS = 0.3
MAX_STEPS = 600
PURSUER_START = (0.0, 0.0)
EVADER_START = (5.0, 2.0)

MATCHUPS: tuple[
    tuple[type[BaseController], type[BaseController]], ...
] = (
    (PurePursuit, PureEvasion),
    (ProportionalNavigation, ProportionalEvasion),
)


@dataclass
class MatchResult:
    pursuer_path: np.ndarray
    evader_path: np.ndarray
    separations: np.ndarray
    capture_step: int | None


def _make_controller(controller_type: type[BaseController]) -> BaseController:
    return controller_type(
        dim=2,
        dt=DT,
        v_max=MAX_SPEED,
        a_max=MAX_ACCELERATION,
    )


def _apply_walls(position: np.ndarray, velocity: np.ndarray) -> None:
    half_size = ARENA_SIZE / 2.0
    for axis in range(2):
        if position[axis] > half_size:
            position[axis] = half_size
            velocity[axis] *= -0.5
        elif position[axis] < -half_size:
            position[axis] = -half_size
            velocity[axis] *= -0.5


def run_match(
    pursuer_type: type[BaseController],
    evader_type: type[BaseController],
) -> MatchResult:
    pursuer = _make_controller(pursuer_type)
    evader = _make_controller(evader_type)
    pursuer_position = np.asarray(PURSUER_START, dtype=float)
    evader_position = np.asarray(EVADER_START, dtype=float)
    pursuer_velocity = np.zeros(2)
    evader_velocity = np.zeros(2)
    pursuer.reset(pursuer_position.copy())
    evader.reset(evader_position.copy())

    pursuer_path = [pursuer_position.copy()]
    evader_path = [evader_position.copy()]
    separations = [
        float(np.linalg.norm(pursuer_position - evader_position))
    ]
    capture_step: int | None = None

    for step in range(1, MAX_STEPS + 1):
        pursuer_acceleration = np.asarray(
            pursuer.compute(
                pursuer_position,
                pursuer_velocity,
                evader_position,
                evader_velocity,
            ),
            dtype=float,
        )
        evader_acceleration = np.asarray(
            evader.compute(
                evader_position,
                evader_velocity,
                pursuer_position,
                pursuer_velocity,
            ),
            dtype=float,
        )

        for name, acceleration in (
            (pursuer_type.__name__, pursuer_acceleration),
            (evader_type.__name__, evader_acceleration),
        ):
            assert acceleration.shape == (2,), (
                f"{name} returned acceleration shape {acceleration.shape}"
            )
            assert np.isfinite(acceleration).all(), (
                f"{name} returned non-finite acceleration at step {step}"
            )
            assert np.linalg.norm(acceleration) <= MAX_ACCELERATION + 1e-9, (
                f"{name} exceeded the acceleration limit at step {step}"
            )

        pursuer_velocity += np.clip(
            pursuer_acceleration, -MAX_ACCELERATION, MAX_ACCELERATION
        ) * DT
        evader_velocity += np.clip(
            evader_acceleration, -MAX_ACCELERATION, MAX_ACCELERATION
        ) * DT
        for velocity in (pursuer_velocity, evader_velocity):
            speed = float(np.linalg.norm(velocity))
            if speed > MAX_SPEED:
                velocity *= MAX_SPEED / speed

        pursuer_position += pursuer_velocity * DT
        evader_position += evader_velocity * DT
        _apply_walls(pursuer_position, pursuer_velocity)
        _apply_walls(evader_position, evader_velocity)

        assert np.isfinite(pursuer_position).all(), (
            f"{pursuer_type.__name__} produced a non-finite position"
        )
        assert np.isfinite(evader_position).all(), (
            f"{evader_type.__name__} produced a non-finite position"
        )
        assert np.linalg.norm(pursuer_velocity) <= MAX_SPEED + 1e-9
        assert np.linalg.norm(evader_velocity) <= MAX_SPEED + 1e-9

        separation = float(
            np.linalg.norm(pursuer_position - evader_position)
        )
        pursuer_path.append(pursuer_position.copy())
        evader_path.append(evader_position.copy())
        separations.append(separation)

        if separation < CAPTURE_RADIUS:
            capture_step = step
            break

    assert pursuer._t == len(pursuer_path) - 1
    assert evader._t == len(evader_path) - 1
    return MatchResult(
        pursuer_path=np.asarray(pursuer_path),
        evader_path=np.asarray(evader_path),
        separations=np.asarray(separations),
        capture_step=capture_step,
    )


def plot_results(results: list[tuple[str, MatchResult]]) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    half_size = ARENA_SIZE / 2.0

    for axis, (title, result) in zip(axes[:2], results):
        axis.plot(
            result.pursuer_path[:, 0],
            result.pursuer_path[:, 1],
            color="tab:blue",
            label="Pursuer",
        )
        axis.plot(
            result.evader_path[:, 0],
            result.evader_path[:, 1],
            color="tab:orange",
            label="Evader",
        )
        axis.scatter(*result.pursuer_path[0], color="tab:blue", marker="o")
        axis.scatter(*result.evader_path[0], color="tab:orange", marker="o")
        if result.capture_step is not None:
            capture_position = result.pursuer_path[result.capture_step]
            axis.scatter(
                *capture_position,
                color="red",
                marker="X",
                s=90,
                label="Capture",
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("x position")
        axis.set_ylabel("y position")
        axis.set_xlim(-half_size, half_size)
        axis.set_ylim(-half_size, half_size)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
        axis.legend()

    distance_axis = axes[2]
    for title, result in results:
        times = np.arange(len(result.separations)) * DT
        distance_axis.plot(times, result.separations, label=title)
    distance_axis.axhline(
        CAPTURE_RADIUS,
        color="red",
        linestyle="--",
        label=f"Capture radius ({CAPTURE_RADIUS})",
    )
    distance_axis.set_title("Distance between agents")
    distance_axis.set_xlabel("Time (seconds)")
    distance_axis.set_ylabel("Separation")
    distance_axis.grid(True, alpha=0.3)
    distance_axis.legend(fontsize=8)

    figure.tight_layout()
    output_path = Path(__file__).with_name("vs.png")
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
    print("1v1 controller matches")
    print(
        f"Starts: pursuer={PURSUER_START}, evader={EVADER_START}; "
        f"capture radius={CAPTURE_RADIUS}; max time={MAX_STEPS * DT:.1f}s"
    )
    results: list[tuple[str, MatchResult]] = []

    for pursuer_type, evader_type in MATCHUPS:
        title = f"{pursuer_type.__name__} vs {evader_type.__name__}"
        result = run_match(pursuer_type, evader_type)
        results.append((title, result))
        minimum_distance = float(np.min(result.separations))
        if result.capture_step is None:
            outcome = "no capture"
            elapsed = len(result.separations) - 1
        else:
            outcome = "pursuer captured evader"
            elapsed = result.capture_step

        print(
            f"[PASS] {title}: {outcome} after {elapsed} steps "
            f"({elapsed * DT:.1f}s); "
            f"minimum distance={minimum_distance:.3f}; "
            f"final distance={result.separations[-1]:.3f}"
        )

    plot_results(results)
    print("All 1v1 simulations completed.")


if __name__ == "__main__":
    main()
