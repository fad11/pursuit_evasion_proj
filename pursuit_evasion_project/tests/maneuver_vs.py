from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from pursuit_evasion.controllers.evader import (  # noqa: E402
    ManeuveringEvasion,
    ProportionalEvasion,
    PureEvasion,
)
from pursuit_evasion.controllers.pursuer import (  # noqa: E402
    ProportionalNavigation,
    PurePursuit,
)
from vs import MAX_STEPS, MatchResult, run_match  # noqa: E402

MATCHUPS = (
    (PurePursuit, PureEvasion),
    (ProportionalNavigation, ProportionalEvasion),
)


def _capture_time(result: MatchResult) -> float | None:
    if result.capture_step is None:
        return None
    return result.capture_step * 0.1


def _plot_results(
    results: list[tuple[str, MatchResult, str, MatchResult]],
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(14, 12))

    for row, (pursuer_name, baseline, maneuver_name, maneuver) in enumerate(results):
        path_axis = axes[row, 0]
        for result, label, style in (
            (baseline, "Original evader", "-"),
            (maneuver, "ManeuveringEvasion", "--"),
        ):
            path_axis.plot(
                result.pursuer_path[:, 0],
                result.pursuer_path[:, 1],
                linestyle=style,
                label=f"Pursuer ({label})",
            )
            path_axis.plot(
                result.evader_path[:, 0],
                result.evader_path[:, 1],
                linestyle=style,
                label=f"Evader ({label})",
            )
        path_axis.set_title(f"{pursuer_name} - paths")
        path_axis.set_xlabel("x position")
        path_axis.set_ylabel("y position")
        path_axis.set_xlim(-10.0, 10.0)
        path_axis.set_ylim(-10.0, 10.0)
        path_axis.set_aspect("equal", adjustable="box")
        path_axis.grid(True, alpha=0.3)
        path_axis.legend(fontsize=8)

        distance_axis = axes[row, 1]
        for result, label in (
            (baseline, "Original evader"),
            (maneuver, "ManeuveringEvasion"),
        ):
            times = np.arange(len(result.separations)) * 0.1
            distance_axis.plot(times, result.separations, label=label)
        distance_axis.axhline(
            0.3,
            color="red",
            linestyle=":",
            label="Capture radius (0.3)",
        )
        distance_axis.set_title(f"{pursuer_name} - separation")
        distance_axis.set_xlabel("Time (seconds)")
        distance_axis.set_ylabel("Distance")
        distance_axis.grid(True, alpha=0.3)
        distance_axis.legend(fontsize=8)

    figure.tight_layout()
    output_path = Path(__file__).with_name("maneuver_vs.png")
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


def _check_three_dimensional_call() -> None:
    controller = ManeuveringEvasion(
        dim=3,
        dt=0.1,
        v_max=1.0,
        a_max=1.0,
    )
    position = np.array([4.0, 2.0, 1.0])
    controller.reset(position.copy())
    acceleration = controller.compute(
        position,
        np.zeros(3),
        np.zeros(3),
        np.array([0.1, 0.2, 0.3]),
    )
    assert acceleration.shape == (3,)
    assert np.isfinite(acceleration).all()
    assert np.linalg.norm(acceleration) <= controller.a_max + 1e-9
    print("[PASS] ManeuveringEvasion: finite, bounded 3D acceleration")


def main() -> None:
    print("Maneuvering evader vs original evasion controllers")
    results: list[tuple[str, MatchResult, str, MatchResult]] = []

    for pursuer_type, baseline_evader_type in MATCHUPS:
        baseline = run_match(pursuer_type, baseline_evader_type)
        maneuver = run_match(pursuer_type, ManeuveringEvasion)
        pursuer_name = pursuer_type.__name__
        baseline_name = baseline_evader_type.__name__
        baseline_time = _capture_time(baseline)
        maneuver_time = _capture_time(maneuver)

        assert baseline_time is not None, (
            f"Baseline {pursuer_name} vs {baseline_name} did not capture "
            f"within {MAX_STEPS} steps"
        )
        if maneuver_time is None:
            improvement = "no capture within the simulation limit"
        else:
            improvement = (
                f"{maneuver_time - baseline_time:+.1f}s until capture"
            )
            assert maneuver_time > baseline_time, (
                f"ManeuveringEvasion did not outlast {baseline_name} against "
                f"{pursuer_name}: baseline={baseline_time:.1f}s, "
                f"maneuver={maneuver_time:.1f}s"
            )

        maneuver_outcome = (
            f"{maneuver_time:.1f}s"
            if maneuver_time is not None
            else "not captured"
        )
        print(
            f"[PASS] {pursuer_name}: {baseline_name} capture at "
            f"{baseline_time:.1f}s; ManeuveringEvasion capture at "
            f"{maneuver_outcome} "
            f"({improvement})"
        )
        results.append(
            (pursuer_name, baseline, baseline_name, maneuver)
        )

    _check_three_dimensional_call()
    _plot_results(results)
    print("All maneuver comparisons passed.")


if __name__ == "__main__":
    main()
