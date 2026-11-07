from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pursuit_evasion.envs._assets import resolve_asset

DT = 1.0 / 240.0
MASS = 1.0
GRAVITY = 9.81
TARGET_HEIGHT = 2.0
TARGET_PITCH = np.deg2rad(22.0)
PITCH_RATE = 0.35
DEFAULT_VIDEO_PATH = Path(__file__).with_name("evader_action.mp4")
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30
VIDEO_FRAME_INTERVAL = round(1.0 / (DT * VIDEO_FPS))


def _local_up(quaternion: tuple[float, ...]) -> np.ndarray:
    rotation = p.getMatrixFromQuaternion(quaternion)
    return np.array([rotation[2], rotation[5], rotation[8]])


def _capture_frame(
    client: int, view_matrix: list[float], projection_matrix: list[float]
) -> np.ndarray:
    _, _, rgba, _, _ = p.getCameraImage(
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        physicsClientId=client,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(
        VIDEO_HEIGHT, VIDEO_WIDTH, 4
    )[:, :, :3]


def run_demo(video_path: Path = DEFAULT_VIDEO_PATH) -> None:
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect to the PyBullet GUI.")

    video_writer = None
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=client)
        p.setTimeStep(DT, physicsClientId=client)
        p.setRealTimeSimulation(0, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
        p.resetDebugVisualizerCamera(
            cameraDistance=10.0,
            cameraYaw=35.0,
            cameraPitch=-20.0,
            cameraTargetPosition=[3.0, 0.0, 1.5],
            physicsClientId=client,
        )

        p.loadURDF("plane.urdf", physicsClientId=client)
        evader = p.loadURDF(
            resolve_asset("evader.urdf"),
            basePosition=[0.0, 0.0, 0.2],
            physicsClientId=client,
        )
        p.changeDynamics(
            evader,
            -1,
            linearDamping=0.04,
            angularDamping=0.08,
            physicsClientId=client,
        )

        video_path = video_path.resolve()
        if video_path.suffix.lower() != ".mp4":
            raise ValueError("Video output path must use the .mp4 extension.")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[3.0, 0.0, 1.5],
            distance=10.0,
            yaw=35.0,
            pitch=-20.0,
            roll=0.0,
            upAxisIndex=2,
            physicsClientId=client,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=VIDEO_WIDTH / VIDEO_HEIGHT,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=client,
        )
        video_writer = imageio.get_writer(
            str(video_path),
            format="FFMPEG",
            mode="I",
            fps=VIDEO_FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        video_writer.append_data(
            _capture_frame(client, view_matrix, projection_matrix)
        )
        print(f"Recording video to: {video_path}")

        # The asset is spherical, so this debug line makes the thrust direction visible.
        thrust_axis_line = -1
        target_pitch = 0.0
        pitch_inertia = 0.1
        pitch_kp = 20.0
        pitch_kd = 8.0
        altitude_kp = 4.0
        altitude_kd = 3.0
        simulation_step = 0
        stages = (
            ("Rise", 3.0, 0.0),
            ("Hover", 2.0, 0.0),
            ("Tilt forward", 1.1, PITCH_RATE),
            ("Forward thrust", 1.5, 0.0),
        )

        for label, duration, body_rate_command in stages:
            print(f"{label}: {duration:.1f} s")
            for _ in range(round(duration / DT)):
                if not p.isConnected(client):
                    return

                position, orientation = p.getBasePositionAndOrientation(
                    evader, physicsClientId=client
                )
                linear_velocity, angular_velocity = p.getBaseVelocity(
                    evader, physicsClientId=client
                )
                pitch = p.getEulerFromQuaternion(orientation)[1]
                up = _local_up(orientation)

                target_pitch = min(
                    target_pitch + body_rate_command * DT,
                    TARGET_PITCH,
                )
                torque_y = pitch_inertia * (
                    pitch_kp * (target_pitch - pitch)
                    + pitch_kd * (body_rate_command - angular_velocity[1])
                )
                p.applyExternalTorque(
                    evader,
                    -1,
                    [0.0, torque_y, 0.0],
                    p.WORLD_FRAME,
                    physicsClientId=client,
                )

                height_error = TARGET_HEIGHT - position[2]
                vertical_force = MASS * (
                    GRAVITY
                    + altitude_kp * height_error
                    - altitude_kd * linear_velocity[2]
                )
                vertical_force = float(np.clip(vertical_force, 0.0, 2.5 * MASS * GRAVITY))
                thrust = vertical_force / max(float(up[2]), 0.25)
                thrust = min(thrust, 3.0 * MASS * GRAVITY)
                p.applyExternalForce(
                    evader,
                    -1,
                    [0.0, 0.0, thrust],
                    [0.0, 0.0, 0.0],
                    p.LINK_FRAME,
                    physicsClientId=client,
                )

                p.stepSimulation(physicsClientId=client)
                simulation_step += 1
                position, orientation = p.getBasePositionAndOrientation(
                    evader, physicsClientId=client
                )
                up = _local_up(orientation)
                endpoint = np.asarray(position) + 0.8 * up
                if thrust_axis_line >= 0:
                    p.removeUserDebugItem(thrust_axis_line, physicsClientId=client)
                thrust_axis_line = p.addUserDebugLine(
                    position,
                    endpoint.tolist(),
                    [1.0, 0.2, 0.2],
                    lineWidth=4.0,
                    lifeTime=0,
                    physicsClientId=client,
                )
                if simulation_step % VIDEO_FRAME_INTERVAL == 0:
                    video_writer.append_data(
                        _capture_frame(client, view_matrix, projection_matrix)
                    )
                time.sleep(DT)

        video_writer.close()
        video_writer = None
        print(f"Sequence complete. Video saved to: {video_path}")
        print("Close the PyBullet window to exit.")
        while p.isConnected(client):
            time.sleep(0.1)
    finally:
        try:
            if video_writer is not None:
                video_writer.close()
        finally:
            if p.isConnected(client):
                p.disconnect(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PyBullet evader action demo.")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="MP4 output path (default: tests/evader_action.mp4).",
    )
    run_demo(video_path=parser.parse_args().video)
