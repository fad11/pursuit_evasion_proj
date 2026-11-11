import numpy as np
import pybullet as p
import pytest

from control.evader_control import BaseEvaderController, RandomEvaderController
from envs.pybullet_env import PursuitEvasionEnv


class ForwardEvaderController(BaseEvaderController):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_reset_returns_12d_observation_and_safe_spawns() -> None:
    env = PursuitEvasionEnv(render_mode="DIRECT", gravity=0.0)
    try:
        for seed in range(50):
            obs, _ = env.reset(seed=seed)
            assert obs.shape == (12,)
            assert env.observation_space.shape == (12,)
            assert env.observation_space.contains(obs)

            pursuer, _ = p.getBasePositionAndOrientation(
                env.pursuer_id,
                physicsClientId=env.physics_client_id,
            )
            evader, _ = p.getBasePositionAndOrientation(
                env.evader_id,
                physicsClientId=env.physics_client_id,
            )
            assert pursuer[2] >= 0.5
            assert evader[2] >= 0.5
            assert np.hypot(evader[0] - pursuer[0], evader[2] - pursuer[2]) > 3.0
            np.testing.assert_allclose(
                obs[8:10],
                [evader[0] - pursuer[0], evader[2] - pursuer[2]],
            )
    finally:
        env.close()


def test_step_uses_frame_skip_and_progress_reward() -> None:
    env = PursuitEvasionEnv(
        render_mode="DIRECT",
        gravity=0.0,
        evader_controller=BaseEvaderController(),
    )
    try:
        env.reset(seed=0)
        p.resetBasePositionAndOrientation(
            env.pursuer_id,
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            env.evader_id,
            [5.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.physics_client_id,
        )
        p.resetBaseVelocity(
            env.pursuer_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=env.physics_client_id,
        )
        p.resetBaseVelocity(
            env.evader_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=env.physics_client_id,
        )

        obs, reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )
        velocity, _ = p.getBaseVelocity(
            env.pursuer_id,
            physicsClientId=env.physics_client_id,
        )

        assert obs.shape == (12,)
        assert velocity[0] == pytest.approx(env.max_accel * env.frame_skip / 240)
        assert reward > -0.01
        assert not terminated
        assert not truncated
        assert info == {"is_success": False}
    finally:
        env.close()


def test_catch_is_range_based_without_agent_collision() -> None:
    env = PursuitEvasionEnv(
        render_mode="DIRECT",
        gravity=0.0,
        evader_controller=BaseEvaderController(),
    )
    try:
        env.reset(seed=0)
        p.resetBasePositionAndOrientation(
            env.pursuer_id,
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            env.evader_id,
            [0.4, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=env.physics_client_id,
        )
        p.performCollisionDetection(physicsClientId=env.physics_client_id)
        assert not p.getContactPoints(
            bodyA=env.pursuer_id,
            bodyB=env.evader_id,
            physicsClientId=env.physics_client_id,
        )

        _, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )

        assert terminated
        assert not truncated
        assert reward == pytest.approx(env.CATCH_REWARD - env.TIME_PENALTY)
        assert info == {"is_success": True}
    finally:
        env.close()


def test_random_evader_is_seeded_by_environment_reset() -> None:
    env = PursuitEvasionEnv(
        render_mode="DIRECT",
        gravity=0.0,
        evader_controller=RandomEvaderController(),
    )
    try:
        env.reset(seed=123)
        first_action = env.evader_controller.get_action(env._get_evader_obs())

        env.reset(seed=123)
        repeated_action = env.evader_controller.get_action(env._get_evader_obs())

        np.testing.assert_array_equal(first_action, repeated_action)
    finally:
        env.close()


def test_controller_list_sampling_and_evader_acceleration_scale() -> None:
    controllers = [BaseEvaderController(), ForwardEvaderController()]
    env = PursuitEvasionEnv(
        render_mode="DIRECT",
        gravity=0.0,
        evader_accel_scale=0.5,
        evader_controllers=controllers,
    )
    try:
        env.reset(seed=0)
        assert env.max_steps == 400
        assert env.evader_controller in controllers

        env.step(np.zeros(2, dtype=np.float32))
        velocity, _ = p.getBaseVelocity(
            env.evader_id,
            physicsClientId=env.physics_client_id,
        )
        expected_x_velocity = (
            env.max_accel * env.evader_accel_scale * env.frame_skip / 240
            if isinstance(env.evader_controller, ForwardEvaderController)
            else 0.0
        )
        assert velocity[0] == pytest.approx(expected_x_velocity)
    finally:
        env.close()
