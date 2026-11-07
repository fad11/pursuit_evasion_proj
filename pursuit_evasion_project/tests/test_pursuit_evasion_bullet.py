from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import pybullet as p

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pursuit_evasion.envs.pursuit_evasion_bullet as bullet_module
from pursuit_evasion.controllers.base import Idle, StraightLine
from pursuit_evasion.controllers.evader import PureEvasion
from pursuit_evasion.controllers.pursuer import PurePursuit
from pursuit_evasion.envs.pursuit_evasion_bullet import PursuitEvasionBullet


class PursuitEvasionBulletTests(unittest.TestCase):
    def make_env(self, **kwargs) -> PursuitEvasionBullet:
        kwargs.setdefault("gravity", 0.0)
        env = PursuitEvasionBullet(**kwargs)
        self.addCleanup(env.close)
        return env

    @staticmethod
    def place_agents(
        env: PursuitEvasionBullet,
        pursuer=(0.0, 1.0),
        evader=(6.0, 1.0),
    ) -> None:
        env.p_pos = np.asarray(pursuer, dtype=float)
        env.e_pos = np.asarray(evader, dtype=float)
        env.p_vel = np.zeros(2)
        env.e_vel = np.zeros(2)
        env._sync()

    def test_rejects_unsupported_configuration(self):
        for kwargs in (
            {"dim": 3},
            {"mode": "bad"},
            {"obs_type": "bad"},
            {"action_type": "bad"},
            {"render_mode": "bad"},
            {"arena_size": 0},
            {"max_steps": 0},
            {"thrust_max": 0},
            {"body_rate_max": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PursuitEvasionBullet(**kwargs)

    def test_reset_seed_options_and_initial_physics_state(self):
        env = self.make_env(reset_range=(2.0, 4.0))
        obs, info = env.reset(
            seed=22,
            options={"center": [1.0, 2.0], "distance": 3.0},
        )
        self.assertEqual(info, {})
        self.assertTrue(env.observation_space.contains(obs))
        np.testing.assert_allclose(env.p_pos, [1.0, 2.0])
        self.assertAlmostEqual(np.linalg.norm(env.e_pos - env.p_pos), 3.0, places=5)
        np.testing.assert_array_equal(env.p_vel, [0.0, 0.0])
        np.testing.assert_array_equal(env.e_vel, [0.0, 0.0])

        position, _ = p.getBasePositionAndOrientation(
            env.p_body,
            physicsClientId=env.client,
        )
        np.testing.assert_allclose([position[0], position[2]], env.p_pos)

        env.reset(seed=22, options={"center": [1.0, 2.0], "distance": 3.0})
        first_target = env.e_pos.copy()
        env.reset(seed=22, options={"center": [1.0, 2.0], "distance": 3.0})
        np.testing.assert_allclose(env.e_pos, first_target)

    def test_reset_random_spawn_stays_in_range_and_arena(self):
        env = self.make_env(reset_range=(2.0, 4.0), arena_size=20.0)
        env.reset(seed=8)
        distance = np.linalg.norm(env.e_pos - env.p_pos)
        self.assertGreaterEqual(distance, 2.0 - 1e-5)
        self.assertLessEqual(distance, 4.0 + 1e-5)
        self.assertTrue(np.all(np.abs(env.p_pos) <= env.arena_size / 2))
        self.assertTrue(np.all(np.abs(env.e_pos) <= env.arena_size / 2))

    def test_reset_and_housekeeping_reject_out_of_arena_center(self):
        env = self.make_env()
        with self.assertRaisesRegex(ValueError, "center must be inside"):
            env.reset(options={"center": [env.arena_size, 0.0]})
        with self.assertRaisesRegex(ValueError, "center must be inside"):
            env.housekeeping(center=[env.arena_size, 0.0])

    def test_single_agent_spaces_and_step_contract(self):
        for mode, controller_key in (
            ("pursuer", "evader_controller"),
            ("evader", "pursuer_controller"),
        ):
            with self.subTest(mode=mode):
                controller = Idle(dim=2)
                env = self.make_env(mode=mode, **{controller_key: controller})
                obs, info = env.reset(seed=4)
                self.assertEqual(info, {})
                self.assertTrue(env.observation_space.contains(obs))
                action = np.zeros(2, dtype=np.float32)
                self.assertTrue(env.action_space.contains(action))
                result = env.step(action)
                self.assertEqual(len(result), 5)
                self.assertTrue(env.observation_space.contains(result[0]))
                self.assertIn("distance", result[4])
                self.assertIn("captured", result[4])
                self.assertIn("is_success", result[4])

    def test_pursuer_rl_action_moves_pursuer_and_idle_opponent_stays(self):
        env = self.make_env(
            mode="pursuer",
            v_max=5.0,
            pursuer_controller=None,
            evader_controller=Idle(dim=2),
        )
        env.reset(seed=1)
        self.place_agents(env)

        env.step(np.array([1.0, 0.0], dtype=np.float32))

        self.assertGreater(env.p_vel[0], 0.08)
        self.assertLess(env.p_vel[0], 0.12)
        self.assertAlmostEqual(env.e_vel[0], 0.0, places=5)

    def test_evader_rl_action_moves_evader_and_idle_opponent_stays(self):
        env = self.make_env(
            mode="evader",
            v_max=5.0,
            pursuer_controller=Idle(dim=2),
        )
        env.reset(seed=2)
        self.place_agents(env)

        env.step(np.array([0.0, 1.0], dtype=np.float32))

        self.assertGreater(env.e_vel[1], 0.08)
        self.assertLess(env.e_vel[1], 0.12)
        self.assertAlmostEqual(env.p_vel[1], 0.0, places=5)

    def test_evader_controller_uses_evader_state_and_moves_away(self):
        controller = PureEvasion(dim=2, v_max=1.0, a_max=1.0)
        env = self.make_env(
            mode="pursuer",
            v_max=5.0,
            evader_controller=controller,
        )
        env.reset(seed=3)
        self.place_agents(env)

        env.step(np.zeros(2, dtype=np.float32))

        self.assertGreater(env.e_vel[0], 0.0)
        self.assertEqual(controller._t, 1)

    def test_pursuer_controller_uses_pursuer_state_and_chases(self):
        controller = PurePursuit(dim=2, v_max=1.0, a_max=1.0)
        env = self.make_env(
            mode="evader",
            v_max=5.0,
            pursuer_controller=controller,
        )
        env.reset(seed=4)
        self.place_agents(env)

        env.step(np.zeros(2, dtype=np.float32))

        self.assertGreater(env.p_vel[0], 0.0)
        self.assertEqual(controller._t, 1)

    def test_reset_resets_scripted_controller_state(self):
        pursuer = PurePursuit(dim=2)
        evader = PureEvasion(dim=2)
        env = self.make_env(
            mode="multi",
            pursuer_controller=pursuer,
            evader_controller=evader,
        )
        env.reset(seed=5)
        env.step({})
        self.assertEqual(pursuer._t, 1)
        self.assertEqual(evader._t, 1)

        env.reset(seed=6)

        self.assertEqual(pursuer._t, 0)
        self.assertEqual(evader._t, 0)

    def test_multi_both_rl_actions_and_returns_match_spaces(self):
        env = self.make_env(mode="multi", pursuer_rl=True, evader_rl=True)
        obs, _ = env.reset(seed=7)
        action = {
            "pursuer": np.array([1.0, 0.0], dtype=np.float32),
            "evader": np.array([-1.0, 0.0], dtype=np.float32),
        }
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(env.action_space.contains(action))

        result = env.step(action)

        self.assertTrue(env.observation_space.contains(result[0]))
        self.assertEqual(set(result[1]), {"pursuer", "evader"})
        self.assertEqual(set(result[2]), {"pursuer", "evader"})
        self.assertEqual(set(result[3]), {"pursuer", "evader"})
        self.assertIn("distance", result[4])
        self.assertGreater(env.p_vel[0], 0.0)
        self.assertLess(env.e_vel[0], 0.0)

    def test_multi_pursuer_rl_evader_controller(self):
        env = self.make_env(
            mode="multi",
            pursuer_rl=True,
            evader_controller=PureEvasion(dim=2),
        )
        obs, _ = env.reset(seed=8)
        self.assertEqual(set(env.action_space.spaces), {"pursuer"})
        self.assertTrue(env.observation_space.contains(obs))
        self.place_agents(env)

        env.step({"pursuer": np.zeros(2, dtype=np.float32)})

        self.assertGreater(env.e_vel[0], 0.0)

    def test_multi_evader_rl_pursuer_controller(self):
        env = self.make_env(
            mode="multi",
            evader_rl=True,
            pursuer_controller=PurePursuit(dim=2),
        )
        env.reset(seed=9)
        self.assertEqual(set(env.action_space.spaces), {"evader"})
        self.place_agents(env)

        env.step({"evader": np.zeros(2, dtype=np.float32)})

        self.assertGreater(env.p_vel[0], 0.0)

    def test_multi_both_controllers_needs_no_rl_actions(self):
        env = self.make_env(
            mode="multi",
            pursuer_controller=StraightLine(dim=2, direction=[1.0, 0.0]),
            evader_controller=StraightLine(dim=2, direction=[-1.0, 0.0]),
        )
        env.reset(seed=10)
        self.assertTrue(env.action_space.contains({}))
        self.place_agents(env)

        env.step({})

        self.assertGreater(env.p_vel[0], 0.0)
        self.assertLess(env.e_vel[0], 0.0)

    def test_multi_uncontrolled_non_rl_agent_uses_seeded_random_action(self):
        env = self.make_env(mode="multi")
        action = {}
        env.reset(seed=30)
        env.step(action)
        state_one = (env.p_vel.copy(), env.e_vel.copy())
        env.reset(seed=30)
        env.step(action)

        np.testing.assert_allclose(env.p_vel, state_one[0], atol=1e-8)
        np.testing.assert_allclose(env.e_vel, state_one[1], atol=1e-8)

    def test_thrust_action_moves_along_tilted_body_axis(self):
        env = self.make_env(
            mode="pursuer",
            action_type="thrust",
            v_max=5.0,
            thrust_max=5.0,
            body_rate_max=1.0,
        )
        env.reset(seed=11)
        self.place_agents(env)

        for _ in range(10):
            env.step(np.array([5.0, 0.5], dtype=np.float32))

        _, orientation = p.getBasePositionAndOrientation(
            env.p_body,
            physicsClientId=env.client,
        )
        pitch = p.getEulerFromQuaternion(orientation)[1]
        self.assertGreater(pitch, 0.1)
        self.assertGreater(env.p_pos[0], 0.0)

    def test_use_torque_false_disables_body_rate_rotation(self):
        env = self.make_env(
            mode="pursuer",
            action_type="thrust",
            use_torque=False,
            v_max=5.0,
            thrust_max=5.0,
            body_rate_max=1.0,
        )
        env.reset(seed=24)
        self.place_agents(env)

        for _ in range(10):
            env.step(np.array([5.0, 1.0], dtype=np.float32))

        _, orientation = p.getBasePositionAndOrientation(
            env.p_body,
            physicsClientId=env.client,
        )
        pitch = p.getEulerFromQuaternion(orientation)[1]
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(env.p_pos[0], 0.0, places=4)

    def test_thrust_limits_default_to_acceleration_limit(self):
        env = self.make_env(mode="pursuer", action_type="thrust", a_max=2.5)
        np.testing.assert_allclose(env.action_space.low, [-2.5, -2.5])
        np.testing.assert_allclose(env.action_space.high, [2.5, 2.5])

    def test_action_shape_and_multi_keys_are_validated(self):
        single = self.make_env(
            mode="pursuer",
            evader_controller=Idle(dim=2),
        )
        single.reset(seed=12)
        with self.assertRaisesRegex(ValueError, "action must have shape"):
            single.step([0.0])

        multi = self.make_env(mode="multi", pursuer_rl=True)
        multi.reset(seed=13)
        with self.assertRaisesRegex(ValueError, "action keys"):
            multi.step({})

    def test_speed_limit_and_arena_bounce(self):
        env = self.make_env(
            mode="multi",
            pursuer_rl=True,
            evader_rl=True,
            arena_size=10.0,
            a_max=100.0,
            v_max=1.0,
        )
        env.reset(seed=14)
        self.place_agents(env, pursuer=(4.9, 1.0), evader=(-4.9, 1.0))

        env.step({
            "pursuer": np.array([100.0, 0.0], dtype=np.float32),
            "evader": np.array([-100.0, 0.0], dtype=np.float32),
        })

        self.assertAlmostEqual(env.p_pos[0], 5.0)
        self.assertAlmostEqual(env.e_pos[0], -5.0)
        self.assertLess(env.p_vel[0], 0.0)
        self.assertGreater(env.e_vel[0], 0.0)
        self.assertLessEqual(np.linalg.norm(env.p_vel), env.v_max + 1e-8)
        self.assertLessEqual(np.linalg.norm(env.e_vel), env.v_max + 1e-8)

    def test_gravity_changes_vertical_motion(self):
        env = self.make_env(
            mode="pursuer",
            gravity=9.81,
            evader_controller=Idle(dim=2),
        )
        env.reset(seed=15)
        self.place_agents(env, pursuer=(0.0, 3.0), evader=(6.0, 3.0))

        env.step(np.zeros(2, dtype=np.float32))

        self.assertLess(env.p_vel[1], 0.0)

    def test_los_observation_and_rate_are_finite(self):
        env = self.make_env(
            mode="multi",
            obs_type="los",
            pursuer_rl=True,
            evader_rl=True,
        )
        env.reset(seed=16)
        self.place_agents(env)
        env.p_vel = np.array([0.0, 0.0])
        env.e_vel = np.array([0.0, 0.5])
        env._sync()

        obs, _, _, _, _ = env.step({
            "pursuer": np.zeros(2, dtype=np.float32),
            "evader": np.zeros(2, dtype=np.float32),
        })

        self.assertEqual(obs["pursuer"].shape, (4,))
        self.assertEqual(obs["evader"].shape, (4,))
        self.assertTrue(np.all(np.isfinite(obs["pursuer"])))
        self.assertTrue(np.all(np.isfinite(obs["evader"])))
        self.assertNotEqual(float(obs["pursuer"][2]), 0.0)

    def test_capture_rewards_and_truncation(self):
        captured_env = self.make_env(
            mode="multi",
            pursuer_rl=True,
            evader_rl=True,
            capture_radius=10.0,
        )
        captured_env.reset(seed=17)
        _, rewards, terminated, truncated, info = captured_env.step({
            "pursuer": np.zeros(2, dtype=np.float32),
            "evader": np.zeros(2, dtype=np.float32),
        })
        self.assertEqual(rewards, {"pursuer": 10.0, "evader": -10.0})
        self.assertTrue(terminated["pursuer"])
        self.assertFalse(truncated["pursuer"])
        self.assertTrue(info["captured"])

        truncated_env = self.make_env(
            mode="pursuer",
            max_steps=1,
            capture_radius=0.01,
            evader_controller=Idle(dim=2),
        )
        truncated_env.reset(seed=18)
        self.place_agents(truncated_env)
        _, _, terminated, truncated, _ = truncated_env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_rgb_array_render_dimensions_and_dtype(self):
        env = self.make_env(mode="pursuer", render_mode="rgb_array")
        env.reset(seed=19)

        frame = env.render()

        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_headless_recording_keeps_frames_across_resets_and_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_dir = Path(temp_dir)
            first = self.make_env(
                mode="pursuer",
                record=True,
                frame_dir=str(frame_dir),
                evader_controller=Idle(dim=2),
            )
            first._capture_rgb = lambda: np.zeros((480, 640, 3), dtype=np.uint8)
            first.reset(seed=20)
            first.step(np.zeros(2, dtype=np.float32))
            first.reset(seed=21)
            names = sorted(path.name for path in frame_dir.glob("frame_*.png"))
            self.assertEqual(
                names,
                ["frame_00000.png", "frame_00001.png", "frame_00002.png"],
            )

            second = self.make_env(
                mode="pursuer",
                record=True,
                frame_dir=str(frame_dir),
                evader_controller=Idle(dim=2),
            )
            second._capture_rgb = lambda: np.zeros((480, 640, 3), dtype=np.uint8)
            second.reset(seed=22)
            names = sorted(path.name for path in frame_dir.glob("frame_*.png"))
            self.assertEqual(names[-1], "frame_00003.png")
            self.assertEqual(len(names), 4)

    def test_gui_mp4_recording_saves_moving_demo_video(self):
        video_path = PROJECT_ROOT / "tests" / "recordings" / "pursuit_evasion_bullet.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        real_connect = p.connect
        with patch.object(
            bullet_module.p,
            "connect",
            side_effect=lambda mode: real_connect(
                p.DIRECT if mode == p.GUI else mode
            ),
        ):
            env = self.make_env(
                mode="multi",
                gui=True,
                record=True,
                video_path=str(video_path),
                pursuer_rl=True,
                evader_rl=True,
                v_max=2.0,
            )
            try:
                env.reset(
                    seed=23,
                    options={"center": [0.0, 1.0], "distance": 8.0},
                )
                initial_positions = (env.p_pos.copy(), env.e_pos.copy())
                for step in range(40):
                    pursuer_action = (
                        [1.0, 0.2] if step < 20 else [-1.0, -0.2]
                    )
                    evader_action = (
                        [-0.7, -0.15] if step < 20 else [0.7, 0.15]
                    )
                    env.step({
                        "pursuer": np.asarray(pursuer_action, dtype=np.float32),
                        "evader": np.asarray(evader_action, dtype=np.float32),
                    })
                self.assertGreater(
                    np.linalg.norm(env.p_pos - initial_positions[0]), 0.5
                )
                self.assertGreater(
                    np.linalg.norm(env.e_pos - initial_positions[1]), 0.5
                )
            finally:
                env.close()

        self.assertTrue(video_path.is_file())
        self.assertGreater(video_path.stat().st_size, 1000)
        reader = imageio.get_reader(str(video_path), format="FFMPEG")
        try:
            metadata = reader.get_meta_data()
            frame = reader.get_data(0)
            frame_count = reader.count_frames()
        finally:
            reader.close()
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertAlmostEqual(metadata["fps"], 10.0)
        self.assertGreaterEqual(frame_count, 30)
        print(f"Saved movement recording: {video_path}")


if __name__ == "__main__":
    unittest.main()
