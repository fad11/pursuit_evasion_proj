import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BasePursuitEvasionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, mode="pursuer", obs_type="relative", dim=2, arena_size=20.0, 
                 max_steps=300, v_max=1.0, a_max=1.0, thrust_max=None, body_rate_max=None, capture_radius=0.3,
                 gravity=0.0, reset_range=(1.0, 5.0),
                 action_type="acceleration",
                 pursuer_controller=None, evader_controller=None, render_mode=None):
        """Initialize the environment using its mode, observation, arena, and dynamics settings.

        Args:
            mode: Which agent acts, either "pursuer", "evader", or both ("multi").
            obs_type: Observation format, either "relative" (3*dim values) or "los" (4 scalars).
            dim: Number of spatial dimensions: 2 uses x and z axes; 3 uses x, y, and z axes.
            arena_size: Length of each side of the square arena.
            max_steps: Maximum number of steps before an episode is truncated.
            v_max: Maximum speed of either agent.
            a_max: Maximum acceleration magnitude allowed for each action component.
            thrust_max: Maximum thrust magnitude when action_type is "thrust".
            body_rate_max: Maximum body-rate magnitude when action_type is "thrust".
            capture_radius: Distance threshold at which the pursuer captures the evader.
            gravity: Gravity setting for the physics backend.
            reset_range: Inclusive minimum and maximum initial distance between the agents.
            action_type: "acceleration" (dim acceleration components) or "thrust" (thrust + body rates).
            pursuer_controller: Optional controller that chooses pursuer actions.
            evader_controller: Optional controller that chooses evader actions.
            render_mode: Optional Gymnasium rendering mode.
        """
        super().__init__()
        self.mode, self.obs_type, self.dim, self.arena_size = mode, obs_type, dim, arena_size
        self.action_type = action_type
        self.max_steps, self.v_max, self.a_max, self.capture_radius = max_steps, v_max, a_max, capture_radius
        self.thrust_max = a_max if thrust_max is None else thrust_max
        self.body_rate_max = a_max if body_rate_max is None else body_rate_max
        self.gravity, self.reset_range = gravity, reset_range
        self.render_mode, self.pursuer_controller, self.evader_controller = render_mode, pursuer_controller, evader_controller
        self.dt, self.use_physics_backend = 0.1, False

        if self.pursuer_controller:
            self.pursuer_controller.v_max, self.pursuer_controller.a_max = self.v_max, self.a_max
            self.pursuer_controller.dt = self.dt
            self.pursuer_controller.set_dim(self.dim)
        if self.evader_controller:
            self.evader_controller.v_max, self.evader_controller.a_max = self.v_max, self.a_max
            self.evader_controller.dt = self.dt
            self.evader_controller.set_dim(self.dim)

        if self.action_type == "acceleration":
            action_dim = self.dim
            low = np.full(action_dim, -a_max); high = np.full(action_dim, a_max)
        else:
            action_dim = 2 if self.dim == 2 else 4
            low = np.array([-self.thrust_max] + [-self.body_rate_max]*(action_dim-1))
            high = np.array([self.thrust_max] + [self.body_rate_max]*(action_dim-1))
        self.action_space = spaces.Box(low=low, high=high, shape=(action_dim,), dtype=np.float32)
        
        los_dim = 4 if self.dim == 2 else 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((3*dim,) if self.obs_type=="relative" else (los_dim,)), dtype=np.float32)

        self.step_count, self.prev_los_angle = 0, None
        self.current_los_rate = 0.0

    def _get_obs(self, role):
        """Build an observation from `role` perspective: relative mode returns [rel_pos, rel_vel, self_vel]; LOS mode returns [range, angle, los_rate, closing]."""
        if role == "pursuer": pos_self, vel_self, pos_target, vel_target = self.p_pos, self.p_vel, self.e_pos, self.e_vel
        else: pos_self, vel_self, pos_target, vel_target = self.e_pos, self.e_vel, self.p_pos, self.p_vel
        rel_pos, rel_vel = pos_target - pos_self, vel_target - vel_self
        if self.obs_type == "relative": return np.concatenate([rel_pos, rel_vel, vel_self]).astype(np.float32)
        elif self.obs_type == "los":
            rng = np.linalg.norm(rel_pos); closing = np.dot(rel_pos, rel_vel) / (rng + 1e-6)
            if self.dim == 2:
                los = np.arctan2(rel_pos[1], rel_pos[0])
                return np.array([rng, los, self.current_los_rate, closing], dtype=np.float32)
            azimuth = np.arctan2(rel_pos[1], rel_pos[0])
            elevation = np.arctan2(rel_pos[2], np.hypot(rel_pos[0], rel_pos[1]))
            return np.array([rng, azimuth, elevation, self.current_los_rate, closing], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset episode state, randomize positions within `reset_range`, zero velocities, reset controllers, and return initial observation."""
        super().reset(seed=seed)
        self.step_count, self.prev_los_angle, self.current_los_rate = 0, None, 0.0
        self.p_pos = np.zeros(self.dim)
        dist = self.np_random.uniform(self.reset_range[0], self.reset_range[1])
        direction = self.np_random.normal(size=self.dim)
        direction /= (np.linalg.norm(direction) + 1e-6)
        self.e_pos = self.p_pos + direction * dist
        for i in range(self.dim):
            if self.e_pos[i] > self.arena_size/2: self.e_pos[i] = self.arena_size/2
            elif self.e_pos[i] < -self.arena_size/2: self.e_pos[i] = -self.arena_size/2

        self.p_vel, self.e_vel = np.zeros(self.dim), np.zeros(self.dim)
        if self.pursuer_controller: self.pursuer_controller.reset(self.p_pos)
        if self.evader_controller: self.evader_controller.reset(self.e_pos)
        
        if self.mode == "multi": return {"pursuer": self._get_obs("pursuer"), "evader": self._get_obs("evader")}, {}
        return self._get_obs(self.mode), {}

    def set_gravity(self, gravity):
        self.gravity = float(gravity)
        self._set_gravity(self.gravity)

    def _set_gravity(self, gravity):
        pass

    def configure_curriculum(self, gravity, reset_range, role, opponent_class):
        self.set_gravity(gravity)
        self.reset_range = tuple(reset_range)
        controller = opponent_class(
            dim=self.dim, v_max=self.v_max, a_max=self.a_max, dt=self.dt
        )
        if role == "pursuer":
            self.evader_controller = controller
            controller.reset(getattr(self, "e_pos", None))
        else:
            self.pursuer_controller = controller
            controller.reset(getattr(self, "p_pos", None))

    def step(self, action):
        self.step_count += 1

        if self.mode == "multi":
            a_p, a_e = action["pursuer"], action["evader"]
        elif self.mode == "pursuer":
            a_p = action
            if self.evader_controller:
                raw_acc = self.evader_controller.compute(self.e_pos, self.e_vel, self.p_pos, self.p_vel)
                a_e = self._accel_to_action(raw_acc, self.e_vel) if self.action_type != "acceleration" else raw_acc
            else:
                a_e = self.action_space.sample()
        elif self.mode == "evader":
            a_e = action
            if self.pursuer_controller:
                raw_acc = self.pursuer_controller.compute(self.p_pos, self.p_vel, self.e_pos, self.e_vel)
                a_p = self._accel_to_action(raw_acc, self.p_vel) if self.action_type != "acceleration" else raw_acc
            else:
                a_p = self.action_space.sample()

        if self.action_type == "acceleration":
            a_p = np.clip(a_p, -self.a_max, self.a_max)
            a_e = np.clip(a_e, -self.a_max, self.a_max)
        else:
            lo = self.action_space.low
            hi = self.action_space.high
            a_p = np.clip(a_p, lo, hi)
            a_e = np.clip(a_e, lo, hi)

        if not self.use_physics_backend:
            if self.action_type == "acceleration":
                self.p_vel += np.asarray(a_p, dtype=float) * self.dt
                self.e_vel += np.asarray(a_e, dtype=float) * self.dt
                for vv in (self.p_vel, self.e_vel):
                    n = np.linalg.norm(vv)
                    if n > self.v_max: vv[:] = (vv / n) * self.v_max
            else:
                self.p_vel = self._apply_action(self.p_vel, a_p)
                self.e_vel = self._apply_action(self.e_vel, a_e)

            self.p_pos += self.p_vel * self.dt
            self.e_pos += self.e_vel * self.dt
            self.p_pos, self.p_vel = self._apply_walls(self.p_pos, self.p_vel)
            self.e_pos, self.e_vel = self._apply_walls(self.e_pos, self.e_vel)
        else:
            if self.action_type == "acceleration":
                accel_p, accel_e = a_p, a_e
            else:
                accel_p = (self._apply_action(self.p_vel, a_p) - self.p_vel) / self.dt
                accel_e = (self._apply_action(self.e_vel, a_e) - self.e_vel) / self.dt
            self._step_physics(accel_p, accel_e)
            for velocity in (self.p_vel, self.e_vel):
                speed = np.linalg.norm(velocity)
                if speed > self.v_max:
                    velocity[:] = velocity * (self.v_max / speed)
            self.p_pos, self.p_vel = self._apply_walls(self.p_pos, self.p_vel)
            self.e_pos, self.e_vel = self._apply_walls(self.e_pos, self.e_vel)
            self._sync_physics_state()

        if self.obs_type == "los":
            rel_pos = self.e_pos - self.p_pos
            rel_vel = self.e_vel - self.p_vel
            cross = (rel_pos[0]*rel_vel[1] - rel_pos[1]*rel_vel[0]) if self.dim == 2 else np.cross(rel_pos, rel_vel)
            denom = np.dot(rel_pos, rel_pos) + 1e-6
            self.current_los_rate = (cross / denom) if np.isscalar(cross) else (np.linalg.norm(cross) / denom)

        dist = np.linalg.norm(self.p_pos - self.e_pos)
        captured = dist < self.capture_radius
        truncated = self.step_count >= self.max_steps

        if captured:
            reward_p = 10.0
            reward_e = -10.0
        else:
            reward_p = -0.1 - (dist / self.arena_size)
            dist_bonus = min(0.5, dist / self.arena_size)
            reward_e = 0.1 + dist_bonus

        if self.mode == "multi":
            info = {
                "captured": captured,
                "is_success": {
                    "pursuer": captured,
                    "evader": not captured,
                },
            }
            return (
                {"pursuer": self._get_obs("pursuer"), "evader": self._get_obs("evader")},
                {"pursuer": reward_p, "evader": reward_e},
                {"pursuer": captured, "evader": captured},
                {"pursuer": truncated, "evader": truncated},
                info
            )

        is_success = captured if self.mode == "pursuer" else not captured
        return (
            self._get_obs(self.mode),
            (reward_p if self.mode == "pursuer" else reward_e),
            captured,
            truncated,
            {"captured": captured, "is_success": is_success}
        )

    def _apply_action(self, vel, action):
        action = np.asarray(action, dtype=float)
        thrust = action[0]; rates = action[1:]
        speed = np.linalg.norm(vel)
        speed = float(np.clip(speed + thrust * self.dt, 0.0, self.v_max))
        if self.dim == 2:
            theta = np.arctan2(vel[1], vel[0]) if speed > 1e-6 else 0.0
            theta += rates[0] * self.dt
            return np.array([speed * np.cos(theta), speed * np.sin(theta)], dtype=float)
        omega = np.asarray(rates, dtype=float)
        vhat = vel / (np.linalg.norm(vel) + 1e-6) if np.linalg.norm(vel) > 1e-6 else np.array([1.0, 0.0, 0.0])
        vhat = vhat + np.cross(omega * self.dt, vhat)
        n = np.linalg.norm(vhat)
        if n > 1e-6: vhat = vhat / n
        return speed * vhat

    def _accel_to_action(self, acc, vel):
        acc = np.asarray(acc, dtype=float); vel = np.asarray(vel, dtype=float)
        speed = np.linalg.norm(vel)
        if speed < 1e-6:
            thrust = np.linalg.norm(acc)
            rates = np.zeros(1 if self.dim == 2 else 3)
            return np.concatenate(([thrust], rates))
        vhat = vel / speed
        thrust = float(np.dot(acc, vhat))
        if self.dim == 2:
            rate = (vel[0] * acc[1] - vel[1] * acc[0]) / (speed * speed)
            return np.array([thrust, rate])
        omega = np.cross(vel, acc) / (speed * speed)
        return np.concatenate(([thrust], omega))

    def _apply_walls(self, pos, vel):
        """Clamp positions to arena bounds and reverse/bounce the corresponding velocity component at each wall."""
        for i in range(self.dim):
            if pos[i] > self.arena_size/2: pos[i], vel[i] = self.arena_size/2, vel[i] * -0.5
            elif pos[i] < -self.arena_size/2: pos[i], vel[i] = -self.arena_size/2, vel[i] * -0.5
        return pos, vel

    def _sync_physics_state(self):
        pass
