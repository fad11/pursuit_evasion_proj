import numpy as np


class BaseEvaderController:
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        pass


class RandomEvaderController(BaseEvaderController):
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(low=-1.0, high=1.0, size=2).astype(np.float32)


class PureEvaderController(BaseEvaderController):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array(
            [-float(obs[1]), -float(obs[2])],
            dtype=np.float32,
        )


class ManeuveringEvaderController(PureEvaderController):
    def __init__(self) -> None:
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        away = np.array(
            [-float(obs[1]), -float(obs[2])],
            dtype=np.float32,
        )
        lateral = np.array([-away[1], away[0]], dtype=np.float32)
        maneuver = 0.65 * np.sin(self.step * 0.17)
        self.step += 1
        action = away + maneuver * lateral
        return (action / np.linalg.norm(action)).astype(np.float32)


class StraightLineEvaderController(BaseEvaderController):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class CircularEvaderController(BaseEvaderController):
    def __init__(self, angular_speed: float = 0.12) -> None:
        self.angular_speed = angular_speed
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        angle = self.step * self.angular_speed
        self.step += 1
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


class SpiralEvaderController(BaseEvaderController):
    def __init__(
        self,
        angular_speed: float = 0.12,
        radial_growth: float = 0.01,
    ) -> None:
        self.angular_speed = angular_speed
        self.radial_growth = radial_growth
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        angle = self.step * self.angular_speed
        magnitude = min(1.0, (self.step + 1) * self.radial_growth)
        self.step += 1
        return np.array(
            [magnitude * np.cos(angle), magnitude * np.sin(angle)],
            dtype=np.float32,
        )


class ZigZagEvaderController(BaseEvaderController):
    def __init__(self, steps_per_switch: int = 20) -> None:
        if steps_per_switch <= 0:
            raise ValueError("steps_per_switch must be positive")
        self.steps_per_switch = steps_per_switch
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        side = 1.0 if (self.step // self.steps_per_switch) % 2 == 0 else -1.0
        self.step += 1
        return (np.array([1.0, side], dtype=np.float32) / np.sqrt(2.0)).astype(
            np.float32
        )
