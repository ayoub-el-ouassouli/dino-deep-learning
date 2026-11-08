import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.game import STATE_SIZE, DinoGame

SURVIVE_REWARD = 0.1
CRASH_PENALTY = -10.0
MAX_EPISODE_STEPS = 5000


class DinoEnv(gym.Env):
    """Gymnasium wrapper: turns DinoGame into a standard RL environment."""

    def __init__(self) -> None:
        super().__init__()
        self.game = DinoGame()
        self.action_space = spaces.Discrete(2)  # 0 = nothing, 1 = jump
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.steps = 0
        return self.game.get_state(), {}

    def step(self, action):
        crashed = self.game.step(int(action))
        self.steps += 1

        reward = CRASH_PENALTY if crashed else SURVIVE_REWARD
        terminated = crashed
        truncated = self.steps >= MAX_EPISODE_STEPS
        info = {"score": int(self.game.score)}

        return self.game.get_state(), reward, terminated, truncated, info