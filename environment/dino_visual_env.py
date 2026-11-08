from collections import deque

import numpy as np

from environment.game import DinoGame
from environment.renderer import Renderer
from vision.image_processing import preprocess

SURVIVE_REWARD = 0.1
CRASH_PENALTY = -10.0
MAX_EPISODE_STEPS = 5000
STACK_SIZE = 4


class DinoVisualEnv:
    """Same game, but the observation is the last 4 screenshots, not 4 numbers."""

    def __init__(self) -> None:
        self.game = DinoGame()
        self.renderer = Renderer(headless=True)
        self.frames: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
        self.steps = 0

    def _capture(self) -> np.ndarray:
        """Draw the game and return the preprocessed picture (60, 160)."""
        self.renderer.draw(self.game)
        return preprocess(self.renderer.get_frame())

    def _observation(self) -> np.ndarray:
        """The stack of the last 4 frames, shape (4, 60, 160)."""
        return np.stack(self.frames)

    def reset(self):
        self.game.reset()
        self.steps = 0
        first = self._capture()
        for _ in range(STACK_SIZE):      # at the start, repeat the first frame
            self.frames.append(first)
        return self._observation(), {}

    def step(self, action: int):
        crashed = self.game.step(int(action))
        self.steps += 1
        self.frames.append(self._capture())   # the oldest frame drops out

        reward = CRASH_PENALTY if crashed else SURVIVE_REWARD
        terminated = crashed
        truncated = self.steps >= MAX_EPISODE_STEPS
        info = {"score": int(self.game.score)}
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.renderer.close()