from collections import deque

import numpy as np
import pyautogui
import pygetwindow

from vision.image_processing import preprocess

pyautogui.PAUSE = 0      # no built-in delay after each PyAutoGUI call
STACK_SIZE = 4
MOVING_FRAMES_TO_START = 10         # consecutive changing frames = the game is running
STATIC_FRAMES_FOR_GAME_OVER = 20    # consecutive identical frames = the game stopped


def active_window_title() -> str:
    """Title of the window that currently has the keyboard, or '' if unknown."""
    try:
        window = pygetwindow.getActiveWindow()
        return window.title if window is not None else ""
    except Exception:
        return ""


class BrowserDino:
    """Watches a region of the screen and presses Space, only when the browser is focused."""

    def __init__(self, left: int, top: int, width: int, height: int, window_keyword: str) -> None:
        self.region = (left, top, width, height)
        self.window_keyword = window_keyword.lower()
        self.frames: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
        self.previous_raw: np.ndarray | None = None
        self.static_count = 0
        self.moving_count = 0
        self.seen_motion = False

    def observe(self) -> np.ndarray:
        """Take a screenshot and return the stack of the last 4 frames, (4, 60, 160)."""
        raw = np.array(pyautogui.screenshot(region=self.region))

        if self.previous_raw is not None:
            if np.array_equal(raw, self.previous_raw):
                self.static_count += 1
                self.moving_count = 0
            else:
                self.static_count = 0
                self.moving_count += 1
                if self.moving_count >= MOVING_FRAMES_TO_START:
                    self.seen_motion = True
        self.previous_raw = raw

        frame = preprocess(raw)
        if not self.frames:
            for _ in range(STACK_SIZE):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
        return np.stack(self.frames)

    @property
    def game_over(self) -> bool:
        return self.seen_motion and self.static_count >= STATIC_FRAMES_FOR_GAME_OVER

    def focused(self) -> bool:
        return self.window_keyword in active_window_title().lower()

    def press_space(self) -> bool:
        """Press Space only if the browser has the keyboard focus. True if pressed."""
        if not self.focused():
            return False
        pyautogui.press("space")
        return True