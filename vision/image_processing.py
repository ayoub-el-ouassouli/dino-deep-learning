import cv2
import numpy as np

FRAME_WIDTH = 160
FRAME_HEIGHT = 60


def preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB image of shape (height, width, 3) -> grayscale (60, 160), values 0..1."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0