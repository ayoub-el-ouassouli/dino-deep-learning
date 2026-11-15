"""Detect a closed fist in webcam frames with MediaPipe's Hand Landmarker."""

import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path("assets/hand_landmarker.task")

WRIST = 0
FINGER_JOINTS = {                 # finger: (middle joint, fingertip) landmark numbers
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}
CURL_RATIO = 1.1            # bigger = easier to count a finger as curled
MIN_CURLED_FINGERS = 4      # all four fingers must be curled for a fist


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the hand model the first time (a few megabytes)."""
    if path.exists():
        return path
    print(f"Downloading the hand model to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as error:
        path.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed ({error}). Download this file manually and save it as {path}:\n{MODEL_URL}"
        )
    return path


@dataclass
class HandReading:
    found: bool = False
    is_fist: bool = False
    curled_fingers: int = 0
    tips: list[tuple[int, int, bool]] = field(default_factory=list)   # (x, y, curled)


class FistDetector:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def analyse(self, frame_bgr) -> HandReading:
        """Look for a hand in one webcam frame and decide whether it is a fist."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = max(int(time.monotonic() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp
        result = self.landmarker.detect_for_video(image, timestamp)
        if not result.hand_landmarks:
            return HandReading()

        landmarks = result.hand_landmarks[0]

        def point(index: int) -> tuple[float, float]:
            return landmarks[index].x * width, landmarks[index].y * height

        wrist = point(WRIST)
        tips: list[tuple[int, int, bool]] = []
        curled = 0
        for middle_joint, fingertip in FINGER_JOINTS.values():
            tip, joint = point(fingertip), point(middle_joint)
            is_curled = math.dist(tip, wrist) < CURL_RATIO * math.dist(joint, wrist)
            curled += is_curled
            tips.append((int(tip[0]), int(tip[1]), is_curled))

        return HandReading(True, curled >= MIN_CURLED_FINGERS, curled, tips)

    def close(self) -> None:
        self.landmarker.close()