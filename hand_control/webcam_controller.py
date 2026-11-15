"""Reads the webcam in a background thread so the game keeps running at 60 FPS."""

import os
import threading
import time

import cv2
import numpy as np

from hand_control.fist_detector import FistDetector, HandReading
from hand_control.jump_trigger import JumpTrigger


class WebcamJumpController:
    def __init__(self, camera_index: int = 0) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(camera_index, backend)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open the webcam. Is another app using it?")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.detector = FistDetector()
        self.trigger = JumpTrigger()
        self.jump_event = threading.Event()
        self.preview: np.ndarray | None = None    # latest camera picture with overlays
        self.error: str | None = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        try:
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)               # mirror view feels natural
                reading = self.detector.analyse(frame)
                if self.trigger.update(reading.is_fist):
                    self.jump_event.set()
                self.preview = self._annotate(frame, reading)
        except Exception as error:                        # report it in the main thread
            self.error = str(error)

    @staticmethod
    def _annotate(frame: np.ndarray, reading: HandReading) -> np.ndarray:
        picture = frame.copy()
        for x, y, curled in reading.tips:
            cv2.circle(picture, (x, y), 8, (0, 200, 0) if curled else (0, 0, 255), -1)
        if not reading.found:
            text, color = "no hand", (0, 165, 255)
        elif reading.is_fist:
            text, color = "FIST -> JUMP", (0, 200, 0)
        else:
            text, color = f"open ({reading.curled_fingers}/4 curled)", (255, 255, 255)
        cv2.putText(picture, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return picture

    def consume_jump(self) -> bool:
        """True once per fist; the game calls this every frame."""
        if self.jump_event.is_set():
            self.jump_event.clear()
            return True
        return False

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=2)
        self.capture.release()
        self.detector.close()