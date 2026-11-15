import time

import cv2
import numpy as np
import pyautogui

# The numbers you measured with select_region
LEFT, TOP = 551, 456
WIDTH, HEIGHT = 810, 227
DURATION = 10

print("Starting in 5 seconds. Click on the game page, press Space, and play normally.")
time.sleep(5)

previous = None
log: list[tuple[float, float]] = []
start = time.perf_counter()
saved = 0
next_save = 0.0

while time.perf_counter() - start < DURATION:
    raw = np.array(pyautogui.screenshot(region=(LEFT, TOP, WIDTH, HEIGHT)))
    now = time.perf_counter() - start
    if previous is not None:
        change = float(np.abs(raw.astype(np.int16) - previous.astype(np.int16)).mean())
        log.append((now, change))
    previous = raw
    if now >= next_save:
        cv2.imwrite(f"results/diag_{saved}.png", cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        saved += 1
        next_save += 1.0

for second in range(DURATION):
    bucket = [change for t, change in log if second <= t < second + 1]
    if bucket:
        identical = sum(1 for change in bucket if change == 0)
        print(
            f"second {second}: {len(bucket)} frames, {identical} identical to the previous, "
            f"mean change {sum(bucket) / len(bucket):.3f}"
        )