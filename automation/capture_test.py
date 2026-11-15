import time

import cv2
import numpy as np
import pyautogui

from vision.image_processing import preprocess

# Replace these with the numbers printed by find_region
LEFT, TOP = 551, 456
WIDTH, HEIGHT = 810, 227

print("Click on the Chrome window with the game now. Capturing in 5 seconds...")
time.sleep(5)

screenshot = pyautogui.screenshot(region=(LEFT, TOP, WIDTH, HEIGHT))
frame = np.array(screenshot)
print("Captured frame shape:", frame.shape)

processed = preprocess(frame)
print("Processed shape:", processed.shape)

cv2.imwrite("results/real_frame_full.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
enlarged = cv2.resize(
    (processed * 255).astype(np.uint8), None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST
)
cv2.imwrite("results/real_frame_processed.png", enlarged)
print("Saved results/real_frame_full.png and results/real_frame_processed.png")