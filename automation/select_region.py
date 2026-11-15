import time

import cv2
import numpy as np
import pyautogui

print("Make sure Chrome shows the game and nothing covers it.")
print("Capturing the whole screen in 5 seconds...")
time.sleep(5)

screenshot = np.array(pyautogui.screenshot())      # whole screen, RGB
height, width = screenshot.shape[:2]
print(f"Screen size: {width} x {height}")

# Shrink big screens so the selection window fits on the monitor
scale = min(1.0, 1400 / width)
preview = cv2.resize(screenshot, None, fx=scale, fy=scale)
preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

print("Drag a rectangle around the game strip, then press ENTER. (C = cancel)")
x, y, w, h = cv2.selectROI("Select the game area", preview, showCrosshair=False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    print("No area selected.")
else:
    left, top = int(x / scale), int(y / scale)
    region_width, region_height = int(w / scale), int(h / scale)
    print()
    print(f"LEFT, TOP = {left}, {top}")
    print(f"WIDTH, HEIGHT = {region_width}, {region_height}")

    crop = screenshot[top:top + region_height, left:left + region_width]
    cv2.imwrite("results/selected_region.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print("Saved results/selected_region.png (check that it shows only the game)")