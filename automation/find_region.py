import time

import pyautogui


def read_corner(name: str) -> tuple[int, int]:
    print(f"Move the mouse to the {name} corner of the game area and keep it there...")
    for seconds in range(5, 0, -1):
        print(f"  reading in {seconds}...")
        time.sleep(1)
    x, y = pyautogui.position()
    print(f"  {name}: x={x}, y={y}")
    return x, y


left, top = read_corner("TOP-LEFT")
right, bottom = read_corner("BOTTOM-RIGHT")

print()
print(f"LEFT, TOP = {left}, {top}")
print(f"WIDTH, HEIGHT = {right - left}, {bottom - top}")