import random
import time

import cv2
import numpy as np

from environment.dino_visual_env import DinoVisualEnv

env = DinoVisualEnv()
obs, _ = env.reset()
print("Observation shape:", obs.shape, obs.dtype)

# Jump, then wait: the 4 stacked frames show the dino rising
env.reset()
for action in (1, 0, 0, 0):
    obs, *_ = env.step(action)
strip = np.hstack(obs)   # put the 4 frames side by side
strip = cv2.resize(
    (strip * 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST
)
cv2.imwrite("results/stacked_frames.png", strip)
print("Saved results/stacked_frames.png")

# Speed test: how many environment steps per second?
env.reset()
start = time.perf_counter()
steps = 0
while steps < 500:
    obs, reward, terminated, truncated, _ = env.step(random.choice([0, 1]))
    steps += 1
    if terminated or truncated:
        env.reset()
elapsed = time.perf_counter() - start
print(f"{steps / elapsed:.0f} environment steps per second")
env.close()