import cv2
import numpy as np

from environment.game import DinoGame
from environment.renderer import Renderer
from vision.image_processing import preprocess

game = DinoGame()
renderer = Renderer()

for _ in range(75):          # the first cactus is on screen by now
    game.step(0)
    renderer.draw(game)

frame = renderer.get_frame()
processed = preprocess(frame)

print("Raw frame shape:      ", frame.shape, frame.dtype)
print("Processed shape:      ", processed.shape, processed.dtype)
print("Value range:          ", processed.min(), "to", processed.max())

cv2.imwrite("results/frame_full.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
enlarged = cv2.resize(
    (processed * 255).astype(np.uint8), None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST
)
cv2.imwrite("results/frame_processed.png", enlarged)
print("Saved two images in results/")
renderer.close()