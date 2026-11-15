import sys
from pathlib import Path

import cv2
import numpy as np

SPRITES = ("dino", "cactus_small", "cactus_large")
MIN_WIDTH, MIN_HEIGHT = 6, 20     # smaller objects are ground dots, digits, etc.
MAX_SIZE = 120                    # bigger objects are not single sprites
DISPLAY_SCALE = 2


def find_candidates(gray: np.ndarray):
    """Return the separate dark objects in the picture, sorted left to right."""
    dark = (gray < 200).astype(np.uint8)

    # The ground line is a long dark row: erase it so it cannot glue objects together
    dark[dark.mean(axis=1) > 0.5, :] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    candidates = []
    for index in range(1, count):          # index 0 is the background
        x, y, w, h, _area = stats[index]
        if MIN_WIDTH <= w <= MAX_SIZE and MIN_HEIGHT <= h <= MAX_SIZE:
            candidates.append((int(x), int(y), int(w), int(h), index))
    candidates.sort()
    return candidates, labels


def save_overview(image: np.ndarray, candidates) -> None:
    """Save an enlarged picture with a red box and a number around every object."""
    s = DISPLAY_SCALE
    big = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    for number, (x, y, w, h, _) in enumerate(candidates, start=1):
        cv2.rectangle(big, (x * s, y * s), ((x + w) * s, (y + h) * s), (0, 0, 255), 1)
        cv2.putText(big, str(number), (x * s, max(14, y * s - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite("results/sprite_candidates.png", big)


def main(image_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Could not read {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    candidates, labels = find_candidates(gray)
    if not candidates:
        raise SystemExit("No objects found in this picture.")

    save_overview(image, candidates)
    print(f"Found {len(candidates)} objects. Open results/sprite_candidates.png to see the numbers:")
    for number, (x, y, w, h, _) in enumerate(candidates, start=1):
        print(f"  {number}: {w} x {h} pixels (left edge at x={x})")

    Path("assets").mkdir(exist_ok=True)
    for name in SPRITES:
        answer = input(f"Number of the {name} (ENTER to skip): ").strip()
        if not answer:
            continue
        if not answer.isdigit() or not 1 <= int(answer) <= len(candidates):
            print("  not a valid number, skipped")
            continue

        x, y, w, h, index = candidates[int(answer) - 1]
        alpha = np.where(labels[y:y + h, x:x + w] == index, 255, 0).astype(np.uint8)
        part = gray[y:y + h, x:x + w]
        cv2.imwrite(f"assets/{name}.png", cv2.merge([part, part, part, alpha]))
        print(f"  saved assets/{name}.png ({w} x {h} pixels)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m vision.extract_sprites <image>")
    main(sys.argv[1])