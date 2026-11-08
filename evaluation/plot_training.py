import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "logs/dino_cnn_log.csv"
OUTPUT_PATH = "results/learning_curve_cnn.png"
RANDOM_BASELINE = 177
WINDOW = 10


def load_scores(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read episode numbers and scores from the training log."""
    episodes: list[int] = []
    scores: list[float] = []
    with open(path, newline="") as log_file:
        for row in csv.DictReader(log_file):
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
    return np.array(episodes), np.array(scores)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Average each value with its neighbours to smooth out the noise."""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main() -> None:
    episodes, scores = load_scores(LOG_PATH)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, scores, alpha=0.3, label="Score per episode")
    if len(scores) >= WINDOW:
        ax.plot(
            episodes[WINDOW - 1:],
            moving_average(scores, WINDOW),
            linewidth=2,
            label=f"{WINDOW}-episode average",
        )
    ax.axhline(RANDOM_BASELINE, linestyle="--", color="gray", label="Random agent (~177)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (frames survived)")
    ax.set_title("Dino DQN learning curve")
    ax.legend()
    ax.grid(alpha=0.3)

    Path("results").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print("Saved", OUTPUT_PATH)
    plt.show()


if __name__ == "__main__":
    main()