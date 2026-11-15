import time

import cv2

from agent.agent import Agent
from automation.browser_controller import BrowserDino, active_window_title
from training.config import VisualConfig

# The numbers you measured with select_region
LEFT, TOP = 551, 456
WIDTH, HEIGHT = 810, 227

MODEL_PATH = "models/dino_cnn_best.pt"
WINDOW_KEYWORD = "Brave"   # must appear in your browser window's title
DURATION = 60              # maximum seconds of play
START_TIMEOUT = 10         # seconds to wait for the game to start
MAX_SAVED_JUMPS = 5        # save a screenshot for the first few jump decisions


def save(path: str, raw) -> None:
    cv2.imwrite(path, cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))


def main() -> None:
    agent = Agent(VisualConfig())
    agent.load(MODEL_PATH)
    dino = BrowserDino(LEFT, TOP, WIDTH, HEIGHT, WINDOW_KEYWORD)

    print("In 5 seconds the AGENT will press Space to start the game, then play.")
    print("Click on the game PAGE now (not the address bar) and don't touch the keyboard.")
    print("To abort: move the mouse into a screen corner, or press Ctrl+C.")
    time.sleep(5)

    print("Active window:", repr(active_window_title()))
    if not dino.press_space():           # this press also starts the game
        print(f"The active window does not contain '{WINDOW_KEYWORD}'. No keys were pressed.")
        print("Set WINDOW_KEYWORD to a word from the title printed above.")
        return

    wait_start = time.perf_counter()
    while not dino.seen_motion:
        dino.observe()
        if time.perf_counter() - wait_start > START_TIMEOUT:
            print("The game never started. Is it visible inside the selected area?")
            return

    print("Game running. The agent is playing...")
    start = time.perf_counter()
    steps = 0
    jump_times: list[float] = []

    while time.perf_counter() - start < DURATION:
        state = dino.observe()
        action = agent.select_action(state, explore=False)
        steps += 1
        if action == 1:
            if not dino.press_space():
                print("The browser lost the keyboard focus. Stopping.")
                break
            elapsed_now = time.perf_counter() - start
            jump_times.append(elapsed_now)
            print(f"  t={elapsed_now:5.2f}s  step {steps}: JUMP")
            if len(jump_times) <= MAX_SAVED_JUMPS:
                save(f"results/real_jump_{len(jump_times)}.png", dino.previous_raw)
        if dino.game_over:
            print("Game over detected.")
            break

    elapsed = time.perf_counter() - start
    save("results/real_last_frame.png", dino.previous_raw)
    print(f"Survived about {elapsed:.1f} s, {steps} frames ({steps / elapsed:.1f} per second)")
    print(f"Jumps: {len(jump_times)}")


if __name__ == "__main__":
    main()