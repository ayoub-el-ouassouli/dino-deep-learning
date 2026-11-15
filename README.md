# Chrome Dino AI

An AI that learns to play a Chrome-Dino-style runner game **from scratch** using **Deep Reinforcement Learning**.

The project is built in two stages, entirely with a hand-written **DQN (Deep Q-Network)** in PyTorch (no ready-made RL library):

| Stage | What the AI sees | Network | Result |
|---|---|---|---|
| 1. Numeric agent | 4 numbers (distance to the next cactus, cactus width, speed, "is the dino in the air") | MLP (small fully connected network) | Averages ~2140 frames survived vs ~169 for an untrained agent |
| 2. Visual agent | The **pixels** of the screen (last 4 frames, grayscale, 160x60) | CNN (convolutional network) | Survives the 5000-frame limit in 10/10 test games vs ~169 untrained |

A purely random player survives about **177 frames** on average, so that is the baseline to beat.

> **Status of the "real Chrome game" part:** the code to capture the real game from the screen and press Space is included, but the agent trained on our simulated game does **not** yet play the real game well (see [Real Chrome Dino](#real-chrome-dino-experimental)). This is a learning project and an honest work in progress.

This is an independent educational project. It is not affiliated with Google or Chrome.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Project structure (what each file does)](#project-structure-what-each-file-does)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Play the game yourself](#1-play-the-game-yourself)
   - [Train the numeric agent](#2-train-the-numeric-agent-stage-1)
   - [Train the visual agent](#3-train-the-visual-agent-stage-2)
   - [Evaluate a trained agent](#4-evaluate-a-trained-agent)
   - [Plot the learning curve](#5-plot-the-learning-curve)
   - [Watch the AI play](#6-watch-the-ai-play)
   - [Sanity-check scripts](#7-sanity-check-scripts)
5. [Real Chrome Dino (experimental)](#real-chrome-dino-experimental)
6. [Hyperparameters](#hyperparameters)
7. [Troubleshooting](#troubleshooting)
8. [Roadmap](#roadmap)

---

## How it works

Reinforcement learning is a loop between an **agent** (the AI) and an **environment** (the game):

```
   ┌──────────────┐   action (0 = do nothing, 1 = jump)   ┌──────────────┐
   │    AGENT     │ ────────────────────────────────────► │ ENVIRONMENT  │
   │ (neural net) │ ◄──────────────────────────────────── │ (Dino game)  │
   └──────────────┘      state (or pixels) + reward       └──────────────┘
```

- **State:** what the agent can see. Stage 1: 4 numbers. Stage 2: the last 4 screenshots stacked together (so the network can see motion).
- **Action:** `0` = keep running, `1` = jump.
- **Reward:** `+0.1` for every frame survived, `-10` for crashing into a cactus.
- **Episode:** one attempt, from the start until the crash (or until the 5000-frame limit).
- **Goal:** collect as much total reward as possible, which means surviving as long as possible.

### The DQN algorithm (implemented by hand)

The network outputs two numbers, `Q(do nothing)` and `Q(jump)`: how much future reward the agent expects from each action. It picks the larger one. Learning uses:

- **Bellman equation:** target = `reward + gamma * (best Q-value of the next state)`; the network is trained to match this target.
- **Experience replay:** past transitions are stored in a memory, and the network trains on random batches of them (this breaks the correlation between consecutive frames).
- **Target network:** a frozen copy of the network used to compute the targets, refreshed every N steps (this keeps training stable).
- **Epsilon-greedy exploration:** with probability epsilon the agent acts randomly; epsilon shrinks from 1.0 to 0.05 during training.
- **Huber loss + gradient clipping** for stable updates, and the **Adam** optimizer.

### The networks

- **MLP (Stage 1):** `4 -> 64 -> 64 -> 2` (4,610 weights).
- **CNN (Stage 2):** input `(4, 60, 160)` -> Conv 16 filters 8x8 stride 4 -> Conv 32 filters 4x4 stride 2 -> Conv 32 filters 3x3 stride 1 -> Flatten (2048) -> Linear 256 -> Linear 2 (546,642 weights). Strided convolutions are used instead of pooling because the exact position of the cactus matters.

### Image preprocessing (Stage 2)

Screenshot -> grayscale -> resize to 160x60 -> divide by 255 (values 0 to 1) -> stack the last 4 frames.

---

## Project structure (what each file does)

```
Chrome-Dino-AI/
├── main.py
├── watch_agent.py
├── requirements.txt
├── environment/
│   ├── physics.py
│   ├── game.py
│   ├── renderer.py
│   ├── dino_env.py
│   └── dino_visual_env.py
├── agent/
│   ├── dqn.py
│   ├── cnn_dqn.py
│   ├── replay_memory.py
│   └── agent.py
├── training/
│   ├── config.py
│   ├── train.py
│   └── train_visual.py
├── evaluation/
│   ├── evaluate.py
│   ├── evaluate_visual.py
│   └── plot_training.py
├── vision/
│   ├── image_processing.py
│   └── extract_sprites.py
├── automation/
│   ├── browser_controller.py
│   ├── run_real.py
│   ├── select_region.py
│   ├── find_region.py
│   ├── capture_test.py
│   └── diagnose_capture.py
├── models/      (saved neural networks, .pt files)
├── logs/        (training logs, .csv files)
├── results/     (graphs and pictures)
├── assets/      (experimental sprites)
└── check_setup.py, test_*.py   (small sanity-check scripts)
```

### Root

| File | What it does |
|---|---|
| `main.py` | **Play the game yourself** with the keyboard (Space = jump). Shows your score. |
| `watch_agent.py` | **Watch the trained CNN agent play** the simulated game in a window. |
| `requirements.txt` | The Python libraries to install. |

### `environment/`: the game

| File | What it does |
|---|---|
| `physics.py` | The rules of the world, with no drawing: the `Dino` (position, gravity, jumping), the `Obstacle` (cactus moving left) and the collision test `collides()`. |
| `game.py` | `DinoGame`: the complete game logic, with no window. It has `reset()` (start a new game), `step(action)` (advance one frame), and `get_state()` (the 4 numbers the Stage 1 agent sees). It also handles cactus spawning, the score, and the speed that increases over time. |
| `renderer.py` | `Renderer`: draws a `DinoGame` with Pygame. It can run **headless** (draws on an invisible surface, no window) and gives the screen pixels as an array via `get_frame()`. |
| `dino_env.py` | `DinoEnv`: wraps the game in the standard reinforcement-learning interface (`reset()` / `step()` returning observation, reward, terminated, truncated, info). Observation = the 4 numbers. Defines the rewards. |
| `dino_visual_env.py` | `DinoVisualEnv`: the same, but the observation is the **last 4 screenshots** (shape `4 x 60 x 160`). |

### `agent/`: the AI

| File | What it does |
|---|---|
| `dqn.py` | `DQN`: the small fully connected network used in Stage 1. |
| `cnn_dqn.py` | `CNNDQN`: the convolutional network used in Stage 2. |
| `replay_memory.py` | `ReplayMemory` (Stage 1) and `ImageReplayMemory` (Stage 2, stores frames as small integers to save RAM): the agent's memory of past experiences. |
| `agent.py` | `Agent`: owns the networks, the memory and the optimizer. Chooses actions (`select_action`), learns from a batch (`learn`: prediction, Bellman target, loss, backpropagation), updates the target network, decays epsilon, and saves/loads weights. |

### `training/`: learning

| File | What it does |
|---|---|
| `config.py` | All the tunable numbers (`Config` for the numeric agent, `VisualConfig` for the CNN agent). |
| `train.py` | The training loop (play, remember, learn). Running it trains the **numeric** agent. |
| `train_visual.py` | Trains the **visual (CNN)** agent with the same loop. |

### `evaluation/`: measuring and plotting

| File | What it does |
|---|---|
| `evaluate.py` | Plays games with an untrained and a trained numeric agent (no random exploration) and prints the scores. |
| `evaluate_visual.py` | The same for the CNN agent. |
| `plot_training.py` | Draws the learning curve from a training log and saves it in `results/`. |

### `vision/`: image processing

| File | What it does |
|---|---|
| `image_processing.py` | `preprocess()`: color screenshot -> grayscale -> 160x60 -> values from 0 to 1. Used by the visual environment and by the real-game controller. |
| `extract_sprites.py` | Experimental tool that cuts the dinosaur and cactus sprites out of a screenshot (to make the simulation look like the real game). |

### `automation/`: the real browser game (experimental)

| File | What it does |
|---|---|
| `browser_controller.py` | `BrowserDino`: takes screenshots of a screen region, builds the 4-frame stack, detects "game over" (the screen stops changing), and presses Space only if the browser window has the keyboard focus. |
| `run_real.py` | Lets the trained agent play the real game in your browser. |
| `select_region.py` | Take a screenshot and **drag a rectangle** around the game to get its screen coordinates. |
| `find_region.py` | Older helper that prints mouse coordinates (`select_region.py` is easier). |
| `capture_test.py` | Captures the game area once and saves the raw and processed pictures in `results/`. |
| `diagnose_capture.py` | Measures how fast screenshots can be taken and whether the screen is changing. |

### Sanity-check scripts (root)

Small scripts written step by step while building the project. Each checks one piece. See [Sanity-check scripts](#7-sanity-check-scripts).

---

## Installation

You need **Python 3.11** and **git**. The commands below are for **Windows PowerShell**.

```powershell
git clone https://github.com/<your-username>/Chrome-Dino-AI.git
cd Chrome-Dino-AI

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS / Linux, activate the environment with `source .venv/bin/activate` instead.

Check that everything works:

```powershell
python check_setup.py
```

It prints the Python, NumPy and PyTorch versions and a small tensor calculation. The project runs on the **CPU**; no GPU is needed.

> Always run the commands from the project's root folder, and run the modules with `python -m folder.file` as shown below.

---

## Usage

### 1. Play the game yourself

```powershell
python main.py
```

A window opens. Press **Space** to jump over the cacti. The score is shown at the top right, the game speeds up over time, and it restarts when you crash.

### 2. Train the numeric agent (Stage 1)

```powershell
python -m training.train
```

- No window opens; the progress is printed every 10 episodes (average score, epsilon, loss).
- Stop it any time with `Ctrl+C`. A "best" model is saved once 20 episodes are done, whenever the average of the last 20 episodes improves.
- Output: `models/dino_dqn_best.pt`, `models/dino_dqn_final.pt`, `logs/dino_dqn_log.csv`.
- Early scores are close to the random baseline (~170). Scores rise as epsilon falls.

> Running a training command again **starts from zero and overwrites** the model files of the same name. Copy a good model to another name first.

### 3. Train the visual agent (Stage 2)

```powershell
python -m training.train_visual
```

- The agent only receives pixels (no distances).
- 300 episodes took about 13.5 minutes on a laptop CPU in the author's run. The game is drawn off-screen, so no window appears.
- Output: `models/dino_cnn_best.pt`, `models/dino_cnn_final.pt`, `logs/dino_cnn_log.csv`.
- Scores are noisy and drop from time to time (typical of DQN). That is why the **best** model (by the 20-episode average) is saved, and the final one can be worse.

### 4. Evaluate a trained agent

```powershell
python -m evaluation.evaluate          # numeric agent: untrained vs trained
python -m evaluation.evaluate_visual   # CNN agent: untrained vs trained
```

These play full games with **no random exploration** and print the average and best score. Example results from the author's runs:

```
Untrained agent: average 169.0, best 169
Trained agent:   average 2140.4, best 2358     (numeric agent)

Untrained CNN: average 169.0, best 169
Trained CNN:   average 5000.0, best 5000       (CNN agent)
```

Note that a game is **capped at 5000 frames**, so "5000" means "survived to the limit", and the real ceiling is higher. The evaluation only tests this specific simulated game (one cactus size, fixed physics).

### 5. Plot the learning curve

Open `evaluation/plot_training.py` and set the two constants at the top:

```python
# numeric agent
LOG_PATH = "logs/dino_dqn_log.csv"
OUTPUT_PATH = "results/learning_curve.png"

# CNN agent
LOG_PATH = "logs/dino_cnn_log.csv"
OUTPUT_PATH = "results/learning_curve_cnn.png"
```

Then run:

```powershell
python -m evaluation.plot_training
```

A window opens and the picture is saved in `results/`. It shows the score of each episode (thin line), the 10-episode moving average (thick line), and the random-agent baseline (dashed line).

### 6. Watch the AI play

After training the CNN agent (or after copying a `models/dino_cnn_best.pt` file into `models/`):

```powershell
python watch_agent.py
```

A window shows the simulated game, played by the trained agent. The agent decides from the pixels of an invisible copy of the game; the window is only for you to watch. Close it with the **X**. Every time an episode ends, the score is printed in the terminal (5000 = reached the frame limit).

### 7. Sanity-check scripts

Run from the root with `python <file>.py`. They are not needed to use the project, but they show how each part works.

| Script | Checks |
|---|---|
| `check_setup.py` | Python, NumPy and PyTorch are installed correctly. |
| `test_random_agent.py` | Score of a purely random player (the ~177 baseline). |
| `test_state.py` | The 4-number state looks correct as the game advances. |
| `test_network.py` | The MLP gives two Q-values for a state. |
| `test_env.py` | The `DinoEnv` interface and the rewards. |
| `test_memory.py` | The replay memory stores and samples transitions. |
| `test_agent.py` | Epsilon decay, target-network copy and action selection. |
| `test_learning.py` | The learning step: the loss falls when training on a small fixed batch. |
| `test_vision.py` | Screenshot capture and preprocessing; saves pictures in `results/`. |
| `test_visual_env.py` | The visual environment (4 stacked frames) and its speed. |
| `test_cnn.py` | The CNN's layer shapes and number of parameters. |
| `test_visual_agent.py` | The CNN agent can learn from the image memory (and the time per learning step). |

---

## Real Chrome Dino (experimental)

**What exists:** the tools to point the agent at the real game running in your browser: screen capture of the game area, the same preprocessing, a safe Space key press (only when the browser window has the focus), and game-over detection by the screen no longer changing.

**What does not work yet:** the agent trained in our simulation **does not play the real game**. In the first live test it never jumped and crashed into the first cactus. This is a **domain gap** (sim-to-real gap): the network learned plain gray boxes on flat ground, and the real game has a pixel-art dinosaur, cacti of different shapes and groups, clouds, textured ground and score text. Also, screenshots arrive at ~35 per second while the simulation ran at 60, so the motion between frames looks different.

### Try it yourself

1. Open the game in your browser (`chrome://dino` or `brave://dino` offline, or any Dino web game). Don't use a site with a shared leaderboard.
2. Find the game area:

```powershell
python -m automation.select_region
```

   Drag a rectangle around the dinosaur, the ground and the score, and press ENTER. It prints `LEFT, TOP` and `WIDTH, HEIGHT` and saves `results/selected_region.png`; check that this picture shows only the game.
3. Put the numbers in the constants at the top of `automation/run_real.py` (`LEFT, TOP`, `WIDTH, HEIGHT`) and set `WINDOW_KEYWORD` to a word from your browser window's title (for example `"Chrome"` or `"Brave"`).
4. Run it, then click on the game page during the 5-second countdown and don't touch the keyboard:

```powershell
python -m automation.run_real
```

The agent presses Space to start the game and plays until it crashes. It saves `results/real_last_frame.png`.

**Safety:** PyAutoGUI has a fail-safe: move the mouse into a corner of the screen to abort. The script only presses Space while the browser window is focused. Please don't use it to post bot scores on online leaderboards.

---

## Hyperparameters

All values live in `training/config.py`.

| Setting | Numeric agent | Visual agent |
|---|---|---|
| Learning rate | 1e-3 | 2.5e-4 |
| Discount factor (gamma) | 0.99 | 0.99 |
| Batch size | 64 | 32 |
| Replay memory size | 50,000 | 10,000 |
| Epsilon (start -> end) | 1.0 -> 0.05 | 1.0 -> 0.05 |
| Epsilon decay per step | 0.9995 | 0.9999 |
| Target network update | every 500 steps | every 1000 steps |
| Learn once every | 1 step | 4 steps |
| Learning starts after | 64 steps | 1000 steps |
| Episodes | 300 | 300 |

Game settings (in `environment/`): window 800x300 at 60 FPS, starting speed 6 (growing by 0.002 per frame up to 14), a new cactus every 60 to 120 frames, episode limit 5000 frames, reward +0.1 per frame and -10 for a crash.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'environment'` (or `agent`, `training`...)**: run from the project's root folder and use the `python -m folder.file` form shown above.
- **PowerShell won't activate the virtual environment** ("running scripts is disabled"): run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and activate again.
- **`pip.exe` is blocked or "Application Control policy has blocked this file" (Windows)**: use `python -m pip install ...` instead of `pip install ...`. If a compiled library is blocked, try an older, more established version of it. In the author's setup NumPy 2.2.6 worked where 2.4.6 was blocked.
- **`FileNotFoundError: models/dino_cnn_best.pt`**: train first (`python -m training.train_visual`), or place a trained model file in `models/`.
- **The training scores drop or bounce around**: this is normal for DQN. Use the best saved model, and look at the moving average in the plot.
- **`ImportError` about OpenCV (`cv2`) or PyAutoGUI**: make sure `opencv-python` and `pyautogui` are installed (they are only needed for the visual agent and the real-game tools).

---

## Roadmap

- [ ] Draw the simulation with real sprites (dinosaur, cacti of several sizes and groups, clouds, textured ground, score) so it looks like the real game.
- [ ] Match the real game's speed and jump, and mimic the slower screenshot rate.
- [ ] Retrain the CNN on the realistic simulation and retest it on the real game.
- [ ] Fine-tune on the real game if a gap remains.
- [ ] Add birds / ducking, night mode, and a test with different game settings to measure generalization.
- [ ] Training monitoring with TensorBoard, and more hyperparameter experiments (Double DQN, other learning rates).

---

## License and credits

Add a license (for example MIT) before publishing if you want others to reuse the code.

The Chrome Dino game belongs to Google; this project only uses the idea (and, experimentally, screenshots of the public game) for learning.
