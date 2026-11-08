from dataclasses import dataclass

from environment.game import STATE_SIZE


@dataclass
class Config:
    state_size: int = STATE_SIZE
    action_size: int = 2
    hidden_size: int = 64

    visual: bool = False
    input_shape: tuple = (4, 60, 160)   # only used when visual=True

    learning_rate: float = 1e-3     # how big each weight adjustment is
    gamma: float = 0.99             # discount factor from the Bellman equation
    batch_size: int = 64
    memory_capacity: int = 50_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995   # multiplied in at every step

    target_update_every: int = 500  # steps between target network copies
    learn_every: int = 1            # learn once per this many steps
    learning_starts: int = 64       # don't learn before this many memories
    episodes: int = 300


@dataclass
class VisualConfig(Config):
    """Settings for the CNN agent that learns from screenshots."""

    visual: bool = True
    learning_rate: float = 2.5e-4
    batch_size: int = 32
    memory_capacity: int = 10_000
    epsilon_decay: float = 0.9999   # slower decay: floor after ~30,000 steps
    target_update_every: int = 1000
    learn_every: int = 4
    learning_starts: int = 1000
    episodes: int = 300