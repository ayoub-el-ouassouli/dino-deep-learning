from dataclasses import dataclass

from environment.game import STATE_SIZE


@dataclass
class Config:
    state_size: int = STATE_SIZE
    action_size: int = 2
    hidden_size: int = 64

    learning_rate: float = 1e-3     # how big each weight adjustment is
    gamma: float = 0.99             # discount factor from the Bellman equation
    batch_size: int = 64
    memory_capacity: int = 50_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995   # multiplied in at every step

    target_update_every: int = 500  # steps between target network copies
    episodes: int = 300