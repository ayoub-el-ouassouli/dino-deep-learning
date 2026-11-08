import random

import numpy as np
import torch


class ReplayMemory:
    """Stores past transitions and returns random batches of them."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: list[tuple] = []
        self.position = 0   # where the next transition will be written

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Save one transition."""
        transition = (state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition   # overwrite the oldest
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Return a random batch as PyTorch tensors."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.from_numpy(np.stack(next_states)),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.memory)