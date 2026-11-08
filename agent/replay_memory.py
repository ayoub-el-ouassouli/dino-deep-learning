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
def _to_uint8(frames: np.ndarray) -> np.ndarray:
    """Convert pixel values from 0..1 decimals to 0..255 whole numbers."""
    return np.rint(frames * 255).astype(np.uint8)


class ImageReplayMemory:
    """Replay memory for image observations, stored as uint8 to save RAM."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done) -> None:
        """Save one transition (same signature as ReplayMemory.push)."""
        self.states[self.position] = _to_uint8(state)
        self.next_states[self.position] = _to_uint8(next_state)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Return a random batch as PyTorch tensors (same format as ReplayMemory)."""
        indices = random.sample(range(self.size), batch_size)
        return (
            torch.from_numpy(self.states[indices].astype(np.float32) / 255.0),
            torch.from_numpy(self.actions[indices]).unsqueeze(1),
            torch.from_numpy(self.rewards[indices]).unsqueeze(1),
            torch.from_numpy(self.next_states[indices].astype(np.float32) / 255.0),
            torch.from_numpy(self.dones[indices]).unsqueeze(1),
        )

    def __len__(self) -> int:
        return self.size