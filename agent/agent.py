import random

import numpy as np
import torch
import torch.nn.functional as F

from agent.dqn import DQN
from agent.replay_memory import ReplayMemory
from training.config import Config


class Agent:
    """Chooses actions, owns the networks and the memory, and learns."""

    def __init__(self, config: Config) -> None:
        self.config = config

        self.policy_net = DQN(config.state_size, config.action_size, config.hidden_size)
        self.target_net = DQN(config.state_size, config.action_size, config.hidden_size)
        self.update_target()
        self.target_net.eval()   # it is never trained directly

        self.memory = ReplayMemory(config.memory_capacity)
        self.epsilon = config.epsilon_start

        # NEW: the optimizer adjusts the policy network's weights
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )

    def update_target(self) -> None:
        """Copy the policy network's weights into the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Epsilon-greedy: random with probability epsilon, else the best Q-value."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.config.action_size)

        with torch.no_grad():
            q_values = self.policy_net(torch.from_numpy(state))
        return int(torch.argmax(q_values).item())

    def decay_epsilon(self) -> None:
        """Shrink epsilon a little, never below its minimum."""
        self.epsilon = max(
            self.config.epsilon_end, self.epsilon * self.config.epsilon_decay
        )

    # NEW: one learning step
    def learn(self) -> float | None:
        """Train on one random batch. Returns the loss, or None if memory is too small."""
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )

        # 1. Prediction: Q-value of the action that was actually taken
        predicted_q = self.policy_net(states).gather(1, actions)

        # 2. Target: Bellman equation, computed by the frozen target network
        with torch.no_grad():
            best_next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + self.config.gamma * best_next_q * (1 - dones)

        # 3. Loss: how far the prediction is from the target
        loss = F.smooth_l1_loss(predicted_q, target_q)

        # 4. Backpropagation and weight update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())


    def save(self, path: str) -> None:
        """Save the policy network's weights to a file."""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load weights from a file into both networks."""
        weights = torch.load(path, map_location="cpu", weights_only=True)
        self.policy_net.load_state_dict(weights)
        self.update_target()    