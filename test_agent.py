import torch

from agent.agent import Agent
from environment.dino_env import DinoEnv
from training.config import Config

agent = Agent(Config())
env = DinoEnv()
state, _ = env.reset()

same = all(
    torch.equal(p, t)
    for p, t in zip(agent.policy_net.parameters(), agent.target_net.parameters())
)
print("Target equals policy at start:", same)

for step in range(1, 6001):
    agent.decay_epsilon()
    if step % 1000 == 0:
        print(f"After {step} steps: epsilon = {agent.epsilon:.3f}")

agent.epsilon = 1.0
actions = [agent.select_action(state) for _ in range(1000)]
print("Share of jumps with epsilon = 1.0:", sum(actions) / len(actions))

agent.epsilon = 0.0
print("Greedy action (epsilon = 0.0):", agent.select_action(state))