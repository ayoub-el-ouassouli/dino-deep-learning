import random

from agent.agent import Agent
from environment.dino_env import DinoEnv
from training.config import Config

agent = Agent(Config())
env = DinoEnv()

# Record the first 64 transitions of an episode (no crash this early)
state, _ = env.reset()
for _ in range(64):
    action = random.choice([0, 1])
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.memory.push(state, action, reward, next_state, terminated)
    state = next_state

print("Memory size:", len(agent.memory))

# batch_size is 64, so every step trains on these same 64 transitions
for step in range(1, 2001):
    loss = agent.learn()
    if step in (1, 100, 500, 1000, 2000):
        print(f"Step {step}: loss = {loss:.6f}")