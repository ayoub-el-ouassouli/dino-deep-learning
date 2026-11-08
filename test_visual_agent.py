import time

from agent.agent import Agent
from environment.dino_visual_env import DinoVisualEnv
from training.config import VisualConfig

config = VisualConfig(memory_capacity=2000)
agent = Agent(config)
env = DinoVisualEnv()

state, _ = env.reset()
for _ in range(500):
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.memory.push(state, action, reward, next_state, terminated)
    state = next_state
    if terminated or truncated:
        state, _ = env.reset()
print("Memory size:", len(agent.memory))

start = time.perf_counter()
for _ in range(20):
    loss = agent.learn()
elapsed = time.perf_counter() - start
print(f"Loss after 20 learning steps: {loss:.4f}")
print(f"{elapsed / 20 * 1000:.0f} ms per learning step")
env.close()