import random

from agent.replay_memory import ReplayMemory
from environment.dino_env import DinoEnv

env = DinoEnv()
memory = ReplayMemory(capacity=1000)

state, _ = env.reset()
for _ in range(300):
    action = random.choice([0, 1])
    next_state, reward, terminated, truncated, _ = env.step(action)
    memory.push(state, action, reward, next_state, terminated)
    state = next_state
    if terminated or truncated:
        state, _ = env.reset()

print("Stored transitions:", len(memory))

states, actions, rewards, next_states, dones = memory.sample(batch_size=5)
print("states shape:     ", states.shape)
print("actions shape:    ", actions.shape)
print("rewards:          ", rewards.squeeze())
print("dones:            ", dones.squeeze())