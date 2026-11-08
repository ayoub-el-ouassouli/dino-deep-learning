from environment.dino_env import DinoEnv

env = DinoEnv()

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0.0
    while True:
        action = env.action_space.sample()   # random action
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Episode {episode + 1}: score {info['score']}, total reward {total_reward:.1f}")