import torch

from agent.cnn_dqn import CNNDQN
from agent.dqn import DQN
from environment.dino_visual_env import DinoVisualEnv

model = CNNDQN(input_shape=(4, 60, 160), action_size=2)

# Follow one input through every layer
x = torch.zeros(1, 4, 60, 160)
for layer in model.features:
    x = layer(x)
    print(f"{layer.__class__.__name__:8s} -> {tuple(x.shape)}")

cnn_params = sum(p.numel() for p in model.parameters())
mlp_params = sum(p.numel() for p in DQN(4, 2).parameters())
print("CNN parameters:       ", cnn_params)
print("Stage 1 MLP parameters:", mlp_params)

# Feed a real observation from the visual environment
env = DinoVisualEnv()
obs, _ = env.reset()
q_values = model(torch.from_numpy(obs).unsqueeze(0))
print("Q-values shape:", tuple(q_values.shape))
print("Q-values:", q_values)
env.close()