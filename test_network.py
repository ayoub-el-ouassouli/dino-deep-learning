import torch

from agent.dqn import DQN
from environment.game import STATE_SIZE, DinoGame

game = DinoGame()
model = DQN(state_size=STATE_SIZE, action_size=2)

state = torch.from_numpy(game.get_state())
q_values = model(state)

print("Q-values:", q_values)
print("Best action:", int(torch.argmax(q_values)))