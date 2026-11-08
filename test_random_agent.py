import random

from environment.game import DinoGame

game = DinoGame()
scores: list[int] = []

for episode in range(20):
    game.reset()
    while True:
        action = random.choice([0, 1])   # pick jump or nothing at random
        if game.step(action):            # True means we crashed
            break
    scores.append(int(game.score))
    print(f"Episode {episode + 1}: score {scores[-1]}")

print("Average score:", sum(scores) / len(scores))