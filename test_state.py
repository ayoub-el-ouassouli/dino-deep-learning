from environment.game import DinoGame

game = DinoGame()
for frame in range(1, 121):
    game.step(0)
    if frame % 20 == 0:
        print(f"Frame {frame}: {game.get_state()}")