import pygame

from environment.game import DinoGame
from environment.renderer import Renderer

FPS = 60


def main() -> None:
    game = DinoGame()
    renderer = Renderer()
    clock = pygame.time.Clock()

    running = True
    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1

        if game.step(action):
            print(f"Crash! Final score: {int(game.score)}")
            game.reset()

        renderer.draw(game)
        clock.tick(FPS)

    renderer.close()


if __name__ == "__main__":
    main()