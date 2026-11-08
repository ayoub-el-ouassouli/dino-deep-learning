import numpy as np
import pygame

from environment.game import HEIGHT, WIDTH, DinoGame
from environment.physics import GROUND_Y

BACKGROUND = (247, 247, 247)
FOREGROUND = (83, 83, 83)


class Renderer:
    """Draws a DinoGame. headless=True draws on an invisible surface (no window)."""

    def __init__(self, headless: bool = False) -> None:
        pygame.init()
        self.headless = headless
        if headless:
            self.screen = pygame.Surface((WIDTH, HEIGHT))
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Chrome Dino AI")
        self.font = pygame.font.SysFont("consolas", 20)

    def draw(self, game: DinoGame) -> None:
        self.screen.fill(BACKGROUND)
        pygame.draw.line(self.screen, FOREGROUND, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

        dino = game.dino
        pygame.draw.rect(
            self.screen, FOREGROUND, pygame.Rect(dino.x, dino.y, dino.width, dino.height)
        )
        for o in game.obstacles:
            pygame.draw.rect(
                self.screen, FOREGROUND, pygame.Rect(o.x, o.y, o.width, o.height)
            )

        if not self.headless:
            text = self.font.render(f"Score: {int(game.score)}", True, FOREGROUND)
            self.screen.blit(text, (WIDTH - 160, 10))
            pygame.display.flip()

    def get_frame(self) -> np.ndarray:
        """The current screen as an RGB array of shape (height, width, 3)."""
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self) -> None:
        pygame.quit()