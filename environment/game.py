import numpy as np
import random

from environment.physics import Dino, Obstacle, collides

WIDTH, HEIGHT = 800, 300

INITIAL_SPEED = 6.0
SPEED_INCREASE = 0.002   # speed gained every frame
MAX_SPEED = 14.0
STATE_SIZE = 4   # how many numbers the agent sees


class DinoGame:
    """The rules of the game. No drawing, no keyboard: just numbers."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start a fresh episode."""
        self.dino = Dino()
        self.obstacles: list[Obstacle] = []
        self.speed = INITIAL_SPEED
        self.score = 0.0
        self.frames_until_spawn = 60
        self.game_over = False

    def step(self, action: int) -> bool:
        """Advance one frame. action: 0 = do nothing, 1 = jump.

        Returns True if the dino crashed during this frame.
        """
        if action == 1:
            self.dino.jump()
        self.dino.update()

        self.frames_until_spawn -= 1
        if self.frames_until_spawn <= 0:
            self.obstacles.append(Obstacle(x=WIDTH))
            self.frames_until_spawn = random.randint(60, 120)

        for obstacle in self.obstacles:
            obstacle.update(self.speed)
        self.obstacles = [o for o in self.obstacles if not o.is_off_screen]

        self.speed = min(self.speed + SPEED_INCREASE, MAX_SPEED)
        self.score += 1

        self.game_over = any(collides(self.dino, o) for o in self.obstacles)
        return self.game_over

    def get_state(self) -> np.ndarray:
        """What the agent can see, as 4 numbers between 0 and 1."""
        dino = self.dino
        upcoming = [o for o in self.obstacles if o.x + o.width > dino.x]

        if upcoming:
            nearest = min(upcoming, key=lambda o: o.x)
            distance = max(nearest.x - (dino.x + dino.width), 0.0)
            width = nearest.width
        else:
            distance = WIDTH      # nothing ahead: treat as "very far"
            width = 0

        return np.array(
            [
                distance / WIDTH,
                width / 50,
                self.speed / MAX_SPEED,
                0.0 if dino.on_ground else 1.0,
            ],
            dtype=np.float32,
        )