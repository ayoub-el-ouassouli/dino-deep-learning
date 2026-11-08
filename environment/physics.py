from dataclasses import dataclass

GROUND_Y = 250        # y position of the ground line (pixels from the top)
GRAVITY = 0.8         # added to velocity every frame
JUMP_SPEED = -14.0    # negative = upwards


@dataclass
class Dino:
    """The dinosaur's position and jumping physics (no drawing here)."""

    x: float = 80.0
    width: int = 40
    height: int = 50
    y: float = GROUND_Y - 50   # y of the dino's TOP edge
    velocity_y: float = 0.0

    @property
    def floor_y(self) -> float:
        """The y value the dino's top edge has when standing on the ground."""
        return GROUND_Y - self.height

    @property
    def on_ground(self) -> bool:
        return self.y >= self.floor_y

    def jump(self) -> None:
        """Start a jump, but only if we are standing on the ground."""
        if self.on_ground:
            self.velocity_y = JUMP_SPEED

    def update(self) -> None:
        """Advance the physics by one frame."""
        self.velocity_y += GRAVITY
        self.y += self.velocity_y

        if self.y >= self.floor_y:      # landed (or never left the ground)
            self.y = self.floor_y
            self.velocity_y = 0.0


@dataclass
class Obstacle:
    """A cactus sliding toward the dinosaur (no drawing here)."""

    x: float
    width: int = 20
    height: int = 40

    @property
    def y(self) -> float:
        """y of the cactus's top edge (it stands on the ground)."""
        return GROUND_Y - self.height

    @property
    def is_off_screen(self) -> bool:
        return self.x + self.width < 0

    def update(self, speed: float) -> None:
        """Move one frame to the left at the given speed."""
        self.x -= speed


def collides(dino: Dino, obstacle: Obstacle) -> bool:
    """Return True if the dino and the obstacle rectangles overlap."""
    horizontal_overlap = (
        dino.x < obstacle.x + obstacle.width and dino.x + dino.width > obstacle.x
    )
    vertical_overlap = (
        dino.y < obstacle.y + obstacle.height and dino.y + dino.height > obstacle.y
    )
    return horizontal_overlap and vertical_overlap