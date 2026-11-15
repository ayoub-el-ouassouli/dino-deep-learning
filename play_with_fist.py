"""Play the Dino game by making a fist in front of the webcam."""

import cv2
import pygame

from environment.game import DinoGame
from environment.renderer import Renderer
from hand_control.webcam_controller import WebcamJumpController

FPS = 60


def main() -> None:
    game = DinoGame()
    renderer = Renderer()
    controller = WebcamJumpController()
    clock = pygame.time.Clock()

    print("Make a FIST to jump, then open your hand before the next jump.")
    print("Quit: close the game window, or press Q in the camera window.")
    running = True
    while running:
        action = 1 if controller.consume_jump() else 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1                                # the keyboard still works

        if game.step(action):
            print(f"Crash! Final score: {int(game.score)}")
            game.reset()
        renderer.draw(game)

        if controller.error:
            print("Webcam error:", controller.error)
            break
        if controller.preview is not None:
            cv2.imshow("Hand control (Q to quit)", controller.preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
        clock.tick(FPS)

    controller.close()
    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()