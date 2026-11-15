import pygame

from agent.agent import Agent
from environment.dino_visual_env import DinoVisualEnv
from environment.renderer import Renderer
from training.config import VisualConfig

MODEL_PATH = "models/dino_cnn_best.pt"
FPS = 60


def main() -> None:
    agent = Agent(VisualConfig())
    agent.load(MODEL_PATH)

    env = DinoVisualEnv()          # the agent sees this (headless pixels)
    window = Renderer()            # this one is only for us, to watch
    clock = pygame.time.Clock()

    state, _ = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.select_action(state, explore=False)
        state, _, terminated, truncated, info = env.step(action)
        window.draw(env.game)

        if terminated or truncated:
            print("Episode ended, score:", info["score"])
            state, _ = env.reset()
        clock.tick(FPS)

    window.close()


if __name__ == "__main__":
    main()