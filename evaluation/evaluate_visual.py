from agent.agent import Agent
from environment.dino_visual_env import DinoVisualEnv
from evaluation.evaluate import play
from training.config import VisualConfig

MODEL_PATH = "models/dino_cnn_best.pt"
EPISODES = 10


def main() -> None:
    env = DinoVisualEnv()

    untrained = Agent(VisualConfig())     # random weights
    trained = Agent(VisualConfig())
    trained.load(MODEL_PATH)              # weights learned in training

    untrained_scores = play(untrained, env, EPISODES)
    trained_scores = play(trained, env, EPISODES)

    print(f"Untrained CNN: average {sum(untrained_scores) / EPISODES:.1f}, "
          f"best {max(untrained_scores)}")
    print(f"Trained CNN:   average {sum(trained_scores) / EPISODES:.1f}, "
          f"best {max(trained_scores)}")
    env.close()


if __name__ == "__main__":
    main()