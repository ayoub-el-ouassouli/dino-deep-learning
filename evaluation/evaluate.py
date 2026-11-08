from agent.agent import Agent
from environment.dino_env import DinoEnv
from training.config import Config

MODEL_PATH = "models/dino_dqn_best.pt"
EPISODES = 20


def play(agent: Agent, env: DinoEnv, episodes: int) -> list[int]:
    """Play full episodes with no random exploration; return the scores."""
    scores: list[int] = []
    for _ in range(episodes):
        state, _ = env.reset()
        while True:
            action = agent.select_action(state, explore=False)
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        scores.append(info["score"])
    return scores


def main() -> None:
    env = DinoEnv()

    untrained = Agent(Config())          # random weights
    trained = Agent(Config())
    trained.load(MODEL_PATH)             # weights learned in training

    untrained_scores = play(untrained, env, EPISODES)
    trained_scores = play(trained, env, EPISODES)

    print(f"Untrained agent: average {sum(untrained_scores) / EPISODES:.1f}, "
          f"best {max(untrained_scores)}")
    print(f"Trained agent:   average {sum(trained_scores) / EPISODES:.1f}, "
          f"best {max(trained_scores)}")


if __name__ == "__main__":
    main()