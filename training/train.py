import csv
from pathlib import Path

from agent.agent import Agent
from environment.dino_env import DinoEnv
from training.config import Config

MODEL_PATH = "models/dino_dqn_best.pt"
FINAL_MODEL_PATH = "models/dino_dqn_final.pt"
LOG_PATH = "logs/training_log.csv"


def train(config: Config) -> None:
    env = DinoEnv()
    agent = Agent(config)

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    scores: list[int] = []
    best_average = float("-inf")
    total_steps = 0

    with open(LOG_PATH, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["episode", "score", "total_reward", "epsilon", "avg_loss"])

        for episode in range(1, config.episodes + 1):
            state, _ = env.reset()
            total_reward = 0.0
            losses: list[float] = []

            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                agent.memory.push(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                total_steps += 1

                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

                agent.decay_epsilon()
                if total_steps % config.target_update_every == 0:
                    agent.update_target()

                if terminated or truncated:
                    break

            score = info["score"]
            scores.append(score)
            average_loss = sum(losses) / len(losses) if losses else 0.0
            writer.writerow(
                [episode, score, round(total_reward, 2), round(agent.epsilon, 4), round(average_loss, 5)]
            )
            log_file.flush()

            recent_average = sum(scores[-20:]) / len(scores[-20:])
            if len(scores) >= 20 and recent_average > best_average:
                best_average = recent_average
                agent.save(MODEL_PATH)

            if episode % 10 == 0:
                print(
                    f"Episode {episode:4d} | last-10 avg score: "
                    f"{sum(scores[-10:]) / 10:7.1f} | epsilon: {agent.epsilon:.3f} "
                    f"| loss: {average_loss:.4f}"
                )

    agent.save(FINAL_MODEL_PATH)
    print(f"Done. Best 20-episode average: {best_average:.1f}")


if __name__ == "__main__":
    train(Config())