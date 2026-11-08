import csv
import time
from pathlib import Path

from agent.agent import Agent
from environment.dino_env import DinoEnv
from training.config import Config


def train(config: Config, env, name: str) -> None:
    """Train an agent on the given environment. Files are named after `name`."""
    agent = Agent(config)

    model_path = f"models/{name}_best.pt"
    final_path = f"models/{name}_final.pt"
    log_path = f"logs/{name}_log.csv"

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    scores: list[int] = []
    best_average = float("-inf")
    total_steps = 0
    start_time = time.perf_counter()

    with open(log_path, "w", newline="") as log_file:
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

                if (
                    total_steps >= config.learning_starts
                    and total_steps % config.learn_every == 0
                ):
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
                agent.save(model_path)

            if episode % 10 == 0:
                minutes = (time.perf_counter() - start_time) / 60
                print(
                    f"Episode {episode:4d} | steps {total_steps:7d} | "
                    f"last-10 avg score: {sum(scores[-10:]) / 10:7.1f} | "
                    f"epsilon: {agent.epsilon:.3f} | loss: {average_loss:.4f} | "
                    f"{minutes:.1f} min"
                )

    agent.save(final_path)
    print(f"Done. Best 20-episode average: {best_average:.1f}")


if __name__ == "__main__":
    train(Config(), DinoEnv(), "dino_dqn")