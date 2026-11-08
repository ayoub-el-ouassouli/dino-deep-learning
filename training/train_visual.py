from environment.dino_visual_env import DinoVisualEnv
from training.config import VisualConfig
from training.train import train

if __name__ == "__main__":
    env = DinoVisualEnv()
    try:
        train(VisualConfig(), env, "dino_cnn")
    finally:
        env.close()