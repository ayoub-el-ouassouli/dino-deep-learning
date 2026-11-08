import torch
from torch import nn


class CNNDQN(nn.Module):
    """Convolutional Q-network: stacked frames in, one Q-value per action out."""

    def __init__(self, input_shape: tuple[int, int, int], action_size: int) -> None:
        super().__init__()
        channels = input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Find out how many numbers come out of the conv part by trying a blank image
        with torch.no_grad():
            flat_size = self.features(torch.zeros(1, *input_shape)).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))