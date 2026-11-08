import sys

import numpy as np
import torch

print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

# A tiny tensor calculation to prove PyTorch really works
x = torch.tensor([1.0, 2.0, 3.0])
print("x * 2 =", x * 2)