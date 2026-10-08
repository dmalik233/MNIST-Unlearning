import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(42)

def get_device():
    return torch.device("mps" if torch.mps.is_available() else "cpu")
