import random
import numpy as np
import torch

from model import CNN

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return torch.device("mps" if torch.mps.is_available() else "cpu")

def load_model(path, device):
    model = CNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()
