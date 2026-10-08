import torch
from torch.utils.data import DataLoader

batch_size = 1000

def predict(model, dataset, device):
    model.eval()
    preds, labels = [], []

    for x, y in DataLoader(dataset, batch_size=batch_size):
        with torch.no_grad():
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()
    