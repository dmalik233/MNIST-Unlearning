import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

batch_size = 1000

def predict(model, dataset, device):
    model.eval()
    preds, labels = [], []

    for x, y in DataLoader(dataset, batch_size=batch_size):
        with torch.no_grad():
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()

def get_accuracy(model, dataset, device):
    preds, labels = predict(model, dataset, device)
    return accuracy_score(labels, preds)