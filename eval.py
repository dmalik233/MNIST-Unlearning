import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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

def get_probs(model, dataset, device):
    model.eval()
    probs = []

    for x, _ in DataLoader(dataset, batch_size=batch_size):
        with torch.no_grad():
            probs.append(F.softmax(model(x.to(device)), dim=1).cpu())

    return torch.cat(probs)

def get_features(model, dataset, device):
    model.eval()
    features, labels = [], []

    for x, y in DataLoader(dataset, batch_size=batch_size):
        with torch.no_grad():
            features.append(model.features(x.to(device)).cpu())
            labels.append(y)

    return torch.cat(features).numpy(), torch.cat(labels).numpy()

def pred_distribution(model, dataset, device):
    preds, _ = predict(model, dataset, device)
    return np.bincount(preds, minlength=10)/len(preds)

def js_div(p, q, eps=1e-8):

    log_m = (0.5*(p+q)).clamp(min=eps).log()
    kl_pm = F.kl_div(log_m, p, reduction="batchmean")
    kl_qm = F.kl_div(log_m, q, reduction="batchmean")

    return max(0.0, (0.5*(kl_pm + kl_qm)).item())

def save_confusion_matrix(model, dataset, device, title, path):
    preds, labels = predict(model, dataset, device)
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap="Blues")
    plt.title(title)
    plt.savefig(path)
    plt.close()