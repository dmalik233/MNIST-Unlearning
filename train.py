import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data import load_mnist, splits
from model import CNN
from eval import predict, get_accuracy
from utils import set_seed, get_device

num_epochs = 5
lr = 1e-3
batch_size = 64

def train_model(train_set, num_epochs=num_epochs, lr=lr, batch_size=batch_size, device=None):
    model = CNN().to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} | loss = {epoch_loss/len(train_set):.4f}")
    return model

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"Using device {device}\n")

    train, test = load_mnist()
    _, _, test_retain, test_forget = splits()

    model = train_model(train, num_epochs=5, device=device)

    print("\n--Test accuracy--")
    print(f"Retain digits: {get_accuracy(model, test_retain, device):.4f}")
    print(f"Digit 7: {get_accuracy(model, test_forget, device):.4f}")
    print(f"All digits: {get_accuracy(model, test, device):.4f}\n")

    torch.save(model.state_dict(), "checkpoints/original_model.pt")
    print("Original model checkpoint saved.")