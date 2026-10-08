import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data import load_mnist, splits
from model import CNN
from eval import predict
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
    for name, ds in [("retain digits", test_retain), ("7s", test_forget), ("all", test)]:
        preds, labels = predict(model, ds, device)

        print(f"{name}: {accuracy_score(labels, preds):.4f}")

    preds, labels = predict(model, test, device)
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    print(cm)

    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap="Blues")
    plt.title("Original Model")
    plt.savefig("results/cm_original_model.png")
    plt.close()

    torch.save(model.state_dict(), "checkpoints/original_model.pt")
    print("Original model checkpoint saved.")