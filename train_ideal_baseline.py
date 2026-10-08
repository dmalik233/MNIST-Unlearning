import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from data import load_mnist, splits
from eval import predict, get_accuracy
from utils import set_seed, get_device
from train import train_model

if __name__ == "__main__":
    set_seed(42)

    device = get_device()
    print(f"Using device {device}")

    _, test = load_mnist()
    train_retain, _, test_retain, test_forget = splits()
    print(f"Training on {len(train_retain)} images (no 7s)..")

    model = train_model(train_retain, device=device)

    print("--Test accuracy--")
    print(f"Retain digits: {get_accuracy(model, test_retain, device):.4f}")
    print(f"Digit 7: {get_accuracy(model, test_forget, device):.4f}")
    print(f"All digits: {get_accuracy(model, test, device):.4f}\n")

    preds, labels = predict(model, test, device)
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    print(cm)
    print("Where the 7s go:", cm[7])

    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap="Blues")
    plt.title("Retrained baseline")
    plt.savefig("results/cm_retrained.png")
    plt.close()

    torch.save(model.state_dict(), "checkpoints/retrained_model.pt")
    print("Retrained model checkpoint saved.")

