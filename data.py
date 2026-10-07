import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

forget_digit = 7

def load_mnist(root="data"):
    train = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())

    return train, test

def class_split(dataset, forget_digit=forget_digit):
    targets = dataset.targets
    forget_index = torch.where(targets == forget_digit)[0].tolist()
    retain_index = torch.where(targets != forget_digit)[0].tolist()

    return Subset(dataset, retain_index), Subset(dataset, forget_index)

def splits(root="data", forget_digit=forget_digit):
    train, test = load_mnist(root)
    train_retain, train_forget = class_split(train, forget_digit)
    test_retain, test_forget = class_split(test, forget_digit)
    return train_retain, train_forget, test_retain, test_forget

if __name__ == "__main__":
    train_retain, train_forget, test_retain, test_forget = splits()
    print(f"Digit to forget: {forget_digit}")
    print(f"Train- Retain: {len(train_retain)}, Forget: {len(train_forget)}")
    print(f"Test- Retain: {len(test_retain)}, Forget: {len(test_forget)}")