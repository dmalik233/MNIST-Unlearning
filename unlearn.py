import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from model import CNN
from data import splits
from eval import get_accuracy
from utils import set_seed, get_device, load_model

def grad_asc(model, train_forget, train_retain, device, max_batch=100, stop_acc=0.01, 
            lr_forget=1e-4, lr_retain=1e-4, forget_batch=64, retain_batch=64):

    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()

    #different optimizers to not mix up the different forget and retain losses
    optim_forget = torch.optim.Adam(model.parameters(), lr=lr_forget)
    optim_retain = torch.optim.Adam(model.parameters(), lr=lr_retain)

    loader_forget = DataLoader(train_forget, batch_size=forget_batch, shuffle=True)
    loader_retain = DataLoader(train_retain, batch_size=retain_batch, shuffle=True)

    batch_count = 0
    print(f"Max batches: {max_batch}, stop forget accuracy: {stop_acc}")

    #loop to prevent running out of batches for larger max_batches
    # batch_size=64 gives 98 batches for forget digit 7
    while True:

        for (xf, yf), (xr, yr) in zip(loader_forget, loader_retain):
            xf, yf = xf.to(device), yf.to(device)
            xr, yr = xr.to(device), yr.to(device)
            model.train()
            batch_count+=1

            #ascent on forget
            loss = -criterion(model(xf), yf)
            optim_forget.zero_grad()
            loss.backward()
            optim_forget.step()

            #descent on retain
            loss = criterion(model(xr), yr)
            optim_retain.zero_grad()
            loss.backward()
            optim_retain.step()

            forget_acc = get_accuracy(model, train_forget, device)

            if (batch_count%10 == 0):
                retain_acc = get_accuracy(model, train_retain, device)
                print(f"Batch {batch_count} | forget acc: {forget_acc:.4f} | retain acc: {retain_acc:.4f}")

            if forget_acc <= stop_acc or batch_count == max_batch:
                retain_acc = get_accuracy(model, train_retain, device)

                reason = f"Target forget accuracy {stop_acc} reached" if forget_acc <= stop_acc else "Max batch count reached."
                print(reason)
                print(f"\nForget_acc {forget_acc:.4f}, retain acc: {retain_acc:.4f} at batch {batch_count}. Stopping.")
                return model.eval()


def bad_teacher(model, train_forget, train_retain, device, 
                epochs=1, lr=1e-4, forget_batch=64, retain_batch=64):
    
    student = copy.deepcopy(model).to(device)
    good = copy.deepcopy(model).to(device).eval()
    dumb = CNN().to(device).eval() #randomly initialised base model

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    loader_forget = DataLoader(train_forget, batch_size=forget_batch, shuffle=True)
    loader_retain = DataLoader(train_retain, batch_size=retain_batch, shuffle=True)

    for epoch in range(epochs):
        for (xf, _), (xr, _) in zip(loader_forget, loader_retain):
            xf, xr = xf.to(device), xr.to(device)

            student.train()

            with torch.no_grad():
                target_forget = F.softmax(dumb(xf), dim=1)
                target_retain = F.softmax(good(xr), dim=1)

            loss_forget = F.kl_div(F.log_softmax(student(xf), dim=1), target_forget, reduction="batchmean")
            loss_retain = F.kl_div(F.log_softmax(student(xr), dim=1), target_retain, reduction="batchmean")

            loss = loss_forget + loss_retain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        forget_acc = get_accuracy(student, train_forget, device)
        retain_acc = get_accuracy(student, train_retain, device)

        print(f"Epoch {epoch + 1} | forget acc: {forget_acc:.4f} | retain acc: {retain_acc:.4f}")

    return student.eval()


if __name__=="__main__":
    set_seed(42)
    device = get_device()
    train_retain, train_forget, _, _ = splits()

    original = load_model("checkpoints/original_model.pt", device)
    model = grad_asc(original, train_forget, train_retain, device)
    torch.save(model.state_dict(), "checkpoints/unlearn_gradient_ascent.pt")
    print("Gradient ascent model checkpoint saved.")

    set_seed(42) 
    model = bad_teacher(original, train_forget, train_retain, device, epochs=5)
    torch.save(model.state_dict(), "checkpoints/unlearn_bad_teacher.pt")
    print("Bad teacher model checkpoint saved.")