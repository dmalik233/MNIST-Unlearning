import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from model import CNN
from data import splits
from evalutils import get_accuracy
from utils import set_seed, get_device, load_model

def grad_asc(model, train_forget, train_retain, device, max_batch=100, stop_acc=0.01, 
            lr_forget=1e-4, lr_retain=1e-4, forget_batch=64, retain_batch=64):

    '''
    Implements gradient ascent based unlearning.
    Maximizes CE loss on forget data, and supplements it with minimising loss on retain data
    to prevent catastrophic forgetting and for maintaining model utility.

    stop_acc- we use this to stop the moment forget accuracy reaches this threshold, to prevent
    further decrease in retain accuracy.
    max_batch- for batch level control instead of epoch level, to restrict the number of destructive updates.
    '''

    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()

    #different optimizers to not mix up the different forget and retain losses
    optim_forget = torch.optim.Adam(model.parameters(), lr=lr_forget)
    optim_retain = torch.optim.Adam(model.parameters(), lr=lr_retain)

    loader_forget = DataLoader(train_forget, batch_size=forget_batch, shuffle=True)
    loader_retain = DataLoader(train_retain, batch_size=retain_batch, shuffle=True)

    batch_count = 0
    print(f"Max batches: {max_batch}, stop at forget accuracy: {stop_acc}")

    #loop to prevent running out of batches for larger max_batches
    # batch_size=64 gives 98 batches for forget digit 7 (total 6265 7s for training in MNIST)
    while True:

        for (xf, yf), (xr, yr) in zip(loader_forget, loader_retain):
            xf, yf = xf.to(device), yf.to(device)
            xr, yr = xr.to(device), yr.to(device)
            model.train()
            batch_count+=1

            #gradient ascent on forget
            loss = -criterion(model(xf), yf)
            optim_forget.zero_grad()
            loss.backward()
            optim_forget.step()

            #normal gradient descent on retain
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

    '''
    In this distillation based approach, we use 3 models
    1. student- the model being unlearned (a copy of the original model)
    2. good- a frozen copy of the original model
    3. dumb- a randomly initialised, untrained model

    On the forget set- minimise the KL divergence between the student and the dumb
    model (which does not have any knowledge of 7s, the target class).
    On the retain set- minimise the KL divergence between the student and the good teacher,
    essentially acting as a healing and knowledge retention step.
    '''

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


def get_fisher_diagonal(model, dataloader, device):

    '''
    To obtain the diagonal elements of the Fisher matrix for approximating
    the second-order derivatives of the loss wrt each weight.
    Calculated by averaging the squared first order gradients across batches.
    '''

    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    model.eval()
    criterion = nn.CrossEntropyLoss()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += (p.grad.detach() ** 2) / len(dataloader)

    return fisher


def fisher_prune(model, train_forget, train_retain, device,
                 prune_ratio=0.35, heal_epochs=2, lr=1e-3, batch_size=64, retain_size=6265,
                 lock_mask=True):
    '''
    Prune layers conv2 and fc1.
    conv1 is ignored as it just learns simple features important for retain data.
    Calculates an importance score using the ratio F_forget / (F_retain + 1e-8) to find
    weights specialized for the forget class, and zeroes out the top prune_ratio.
    fc2 weights for class 7 are reset to 0 and left unlocked so healing pushes its logits down.

    lock_mask- if True (hard pruning), pruned weights have their gradients zeroed during
    healing so they stay dead. If False (synaptic reset), they are allowed to update on
    retain data and get repurposed.
    '''

    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()

    # match retain size to forget set (6265 samples for digit 7) for balanced fisher estimation
    indices = torch.randperm(len(train_retain))[:retain_size].tolist()
    retain_subset = Subset(train_retain, indices)

    loader_forget = DataLoader(train_forget, batch_size=batch_size, shuffle=True)
    loader_retain = DataLoader(retain_subset, batch_size=batch_size, shuffle=True)

    f_forget = get_fisher_diagonal(model, loader_forget, device)
    f_retain = get_fisher_diagonal(model, loader_retain, device)

    masks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "conv2" in name or "fc1" in name:
                score = f_forget[name] / (f_retain[name] + 1e-8)
                thresh = torch.quantile(score.flatten(), 1.0 - prune_ratio)
                mask = score >= thresh
                
                p[mask] = 0.0
                masks[name] = mask
                print(f"{name}: pruned {mask.sum().item()}/{mask.numel()}")

            elif "fc2" in name:
                # reset class 7 output weights, leave unmasked so healing pushes logits down
                p[7] = 0.0

    forget_acc = get_accuracy(model, train_forget, device)
    retain_acc = get_accuracy(model, train_retain, device)
    print(f"After pruning | forget acc: {forget_acc:.4f} | retain acc: {retain_acc:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(heal_epochs):
        model.train()
        for xr, yr in loader_retain:
            xr, yr = xr.to(device), yr.to(device)
            
            loss = criterion(model(xr), yr)
            optimizer.zero_grad()
            loss.backward()

            # lock_mask=True (hard pruning): zero gradients so pruned weights stay 0
            # lock_mask=False (synaptic reset): let retain gradients reuse the zeroed weights
            if lock_mask:
                for name, p in model.named_parameters():
                    if name in masks and p.grad is not None:
                        p.grad[masks[name]] = 0.0

            optimizer.step()

        forget_acc = get_accuracy(model, train_forget, device)
        retain_acc = get_accuracy(model, train_retain, device)
        print(f"Heal epoch {epoch + 1} | forget acc: {forget_acc:.4f} | retain acc: {retain_acc:.4f}")

    return model.eval()
    
if __name__=="__main__":
    set_seed(42)
    device = get_device()
    train_retain, train_forget, _, _ = splits()

    original = load_model("checkpoints/original_model.pt", device)
    print("\n--Gradient ascent based unlearning--")
    model = grad_asc(original, train_forget, train_retain, device)
    torch.save(model.state_dict(), "checkpoints/unlearn_gradient_ascent.pt")
    print("Gradient ascent model checkpoint saved.")

    set_seed(42)
    print("\n--Distillation based unlearning--") 
    model = bad_teacher(original, train_forget, train_retain, device, epochs=5)
    torch.save(model.state_dict(), "checkpoints/unlearn_bad_teacher.pt")
    print("Bad teacher model checkpoint saved.")

    set_seed(42)
    print("\n--Fisher pruning (hard pruning) unlearning--")
    model = fisher_prune(original, train_forget, train_retain, device, lock_mask=True)
    torch.save(model.state_dict(), "checkpoints/unlearn_fisher_prune_hard.pt")
    print("Fisher prune (hard pruning) model checkpoint saved.")

    set_seed(42)
    print("\n--Fisher pruning (synaptic reset) unlearning")
    model = fisher_prune(original, train_forget, train_retain, device, lock_mask=False)
    torch.save(model.state_dict(), "checkpoints/unlearn_fisher_prune_reset.pt")
    print("Fisher prune (synaptic reset) model checkpoint saved.")