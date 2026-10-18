import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from data import splits
from evalutils import get_accuracy
from utils import set_seed, get_device, load_model

def relearn(model, few_sevens, train_retain, test_forget, device, steps=300, lr=1e-4,
            eval_every=10, seven_batch=16, retain_batch=64):

    """
    Tests how quickly a model relearns the forgotten class when given
    a fraction of the forgotten data, alongside normal retain data.
    
    If a method only masked the final layer but kept the features, 
    the model will relearn the forgotten class very quickly.
    """

    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader_seven = DataLoader(few_sevens, batch_size=seven_batch, shuffle=True)
    loader_retain = DataLoader(train_retain, batch_size=retain_batch, shuffle=True)

    initial_accuracy = get_accuracy(model, test_forget, device)
    curve = [(0, initial_accuracy)]
    step = 0

    #7s loader is tiny (100 images), so loop over it again and again
    while True:
        for (xs, ys), (xr, yr) in zip(loader_seven, loader_retain):
            model.train()
            x = torch.cat([xs, xr]).to(device)
            y = torch.cat([ys, yr]).to(device)

            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % eval_every == 0:
                curve.append((step, get_accuracy(model, test_forget, device)))
            if step == steps:
                return curve

if __name__ == "__main__":
    device = get_device()
    train_retain, train_forget, _, test_forget = splits()
    
    set_seed(42)
    few_sevens = Subset(train_forget, torch.randperm(len(train_forget))[:100].tolist())

    models = {
        "Original": "checkpoints/original_model.pt",
        "Retrained": "checkpoints/retrained_model.pt",
        "Gradient ascent": "checkpoints/unlearn_gradient_ascent.pt",
        "Bad teacher": "checkpoints/unlearn_bad_teacher.pt",
        "Fisher Method (hard)": "checkpoints/unlearn_fisher_prune_hard.pt",
        "Fisher Method (reset)": "checkpoints/unlearn_fisher_prune_reset.pt"
    }

    for name, path in models.items():
        set_seed(42)  
        curve = relearn(load_model(path, device), few_sevens, train_retain, test_forget, device)

        steps_to_90 = next((s for s, acc in curve if acc >= 0.9), None)
        step_str = str(steps_to_90) if steps_to_90 is not None else ">300"
        print(f"{name}: start {curve[0][1]:.4f} | end {curve[-1][1]:.4f} | steps to 90%: {step_str}")

        plt.plot(*zip(*curve), label=name)

    plt.xlabel("fine-tuning steps")
    plt.ylabel("test accuracy on 7s")
    plt.title("Relearning from 100 sevens")
    plt.legend()
    plt.savefig("results/relearn.png")
    plt.close()