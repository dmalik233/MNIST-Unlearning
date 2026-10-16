import numpy as np
from torch.utils.data import ConcatDataset

from data import splits
from eval import get_accuracy, get_probs, pred_distribution, js_div, save_confusion_matrix
from utils import get_device, load_model, set_seed


def report(m_type, model, reference, test_retain, test_forget, device):
    print(f"\n--{m_type} model--")
    print(f"Retain accuracy: {get_accuracy(model, test_retain, device):.4f}")
    print(f"Forget accuracy: {get_accuracy(model, test_forget, device):.4f}")

    js_retain = js_div(get_probs(reference, test_retain, device), get_probs(model, test_retain, device))
    js_forget = js_div(get_probs(reference, test_forget, device), get_probs(model, test_forget, device))

    #We compare against the ideal baseline (full retrain case)
    print(f"\nJS {m_type} to retrained (retain): {js_retain:.4f}")
    print(f"JS {m_type} to retrained (forget): {js_forget:.4f}")

    distribution = pred_distribution(model, test_forget, device)
    print("\nDistribution of 7s: ")
    print(np.round(distribution * 100, 1), "\n")

    full_test = ConcatDataset([test_retain, test_forget])
    file_name = m_type.lower().replace(" ", "_")
    save_confusion_matrix(model, full_test, device, m_type, f"results/cm_{file_name}.png")

if __name__ == "__main__":

    device = get_device()
    _, _, test_retain, test_forget = splits()
    original = load_model("checkpoints/original_model.pt", device)
    retrained = load_model("checkpoints/retrained_model.pt", device)
    grad_asc = load_model("checkpoints/unlearn_gradient_ascent.pt", device)
    bad_teacher = load_model("checkpoints/unlearn_bad_teacher.pt", device)

    report("Original", original, retrained, test_retain, test_forget, device)
    report("Retrained", retrained, retrained, test_retain, test_forget, device)
    report("Gradient ascent", grad_asc, retrained, test_retain, test_forget, device)
    report("Bad teacher", bad_teacher, retrained, test_retain, test_forget, device)
    