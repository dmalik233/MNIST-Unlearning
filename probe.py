from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import load_mnist, forget_digit
from eval import get_features
from utils import get_device, load_model, set_seed

def linear_probe(model, train, test, device):
    x_train, y_train = get_features(model, train, device)
    x_test, y_test = get_features(model, test, device)

    y_train = y_train == forget_digit
    y_test = y_test == forget_digit

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    probe.fit(x_train, y_train)

    bal_acc = balanced_accuracy_score(y_test, probe.predict(x_test))
    auc = roc_auc_score(y_test, probe.predict_proba(x_test)[:, 1])
    return bal_acc, auc

if __name__=="__main__":
    set_seed(42)
    device = get_device()

    train, test = load_mnist()

    models = {
        "Original": "checkpoints/original_model.pt",
        "Retrained": "checkpoints/retrained_model.pt",
        "Gradient ascent": "checkpoints/unlearn_gradient_ascent.pt",
        "Bad teacher": "checkpoints/unlearn_bad_teacher.pt"
    }

    for name, path in models.items():
        model = load_model(path, device)
        bal_acc, auc = linear_probe(model, train, test, device)
        print(f"{name}: balanced acc {bal_acc:.4f} | AUC {auc:.4f}")