import torch
import torch.nn as nn
import sys
from pathlib import Path

from dataset_loader import get_dataloaders
from model import ResNet18Binary
from train import get_device, run_epoch


def parse_bool_arg(name, default=False):
    prefix = f"{name}="

    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            value = arg[len(prefix):].lower()
            return value in ("1", "true", "yes", "y")

    return default


def load_model(model_path, device):
    model_save = torch.load(model_path, map_location=device)

    if isinstance(model_save, dict) and "model_state_dict" in model_save:
        state_dict = model_save["model_state_dict"]
        threshold = model_save.get("best_threshold", 0.5)
    else:
        state_dict = model_save
        threshold = 0.5

    model = ResNet18Binary().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, threshold


def evaluate_saved_model(model_path, test_dataloader, criterion, device, threshold=None):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Could not find '{model_path}'. Run train.py first or use "
            "`python evaluate.py use_example=True`."
        )

    model, saved_threshold = load_model(model_path, device)

    if threshold is None:
        threshold = saved_threshold

    test_loss, test_acc = run_epoch(
        dataloader=test_dataloader,
        model=model,
        criterion=criterion,
        device=device,
        threshold=threshold
    )

    return test_loss, test_acc, threshold


def print_model_weights_summary(model_path):
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    for name, weights in state.items():
        print(name, weights.shape)


def main():
    use_example = parse_bool_arg("use_example", default=False)
    device = get_device()

    _, test_dataloader, _ = get_dataloaders("retinamnist")
    criterion = nn.BCEWithLogitsLoss()

    if use_example:
        model_paths = {
            "Adam": "weights/example_model_Adam.pt",
            "SGD": "weights/example_model_SGD.pt",
        }
        print("-------------------------------------------------------------")
        print(f"Using example weights:")
    else:
        model_paths = {
            "Adam": "weights/best_model_Adam.pt",
            "SGD": "weights/best_model_SGD.pt",
        }
        print("-------------------------------------------------------------")
        print(f"Using self-trained weights:")


    for optimizer_name, model_path in model_paths.items():
        test_loss, test_acc, threshold = evaluate_saved_model(
            model_path=model_path,
            test_dataloader=test_dataloader,
            criterion=criterion,
            device=device
        )

        print(
            f"{optimizer_name}: "
            f"Test Loss = {test_loss:.4f}, "
            f"Test Accuracy = {test_acc:.4f}, "
            f"Threshold = {threshold:.2f}"
        )
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()
