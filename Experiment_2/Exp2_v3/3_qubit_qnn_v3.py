import math
from pathlib import Path
import sys
import os
import random
import signal
import atexit
import time

import cudaq
from cudaq import spin
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import trange

# -----------------------------------------------------------------------------
# Interrupt Handler
# -----------------------------------------------------------------------------
def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nCtrl+C received. Stopping after current batch...")

def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED_CUDAQ = 44
cudaq.set_random_seed(SEED_CUDAQ)

SEED = 22

def seed_everything(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
            worker_seed = SEED + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TARGET_DIGITS = [1, 2, 3, 4, 5, 6, 7, 8]
SAMPLE_COUNT = 4000
TEST_SIZE_PCT = 30

BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 120
EVAL_EVERY = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

QUBIT_COUNT = 3
SHIFT = math.pi / 2
DATA_DIR = Path("./data")

SET_DETERMINISTIC = True
TORCH_DEVICE = "cpu" # Options: "cpu", "cuda"
CUDA_DEVICE = "qpp-cpu" # Options: "qpp-cpu", "nvidia"
DEVICE = None

STOP_REQUESTED = False
# -----------------------------------------------------------------------------
# Backend selection
# -----------------------------------------------------------------------------
def configure_backend():

    if TORCH_DEVICE == "cuda" and torch.cuda.is_available():
        try:
            cudaq.set_target(CUDA_DEVICE)
            device = torch.device(TORCH_DEVICE)
            return device
        except Exception as exc:
            print(f"Falling back to CPU backend because CUDA-Q nvidia target failed: {exc}")

    cudaq.set_target("qpp-cpu")
    return torch.device("cpu")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
class RemappedSubset(Dataset):
    def __init__(self, base_dataset, indices, label_map):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]
        return x, self.label_map[int(y)]


class DeviceDataLoader:
    """Move mini-batches to the target device lazily."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for x, y in self.loader:
            yield (
                x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True),
            )

    def __len__(self):
        return len(self.loader)



def build_dataloaders(target_digits, sample_count, test_size_pct, batch_size, device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    label_map = {digit: i for i, digit in enumerate(target_digits)}

    candidate_indices = [
        i for i, y in enumerate(dataset.targets.tolist()) if int(y) in label_map
    ]

    if sample_count > len(candidate_indices):
        sample_count = len(candidate_indices)

    rng = np.random.default_rng(SEED)
    sampled_indices = rng.choice(candidate_indices, size=sample_count, replace=False)
    sampled_labels = [label_map[int(dataset.targets[i])] for i in sampled_indices]

    train_idx, test_idx = train_test_split(
        sampled_indices,
        test_size=test_size_pct / 100.0,
        shuffle=True,
        stratify=sampled_labels,
        random_state=42,
    )

    train_ds = RemappedSubset(dataset, train_idx, label_map)
    test_ds = RemappedSubset(dataset, test_idx, label_map)

    pin_memory = device.type == "cuda"

    if SET_DETERMINISTIC:
        g = torch.Generator()
        g.manual_seed(SEED)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(NUM_WORKERS > 0),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(NUM_WORKERS > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

    return DeviceDataLoader(train_loader, device), DeviceDataLoader(test_loader, device)


# -----------------------------------------------------------------------------
# Quantum circuit wrapper
# -----------------------------------------------------------------------------
def build_observables(qubit_count: int):
    if qubit_count == 1:
        return [
            spin.z(0),
            spin.x(0),
        ]
    elif qubit_count == 2:
        return [
            spin.z(0),
            spin.z(1),
            spin.z(0) * spin.z(1),
            spin.x(0) * spin.x(1),
        ]
    elif qubit_count == 3:
        return [
            spin.z(0),
            spin.z(1),
            spin.z(2),
            spin.z(0) * spin.z(1),
            spin.z(1) * spin.z(2),
            spin.z(0) * spin.z(2),
            spin.x(0) * spin.x(1),
            spin.x(1) * spin.x(2),
        ]
    else:
        raise ValueError("Only qubit counts 1, 2, and 3 are supported.")
    
class QuantumCircuitRunner:
    def __init__(self, qubit_count: int):
        @cudaq.kernel
        def kernel(qubit_count: int, thetas: np.ndarray):
            qubits = cudaq.qvector(qubit_count)

            # First rotation layer
            for i in range(qubit_count):
                ry(thetas[i], qubits[i])

            # Nearest-neighbor entanglement
            for i in range(qubit_count - 1):
                x.ctrl(qubits[i], qubits[i + 1])

            # Second rotation layer
            for i in range(qubit_count):
                rx(thetas[qubit_count + i], qubits[i])

        self.kernel = kernel
        self.qubit_count = qubit_count
        self.param_count = 2 * qubit_count
        self.hamiltonians = build_observables(qubit_count)

    def run(self, theta_vals: torch.Tensor) -> torch.Tensor:
        theta_np = theta_vals.detach().cpu().numpy()
        batch_qubits = [self.qubit_count] * theta_np.shape[0]

        outputs = []
        for hamiltonian in self.hamiltonians:
            results = cudaq.observe(self.kernel, hamiltonian, batch_qubits, theta_np)
            outputs.append(
                torch.tensor(
                    [result.expectation() for result in results],
                    dtype=theta_vals.dtype,
                )
            )

        return torch.stack(outputs, dim=1).to(theta_vals.device)


class QuantumAutograd(Function):
    @staticmethod
    def forward(ctx, thetas: torch.Tensor, runner: QuantumCircuitRunner, shift: float):
        ctx.runner = runner
        ctx.shift = float(shift)
        ctx.save_for_backward(thetas)
        return runner.run(thetas)

    @staticmethod
    def backward(ctx, grad_output):
        (thetas,) = ctx.saved_tensors
        batch_size, param_count = thetas.shape
        shift = ctx.shift

        # Vectorized parameter-shift:
        # Instead of 2 * P separate quantum runs, build all shifted parameter sets
        # in two large batches and call the circuit runner only twice.
        eye = torch.eye(param_count, device=thetas.device, dtype=thetas.dtype).unsqueeze(0)
        theta_expanded = thetas.unsqueeze(1)  # (B, 1, P)

        plus = (theta_expanded + shift * eye).reshape(batch_size * param_count, param_count)
        minus = (theta_expanded - shift * eye).reshape(batch_size * param_count, param_count)

        exp_plus = ctx.runner.run(plus).reshape(batch_size, param_count, -1)
        exp_minus = ctx.runner.run(minus).reshape(batch_size, param_count, -1)

        deriv = (exp_plus - exp_minus) / (2.0 * shift)  # (B, P, O)
        grad_thetas = (grad_output.unsqueeze(1) * deriv).sum(dim=2)  # (B, P)

        return grad_thetas, None, None


class QuantumLayer(nn.Module):
    def __init__(self, qubit_count: int, shift: float):
        super().__init__()
        self.runner = QuantumCircuitRunner(qubit_count)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumAutograd.apply(inputs, self.runner, self.shift)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class HybridQNN(nn.Module):
    def __init__(self, qubit_count: int, shift: float):
        super().__init__()

        param_count = 2 * qubit_count

        self.classical = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, param_count),
        )

        self.quantum = QuantumLayer(qubit_count, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta_params = self.classical(x)
        return self.quantum(theta_params)


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------
def print_loader_distribution(loader, name):
    counts = torch.zeros(len(TARGET_DIGITS), dtype=torch.long)
    for _, y in loader:
        counts += torch.bincount(y.cpu(), minlength=len(TARGET_DIGITS))

    print(f"\n{name} distribution:")
    for cls, count in enumerate(counts.tolist()):
        print(f"Class {cls} ({TARGET_DIGITS[cls]}): {count} samples")


@torch.inference_mode()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        logits = model(x)
        loss = loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        batch_size = y.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == y).sum().item()
        total_examples += batch_size

        y_true_all.append(y.cpu())
        y_pred_all.append(preds.cpu())

    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)
    return avg_loss, avg_acc, y_true_all, y_pred_all


def train_model(model, train_loader, test_loader, epochs, eval_every):
    signal.signal(signal.SIGINT, handle_sigint)
    atexit.register(cleanup)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    eval_epochs = []

    progress = trange(epochs, desc="Training", leave=True)

    for epoch in progress:
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        if STOP_REQUESTED:
            print("Stopping before next epoch.")
            break

        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == y).sum().item()
            running_examples += batch_size

        if STOP_REQUESTED:
            print("Training interrupted cleanly.")
            break

        train_loss = running_loss / max(running_examples, 1)
        train_acc = running_correct / max(running_examples, 1)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        if (epoch + 1) % eval_every == 0 or epoch == 0 or (epoch + 1) == epochs:
            test_loss, test_acc, _, _ = evaluate(model, test_loader, loss_fn)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            eval_epochs.append(epoch + 1)
            progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                test_loss=f"{test_loss:.4f}",
                test_acc=f"{test_acc:.4f}",
            )
        else:
            progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
            )

    return {
        "train_loss": train_loss_hist,
        "train_acc": train_acc_hist,
        "test_loss": test_loss_hist,
        "test_acc": test_acc_hist,
        "eval_epochs": eval_epochs,
        "loss_fn": loss_fn,
    }


# -----------------------------------------------------------------------------
# Reporting / plots
# -----------------------------------------------------------------------------
def per_class_accuracy(y_true, y_pred, num_classes):
    results = {}
    for cls in range(num_classes):
        cls_mask = y_true == cls
        cls_total = cls_mask.sum().item()
        cls_correct = ((y_pred == cls) & cls_mask).sum().item()
        results[cls] = cls_correct / cls_total if cls_total > 0 else 0.0
    return results


def save_metrics_plot(history, path="optimized_metrics.png"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(
        [epoch - 1 for epoch in history["eval_epochs"]],
        history["test_loss"],
        marker="o",
        label="Test",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(
        [epoch - 1 for epoch in history["eval_epochs"]],
        history["test_acc"],
        marker="o",
        label="Test",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, path="optimized_confusion_matrix.png"):
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(range(len(TARGET_DIGITS)), TARGET_DIGITS)
    plt.yticks(range(len(TARGET_DIGITS)), TARGET_DIGITS)
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.title("Optimized Hybrid QNN Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    global DEVICE
    DEVICE = configure_backend()

    print(f"Using CUDA-Q target: {CUDA_DEVICE}")
    print(f"Using device: {DEVICE}")

    seed_everything(22)

    observables = build_observables(QUBIT_COUNT)
    num_classes = len(TARGET_DIGITS)

    if num_classes != len(observables):
        raise ValueError(
            f"Mismatch: len(TARGET_DIGITS)={num_classes} but "
            f"QUBIT_COUNT={QUBIT_COUNT} gives {len(observables)} observables."
        )

    train_loader, test_loader = build_dataloaders(
        target_digits=TARGET_DIGITS,
        sample_count=SAMPLE_COUNT,
        test_size_pct=TEST_SIZE_PCT,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    print_loader_distribution(train_loader, "Train")
    print_loader_distribution(test_loader, "Test")

    model = HybridQNN(QUBIT_COUNT, SHIFT).to(DEVICE)

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        eval_every=EVAL_EVERY,
    )

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, history["loss_fn"])

    print("\nFinal metrics:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    per_class = per_class_accuracy(y_true, y_pred, num_classes=len(TARGET_DIGITS))
    print("\nPer-class accuracy:")
    for cls, acc in per_class.items():
        print(f"Class {cls} ({TARGET_DIGITS[cls]}): {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true.numpy(), y_pred.numpy(), digits=4))

    save_confusion_matrix(y_true, y_pred)
    save_metrics_plot(history)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)