"""
Experiment 1 (v4): 2-qubit, 4-class hybrid QNN with increased circuit depth.

Goal:
Test whether adding an additional entanglement + rotation block improves
class separability and overall performance relative to v3.

Hypothesis:
If the main limitation in earlier runs was insufficient circuit expressivity,
then increasing depth should improve per-class learning and reduce confusion
between visually similar digits.
"""

import cudaq
from cudaq import *

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torchvision

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from pathlib import Path

# Reproducibility
torch.manual_seed(22)
cudaq.set_random_seed(44)

# Run on CPU for baseline stability/debugging
device = torch.device("cpu")
cudaq.set_target("qpp-cpu")

# Optional GPU backend
# cudaq.set_target("nvidia")
# device = torch.device("cuda:0")

#More robust 'Project Root' for MNIST dataset
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / ".git").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"


def prepare_data(target_digits, sample_count, test_size):
    """Load MNIST, filter to selected digits, remap labels, and split train/test.

    Args:
        target_digits (list[int]): Digits to include in the experiment
        sample_count (int): Total number of filtered samples to use
        test_size (float): Percent of selected data used for testing

    Returns:
        x_train, x_test, y_train, y_test
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
    )

    # Keep only the requested digits
    idx = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for digit in target_digits:
        idx |= (dataset.targets == digit)

    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]

    # Random subset of the filtered dataset
    subset_indices = torch.randperm(dataset.data.size(0))[:sample_count]

    x = dataset.data[subset_indices].float().unsqueeze(1).to(device)

    # Remap labels to 0..N-1 for CrossEntropyLoss
    raw_y = dataset.targets[subset_indices]
    mapping = {digit: i for i, digit in enumerate(target_digits)}
    y = torch.tensor(
        [mapping[int(label)] for label in raw_y],
        dtype=torch.long,
        device=device
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size / 100,
        shuffle=True,
        random_state=42
    )

    return x_train, x_test, y_train, y_test

def print_class_distribution(y, name):
    unique, counts = torch.unique(y, return_counts=True)
    print(f"\n{name} distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {u.item()}: {c.item()} samples")



# Experiment parameters
sample_count = 2000
target_digits = [5, 6, 3, 4]   # Four MNIST classes for Experiment 1
test_size = 30
epochs = 200

# Quantum parameters
qubit_count = 2                 # Two-qubit circuit
shift = torch.tensor(torch.pi / 2)


x_train, x_test, y_train, y_test = prepare_data(
    target_digits,
    sample_count,
    test_size
)

print_class_distribution(y_train, "Train")
print_class_distribution(y_test, "Test")

# Save a sample grid of inputs
grid_img = torchvision.utils.make_grid(
    x_train[:10],
    nrow=5,
    padding=3,
    normalize=True
)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.savefig("sample_inputs_exp1v4.png", dpi=200, bbox_inches="tight")
plt.close()

class QuantumFunction(Function):
    """PyTorch autograd wrapper for the parameterized quantum circuit."""

    def __init__(self, qubit_count: int):
        """Define a 2-qubit variational circuit with one entangling gate."""

        @cudaq.kernel
        def kernel(qubit_count: int, thetas: np.ndarray):
            qubits = cudaq.qvector(qubit_count)

            # Layer 1
            ry(thetas[0], qubits[0])
            rx(thetas[1], qubits[0])
            ry(thetas[2], qubits[1])
            rx(thetas[3], qubits[1])

            # Entanglement 1
            x.ctrl(qubits[0], qubits[1])

            # Layer 2
            ry(thetas[4], qubits[0])
            rx(thetas[5], qubits[0])
            ry(thetas[6], qubits[1])
            rx(thetas[7], qubits[1])

            # Entanglement 2
            x.ctrl(qubits[0], qubits[1])

            # Layer 3
            ry(thetas[8], qubits[0])
            rx(thetas[9], qubits[0])
            ry(thetas[10], qubits[1])
            rx(thetas[11], qubits[1])

        self.kernel = kernel
        self.qubit_count = qubit_count

    def run(self, theta_vals: torch.Tensor) -> torch.Tensor:
        """Evaluate four observables to produce a 4D quantum output vector."""

        qubit_count_batch = [self.qubit_count for _ in range(theta_vals.shape[0])]

        # Four observables used as class logits/features
        hamiltonians = [
            spin.z(0),
            spin.z(1),
            spin.x(0),
            spin.x(1)
        ]

        outputs = []
        for H in hamiltonians:
            res = cudaq.observe(self.kernel, H, qubit_count_batch, theta_vals)
            outputs.append([r.expectation() for r in res])

        # Shape: (batch_size, 4)
        outputs = torch.tensor(outputs, dtype=torch.float32, device=device).T
        return outputs

    @staticmethod
    def forward(ctx, thetas: torch.Tensor, quantum_circuit, shift) -> torch.Tensor:
        """Forward pass through the quantum circuit."""
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        exp_vals = ctx.quantum_circuit.run(thetas)
        ctx.save_for_backward(thetas, exp_vals)

        return exp_vals

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using the parameter-shift rule."""
        thetas, _ = ctx.saved_tensors
        gradients = torch.zeros_like(thetas, device=device)

        for i in range(thetas.shape[1]):
            thetas_plus = thetas.clone()
            thetas_plus[:, i] += ctx.shift
            exp_vals_plus = ctx.quantum_circuit.run(thetas_plus)   # (batch, 4)

            thetas_minus = thetas.clone()
            thetas_minus[:, i] -= ctx.shift
            exp_vals_minus = ctx.quantum_circuit.run(thetas_minus) # (batch, 4)

            deriv = (exp_vals_plus - exp_vals_minus) / (2 * ctx.shift)

            # Contract output gradients with parameter-shift derivatives
            gradients[:, i] = (grad_output * deriv).sum(dim=1)

        return gradients, None, None


class QuantumLayer(nn.Module):
    """Wrap the quantum autograd function as a PyTorch layer."""

    def __init__(self, qubit_count: int, shift: torch.Tensor):
        super().__init__()
        self.quantum_circuit = QuantumFunction(qubit_count)
        self.shift = shift

    def forward(self, inputs):
        return QuantumFunction.apply(inputs, self.quantum_circuit, self.shift)


class HybridQNN(nn.Module):
    """Hybrid classical-quantum neural network for 4-class MNIST classification."""

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.25)

        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 12)   # eight circuit parameters

        self.quantum = QuantumLayer(qubit_count, shift)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)

        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)

        # Quantum layer returns 4 logits
        x = self.quantum(x)
        return x


def accuracy_score(y, y_hat):
    preds = torch.argmax(y_hat, dim=1)
    return (y == preds).float().mean().item()


hybrid_model = HybridQNN().to(device)

optimizer = optim.Adadelta(
    hybrid_model.parameters(),
    lr=0.001,
    weight_decay=0.8
)

def per_class_accuracy(y_true, y_pred, num_classes):
    results = {}
    for cls in range(num_classes):
        cls_mask = (y_true == cls)
        cls_total = cls_mask.sum().item()
        cls_correct = ((y_pred == cls) & cls_mask).sum().item()
        results[cls] = cls_correct / cls_total if cls_total > 0 else 0.0
    return results

loss_function = nn.CrossEntropyLoss().to(device)

training_cost = []
testing_cost = []
training_accuracy = []
testing_accuracy = []

hybrid_model.train()
for epoch in range(epochs):
    optimizer.zero_grad()

    y_hat_train = hybrid_model(x_train).to(device)
    train_cost = loss_function(y_hat_train, y_train).to(device)

    train_cost.backward()
    optimizer.step()

    training_accuracy.append(accuracy_score(y_train, y_hat_train))
    training_cost.append(train_cost.item())

    hybrid_model.eval()
    with torch.no_grad():
        y_hat_test = hybrid_model(x_test).to(device)
        test_cost = loss_function(y_hat_test, y_test).to(device)

        testing_accuracy.append(accuracy_score(y_test, y_hat_test))
        testing_cost.append(test_cost.item())

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_cost.item():.4f} | "
        f"Test Loss: {test_cost.item():.4f} | "
        f"Train Acc: {training_accuracy[-1]:.4f} | "
        f"Test Acc: {testing_accuracy[-1]:.4f}"
    )

hybrid_model.eval()
with torch.no_grad():
    final_logits = hybrid_model(x_test).to(device)
    final_preds = torch.argmax(final_logits, dim=1)

cm = confusion_matrix(y_test.cpu(), final_preds.cpu())
per_class = per_class_accuracy(y_test, final_preds, num_classes=len(target_digits))

print("\nPer-class accuracy:")
for cls, acc in per_class.items():
    print(f"Class {cls} ({target_digits[cls]}): {acc:.4f}")

print("\nClassification report:")
print(classification_report(y_test.cpu(), final_preds.cpu(), digits=4))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_digits, yticklabels=target_digits)
plt.xlabel("Predicted Digit")
plt.ylabel("True Digit")
plt.title("Experiment 1 (v4) Confusion Matrix")
plt.tight_layout()
plt.savefig("experiment1v4_confusion_matrix.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_cost, label="Train")
plt.plot(testing_cost, label="Test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label="Train")
plt.plot(testing_accuracy, label="Test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig("experiment1v4_metrics.png", dpi=200)
plt.close()
