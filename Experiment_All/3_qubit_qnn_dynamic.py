import math
from pathlib import Path
import os
import sys
import time
import random
import signal
import atexit
import copy

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
from tqdm import trange, tqdm

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
# Quantum circuit settings
QUBIT_COUNT = 3 # Options: 1, 2, 3
SHIFT = math.pi / 2

# Data settings
DATA_DIR = Path("./data")
PRESETS={1: [5, 6],
         2: [3, 4, 5, 6],
         3: [1, 2, 3, 4, 5, 6, 7, 8]}

TARGET_DIGITS = PRESETS[QUBIT_COUNT]
SAMPLE_COUNT = 4000
TEST_SIZE_PCT = 30

# Training hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 40
EVAL_EVERY = EPOCHS // 10
LR = 1e-3
WEIGHT_DECAY = 3e-4
DROPOUT = 0.30

# Early stop settings
EARLY_STOP_METRIC = "accuracy" # Options: "loss", "accuracy"
EARLY_STOP_PATIENCE = 2 # number of eval periods, *not* epochs
SCHEDULER_PATIENCE = 0 # number of eval periods with no improvement before reducing LR

# Device and Determinism Settings
SET_DETERMINISTIC = True
TORCH_DEVICE = "cpu" # Options: "cpu", "cuda"
CUDA_DEVICE = "qpp-cpu" # Options: "qpp-cpu", "nvidia"

# Global variable initializations
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
    """
    Custom dataset wrapper that restricts MNIST to a selected subset of samples
    and remaps the original digit labels to contiguous class indices.

    Functionality:
        - Stores only the sampled indices selected for the experiment.
        - Fetches the corresponding image/label pair from the original MNIST dataset.
        - Converts the original digit label into the remapped class index used by
          the classifier.

    This allows the dataloader to work with arbitrary digit subsets while keeping
    the training labels compatible with the loss function.
    """
    def __init__(self, base_dataset, indices, label_map):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieve one image from the original dataset and remap its digit label to
        the experiment-specific class index.
        """
        x, y = self.base_dataset[self.indices[idx]]
        return x, self.label_map[int(y)]


class DeviceDataLoader:
    """
    Wrapper around a PyTorch DataLoader to move data to the specified device.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        """
        Iterate over the underlying dataloader and move each mini-batch to the
        configured device just before it is used.

        The transfer is done lazily, batch by batch, so only the current mini-batch
        is moved to the target device. This avoids preloading the full dataset onto 
        the device at once.
        """
        for x, y in self.loader:
            yield (
                x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True),
            )

    def __len__(self):
        return len(self.loader)


def build_dataloaders(target_digits, sample_count, test_size_pct, batch_size, device):
    """
    Build deterministic train/test dataloaders for the selected subset of MNIST.

    Only the digits listed in target_digits are retained. Their labels are then
    remapped to contiguous class indices 0..N-1 so they are compatible with
    CrossEntropyLoss.

    The split is stratified so that each class is represented proportionally in
    both the training and test sets.
    """
    # Standard MNIST normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    
    # Load the full MNIST dataset and filter it down to the selected digits.
    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    # Create a mapping from the original digit labels to the remapped class indices.
    label_map = {digit: i for i, digit in enumerate(target_digits)}

    # Identify the indices of samples whose labels are in the target set.
    candidate_indices = [
        i for i, y in enumerate(dataset.targets.tolist()) if int(y) in label_map
    ]

    # If the requested sample count exceeds the available samples for the selected digits, adjust it to the maximum possible.
    if sample_count > len(candidate_indices):
        sample_count = len(candidate_indices)

    # Deterministcally sample the specified number of indices from the candidate set
    rng = np.random.default_rng(SEED)
    sampled_indices = rng.choice(candidate_indices, size=sample_count, replace=False)
    sampled_labels = [label_map[int(dataset.targets[i])] for i in sampled_indices]

    # Create train/test split with stratification to maintain class balance in both sets
    train_idx, test_idx = train_test_split(
        sampled_indices,
        test_size=test_size_pct / 100.0,
        shuffle=True,
        stratify=sampled_labels,
        random_state=42,
    )

    # Wrap the original dataset with RemappedSubset
    train_ds = RemappedSubset(dataset, train_idx, label_map)
    test_ds = RemappedSubset(dataset, test_idx, label_map)

    pin_memory = device.type == "cuda"

    # Create deterministic dataloaders if requested, otherwise use standard dataloaders with shuffling.
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
# Parameter counting and model summary printing
# -----------------------------------------------------------------------------
def count_params(module):
    """
    Count the total and trainable parameters in a given classical module.
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model):
    """
    Print a compact summary of the hybrid model.

    - The classical side contains the trainable PyTorch parameters.
    - The quantum side contributes circuit structure, observables, and
      computational cost
    
    *Does not introduce separate registered nn.Parameters*

    The quantum component increases representational structure and
    simulation cost without adding a large bank of stored weights.
    """
    # Count classical parameters and trainable parameters
    classical_total, classical_trainable = count_params(model.classical)
    total_params, trainable_params = count_params(model)

    # Extract quantum circuit details from the model's quantum runner
    runner = model.quantum.runner
    qubit_count = runner.qubit_count
    circuit_angles = runner.param_count
    observable_count = len(runner.hamiltonians)

    print("\n=== Hybrid QNN Summary ===")
    print(f"Target digits:              {TARGET_DIGITS}")
    print(f"Class count:               {len(TARGET_DIGITS)}")
    print(f"Qubit count:               {qubit_count}")
    print(f"Circuit angles / sample:   {circuit_angles}")
    print(f"Observable outputs:        {observable_count}")
    print(f"Entangling gates:          {max(qubit_count - 1, 0)}")
    print(f"Rotation gates:            {2 * qubit_count}")
    print(f"Shift:                     {SHIFT}")
    print(f"Sample count:              {SAMPLE_COUNT}")
    print(f"Test split (%):            {TEST_SIZE_PCT}")
    print(f"Batch size:                {BATCH_SIZE}")
    print(f"Epochs:                    {EPOCHS}")
    print(f"Eval every:                {EVAL_EVERY}")
    print(f"Learning rate:             {LR}")
    print(f"Weight decay:              {WEIGHT_DECAY}")
    print(f"Dropout:                   {DROPOUT}")
    print(f"Device:                    {DEVICE}")
    print(f"CUDA-Q target setting:     {CUDA_DEVICE}")

    print("\n--- Parameter / circuit summary ---")
    print(f"Classical trainable params:        {classical_trainable:,}")
    print(f"Shared trainable quantum params:   0")
    print(f"Quantum circuit angles / sample:   {circuit_angles}")
    print(f"Observable outputs:                {observable_count}")
    print(f"Qubit count:                       {qubit_count}")

    print("\n--- Quantum cost summary ---")
    print(f"observe() calls / forward pass:         {observable_count}")
    print(f"observe() calls / backward pass:        {2 * observable_count}")
    print(f"observe() calls / train batch total:    {3 * observable_count}")

def print_circuit_diagram(model):
    """
    Print the quantum circuit diagram for the model's quantum runner.
    """
    runner = model.quantum.runner
    theta_preview = np.zeros(runner.param_count, dtype=np.float64)

    print("\n=== Circuit diagram ===")
    try:
        print(cudaq.draw(runner.kernel, runner.qubit_count, theta_preview))
    except Exception as exc:
        print(f"Could not render circuit diagram with cudaq.draw: {exc}")

# -----------------------------------------------------------------------------
# Quantum circuit wrapper
# -----------------------------------------------------------------------------
def build_observables(qubit_count: int):
    """
    Construct the observable set used to read out the quantum circuit.

    The quantum circuit itself prepares a parameterized state, but the model
    still needs classical-valued outputs for classification. Those outputs are
    obtained as expectation values of selected Pauli observables.

    Design rationale:
        - Single-qubit Z terms measure local polarization in the computational basis.
          These terms capture whether each qubit is biased toward |0> or |1>.
        - Two-qubit ZZ terms measure pairwise correlations in the computational basis.
          These terms help detect whether two qubits tend to align or anti-align.
        - Two-qubit XX terms measure correlations in the X basis, which adds sensitivity
          to phase/coherence information that Z-only measurements would miss.

    For this project, the number of observables == the number of target classes:
        1 qubit -> 2 observables -> 2 classes
        2 qubits -> 4 observables -> 4 classes
        3 qubits -> 8 observables -> 8 classes

    This makes the quantum layer output one score per class, so the expectation
    values can be passed directly into CrossEntropyLoss as class logits.

    The expectation values lie in [-1, 1]. They are not probabilities; they are
    treated as class scores that the loss function can separate during training.
    """
    if qubit_count == 1:
        return [
            spin.z(0),  # Local computational-basis polarization
            spin.x(0),  # Local X-basis polarization (adds phase/coherence sensitivity)
        ]
    elif qubit_count == 2:
        return [
            spin.z(0),  # Local computational-basis polarization
            spin.z(1),  # Local computational-basis polarization
            spin.z(0) * spin.z(1),  # Pairwise ZZ correlation
            spin.x(0) * spin.x(1),  # Pairwise XX correlation
        ]
    elif qubit_count == 3:
        return [
            spin.z(0),                  # Local Z polarization of qubit 0
            spin.z(1),                  # Local Z polarization of qubit 1
            spin.z(2),                  # Local Z polarization of qubit 2
            spin.z(0) * spin.z(1),      # Z-basis correlation between qubits 0 and 1
            spin.z(1) * spin.z(2),      # Z-basis correlation between qubits 1 and 2
            spin.z(0) * spin.z(2),      # Z-basis correlation between qubits 0 and 2
            spin.x(0) * spin.x(1),      # X-basis correlation between qubits 0 and 1
            spin.x(1) * spin.x(2),      # X-basis correlation between qubits 1 and 2
        ]
    else:
        raise ValueError("Only qubit counts 1, 2, and 3 are supported.")
    
class QuantumCircuitRunner:
    """
    Build and evaluate the variational quantum circuit used as the quantum part
    of the hybrid classifier.

    Circuit structure:
        1. Apply one RY rotation to each qubit.
        2. Apply nearest-neighbor controlled-X gates to entangle adjacent qubits.
        3. Apply one RX rotation to each qubit.

    Rotation Layers:
        - The first RY layer prepares a tunable superposition on each qubit.
        - The entangling chain allows information to spread across qubits so the
          circuit can represent correlated features rather than independent ones.
        - The second RX layer adds another rotation axis after entanglement, which
          increases expressivity compared to using only one axis of rotation.

    The circuit is a small hardware-efficient ansatz:
    local rotations create flexible single-qubit states, and the controlled-X
    gates introduce multi-qubit correlations.

    The number of circuit angles is 2 * qubit_count because each qubit receives:
        - one RY parameter in the first layer
        - one RX parameter in the second layer
    """
    def __init__(self, qubit_count: int):
        @cudaq.kernel
        def kernel(qubit_count: int, thetas: np.ndarray):
            qubits = cudaq.qvector(qubit_count)

            """
            First rotation layer:
                Apply RY to every qubit so each qubit starts in a tunable
                superposition. Starting with local rotations gives the circuit
                learnable single-qubit degrees of freedom before entanglement.
            """
            for i in range(qubit_count):
                ry(thetas[i], qubits[i])

            """
            Nearest-neighbor entanglement:
                Controlled-X gates correlate adjacent qubits, allowing the circuit
                to represent joint features that can't be written as independent
                single-qubit states.
            """
            for i in range(qubit_count - 1):
                x.ctrl(qubits[i], qubits[i + 1])

            """
            Second rotation layer:
                Apply RX after entanglement to add another trainable rotation axis.
                Using both RY and RX makes the ansatz more expressive than a single
                repeated axis, while still keeping the circuit shallow.
            """
            for i in range(qubit_count):
                rx(thetas[qubit_count + i], qubits[i])

        self.kernel = kernel
        self.qubit_count = qubit_count
        self.param_count = 2 * qubit_count
        self.hamiltonians = build_observables(qubit_count)

    def run(self, theta_vals: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the circuit on a batch of parameter vectors and return one
        expectation value per observable.

        Each row of theta_vals represents one sample-specific set of circuit
        angles produced by the classical network. For each sample, the circuit
        is executed and measured against all observables in self.hamiltonians.

        The returned tensor has shape:
            (batch_size, number_of_observables)

        - each row corresponds to one input image
        - each column is the expectation value of one observable
        - these expectation values serve as the output scores of the quantum layer

        Because the observable count is matched to the class count, the output
        can be used directly as the logits for multiclass classification.
        """
        # Convert the input tensor to a NumPy array for CUDA-Q, and batch qubits
        theta_np = theta_vals.detach().cpu().numpy()
        batch_qubits = [self.qubit_count] * theta_np.shape[0]

        outputs = []

        # For each observable, run the circuit on the full batch of parameter sets and collect the expectation values.
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
    """
    Custom PyTorch autograd bridge for the quantum circuit.

    CUDA-Q circuit evaluations are not automatically differentiable by PyTorch,
    so this class defines how the quantum layer participates in backpropagation.

    - forward: run the parameterized quantum circuit and return its observable
      expectation values
    - backward: compute gradients with respect to the circuit angles using the
      parameter-shift rule

    The quantum circuit behaves like a differentiable layer inside the
    larger hybrid neural network.
    """

    @staticmethod
    def forward(ctx, thetas: torch.Tensor, runner: QuantumCircuitRunner, shift: float):
        """
        Evaluate the quantum circuit for the current batch of circuit angles.

        Parameters:
            thetas: batch of sample-specific circuit parameters produced by the
                    classical network
            runner: helper object that executes the CUDA-Q circuit and measures
                    the selected observables
            shift: parameter-shift amount used later in the backward pass

        Forward Pass:
            1. stores the circuit runner and shift value in the autograd context
            2. saves the input angles so they are available during backpropagation
            3. returns the quantum layer outputs as observable expectation values

        The outputs act as the class scores passed to the loss function.
        """
        ctx.runner = runner
        ctx.shift = float(shift)
        ctx.save_for_backward(thetas)
        return runner.run(thetas)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using a vectorized parameter-shift rule.

        Instead of shifting one parameter at a time in a Python loop, this
        implementation constructs all +shift and -shift parameter sets in two
        larger batches. This preserves the same gradient rule while reducing
        Python overhead.

        Output shape logic:
            - deriv has shape (batch, parameter, observable)
            - grad_output has shape (batch, observable)
            - contracting over the observable dimension gives gradients with respect
              to each circuit parameter for each sample
        """
        (thetas,) = ctx.saved_tensors
        batch_size, param_count = thetas.shape
        shift = ctx.shift

        """
        Vectorized parameter-shift:
            Instead of 2 * P separate quantum runs, build all shifted parameter sets
            in two large batches and call the circuit runner only twice.
        """
        # Create identity matrix to add/subtract shift to each parameter independently
        eye = torch.eye(param_count, device=thetas.device, dtype=thetas.dtype).unsqueeze(0)
        theta_expanded = thetas.unsqueeze(1)  # (B, 1, P)

        # Construct the +shift and -shift parameter sets for all parameters
        plus = (theta_expanded + shift * eye).reshape(batch_size * param_count, param_count)
        minus = (theta_expanded - shift * eye).reshape(batch_size * param_count, param_count)

        # Evaluate the circuit on all shifted parameter sets in two large batches
        exp_plus = ctx.runner.run(plus).reshape(batch_size, param_count, -1)
        exp_minus = ctx.runner.run(minus).reshape(batch_size, param_count, -1)

        # Compute the parameter-shift gradient for each parameter and sample
        deriv = (exp_plus - exp_minus) / (2.0 * shift)  # (B, P, O)
        grad_thetas = (grad_output.unsqueeze(1) * deriv).sum(dim=2)  # (B, P)

        return grad_thetas, None, None


class QuantumLayer(nn.Module):
    """
    PyTorch module that wraps the custom quantum autograd function.

    This layer provides the interface between the classical network and the
    quantum circuit. It accepts a batch of circuit angles produced by the
    classical front end, sends them through the quantum circuit, and returns
    the resulting observable expectation values.

    Unlike a standard neural network layer, this module does not store its own
    trainable weight tensors. The trainable parameters live in the classical 
    network, which learns how to generate useful circuit angles for each input 
    sample.
    """
    def __init__(self, qubit_count: int, shift: float):
        super().__init__()
        self.runner = QuantumCircuitRunner(qubit_count)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the differentiable quantum layer to a batch of circuit parameters.

        Parameters:
            inputs: batch of sample-specific circuit angles generated by the
                    classical network

        Returns:
            a batch of observable expectation values, with one output vector per
            input sample

        This method delegates the actual circuit evaluation and gradient handling
        to QuantumAutograd so the quantum circuit can participate in end-to-end
        PyTorch training.
        """
        return QuantumAutograd.apply(inputs, self.runner, self.shift)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class HybridQNN(nn.Module):
    """
    Hybrid classical-quantum neural network for MNIST classification.

    Architecture:
        - A classical feedforward network first compresses the 28x28 image into a
          low-dimensional parameter vector.
        - The size of that vector is 2 * qubit_count so it matches the number of
          trainable rotation angles used by the quantum circuit.
        - The quantum layer then maps those circuit angles to expectation values of
          the chosen observables, producing one output score per class.

    This design separates the roles of the two model components:
        - the classical front end performs feature extraction from pixel space
        - the quantum back end acts as a small nonlinear feature map/readout layer
    """
    def __init__(self, qubit_count: int, shift: float):
        super().__init__()

        param_count = 2 * qubit_count

        self.classical = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, param_count),
        )

        self.quantum = QuantumLayer(qubit_count, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert the input image into circuit angles with the classical network,
        then pass those angles through the quantum layer to obtain class scores.
        """
        theta_params = self.classical(x)
        return self.quantum(theta_params)


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------
# Loader distribution printing and evaluation functions
def print_loader_distribution(loader, name):
    counts = torch.zeros(len(TARGET_DIGITS), dtype=torch.long)
    for _, y in loader:
        counts += torch.bincount(y.cpu(), minlength=len(TARGET_DIGITS))

    print(f"\n{name} distribution:")
    for cls, count in enumerate(counts.tolist()):
        print(f"Class {cls} ({TARGET_DIGITS[cls]}): {count} samples")

# Evaluation on test set
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
    """
    Train the hybrid model with AdamW and monitor test performance periodically.

    Early stopping is based on test loss rather than test accuracy. This is a
    common choice because loss is more sensitive than accuracy to small changes
    in classifier confidence, especially once accuracy begins to plateau.

    The best model weights are restored at the end of training so the final
    evaluation reflects the best observed test-loss checkpoint rather than
    simply the last epoch.
    """
    signal.signal(signal.SIGINT, handle_sigint)
    atexit.register(cleanup)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # because lower test_loss is better
        factor=0.5,      # cut LR in half when plateauing
        patience=SCHEDULER_PATIENCE,      # number of eval periods with no improvement
        threshold=1e-3,  # ignore tiny meaningless changes
        min_lr=1e-5,
    )

    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    eval_epochs = []

    # Only print the progress bar if running in an interactive terminal
    use_tqdm = sys.stderr.isatty()
    progress = trange(epochs, desc="Training", leave=True, disable=not use_tqdm)

    patience = EARLY_STOP_PATIENCE
    best_test_loss = float("inf")
    best_test_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    eval_epoch_start_time = None

    # Main training loop
    for epoch in progress:
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        

        if STOP_REQUESTED:
            print("Stopping before next epoch.")
            break

        for x, y in train_loader:
            if STOP_REQUESTED:
                break

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

        if use_tqdm:
            progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
            )

        # Evaluation loop every eval_every epochs + first and last
        if (epoch + 1) % eval_every == 0 or epoch == 0 or (epoch + 1) == epochs:
            test_loss, test_acc, _, _ = evaluate(model, test_loader, loss_fn)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            eval_epochs.append(epoch + 1)

            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Early stop on loss block
            if EARLY_STOP_METRIC == "loss":
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    status = "improved"
                else:
                    patience_counter += 1
                    status = f"no_improve={patience_counter}, best={best_test_loss:.4f}@{best_epoch}"
            # Early stop on accuracy block
            elif EARLY_STOP_METRIC == "accuracy":
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    status = "improved"
                else:
                    patience_counter += 1
                    status = f"no_improve={patience_counter}, best={best_test_acc:.4f}@{best_epoch}"
            
            # Eval epoch time
            eval_epoch_end_time = time.perf_counter()

            if eval_epoch_start_time is None:
                epoch_eval_time = 0.0
            else:
                epoch_eval_time = eval_epoch_end_time - eval_epoch_start_time

            eval_epoch_start_time = eval_epoch_end_time

            # Eval print message
            msg = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f} | "
                f"lr={current_lr:.2e} | {status} | {epoch_eval_time:.1f}s"
            )

            # Update tqdm description and print message
            if use_tqdm:
                progress.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    train_acc=f"{train_acc:.4f}",
                    test_loss=f"{test_loss:.4f}",
                    test_acc=f"{test_acc:.4f}",
                )
                tqdm.write(msg)
            else:
                print(msg)

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

            
    
    # Restore best model state if available
    if best_state is not None:
        model.load_state_dict(best_state)
        if EARLY_STOP_METRIC == "loss":
            print(
                f"Restored best model from epoch {best_epoch} "
                f"with test_loss={best_test_loss:.4f}"
            )
        else:
            print(
                f"Restored best model from epoch {best_epoch} "
                f"with test_acc={best_test_acc:.4f}"
            )

    return {
        "train_loss": train_loss_hist,
        "train_acc": train_acc_hist,
        "test_loss": test_loss_hist,
        "test_acc": test_acc_hist,
        "eval_epochs": eval_epochs,
        "loss_fn": loss_fn,
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
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

def save_metrics_plot(history, path=f"optimized_metrics_{QUBIT_COUNT}q_{len(TARGET_DIGITS)}c.png"):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train")
    plt.plot(
        history["eval_epochs"],
        history["test_loss"],
        marker="o",
        label="Test",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history["train_acc"]) + 1), history["train_acc"], label="Train")
    plt.plot(
        history["eval_epochs"],
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


def save_confusion_matrix(y_true, y_pred, path=f"optimized_confusion_matrix_{QUBIT_COUNT}q_{len(TARGET_DIGITS)}c.png"):
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
    print_model_summary(model)

    start_time = time.time()

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        eval_every=EVAL_EVERY,
    )

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, history["loss_fn"])

    end_time = time.time()
    elapsed_time = end_time - start_time

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

    print_circuit_diagram(model)

    print(f"\nTotal training time: {elapsed_time:.2f}s")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)