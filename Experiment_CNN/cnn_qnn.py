"""
Course: Quantum Computing - CS-5463-907
Term: Spring 2026
Group: Quantum Machine Learning Group 2 
Project: Multi-Qubit Expressibility
Authors: Devin Marinelli, Jonathan Miller, John Schneider, Keeban Villarreal

Project Summary:
This project explores the expressibility of variational quantum circuits as the number of qubits 
increases. We implement a hybrid classical-quantum neural network (QNN) for classifying subsets of 
the MNIST dataset. The quantum component consists of a parameterized circuit with local rotations 
and nearest-neighbor entanglement, and a structured set of observables that serve as quantum 
features. The classical front end learns to generate sample-specific circuit angles, and a final 
linear readout layer maps the quantum features to class logits.
"""

# General purpose imports
import math
from pathlib import Path
import os
import sys
import time
import random
import copy
import argparse
from tqdm import trange, tqdm
from typing import Optional

# Interrupt handling imports
import signal
import atexit

# Quantum and machine learning imports
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

# Visualization imports
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
# -----------------------------------------------------------------------------
# Interrupt Handling Setup
# -----------------------------------------------------------------------------
# SIGINT handler
def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nCtrl+C received. Stopping after current batch...")

# Cleanup function to be called on exit
def cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Global Config
# -----------------------------------------------------------------------------
# Quantum circuit settings
SHIFT = math.pi / 2

# Class value presets for MNIST
PRESETS={1: [5, 6],
         2: [3, 4, 5, 6],
         3: [1, 2, 3, 4, 5, 6, 7, 8],
         4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

# Training Hyperparameters
TEST_SIZE_PCT = 30

# Device and Determinism Settings
SET_DETERMINISTIC = True
SEED_CUDAQ = 44
SEED = 22

# Global variable initializations
STOP_REQUESTED = False

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
cudaq.set_random_seed(SEED_CUDAQ)

# Set deterministic behavior for reproducibility
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

# Seed worker function for DataLoader workers if used
def seed_worker(worker_id):
            worker_seed = SEED + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

# -----------------------------------------------------------------------------
# Backend selection
# -----------------------------------------------------------------------------
def configure_backend(classical_choice: str, cudaq_choice: str):

    if classical_choice == "cuda" and torch.cuda.is_available():
        try:
            cudaq.set_target(cudaq_choice)
            device = torch.device(classical_choice)
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


def build_dataloaders(target_digits, test_size_pct, device, args):
    """
    Build deterministic train/test dataloaders for the selected subset of MNIST.

    Only the digits listed in target_digits are retained. Their labels are then
    remapped to contiguous class indices 0..N-1 so they are compatible with
    CrossEntropyLoss.

    The split is stratified so that each class is represented proportionally in
    both the training and test sets.
    """
    batch_size = args.batch_size
    sample_count = args.samples

    # Standard MNIST normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    
    # Load the full MNIST dataset and filter it down to the selected digits.
    dataset = datasets.MNIST(
        root=str(args.data_dir),
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
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(args.num_workers > 0),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
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


def print_model_summary(model, device, target_digits, args):
    """
    Print a compact summary of the hybrid model.

    - The classical side contains the trainable PyTorch parameters.
    - The quantum side contributes circuit structure, observables, and
      computational cost

    The quantum component increases representational structure and
    simulation cost but does not introduce separate registered nn.Parameters.
    """
    # Count classical parameters and trainable parameters
    _, classical_trainable = count_params(model.classical)
    num_classes = len(target_digits)

    # Extract quantum circuit details from the model's quantum runner
    runner = model.quantum.runner
    qubit_count = runner.qubit_count
    circuit_angles = runner.param_count
    observable_count = len(runner.hamiltonians)

    print("\n=== Hybrid QNN Summary ===")
    print(f"Target digits:              {target_digits}")
    print(f"Class count:               {num_classes}")
    print(f"Qubit count:               {qubit_count}")
    print(f"Circuit angles / sample:   {circuit_angles}")
    print(f"Quantum features:        {observable_count}")
    print(f"Entangling gates:          {max(qubit_count - 1, 0)}")
    print(f"Rotation gates:            {2 * qubit_count}")
    print(f"Shift:                     {SHIFT}")
    print(f"Sample count:              {args.samples}")
    print(f"Test split (%):            {TEST_SIZE_PCT}")
    print(f"Batch size:                {args.batch_size}")
    print(f"Epochs:                    {args.epochs}")
    print(f"Eval every:                {(args.epochs // 10) if args.eval_every is None else args.eval_every}")
    print(f"Learning rate:             {args.lr}")
    print(f"Weight decay:              {args.weight_decay}")
    print(f"Dropout:                   {args.dropout}")

    print("\n--- Parameter / circuit summary ---")
    print(f"Classical trainable params:        {classical_trainable:,}")
    print(f"Shared trainable quantum params:   0")
    print(f"Quantum circuit angles / sample:   {circuit_angles}")
    print(f"Quantum features:                  {observable_count}")
    print(f"Final readout classes:             {num_classes}")
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
def pauli_product(axis: str, indices: list[int]):
    """
    Build a multi-qubit Pauli product operator over the requested qubit indices.

    This allows for global parity-like observables be generated dynamically 
    for any qubit count instead of being manually written case by case and 
    keeps the observable builder readable and scalable.

    Parameters:
        axis: Which Pauli axis to use. Supported values are "x" and "z".
        indices: The qubit indices included in the product.

    Returns:
        A CUDA-Q SpinOperator representing the requested Pauli product.
    """
    if not indices:
        raise ValueError("indices must contain at least one qubit index.")

    # Example pauli_product("z", [0, 1, 2]) -> Z0 * Z1 * Z2
    if axis == "z":
        op = spin.z(indices[0])
        for i in indices[1:]:
            op = op * spin.z(i)
        return op

    # Example pauli_product("x", [0, 1, 2, 3]) -> X0 * X1 * X2 * X3
    elif axis == "x":
        op = spin.x(indices[0])
        for i in indices[1:]:
            op = op * spin.x(i)
        return op

    else:
        raise ValueError("axis must be 'x' or 'z'")


def build_observables(qubit_count: int, max_features: Optional[int] = None):
    """
    Dynamically construct a quantum feature set for the variational circuit.

    The observables are treated as quantum features. The final linear readout layer
    learns how to combine those quantum features into class logits.

    Feature-generation policy:
        Tier 1: Local Z terms
            - one per qubit
            - captures each qubit's local computational-basis bias

        Tier 2: Local X terms
            - one per qubit
            - adds local phase/coherence-sensitive information

        Tier 3: Nearest-neighbor ZZ terms
            - one for each adjacent qubit pair
            - captures local pairwise correlations in the Z basis

        Tier 4: Nearest-neighbor XX terms
            - one for each adjacent qubit pair
            - captures local pairwise correlations in the X basis

        Tier 5: Global parity-like terms
            - Z⊗...⊗Z
            - X⊗...⊗X
            - adds one coarse global view of the whole multi-qubit state

    A tiered policy:
        - Keeps the observable builder fully dynamic in qubit_count.
        - Avoids the exponential blow-up of trying to include the full Pauli space.
        - Matches the actual circuit structure: local rotations + nearest-neighbor
          entangling chain.
        - Gives a reasonable balance of local, pairwise, and global information.

    Default feature budget:
        If max_features is not provided, use 2**qubit_count features. This gives
        the quantum layer a feature budget on the same order as the computational
        basis size without attempting full state tomography.

    Parameters:
        qubit_count: Number of qubits in the circuit.
        max_features: Optional cap on the number of observables/features returned.

    Returns:
        A list of CUDA-Q SpinOperator observables to be measured by the quantum layer.
    """
    if qubit_count < 1:
        raise ValueError("qubit_count must be at least 1.")

    if max_features is None:
        max_features = 2 ** qubit_count

    observables = []

    def append_observable(op):
        """
        Append an observable only if the feature budget has not been reached.
        Returns True if the budget is full after appending, otherwise False.
        """
        observables.append(op)
        return len(observables) >= max_features

    # -------------------------------------------------------------------------
    # Tier 1: local Z features
    # -------------------------------------------------------------------------
    for i in range(qubit_count):
        if append_observable(spin.z(i)):
            return observables

    # -------------------------------------------------------------------------
    # Tier 2: local X features
    # -------------------------------------------------------------------------
    for i in range(qubit_count):
        if append_observable(spin.x(i)):
            return observables

    # -------------------------------------------------------------------------
    # Tier 3: nearest-neighbor ZZ features
    # -------------------------------------------------------------------------
    for i in range(qubit_count - 1):
        if append_observable(spin.z(i) * spin.z(i + 1)):
            return observables

    # -------------------------------------------------------------------------
    # Tier 4: nearest-neighbor XX features
    # -------------------------------------------------------------------------
    for i in range(qubit_count - 1):
        if append_observable(spin.x(i) * spin.x(i + 1)):
            return observables

    # -------------------------------------------------------------------------
    # Tier 5: global parity-like features
    # -------------------------------------------------------------------------
    if append_observable(pauli_product("z", list(range(qubit_count)))):
        return observables

    if append_observable(pauli_product("x", list(range(qubit_count)))):
        return observables

    return observables
    
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
        
        Parameters:
            theta_vals: a batch of circuit angle vectors with shape
                        (batch_size, 2 * qubit_count)

        Return shape:
            (batch_size, number_of_observables)

        Interpretation:
            - each row corresponds to one input image
            - each column is one quantum feature
            - each feature is the expectation value of a selected observable

        In the updated architecture, these quantum features are passed into a final
        classical readout layer, which learns how to map them to class logits.
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

    - forward:  run the parameterized quantum circuit and return its observable
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
            shift:  parameter-shift amount used later in the backward pass

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
        constructs all +shift and -shift parameter sets in two larger batches. 
        This preserves the same gradient rule while reducing Python overhead.

        Parameters:
            grad_output: the gradient of the loss with respect to the quantum
                         layer outputs 
                        
        Output shape logic:
            - deriv has shape (batch, parameter, observable)
            - grad_output has shape (batch, observable)
            - contracting over the observable dimension gives gradients with
              respect to each circuit parameter for each sample
        
        Backward Pass:
            1. retrieves the saved input angles and shift value from the context
            2. constructs the shifted parameter sets for all parameters in two
               large batches (one for +shift and one for -shift)
            3. evaluates the circuit on all shifted parameter sets to get the
               corresponding expectation values
            4. applies the parameter-shift formula to compute the gradient of
               each observable with respect to each circuit parameter
            5. combines those gradients with grad_output to get the final
               gradient with respect to the input angles
        """
        (thetas,) = ctx.saved_tensors
        batch_size, param_count = thetas.shape
        shift = ctx.shift

        """
        Vectorized parameter-shift:
            Instead of 2 * P separate quantum runs, build all shifted parameter 
            sets in two large batches and call the circuit runner only twice.
        """
        # Create identity matrix to add/subtract shift to each parameter independently
        eye = torch.eye(param_count, device=thetas.device, dtype=thetas.dtype).unsqueeze(0)
        theta_expanded = thetas.unsqueeze(1)

        # Construct the +shift and -shift parameter sets for all parameters
        plus = (theta_expanded + shift * eye).reshape(batch_size * param_count, param_count)
        minus = (theta_expanded - shift * eye).reshape(batch_size * param_count, param_count)

        # Evaluate the circuit on all shifted parameter sets in two large batches
        exp_plus = ctx.runner.run(plus).reshape(batch_size, param_count, -1)
        exp_minus = ctx.runner.run(minus).reshape(batch_size, param_count, -1)

        # Compute the parameter-shift gradient for each parameter and sample
        deriv = (exp_plus - exp_minus) / (2.0 * shift)
        grad_thetas = (grad_output.unsqueeze(1) * deriv).sum(dim=2)

        return grad_thetas, None, None


class QuantumLayer(nn.Module):
    """
    PyTorch module that wraps the custom quantum autograd function.

    The interface between the classical network and the quantum circuit. 
    It accepts a batch of circuit angles produced by the classical front 
    end, sends them through the quantum circuit, and returns the resulting
    observable expectation values.

    The quantum layer does not store its own trainable weight tensors. The 
    trainable parameters live in the classical network, which learns how to 
    generate useful circuit angles for each input sample.
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

    Flow:
        image -> classical network -> circuit angles -> quantum layer ->
        quantum features -> final readout layer -> class logits

    Design:
        - The classical front end compresses the image into a small set of
          circuit angles.
        - The quantum circuit transforms those angles into a richer feature
          representation by measuring a structured set of observables.
        - A final linear readout layer maps the quantum feature vector to the
          required number of class logits.
    """
    def __init__(self, qubit_count: int, shift: float, dropout: float, num_classes: int):
        super().__init__()

        """
        Number of trainable circuit angles:
            - one RY angle + one RX angle per qubit
        """
        param_count = 2 * qubit_count

        # Number of quantum features returned by the observable builder
        quantum_feature_count = len(build_observables(qubit_count))

        """
        Classical CNN encoder:
            Maps the 28x28 image to a small vector of circuit angles.
            """
        self.classical = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, param_count),
        )

        """
        Quantum feature extractor:
            Runs the parameterized quantum circuit and returns a vector of
            observable expectation values.
            """
        self.quantum = QuantumLayer(qubit_count, shift)

        """
        Final classical readout:
            Learns how to combine the quantum features into class logits.
        """
        self.readout = nn.Linear(quantum_feature_count, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Steps:
            1. Encode the image into circuit angles.
            2. Evaluate the quantum circuit to obtain quantum features.
            3. Map those features to class logits with the final readout layer.
        """
        theta_params = self.classical(x)
        q_features = self.quantum(theta_params)
        logits = self.readout(q_features)
        return logits


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------
# Loader distribution printing and evaluation functions
def print_loader_distribution(loader, name, digit_list):
    num_classes = len(digit_list)
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in loader:
        counts += torch.bincount(y.cpu(), minlength=num_classes)

    print(f"\n{name} distribution:")
    for cls, count in enumerate(counts.tolist()):
        print(f"Class {cls} ({digit_list[cls]}): {count} samples")

# Evaluation on test set
@torch.inference_mode()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    y_true_all = []
    y_pred_all = []

    # Evaluation loop
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

    # Evaluation metric calculation
    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)
    return avg_loss, avg_acc, y_true_all, y_pred_all

# Build selected optimizer
def build_optimizer(model, args):
    """
    Build the optimizer based on the selection.
    """
    name = args.optimizer.lower()

    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif name == "radam":
        return optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif name == "nadam":
        return optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}. Please choose one of the following from the Adam family: AdamW, Adam, RAdam, or NAdam.")

def train_model(model, train_loader, test_loader, args):
    """
    Train the hybrid model with AdamW and monitor test performance periodically.

    Early stopping is based on test loss rather than test accuracy. This is a
    common choice because loss is more sensitive than accuracy to small changes
    in classifier confidence, especially once accuracy begins to plateau.

    The best model weights are restored at the end of training so the final
    evaluation reflects the best observed test-loss checkpoint rather than
    simply the last epoch.
    """
    # Setup for graceful interruption and cleanup
    signal.signal(signal.SIGINT, handle_sigint)
    atexit.register(cleanup)

    # Training configuration
    epochs = args.epochs
    eval_every = args.epochs // 10 if args.eval_every is None else args.eval_every
    optimizer = build_optimizer(model, args)
    loss_fn = nn.CrossEntropyLoss()

    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # because lower test_loss is better
        factor=0.5,      # cut LR in half when plateauing
        patience=args.scheduler_patience,      # number of eval periods with no improvement
        threshold=1e-3,  # ignore tiny meaningless changes
        min_lr=1e-5,
    )

    # Histories for plotting and analysis
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    eval_epochs = []

    # Only print the progress bar if running in an interactive terminal
    use_tqdm = sys.stderr.isatty()
    progress = trange(epochs, desc="Training", leave=True, disable=not use_tqdm)

    # Early stopping variables
    patience = args.early_stop_patience
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
        
        # Graceful interruption check before starting the epoch
        if STOP_REQUESTED:
            print("Stopping before next epoch.")
            break
        
        # Mini-batch training loop
        for x, y in train_loader:
            if STOP_REQUESTED:
                break
            
            # Zero gradients, run forward pass, compute loss, backpropagate, and update weights
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            # Compute training metrics for this batch and accumulate for the epoch
            preds = logits.argmax(dim=1)
            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == y).sum().item()
            running_examples += batch_size

        # Check for graceful interruption after the epoch
        if STOP_REQUESTED:
            print("Training interrupted cleanly.")
            break
        
        # Epoch-level training metrics
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

            # Step the scheduler based on test loss
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Early stop on loss block
            if args.early_stop_metric == "loss":
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
            elif args.early_stop_metric == "accuracy":
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
        if args.early_stop_metric == "loss":
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
# Per-class accuracy calculation
def per_class_accuracy(y_true, y_pred, num_classes):
    results = {}
    for cls in range(num_classes):
        cls_mask = y_true == cls
        cls_total = cls_mask.sum().item()
        cls_correct = ((y_pred == cls) & cls_mask).sum().item()
        results[cls] = cls_correct / cls_total if cls_total > 0 else 0.0
    return results

# Loss and accuracy curves plotting
def save_metrics_plot(history, save_dir, qubit_count, digit_list_length):
    path=f"{save_dir}/optimized_metrics_{qubit_count}q_{digit_list_length}c.png"
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

# Confusion matrix plotting
def save_confusion_matrix(y_true, y_pred, save_dir, qubit_count, digit_list):
    num_classes = len(digit_list)
    path=f"{save_dir}/optimized_confusion_matrix_{qubit_count}q_{num_classes}c.png"
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(range(num_classes), digit_list)
    plt.yticks(range(num_classes), digit_list)
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
    # Command-line argument parsing
    parser = argparse.ArgumentParser()

    # Quantum hyperparameters
    parser.add_argument("--num_qubits", type=int, default=1, choices=[1, 2, 3, 4], help="Number of qubits in the quantum circuit.")
    parser.add_argument("--samples", type=int, default=4000, help="Number of samples to use from the dataset.")
    
    # Data and device hyperparameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to download/load the MNIST dataset. Generally in './data' in the project root.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save outputs like metrics plots and confusion matrices.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for classical component.")
    parser.add_argument("--q_device", type=str, default="qpp-cpu", choices=["qpp-cpu", "cuda"], help="CUDA-Q target device.")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluate on the test set every N epochs.")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "RAdam", "NAdam"], help="Optimizer to use for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization) factor for the optimizer.")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate for the classical encoder.")
    
    # Early stop and scheduler hyperparameters
    parser.add_argument("--early_stop_metric", type=str, default="loss", choices=["loss", "accuracy"], help="Metric to monitor for early stopping.")
    parser.add_argument("--early_stop_patience", type=int, default=1, help="Number of evaluation periods with no improvement before early stopping.")
    parser.add_argument("--scheduler_patience", type=int, default=0, help="Number of evaluation periods with no improvement before reducing learning rate.")

    args = parser.parse_args()

    # Set global constants based on parsed arguments
    device = configure_backend(args.device, args.q_device)

    # Create the save directory if it doesn't exist
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Print device and optimizer configuration summary
    print(f"Using CUDA-Q target: {args.q_device}")
    print(f"Using device: {device}")
    print(f"Optimizer: {args.optimizer}")

    # Set deterministic behavior for reproducibility
    if SET_DETERMINISTIC:
        seed_everything(22)

    # Select MNIST digit subset based on qubit count
    target_digits = PRESETS[args.num_qubits]
    num_classes = len(target_digits)

    # Build the quantum feature set
    observables = build_observables(args.num_qubits)

    # Observable count must be at least equal to the number of classes
    if len(observables) < num_classes:
        raise ValueError(
            f"Need at least as many quantum features as classes. "
            f"Got {len(observables)} observables for {num_classes} classes."
        )

    # Dataloaders
    train_loader, test_loader = build_dataloaders(
        target_digits=target_digits,
        test_size_pct=TEST_SIZE_PCT,
        device=device,
        args=args
    )

    # Print dataset distribution summary
    print_loader_distribution(train_loader, "Train", target_digits)
    print_loader_distribution(test_loader, "Test", target_digits)

    # Build the Hybrid QNN and print the model summary
    model = HybridQNN(args.num_qubits, SHIFT, args.dropout, num_classes).to(device)
    print_model_summary(model, device, target_digits, args)

    # Start time of training loop
    start_time = time.time()

    # Training loop
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        args=args
    )

    # Final evaluation
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, history["loss_fn"])

    # Visualization
    # Streamlit?

    # Elapsed time calculation
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Final metrics and reporting
    print("\nFinal metrics:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    per_class = per_class_accuracy(y_true, y_pred, num_classes=num_classes)
    print("\nPer-class accuracy:")

    for cls, acc in per_class.items():
        print(f"Class {cls} ({target_digits[cls]}): {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true.numpy(), y_pred.numpy(), digits=4))

    # Save plots
    save_confusion_matrix(y_true, y_pred, args.save_dir, args.num_qubits, target_digits)
    save_metrics_plot(history, args.save_dir, args.num_qubits, num_classes)

    # Print the quantum circuit and total training time
    print_circuit_diagram(model)
    print(f"\nTotal training time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)