# Multi-Qubit Expressibility (CUDA-Q Project)

This project explores hybrid quantum-classical neural networks using NVIDIA CUDA-Q and PyTorch.  
The goal is to analyze how increasing qubit count affects model expressibility, performance, and training behavior.

---

## 🚀 Project Overview

We implement a hybrid quantum neural network (QNN) for MNIST classification and evaluate:

- Baseline performance (1-qubit model)
- Multi-qubit scaling (2, 3, 4+ qubits)
- Expressibility vs. classical models
- Training stability and convergence
- Computational cost vs. performance gains

---

## 📂 Project Structure
multi-qubit-expressibility/
│
├── .devcontainer/
│ ├── Dockerfile
│ └── devcontainer.json
│
├── Experiment_0/
│ ├── baseline_qnn.py
│ └── experiment0_metrics.png
│
├──Experiment_1/
│ ├── Exp1_v1/
│ ├── Exp1_v2/
│ ├── Exp1_v3/
│ ├── Exp1_v4/
│ ├── Exp1_v5/
│ └── Exp1_v6/
│
├── requirements.txt
└── README.md

---

## 🧱 Setup Instructions

### Prerequisites

- Docker Desktop (WSL2 enabled)
- Visual Studio Code
- Dev Containers extension

---

### 🔧 Setup Steps

1. Clone the repository:
git clone https://github.com/jgmiller24/multi-qubit-expressibility.git
cd multi-qubit-expressibility

2. Open the project in VS code:
code .

3. Rebuild the container:
ctl + shift + P --> Dev Containers: Rebuild and Reopen in Container.
