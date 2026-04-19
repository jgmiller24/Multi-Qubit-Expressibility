## Experiment 3 (v1c): 4-Qubit Scaling Evaluation – Refined Observable Design

### Observables
- Increased diversity across Z, X, and Y measurement axes
- Introduced higher-order correlation terms to capture more complex feature interactions
- Improved balance in observable selection to avoid dominance of a single measurement basis
- Ensured later qubits contribute meaningful correlations to the feature space

### Results
- Strong overall performance with test accuracy ≈ 86.7%
- Significant improvement in previously underperforming classes
  - Digit 8 recall improved from near-zero to ~84%
- Most classes achieved high and balanced accuracy (e.g., 0, 1, 2, 3, 6, 7, 9)
- Some remaining difficulty observed in digit 5, indicating residual feature overlap
- A notable test/train gap (~11%) is present, indicating some overfitting.

### Key Takeaways
- Increasing qubit count and circuit depth alone does not guarantee improved performance
- Observable design plays a critical role in defining the effective feature space of the quantum model
- Earlier experiments showed class-specific collapse (notably digit 8) despite sufficient circuit capacity
- By improving observable diversity and qubit participation, class separability improved substantially
- This confirms that observable selection is a primary driver of representational power in hybrid quantum neural networks