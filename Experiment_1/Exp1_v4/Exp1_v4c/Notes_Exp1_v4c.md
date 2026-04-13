
## Experiment 1 (v4c): Increased Circuit Depth – EPOCHS = 600

### Circuit Modification:
- Same architecture as v4 (increased circuit depth: 3 rotation layers + 2 entanglement layers)
- Extended training duration from 400 → 600 epochs

### Expected Impact:
- Allow deeper circuit to fully converge
- Determine whether previous performance limits were due to insufficient training
- Improve per-class separability with increased optimization time

---

### Results Summary:
- Test accuracy improved to ~78–79% (from ~70% at 400 epochs)
- Training remained stable with no signs of divergence or overfitting
- Continued improvement observed up to ~500 epochs, with diminishing gains thereafter

---

### Key Observation:
- Increasing training duration enabled deeper circuits to realize their full representational capacity
- Performance gains plateaued despite continued training beyond ~500 epochs

---

### Interpretation:
- Circuit depth is beneficial, but only when sufficiently trained
- Earlier conclusion (v4) that depth was ineffective was due to undertraining
- However, even with extended training, performance saturates below ~80%

- This suggests:
  - Optimization is no longer the limiting factor
  - Circuit expressivity alone cannot overcome performance ceiling

---

### Per-Class Behavior:
- Balanced performance across all classes (no collapse observed)
- Strong improvements in previously weaker classes (e.g., digit '5')
- Remaining errors reflect intrinsic overlap between digit classes rather than learning failure

---

### Conclusion:
- Increasing circuit depth + sufficient training improves performance significantly
- However, model converges to a performance ceiling (~75–80%)

- Primary limitation is now:
  - restricted representational capacity of the 2-qubit system

---

### Key Insight:
- Additional training yields diminishing returns beyond ~400–500 epochs
- The performance ceiling is not due to optimization, but to the limited Hilbert space (2 qubits)

---

### Next Steps:
- Compare against v5c (observable-focused approach at 600 epochs)
- Confirm whether both approaches converge to the same ceiling
- Transition to 3-qubit architecture to test representational scaling