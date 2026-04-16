## Experiment 2 (v2): Enhanced 3-Qubit QNN (Increased Circuit Expressivity)

### Circuit Configuration:
- 3-qubit architecture (same as v1)
- Two rotation layers:
  - Layer 1: RY rotations (3 parameters)
  - Entanglement: q0 → q1 → q2
  - Layer 2: RX rotations (3 parameters)
- Total parameters: 6
- Measurement:
  - Same 8 observables as v1


### Objective:
Evaluate whether increasing circuit expressivity (via additional parameterized layers) improves multi-class classification performance.


### Results Summary:
- Test accuracy: ~45–46%
- Significant improvement over v1 (~22%)
- Majority of classes now receive non-zero predictions


### Key Observations:
- Model successfully transitions from near-random to meaningful learning
- Class collapse reduced:
  - Most classes predicted at least occasionally
- However, uneven performance persists:
  - Strong classes: ~60–85% recall
  - Weak classes: ~0–20% recall


### Interpretation:
- Adding a second parameterized layer significantly improves feature extraction
- Circuit expressivity is a **dominant factor** in performance scaling
- Remaining errors suggest:
  - Limited depth still constrains separability
  - Some class overlaps require deeper representations


### Comparative Insight (v1 → v2):
| Change | Impact |
|------|--------|
| + RX layer | Major performance gain |
| + independent parameters | Improved class coverage |
| Same qubit count | Confirms depth > qubit scaling |


### Conclusion:
- Increasing circuit depth and parameterization dramatically improves performance
- 3-qubit systems can support 8-class classification, but require sufficient expressivity


### Takeaway:
> **Circuit design (depth + parameterization) is more critical than qubit count for scaling performance**