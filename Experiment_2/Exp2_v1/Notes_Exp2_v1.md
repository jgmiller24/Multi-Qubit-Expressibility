
## Experiment 2 (v1): Initial 3-Qubit QNN

### Circuit Configuration:
- Expanded from 2 → 3 qubits
- Single rotation layer:
  - RY applied to each qubit
- Linear entanglement:
  - q0 → q1 → q2
- Measurement:
  - 8 observables (Z and pairwise interactions)
  - Output dimension aligned with 8-class task


### Objective:
Evaluate whether increasing Hilbert space (2 → 3 qubits) improves multi-class classification performance.


### Results Summary:
- Test accuracy: ~21–22%
- Slight improvement over random baseline (~12.5%)
- Model fails to learn several classes (0% recall observed)


### Key Observations:
- Increasing qubit count alone did **not** significantly improve performance
- Severe class imbalance in predictions:
  - Some classes predicted frequently
  - Others never predicted (class collapse)
- Training stable but ineffective


### Interpretation:
- The model lacks sufficient **circuit expressivity**
- Larger Hilbert space is underutilized due to shallow parameterization
- Observable design is sufficient (8 outputs), but circuit cannot generate separable features

### Conclusion:
- Increasing qubit count without increasing circuit depth is insufficient
- Model capacity is limited by **circuit structure**, not just qubit count


### Takeaway:
> **Qubit scaling alone does not solve multi-class classification — expressivity must scale as well**