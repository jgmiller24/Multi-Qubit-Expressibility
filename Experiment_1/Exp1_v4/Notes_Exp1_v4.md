
## Experiment 1 (v4): Increased Circuit Depth

## Circuit Modification:
- Added third rotation layer (total depth increased)
- Added second entanglement layer (CNOT)
- Increased parameter count from 8 → 12

## Expected Impact:
- Improved feature expressivity
- Better class separability
- Reduction in class collapse behavior

### Results Summary
- Test accuracy remained ~53–54% (no improvement over v3)
- Training remained stable with no signs of optimization issues

### Key Observation
- Increasing circuit depth (additional rotation + entanglement layer)
  did NOT improve performance.

### Interpretation
- Model is not limited by circuit depth alone
- Additional parameters and entanglement do not yield better class separation
- Suggests a bottleneck in:
  - observable design
  - low-dimensional quantum representation (2 qubits)

### Per-Class Behavior
- Similar imbalance as v3 persists
- Certain classes remain harder to distinguish
- No class collapse, but no improvement in separability

### Conclusion
The limiting factor is likely:
- insufficient measurement expressivity
- constrained Hilbert space (2 qubits)

rather than insufficient circuit depth.

### Next Steps
- Modify observables (include interaction terms like ZZ, XX)
- Try different digit subsets (more separable classes)
- Explore 3-qubit architectures
- Investigate hybrid architectures with stronger classical feature extraction

