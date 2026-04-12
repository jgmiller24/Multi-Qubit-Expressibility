
## Experiment 1 (v5): Modified Observables

## Circuit Modification:
- Reverted quantum circuit back to v3 configuration (reduced depth)
- Modified observables to improve measurement expressivity
- Maintained 2-qubit architecture (fixed Hilbert space)

## Expected Impact:
- Improved class separability via more informative measurement operators
- Reduction in class-specific collapse observed in earlier experiments
- Better utilization of existing circuit features without increasing depth

### Results Summary:
- Test accuracy improved to ~55–56% (increase over v4 ~53–54%)
- Training remained stable with smooth convergence
- No signs of optimization instability or divergence

### Key Observation:
- Adjusting observables yielded measurable performance gains,
  whereas increasing circuit depth alone (v4) did not

### Interpretation:
- Model performance is more sensitive to **observable design**
  than to additional circuit layers at this scale
- Improved results suggest:
  - Better alignment between quantum measurements and class structure
- Remaining limitations likely stem from:
  - Restricted feature space (2 qubits → limited Hilbert space)
  - Incomplete class separability in encoded representations

### Per-Class Behavior:
- Class performance more balanced compared to v3/v4
- Previously weak classes (e.g., digit '5') improved but still lag behind
- No class collapse observed
- Confusion matrix indicates:
  - Persistent overlap between certain digit pairs
  - Some systematic misclassification patterns remain

### Conclusion:
The primary bottleneck is likely:
- limited representational capacity of the 2-qubit system

However:
- observable design plays a **critical role** in extracting useful information
- improvements from v5 confirm that measurement strategy is a key lever

### Next Steps:
- Continue refining observables (e.g., include interaction terms like ZZ, XX)
- Increase training duration to confirm convergence behavior
- Evaluate whether further observable tuning yields diminishing returns
- Begin exploration of 3-qubit architectures to expand feature space
- Analyze confusion matrix patterns to guide targeted improvements

### Amendment 1
- Like in Exp_v4, previous conclusion is not fully justified.
- See Notes_Exp_v5b for further details.
