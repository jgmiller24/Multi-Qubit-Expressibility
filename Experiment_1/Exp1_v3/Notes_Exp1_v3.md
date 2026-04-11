## Experiment 1 (v3): Observables + Circuit Expressivity

### Results Summary
- Train Accuracy: ~61%
- Test Accuracy: ~55%
- Improvement over v2 (~45–50%)

### Key Observations
- Model no longer exhibits class collapse (all classes predicted)
- Loss decreases steadily → stable training
- No significant overfitting observed

### Per-Class Behavior
- Strong performance on digits: 6, 3, 4
- Weak performance on digit: 5
  - Frequently misclassified as 3, 6, or 4

### Interpretation
- Revised observables (X + Z) significantly improved feature extraction
- Increased circuit depth improved model expressivity
- Remaining errors are due to:
  - visual similarity between digits
  - insufficient separation in learned feature space

### Conclusion
The primary limitation in earlier experiments was not dataset imbalance,
but insufficient observable diversity and circuit expressivity.

This experiment demonstrates that:
- measurement design is critical in hybrid QNNs
- shallow circuits limit multiclass performance

### Next Steps
- Further increase circuit depth (additional entanglement layers)
- Experiment with different observable combinations (e.g., ZZ terms)
- Try more separable digit sets to isolate representational limits
- Explore 3-qubit models for higher-dimensional encoding