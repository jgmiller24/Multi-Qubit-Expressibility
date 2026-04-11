# Experiment 1 (v1): 2-Qubit, 4-Class QNN

## Summary
Initial extension from 1-qubit binary classification (Experiment 0) to a 2-qubit, 4-class MNIST task.

## Results
- Accuracy dropped significantly compared to Experiment 0 (~90% → ~45–50%)
- Loss decreased steadily, indicating the model was learning something
- Training and test performance remained close → no obvious overfitting

## Observations
- Increased task complexity (multiclass vs binary) made classification more difficult
- Larger quantum state space (2 qubits) did not automatically improve performance
- Model struggled to form strong decision boundaries between classes

## Post-hoc Analysis (added after v2)
- Per-class behavior was not initially evaluated
- Later analysis (v2) revealed:
  - Some classes were partially learned
  - At least one class was completely unlearned (collapsed prediction)

## Interpretation
- The issue is not simply "more qubits = better performance"
- Circuit expressibility and observable design are critical
- Model likely under-expressive for this classification task

## Next Steps (defined after v1)
- Add per-class accuracy tracking
- Investigate class separability
- Tune optimizer and regularization
- Increase dataset size
- Explore alternative circuit structures