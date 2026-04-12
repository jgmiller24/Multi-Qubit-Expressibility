
## Experiment 1 (v5b): Modified Observables – EPOCHS = 400

## Updated Results Summary:
- Test accuracy improved to ~74% (increase over v5 ~55–56%)
- Performance slightly exceeds v4b (~70%)
- Training remained stable with continued improvement and no clear plateau at 400 epochs

## Key Observation:
- Observable improvements yield performance comparable to (and slightly better than) increased circuit depth
- Both approaches converge to a similar performance range (~70–75%)

## Interpretation:
- At 400 epochs, both train and test curves were still improving, although the rate of improvement had begun to slow
- This suggests the model has not fully converged, justifying extension to 600 epochs to better estimate the practical performance ceiling
- Observable design and circuit depth act as complementary sources of expressivity
- Improved observables enable more efficient feature extraction without increasing circuit complexity
- However, both approaches appear to converge toward a similar performance ceiling

## Per-Class Behavior:
- More balanced class performance across all digits
- Strong improvements in previously weaker classes
- Remaining errors indicate persistent overlap between certain digit pairs

## Revised Conclusion:
- Performance is no longer primarily limited by training duration or optimization
- Additional training (e.g., to 600 epochs) is expected to yield incremental improvements, though gains may diminish and plateau below ~80%
- The primary bottleneck is now:
  - limited representational capacity of the 2-qubit system

## Key Insight:
- There exists an upper bound on performance imposed by the size of the quantum state space
- Architectural improvements alone (depth or observables) cannot fully overcome this limitation

## Next Steps:
- Incrementally increase epochs (e.g., 600) to identify convergence point
- Track best test accuracy across epochs (not just final epoch)
- Combine best-performing elements:
  - deeper circuit (v4b)
  - improved observables (v5b)
- Evaluate diminishing returns from further architectural tuning
- Transition to 3-qubit architectures to expand representational capacity