# Experiment 1 (v2): 2-Qubit, 4-Class QNN 

## Changes from v1
- Increased sample size (1000 → 2000)
- Added per-class accuracy tracking
- Added classification report and confusion insights
- Evaluated class distribution explicitly

## Results
- Overall accuracy remained ~45–50%
- Increasing dataset size did NOT significantly improve performance
- Loss continued to decrease → optimization is working

## Per-Class Performance
- Class 0 (digit '5'): **0.0 accuracy (not learned)**
- Class 1 (digit '6'): ~0.49 accuracy
- Class 2 (digit '3'): ~0.78 accuracy
- Class 3 (digit '4'): ~0.62 accuracy

## Key Observations
- Model exhibits **class collapse**:
  - One class receives no correct predictions
- Learning is **uneven across classes**
- Dataset is reasonably balanced → imbalance is NOT the issue

## Interpretation
- The model is not failing globally, but **failing selectively**
- Indicates:
  - Limited representational capacity of current circuit
  - Insufficient feature separation in quantum embedding
- Observable set may not provide enough discriminatory power

## Additional Factors
- High weight decay (0.80) may suppress learning
- Digit selection may include visually similar classes
- Circuit depth is shallow for 4-class separation

## Conclusions
- Increasing data alone does not solve the problem
- Performance bottleneck is architectural, not statistical

## Next Steps
- Tune optimizer (lower weight decay, adjust learning rate)
- Test more separable digit sets (e.g., [0,1,4,7])
- Increase circuit depth / add gates
- Expand observable set (more measurement diversity)
- Analyze confusion matrix for misclassification patterns