
## Experiment Optimizer - Using Exp2_v3 model - Decreased optimizer weight decay to 1e-4. 

## Results Summary:
- Test accuracy improved to ~43% at 200 epochs
- Training and test curves show smooth and stable convergence
- Loss decreased consistently with no instability

## Key Observation:
- Reducing weight decay significantly improved learning dynamics
- Model now trains effectively without suppression from regularization

## Per-Class Behavior:
- Strong performance on certain classes (e.g., Class 0: ~90%)
- Persistent underperformance in several classes (e.g., Class 4, Class 6)
- Indicates continued class separability limitations

## Interpretation:
- Previous poor performance was partially due to excessive regularization
- However, improved optimization alone does not resolve classification imbalance
- Remaining limitation is likely due to:
    - insufficient circuit expressivity
    - or suboptimal observable design

## Conclusion:
- Optimization tuning improves convergence speed and stability
- But architectural factors remain the dominant bottleneck for 8-class performance