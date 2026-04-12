
## Experiment 1 (v4b): Increased Circuit Depth – EPOCHS = 400

## Updated Results Summary:
- Test accuracy improved significantly to ~70% (from ~53–54% at 200 epochs)
- Training and test curves continued improving steadily with no clear plateau
- Model benefits from extended training time

## Key Observation:
- Increasing circuit depth *does* improve performance when sufficient training time is provided
- Earlier conclusion (no benefit from depth) was due to undertraining

## Interpretation:
- Deeper quantum circuits introduce:
  - greater representational capacity
  - more complex optimization landscape
- As a result:
  - convergence is slower
  - more epochs are required to fully utilize added expressivity

## Per-Class Behavior:
- All classes improved significantly compared to earlier runs
- More balanced class performance:
  - Class 1 (~0.73), Class 2 (~0.70), Class 0 (~0.68), Class 3 (~0.67)
- Reduced class disparity and improved separability

## Revised Conclusion:
- Circuit depth is a meaningful contributor to model performance
- However, its benefits are only realized with sufficient training duration
- Optimization difficulty increases with circuit depth

## Key Insight:
- There is a tradeoff between:
  - expressivity (depth)
  - trainability (optimization difficulty)

## Next Steps:
- Rerun Exp1_v5 (observable improvements) at 400 epochs for fair comparison
- Determine whether:
  - observables (v5) OR
  - circuit depth (v4b)
  provides greater performance gain
- Explore combined approach (depth + optimized observables)

