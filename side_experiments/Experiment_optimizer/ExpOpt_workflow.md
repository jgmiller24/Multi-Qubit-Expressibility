
## Evaluations with optimizer adjustments and CPU vs GPU runtime comparison

## Experiment Workflow

- ExpOpt_v1 → Adadelta (reduced weight decay)
  - Test whether excessive regularization was limiting learning
  - Establish baseline convergence behavior under weaker constraints

- ExpOpt_v2 → Adam optimizer
  - Evaluate impact of optimizer choice on convergence speed and stability
  - Compare learning dynamics and final performance vs Adadelta

- ExpOpt_v3 → CPU vs GPU runtime comparison (Adam)
  - Measure practical runtime differences under identical settings
  - Assess whether CUDA acceleration improves hybrid QNN training efficiency

## Adadelta vs Adam
- Adadelta
  - Adaptive learning rate based on accumulated gradients
  - Does not require an explicit learning rate (or uses a minimal one)
  - Tends to produce stable but slower updates
  - More sensitive to additional regularization (e.g., weight decay)

- Adam
  - Combines momentum (first moment) and adaptive scaling (second moment)
  - Uses explicit learning rate with bias correction
  - Produces faster and more directed convergence
  - Better suited for complex, non-convex optimization landscapes

## Key findings
- Adadelta provided stable but overly conservative updates under high regularization
- Adam enabled efficient navigation of the optimization landscape, unlocking model performance
- GPU acceleration is not effective for this hybrid workflow at current scale
  - CPU clock time: 40:24 mm:ss
  - GPU clock time: 04:26:34 hh:mm:ss

## Implications

## Insight
- Model performance is nearly identical on CPU and GPU
- The hybrid pipeline overhead dominates any GPU benefit.