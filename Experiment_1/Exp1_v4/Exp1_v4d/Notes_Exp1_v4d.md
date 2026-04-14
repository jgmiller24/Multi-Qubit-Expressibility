
## Experiment 1 (v4d): Increased Circuit Depth – EPOCHS = 1000

### Circuit Modification:
- Same architecture as v4 (3 rotation layers + 2 entanglement layers)
- Extended training duration from 600 → 1000 epochs

### Expected Impact:
- Fully converge deeper circuit representation
- Determine true performance ceiling of depth-based expressivity
- Evaluate whether additional training continues to yield gains


### Results Summary:
- Test accuracy improved to ~87% (from ~78–79% at 600 epochs)
- Training remained stable with no signs of divergence
- Continued improvement observed beyond 600 epochs, with slowing gains after ~800 epochs


### Key Observation:
- Increasing training duration allowed deeper circuits to significantly outperform earlier results
- Unlike v4/v4b conclusions, depth *does* provide meaningful benefit when fully trained
- Performance gains begin to plateau near ~85–88%


### Interpretation:
- Earlier conclusion (v4) that depth was ineffective was due to insufficient training
- Circuit depth increases representational capacity, but:
  - Requires longer optimization time to realize benefits
- However, even with extended training:
  - Performance saturates below ~90%

This suggests:
- Optimization is no longer the bottleneck
- The model is approaching a **capacity limit of the 2-qubit system**


### Per-Class Behavior:
- Strong and balanced performance across all classes
- Significant improvement in previously weaker classes (e.g., digit '5')
- Minimal class confusion compared to earlier experiments
- Remaining errors likely due to intrinsic overlap between digit classes


### Conclusion:
- Increasing circuit depth + sufficient training significantly improves performance
- Final performance converges around ~85–88% test accuracy
- The primary limitation is now:
  - **restricted Hilbert space (2 qubits)**


### Key Insight:
- Circuit depth is effective **only when adequately trained**
- Observable design and circuit depth both improve performance, but:
  - Neither can overcome the **fundamental capacity limit of 2 qubits**
- The system exhibits a clear **performance ceiling**


### Next Steps:
- Transition to **3-qubit architecture** to expand Hilbert space
- Explore:
  - More expressive entanglement patterns
  - Hybrid classical feature preprocessing
- Compare scaling behavior between 2-qubit and 3-qubit models