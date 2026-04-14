
## Experiment 1 (v5d): Full Convergence – EPOCHS = 1000

### Objective
Evaluate full convergence behavior of the 2-qubit hybrid QNN using optimized observables,
and determine whether previously observed performance limitations were due to insufficient
training or representational constraints.


### Results Summary
- Test accuracy improved to ~88–89% (increase over v5c ~80%)
- Training accuracy reached ~93–94%
- Loss curves continued decreasing and began to plateau near the end of training
- Training remained stable with no signs of divergence or instability


### Key Observation
- Significant performance gains were achieved by extending training duration
- The previously observed performance ceiling (~80%) was not intrinsic to the model
- Both circuit depth (v4) and observable design (v5) contribute to performance,
  but require sufficient training to fully realize their benefits


### Interpretation
- Earlier experiments underestimated model capacity due to premature convergence assumptions
- The 2-qubit system is capable of learning complex class boundaries when fully optimized
- Observables improve learning efficiency (faster gains), while circuit depth contributes
  to overall capacity when given enough training time
- The model exhibits **delayed convergence behavior**, requiring extended epochs to reach
  optimal performance


### Per-Class Behavior
- All classes achieved strong performance (no class collapse observed)
- Previously weak classes (e.g., digit '5') improved significantly
- More balanced class accuracy across all digits
- Remaining errors suggest minor overlap between visually similar digits


### Revised Conclusion
- Representational capacity of the 2-qubit system is sufficient for this task,
  but requires extended training to fully utilize
- The 2-qubit hybrid QNN can achieve ~88–90% accuracy on this 4-class MNIST task
- Optimization time is a critical factor in hybrid quantum-classical models


### Key Insight
- Hybrid QNNs may require significantly longer training compared to classical models
  to fully realize their expressive capacity
- Apparent performance ceilings can emerge prematurely if convergence is not fully reached


### Final Takeaway (Experiment 1)
- Observable design improves learning efficiency
- Circuit depth contributes to capacity, but only when sufficiently trained
- The 2-qubit system is more capable than initially assumed
- Performance is ultimately governed by **training convergence**, not just architecture


### Next Steps
- Transition to 3-qubit architecture to evaluate impact of increased Hilbert space
- Compare:
  - Convergence speed (2 vs 3 qubits)
  - Final accuracy ceiling
- Investigate whether additional qubits improve:
  - Class separability
  - Training efficiency