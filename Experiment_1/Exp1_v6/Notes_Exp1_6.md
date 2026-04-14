
## Experiment 1 (v6): Combined Depth + Observables — Full Convergence (1500 Epochs)

## Circuit Modification:
- Combined increased circuit depth (v4) with improved observables (v5)
- Maintained 2-qubit architecture
- Extended training duration to 1500 epochs

## Expected Impact:
- Fully realize combined expressivity of circuit depth and observable design
- Determine whether previous performance ceiling (~75–80%) was due to:
insufficient training or representational limits
- Observe convergence behavior and plateau region

## Results Summary:
- Test accuracy improved to ~92–93%
- Significant increase over previous runs (~70–80%)
- Train and test curves converged with minimal gap
- No instability or overfitting observed
- Performance gains slowed after extended training → approaching plateau

## Key Observation:
- The previously observed performance ceiling was not intrinsic to the 2-qubit system
- Instead, it resulted from insufficient optimization time
- Combining circuit depth and improved observables yields strong performance when fully trained

## Interpretation:
- Model performance depends on the interaction of three factors:
    - circuit depth
    - observable design
    - training duration
- Observable improvements provide faster gains
- Circuit depth contributes additional capacity but requires longer training to realize benefits
- Extended training reveals the true capability of the model

## Per-Class Behavior:
- Strong and balanced performance across all classes
- No class collapse observed
- Remaining errors likely due to:
    - intrinsic overlap between digit classes
    - limitations in feature encoding rather than optimization

## Conclusion:
- A 2-qubit hybrid QNN can achieve >90% accuracy on 4-class MNIST when:
    - circuit expressivity is sufficient
    - observables are well-designed
    - training is allowed to fully converge
- The limiting factor is not solely Hilbert space size, but the ability to effectively optimize within it

## Key Insight:
- Apparent performance limits in hybrid QNNs may stem from optimization constraints 
rather than representational capacity, especially in low-qubit systems.

## GPU acceleration
- GPU acceleration was successfully enabled, but for the current 2-qubit hybrid implementation, runtime increased due to repeated CPU↔GPU transfers between PyTorch tensors and CUDA-Q circuit evaluation. At this scale, transfer overhead outweighed any practical speedup.