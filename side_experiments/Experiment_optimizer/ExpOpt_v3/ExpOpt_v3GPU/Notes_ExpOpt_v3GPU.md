## Experiment Optimizer (v3): CPU vs GPU runtime

## Compute: GPU
- Clock time: 04:26:34 hh:mm:ss
## Result
- GPU execution was successfully enabled (torch.cuda.is_available() = True)
- Model converged rapidly using Adam optimizer (~25 epochs)
- Final performance:
    - Test accuracy: ~89%
    - Comparable to CPU-based runs

## Runtime Observation:
- GPU execution did not provide practical speedup
- Training required ~4+ hours for only 50 epochs
- Per-iteration runtime remained high despite CUDA availability

## Key Insight:
- The dominant computational cost lies in repeated quantum evaluations:
    - cuda.observe() exevuted per observable
    - Requires CPU-based simulation
- Additional overhead introduced by:
    - Tensor conversion (.cpy().numpy())
    - Host-device data transfer
    - Reconstruction of outputs in Pytorch

## Conclusion:
- GPU acceleration is not effective for this hybrid workflow at current scale
- Runtime performance is limited by:
    - quantum simulation cost
    - interface overhead between Pytorch and CUDA-Q

## Implication:
- CPU execution is sufficient and more practical for current experiments
- GPU may only become ceneficial if:
    - quantum simulation is GPU-Accelerated
    - batching reduces per-call overhead
    - transfer costs are minimized