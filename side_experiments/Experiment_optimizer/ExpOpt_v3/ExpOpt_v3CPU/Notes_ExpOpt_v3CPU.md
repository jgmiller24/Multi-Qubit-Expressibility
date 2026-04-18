## Experiment Optimizer (v3): CPU vs GPU runtime

## Compute: CPU
- Clock time: 40:24 mm:ss

## Result
- Final results:
    - Test accuracy: ~89.5%
    - Performance comparable to GPU

## Runtime Observation:
- CPU provided remarkalbe speedup over GPU (~6x)
- Completed 50 epochs in ~40mins
- Performance curves nearly identical

## Key Insight:
- Reduced overhead for host-data transfers (CPU <=> GPU) noticably decreased runtime

## Conclusion:
- CPU is the more practical backend for the current implementation

