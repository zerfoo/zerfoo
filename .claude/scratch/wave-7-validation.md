# Wave 7 — Validation

**Commit**: 73d14342 (fix/wave-7-gpu-reshape-dst)
**DGX pod**: bench-patchtst-20260409-235603-s5000-c10-e3
**Shapes**: samples=5000, channels=10, epochs=3 (default production bench)

```
engine: GPU (CUDA)
config: 5000 samples x 10 channels x 24 input_len x 3 epochs (batch=64, lr=1.0e-03)
model:  dModel=64 nHeads=4 nLayers=2 patchLen=8 stride=4
total:  4.24267115s (1.414223716s/epoch)

epoch 1: loss=0.027676 ok
epoch 2: loss=0.019176 ok
epoch 3: loss=0.018603 ok

convergence: OK (0.027676 -> 0.018603, 32.8% reduction)
```

Loss strictly decreases across all 3 epochs. Frozen-loss symptom
(0.268357 across all epochs) is gone. Definition of done met.
