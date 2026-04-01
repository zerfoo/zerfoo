# DRY Audit Scratchpad

## Key Findings Summary

### Duplicated Primitive Functions Found

1. **GELU** - 7 separate implementations:
   - `geluScalar` (patchtst.go:696) - float32
   - `geluF64` (patchtst_backward.go:678) - float64
   - `geluDerivF64` (patchtst_backward.go:655) - float64 derivative
   - `geluDerivF32` (patchtst_gpu_train.go:1416) - float32 derivative
   - `gelu` (itransformer.go:353) - float64
   - `geluGrad` (itransformer_backward.go:6) - float64 derivative
   - `geluMatrix` (ttm.go:1436) - applies GELU to matrix

2. **Layer Norm** - 11 separate implementations:
   - `layerNormF64WithCache` (patchtst_backward.go:556) - float64 with cache
   - `layerNormBackwardF64` (patchtst_backward.go:593) - float64 backward
   - `layerNormF64` (patchtst_backward.go:993) - float64 forward only
   - `layerNormF32WithCache` (patchtst_gpu_train.go:336) - float32 with cache
   - `layerNormBackwardF32` (patchtst_gpu_train.go:369) - float32 backward
   - `PatchTST.layerNorm` (patchtst.go:509) - tensor-based
   - `layerNormCached` (itransformer_backward.go:290) - float64 different signature
   - `layerNormBackward` (itransformer_backward.go:539) - float64 different signature
   - `layerNorm` (itransformer.go:329) - float64 vector-based
   - `TTM.layerNormF32` (ttm.go:627) - tensor-based
   - `TFT.layerNorm` (tft.go:535) - tensor-based

3. **Softmax** - 4 implementations:
   - Inline in patchtst_backward.go (CPU per-head loop)
   - Inline in patchtst_engine.go (CPU per-head loop)
   - Inline in patchtst_gpu_train.go (engine.Softmax)
   - `softmax` (itransformer.go:309)

4. **Linear forward** - 8 implementations:
   - `linearF64` (patchtst_backward.go:1090) - flat weight float64
   - `PatchTST.linearF64Engine` (patchtst_engine.go:98) - engine-backed
   - `linearForwardVec` (itransformer.go:194) - vector float64
   - `ITransformer.linearBatchEngine` (itransformer_engine.go:15) - engine-backed
   - `linearBatchCPU` (itransformer_engine.go:63) - CPU fallback
   - `TTM.linearF64Engine` (ttm_train_engine.go:72) - engine-backed (DUPLICATE of PatchTST)
   - `NHiTS.linearForward` (nhits.go:260) - tensor-based
   - `NBEATS.linearForward` (nbeats.go:426) - tensor-based

5. **MatMul engine wrapper** - 3 copies:
   - `PatchTST.matMulEngine` (patchtst_engine.go:21)
   - `PatchTST.matMulEngineWithBufs` (patchtst_engine.go:25)
   - `TTM.matMulEngine` (ttm_train_engine.go:15) - EXACT DUPLICATE of PatchTST

6. **AdamW optimizer** - 4 implementations:
   - `adamWUpdate` (training_ops.go:41) - shared float64
   - `NHiTS.adamUpdate` (nhits.go:652) - float32 method
   - Inline in patchtst_gpu_train.go:1366-1378 - float32 inline
   - Inline in patchtst_engine.go:706-711 - float64 inline

7. **Gradient clipping** - 3 implementations:
   - `clipGradients` (training_ops.go:8) - shared float64
   - `NHiTS.clipGradients` (nhits.go:634) - float32 method
   - Inline in patchtst_gpu_train.go:1332-1349 - float32 inline

8. **adamState struct** - 4 definitions:
   - `adamWState` (training_ops.go:26) - float64
   - `adamState` (nhits.go:276) - float32
   - Inline `adamState` (frets_engine.go:113) - float32
   - Inline `adamState` (dlinear_engine.go:52) - float32

9. **Matrix utility functions**:
   - `copyMatrix` (patchtst_backward.go:668) - copies [][]float64
   - `deepCopy2D` (itransformer_backward.go:163) - copies [][]float64 (same purpose)
   - `zeroMatrix` (itransformer_backward.go:119) - only in iTransformer
   - `transposeMatrix` (ttm.go:1425) - only in TTM

### PatchTST: The Worst Offender (6,196 non-test lines across 6 files)

PatchTST has FIVE parallel implementations of the same transformer:
1. `patchtst.go` - tensor-based Forward (inference path)
2. `patchtst_backward.go` - CPU float64 forward + backward (1,104 lines)
3. `patchtst_engine.go` - engine-backed float64 forward + CPU backward + training loop (825 lines)
4. `patchtst_backward_engine.go` - engine-backed backward pass (412 lines)
5. `patchtst_gpu_train.go` - full float32 GPU training with fused forward/backward (1,452 lines)

Each implements the same pattern: patch embedding -> pos embedding -> N x (LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual) -> flatten -> output head.

### What Uses the Shared TrainLoop

Backends using the shared `TrainLoop` (DRY-compliant):
- DLinear (via TrainableBackend interface)
- FreTS (via TrainableBackend interface)
- NHiTS (CPU path only)
- CfC (via TrainableBackend interface)

Backends with their own training loops:
- PatchTST (3 separate loops: CPU, engine, GPU)
- ITransformer (own loop)
- TTM (own loop)
- NHiTS (engine path)
- TimeMixer (own loop with different TrainConfig signature!)
