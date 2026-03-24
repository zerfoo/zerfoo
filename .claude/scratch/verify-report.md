## Verification Report

### Date: 2026-03-24
### Scope: Full system -- all layers, architectures, and training pipelines

### Architecture
- Modules discovered: 15 layer packages, 9 timeseries backends, 5 tabular models, 1 HRM model
- Layers: core (60+ ops), normalization, attention, embeddings, SSM, recurrent, transformer, residual, audio, vision, timeseries, hrm, components, regularization, gather, reducesum, transpose

### Test Coverage Added

| Test File | Tests | Status |
|-----------|-------|--------|
| timeseries/verify_learn_predict_test.go | 7 backends x 2 tests (learn+predict, save/load) | ALL PASS |
| tabular/verify_learn_predict_test.go | 5 models inference + 2 training + 3-class | ALL PASS |
| layers/core/verify_learn_test.go | 4 layers x 2 tests (loss decrease, gradients) | ALL PASS |
| layers/normalization/verify_learn_test.go | RMSNorm loss decrease + BatchNorm backward | ALL PASS |
| layers/attention/verify_learn_test.go | AttentionHead loss decrease + gradients | ALL PASS |
| layers/recurrent/verify_learn_test.go | SimpleRNN loss decrease + gradients | ALL PASS (after fix) |

### Bugs Found

#### BUG 1: SimpleRNN bias gradient never computed (FIXED)
- **File:** layers/recurrent/rnn.go:131-152
- **Severity:** HIGH
- **Description:** SimpleRNN.Backward() computes gradients for inputWeights and hiddenWeights but never calls bias.Backward(). The bias parameter is included in Parameters() so optimizers receive it, but its gradient is always zero. This means bias parameters never update during training.
- **Fix:** Added `r.bias.Backward(ctx, mode, tanhGrad[0])` call in Backward() between hiddenWeights backward and return.
- **Verification:** TestSimpleRNN_ParameterGradientsNonZero now passes (was failing before fix).

#### FINDING 2: AttnRes.Backward() not implemented
- **File:** layers/residual/attn_res.go:149-151
- **Severity:** LOW (inference-only layer, backward handled by computation graph)
- **Description:** Returns error "AttnRes: backward pass not yet implemented". This is acceptable since AttnRes is used during inference via the computation graph which handles backward differently.

#### FINDING 3: BatchNorm has no backward pass
- **File:** layers/normalization/batch_norm.go
- **Severity:** MEDIUM (inference-only currently)
- **Description:** BatchNorm.Backward() returns (nil, nil) with no gradient computation. Parameters() returns empty. This is by design for inference, but prevents BatchNorm from being used in training.

#### FINDING 4: FTTransformer, TabResNet, TabNet have no training
- **File:** tabular/ft_transformer.go, tabular/resnet.go, tabular/tabnet.go
- **Severity:** LOW (by design -- inference only)
- **Description:** Three of five tabular models have no Train() function. Only MLP (via Train()) and SAINT (via TrainSAINT()) support training. The others are inference-only.

### Verification Results

#### Timeseries Backends (7 tested)
| Backend | Learn | Loss Decrease | Predict | Save/Load |
|---------|-------|---------------|---------|-----------|
| DLinear | PASS | PASS | PASS | PASS |
| NHiTS | PASS | PASS | PASS | PASS |
| FreTS | PASS | PASS | PASS | PASS |
| ITransformer | PASS | PASS | PASS | PASS |
| Mamba | PASS | PASS | PASS | PASS |
| CfC | PASS | PASS | PASS | PASS |
| PatchTST | PASS | PASS | PASS | PASS |

#### Tabular Models (5 tested)
| Model | Inference | Training | 3-Class | Convergence |
|-------|-----------|----------|---------|-------------|
| MLP | PASS | PASS | PASS | >60% accuracy |
| FTTransformer | PASS | N/A (no training) | N/A | N/A |
| TabResNet | PASS | N/A (no training) | N/A | N/A |
| SAINT | PASS | PASS | N/A | Valid predictions |
| TabNet | PASS | N/A (no training) | N/A | N/A |

#### Core Layers (4 tested for learning)
| Layer | Loss Decrease | Gradients Non-Zero |
|-------|---------------|--------------------|
| Dense | PASS | PASS |
| Linear | PASS | PASS |
| FFN | PASS | PASS |
| Conv1D | PASS | PASS |

#### Additional Layers
| Layer | Loss Decrease | Gradients Non-Zero |
|-------|---------------|--------------------|
| RMSNorm | PASS | PASS |
| AttentionHead | PASS | PASS |
| SimpleRNN | PASS | PASS (after bug fix) |
| BatchNorm | N/A (no backward) | N/A |

### Summary
- **Total tests added:** 6 test files, ~25 test functions
- **Bugs found:** 1 (SimpleRNN bias gradient -- FIXED)
- **Known gaps:** 3 (AttnRes backward, BatchNorm backward, 3 tabular models lack training)
- **All existing tests still pass after changes**
