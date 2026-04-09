# Wave 5 DGX Validation Result — BLOCKED

Date: 2026-04-09 (UTC) / 2026-04-08 (local)
Branch: docs/wave-5-dgx-validation
Base: main @ 6bde36a6 (includes fix 168a938f)

## Build artifact
- DGX HEAD: `6bde36a6 docs(devlog): consolidated PatchTST GPU training convergence postmortem + cleanup`
- `/var/lib/zerfoo/bin/bench_train` md5: **b2da62ec93225b10be86a10d4b642516**
- Built with `/usr/local/go/bin/go build ./cmd/bench_train` on DGX (aarch64)

## SSH session leak check
- sessions_before: 0
- sessions_after:  0
- Result: **PASS** (no leak — all bench work went through Spark)

## T5.2 Smoke bench 1K x 5 x 2 — **FAILED (panic)**

Engine reported: `GPU (CUDA)` — GPU path taken.

Config: `1000 samples x 5 channels x 24 input_len x 2 epochs (batch=64, lr=1.0e-03)`
Model:  `dModel=64 nHeads=4 nLayers=2 patchLen=8 stride=4`

Outcome: `panic: patchtst gpu sentinel: gradient backing-slice mismatch at index 0`

Relevant sentinel dump:
```
grads.allParamTensors()[i] wrapper: 0x59749b7bf860, Data()[0] ptr: 0x59749bb71000, Data()[:4]: [0 0 0 0]
gradTs[i]                   wrapper: 0x59749b7bf860, Data()[0] ptr: 0x59749bb71800, Data()[:4]: [0 0 0 0]
```

Key observations:
1. The wrapper pointers are **identical** (`0x59749b7bf860`) — the rebuild-per-batch logic correctly pulls the live `*tensor.TensorNumeric[float32]` from the struct.
2. The Data() backing pointers differ by exactly **0x800 (2048 bytes)**. This is the strengthened sentinel firing in `timeseries/gradts_sentinel.go:43`.
3. Meaning: between `gradTs[i] = grads.allParamTensors()[i]` being captured and `verifyGradTsAliasing` running, something reseated the underlying slice of the same wrapper object (arena realloc / `SetData` / backing-buffer swap). The wrapper identity check the old code relied on is insufficient; the new sentinel catches it.
4. AdamW would have been reading stale gradient memory — exactly the bug Wave 4 was meant to close.

Stack (abbreviated):
```
timeseries.verifyGradTsAliasing     gradts_sentinel.go:43
timeseries.(*PatchTST).trainWindowedGPU  patchtst_gpu_train.go:1052
timeseries.(*PatchTST).TrainWindowed     patchtst.go:449
main.main                           cmd/bench_train/main.go:106
```

Per plan directive ("If it fails or loss is frozen: STOP and report. Don't continue to T5.3") — **stopped after T5.2**. T5.3, T4.2, T4.3 not executed.

## Verdict: **BLOCKED**

The PR #365 fix addresses wrapper-identity aliasing (paramTs/gradTs rebuilt per batch) but does **not** address backing-slice reseating. On the DGX GPU path, the grad tensor's underlying `Data()` slice is being reallocated/swapped between the sentinel wrapper-capture point and its immediate verification — the two captures above are of the same wrapper address in the same function call. The strengthened sentinel is working as designed and catching the real bug; the fix merely surfaces it loudly instead of masking it.

### Recommended next step (Wave 6 / follow-up PR)
Investigate what reseats `Data()` on the grad tensor between `allParamTensors()` assembly and the sentinel check inside the same call site. Candidates:
- Arena-backed tensor pool growing/recycling between ops
- An engine op calling `SetData` on the grad tensor
- Gradient zero/accumulate path allocating a fresh backing slice
The sentinel dump points at `timeseries/gradts_sentinel.go:43` and `patchtst_gpu_train.go:1052` — trace who holds the grad backing buffer between those two lines on the GPU path.

Zero-stub policy honored: no T5.3 / T4.2 / T4.3 numbers fabricated.
