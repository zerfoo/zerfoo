# Wave 7 — Localization

## Smoking gun (from `.claude/scratch/wave-7-probe-logs.txt`)

```
post-dFlat-matmul         ptr=0x69325628c000 l2=0.0833596 allZero=false
post-dFlat-reshape-dX     ptr=0x69324f244000 l2=0         allZero=true
pre-encoderBackward:dX    ptr=0x69324f244000 l2=0         allZero=true
encBwd:entry:dX           ptr=0x69324f244000 l2=0         allZero=true
encBwd:layer-1:dX-out     l2=0 allZero=true  (every downstream op is zero)
```

MatMul wrote real gradient data into `fc.dFlat` (l2=0.0836, non-zero). The
very next call, `engine.Reshape(ctx, fc.dFlat, [1600,64], fc.dX)`, produced an
all-zero `fc.dX` whose storage pointer does NOT match `fc.dFlat`'s new ptr.
That zero dX cascades through every encoder-layer backward, every patch-emb
grad op, and every pos-emb grad — exactly matching the frozen-loss symptom.

## Root cause

`ztensor/compute/gpu_engine_memory.go:614` `GPUEngine.Reshape`:

```go
// GPUStorage[T]: zero-copy reshape.
if gs, ok := a.GetStorage().(*tensor.GPUStorage[T]); ok && isFloat32[T]() && newSize == currentSize {
    return tensor.NewWithStorage[T](inferredShape, gs.View(gs.Len()))
}
```

This branch **ignores the `dst ...*tensor.TensorNumeric[T]` parameter entirely**.
It returns a brand-new tensor aliasing `a`'s storage and never touches `dst`.
Any caller that discards the return value and relies on `dst` being the
reshaped view gets stale pre-allocated zero storage.

zerfoo hit this at `timeseries/patchtst_gpu_train.go:999`:

```go
if _, err = m.engine.Reshape(ctx, fc.dFlat, []int{totalRows, dModel}, fc.dX); err != nil {
    return nil, err
}
// ...
dX, err := encoderBackward(ctx, m.engine, fc.dX, ...)  // fc.dX is stale zeros
```

## Fix (routed to E2d-ish — zerfoo call-site fix, minimal/surgical)

Capture the return value of Reshape and pass it into encoderBackward instead
of `fc.dX`. One-line change. `fc.dX` is retained as a pre-allocation slot for
its buffer identity but the returned reshaped view is the authoritative tensor.

Follow-up (separate issue/PR): fix ztensor `GPUEngine.Reshape` to honor `dst`
when provided (either copy data into dst's storage or SetStorage dst to alias
the view). Current behavior silently violates the compute.Engine contract
shared with the CPU engine.
