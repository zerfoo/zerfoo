// Package training: persistent parameter-gradient accumulators (issue #850).
//
// Layer Backward implementations assign engine-op results directly to the
// persistent Parameter.Gradient field (e.g. layers/core/bias.go
// `b.biases.Gradient = biasesGrad`). On the GPU engine those results are
// ARENA-allocated tensors. Trainers that accumulate gradients across multiple
// Backward calls while resetting the arena between samples (the Wolf
// crossasset pattern: forward+backward per sample, ResetPool per sample,
// optimizer step per batch) are then left with Parameter.Gradient pointing at
// recycled arena memory: silent corruption normally, deterministic NaN under
// ZTENSOR_ARENA_POISON=1. This is the fourth lifetime pattern of the
// zerfoo#842 / zerfoo#845 / zerfoo/ztensor#128 bug class: a Backward-written
// Parameter.Gradient read after arena reset.
//
// The fix is a single strategy-level hook, not N layer edits: after every
// graph.Backward, each trainable parameter whose gradient is arena-backed has
// that gradient accumulated into a parameter-owned PERSISTENT (non-arena)
// buffer, and Parameter.Gradient is repointed at that buffer. Layers that
// overwrite .Gradient next sample replace the pointer with a fresh arena
// tensor; the hook then re-accumulates into the same persistent buffer.
// Layers (and the optimizer, zerfoo#845) that write IN PLACE into the
// existing .Gradient now correctly hit the persistent buffer and are skipped
// by the hook via storage identity.
//
// Pinning via the ztensor Pin API (ADR 006) is deliberately NOT used here:
// arena Reset has raise-the-floor semantics, so pinning the gradients would
// retain every sample's activations below the pinned buffers, defeating the
// per-sample reset and re-OOMing the arena (zerfoo/ztensor#118 history).
package training

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// gradAccumulator maintains one persistent (non-arena) gradient buffer per
// trainable parameter. It is embedded in the gradient strategies and lives as
// long as the strategy, so the buffers persist across samples AND across
// optimizer steps: the optimizer zeroes Parameter.Gradient in place after
// Step (zerfoo#845 stepMixedV zeroes via storage Set), and the next batch
// accumulates into the same, now-zeroed buffer.
type gradAccumulator[T tensor.Numeric] struct {
	// engine, when set (see DefaultBackpropStrategy.SetEngine), performs the
	// accumulation as an in-place engine.Add with dst=accumulator -- on the
	// GPU engine this is a device-side kernel writing into the accumulator's
	// existing (non-pool) device pointer, with no host round-trip. When nil,
	// a host read-modify-writeback fallback is used (Data() is a D2H copy on
	// GPU storage and Set() an H2D copy), which is correct on both unified
	// and discrete memory, just slower.
	engine compute.Engine[T]

	// accums maps each parameter to its persistent accumulator tensor.
	accums map[*graph.Parameter[T]]*tensor.TensorNumeric[T]
}

// setEngine configures the optional engine used for in-place device-side
// accumulation.
func (a *gradAccumulator[T]) setEngine(e compute.Engine[T]) { a.engine = e }

// capture is the post-Backward hook: for every trainable parameter of g
// whose Gradient is arena-backed, accumulate the gradient into the
// parameter's persistent buffer and repoint Parameter.Gradient at it.
//
// The hook is unconditional but cheap: parameters with nil gradients (not in
// the active graph this step) and gradients whose storage is not arena-backed
// (the entire CPU path) are skipped without any allocation, so CPU training
// behavior is byte-identical to before.
func (a *gradAccumulator[T]) capture(ctx context.Context, g *graph.Graph[T]) error {
	for _, p := range g.Parameters() {
		grad := p.Gradient
		if grad == nil {
			// First-batch / partial-graph path: the parameter received no
			// gradient this step.
			continue
		}

		accum := a.accums[p]
		if accum != nil && storagesIdentical(accum, grad) {
			// The layer (or optimizer) accumulated in place into the
			// persistent buffer we installed earlier; nothing to migrate.
			// This also makes the hook idempotent for parameters shared by
			// multiple nodes (Graph.Parameters may yield duplicates).
			continue
		}

		if !arenaBackedStorage(grad) {
			// Non-arena storage (CPU engines, raw device allocations,
			// bucketed pools) is never recycled behind a live reference;
			// leave the gradient untouched.
			continue
		}

		if accum == nil {
			var err error
			accum, err = newPersistentGradTensor[T](grad)
			if err != nil {
				return fmt.Errorf("training: allocating persistent gradient accumulator for %q: %w", p.Name, err)
			}
			if a.accums == nil {
				a.accums = make(map[*graph.Parameter[T]]*tensor.TensorNumeric[T])
			}
			a.accums[p] = accum
		}

		if err := a.addInto(ctx, a.engineFor(g, accum, grad), accum, grad, p.Name); err != nil {
			return err
		}

		p.Gradient = accum
	}

	return nil
}

// engineFor selects the engine used to accumulate grad into accum.
//
// An explicitly configured engine (SetEngine) always wins. Otherwise, for
// fully device-resident f32 accumulation -- both tensors backed by
// *tensor.GPUStorage, the only case where the host fallback's
// read-modify-writeback round-trips device memory through the host -- the
// graph's own engine is used, so the add runs as an in-place device kernel
// on the same stream as the graph's kernels (Bug 11: the host round-trip is
// both slow and, before the ztensor host-access ordering fix, racy on
// coherent unified-memory platforms). Non-f32 and non-device cases return
// nil and take the host fallback: GPU engines only run native kernels for
// float32 and their CPU fallback does not honor the in-place dst contract
// for device-backed tensors.
func (a *gradAccumulator[T]) engineFor(g *graph.Graph[T], accum, grad *tensor.TensorNumeric[T]) compute.Engine[T] {
	if a.engine != nil {
		return a.engine
	}
	var zero T
	if _, isF32 := any(zero).(float32); !isF32 {
		return nil
	}
	if _, ok := accum.GetStorage().(*tensor.GPUStorage[T]); !ok {
		return nil
	}
	if _, ok := grad.GetStorage().(*tensor.GPUStorage[T]); !ok {
		return nil
	}
	return g.Engine()
}

// storagesIdentical reports whether two tensors share the same storage
// object (pointer identity through the Storage interface).
func storagesIdentical[T tensor.Numeric](x, y *tensor.TensorNumeric[T]) bool {
	return x.GetStorage() == y.GetStorage()
}

// arenaBackedStorage reports whether t's storage is backed by an arena pool
// that can recycle the memory behind a live reference. It reuses the ADR 006
// save-for-backward detection mechanism (zerfoo/ztensor#128): storage that
// implements tensor.PinnableStorage AND whose backing pool actually takes a
// pin is arena-backed. The probe pin is released immediately; it exists only
// as a detection handshake, never as a retention mechanism (see the package
// comment for why pinning gradients is rejected).
func arenaBackedStorage[T tensor.Numeric](t *tensor.TensorNumeric[T]) bool {
	p, ok := t.GetStorage().(tensor.PinnableStorage)
	if !ok {
		return false
	}
	if !p.PinForBackward() {
		return false
	}
	p.UnpinForBackward()
	return true
}

// newPersistentGradTensor allocates a zeroed accumulator with the same shape
// as grad whose storage is guaranteed NOT to be arena-backed:
//
//   - For device gradients (*tensor.GPUStorage) it allocates a raw
//     runtime.Malloc-backed GPUStorage (pool == nil) on the same device via
//     tensor.NewGPUStorageFromSlice. A nil-pool GPUStorage is never touched
//     by engine ResetPool, and a same-length storage Set/TrySet is an
//     in-place copy (no realloc), so the device pointer is stable across the
//     optimizer's in-place MulScalar / Set-zeroing (zerfoo#845).
//   - For any other pinnable storage (host-backed arenas in tests) it
//     allocates a plain GC-owned host tensor, which cannot be recycled
//     behind a live reference.
func newPersistentGradTensor[T tensor.Numeric](grad *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := grad.Shape()
	if gs, ok := grad.GetStorage().(*tensor.GPUStorage[T]); ok {
		ps, err := tensor.NewGPUStorageFromSlice(make([]T, gs.Len()), gs.DeviceID())
		if err != nil {
			return nil, err
		}
		return tensor.NewWithStorage[T](shape, ps)
	}
	return tensor.New[T](shape, nil)
}

// addInto performs accum += grad without ever swapping accum's storage.
// eng, when non-nil (see engineFor), performs the add as an in-place device
// kernel; otherwise the host read-modify-writeback fallback is used.
func (a *gradAccumulator[T]) addInto(ctx context.Context, eng compute.Engine[T], accum, grad *tensor.TensorNumeric[T], name string) error {
	if accum.GetStorage().Len() != grad.GetStorage().Len() {
		return fmt.Errorf("training: gradient accumulator for %q has %d elements but gradient has %d",
			name, accum.GetStorage().Len(), grad.GetStorage().Len())
	}

	if eng != nil {
		before := accum.GetStorage()
		res, err := eng.Add(ctx, accum, grad, accum)
		if err != nil {
			return fmt.Errorf("training: accumulating gradient for %q: %w", name, err)
		}
		// The engine must write IN PLACE into the provided dst storage
		// (ztensor#84 dst reuse). If it relocated the accumulator (e.g. a
		// host dst handed to a GPU engine gets re-homed into the arena via
		// SetStorage), the persistent buffer would silently become
		// arena-backed -- the exact bug this hook exists to fix.
		if res != accum || res.GetStorage() != before {
			return fmt.Errorf("training: engine.Add relocated the persistent gradient accumulator for %q; the engine must write in place into dst", name)
		}
		return nil
	}

	// Host fallback: read-modify-writeback through the storage interface.
	// Data() is the backing slice for host storage and a D2H copy for GPU
	// storage; the writeback below makes the result land in the accumulator's
	// existing buffer in both cases.
	dst := accum.Data()
	src := grad.Data()
	if err := addSlice(dst, src); err != nil {
		return fmt.Errorf("training: accumulating gradient for %q: %w", name, err)
	}
	if gs, ok := accum.GetStorage().(*tensor.GPUStorage[T]); ok {
		if err := gs.TrySet(dst); err != nil {
			return fmt.Errorf("training: writing back gradient accumulator for %q: %w", name, err)
		}
		return nil
	}
	accum.GetStorage().Set(dst)
	return nil
}

// addSlice performs dst[i] += src[i] for the built-in numeric types.
// Minifloat types (float16/bfloat16/float8) are not supported on this host
// fallback path; configure an engine via SetEngine for those. In practice
// they cannot reach here: arena-backed gradients only come from the GPU
// engine, whose kernels are float32-gated.
func addSlice[T tensor.Numeric](dst, src []T) error {
	if len(dst) != len(src) {
		return errors.New("length mismatch")
	}
	switch d := any(dst).(type) {
	case []float32:
		s := any(src).([]float32)
		for i := range d {
			d[i] += s[i]
		}
	case []float64:
		s := any(src).([]float64)
		for i := range d {
			d[i] += s[i]
		}
	case []int:
		s := any(src).([]int)
		for i := range d {
			d[i] += s[i]
		}
	case []int8:
		s := any(src).([]int8)
		for i := range d {
			d[i] += s[i]
		}
	case []int16:
		s := any(src).([]int16)
		for i := range d {
			d[i] += s[i]
		}
	case []int32:
		s := any(src).([]int32)
		for i := range d {
			d[i] += s[i]
		}
	case []int64:
		s := any(src).([]int64)
		for i := range d {
			d[i] += s[i]
		}
	case []uint:
		s := any(src).([]uint)
		for i := range d {
			d[i] += s[i]
		}
	case []uint8:
		s := any(src).([]uint8)
		for i := range d {
			d[i] += s[i]
		}
	case []uint32:
		s := any(src).([]uint32)
		for i := range d {
			d[i] += s[i]
		}
	case []uint64:
		s := any(src).([]uint64)
		for i := range d {
			d[i] += s[i]
		}
	default:
		return errors.New("unsupported numeric type for host gradient accumulation; set an engine on the strategy")
	}
	return nil
}
