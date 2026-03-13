package graph

import "github.com/zerfoo/zerfoo/tensor"

// BufferLayout describes a contiguous pre-allocated buffer with fixed offsets
// for each slot in an ExecutionPlan. CUDA graph capture requires that device
// memory addresses remain stable across runs; this layout ensures every
// intermediate tensor occupies the same offset in every execution.
type BufferLayout struct {
	// Offsets[i] is the element offset of slot i into the contiguous buffer.
	// A value of -1 means the slot is not part of the contiguous buffer
	// (e.g. frozen or input slots that are managed externally).
	Offsets []int

	// Sizes[i] is the element count for slot i (product of its shape dims).
	// Zero for slots without a known shape.
	Sizes []int

	// TotalElements is the sum of all slot sizes that are part of the
	// contiguous buffer.
	TotalElements int
}

// ComputeBufferLayout computes element offsets for each slot based on the
// slot shapes from compilation. Frozen and input slots are excluded (offset -1)
// since they are managed externally (model weights are constant, inputs change).
func ComputeBufferLayout(slotShapes [][]int, frozenIdx []int, inputIdx []int) BufferLayout {
	excluded := make(map[int]bool, len(frozenIdx)+len(inputIdx))
	for _, idx := range frozenIdx {
		excluded[idx] = true
	}
	for _, idx := range inputIdx {
		excluded[idx] = true
	}

	offsets := make([]int, len(slotShapes))
	sizes := make([]int, len(slotShapes))
	offset := 0

	for i, shape := range slotShapes {
		if excluded[i] || len(shape) == 0 {
			offsets[i] = -1
			continue
		}
		n := 1
		for _, d := range shape {
			n *= d
		}
		sizes[i] = n
		offsets[i] = offset
		offset += n
	}

	return BufferLayout{
		Offsets:       offsets,
		Sizes:         sizes,
		TotalElements: offset,
	}
}

// PreallocateBuffers creates pre-allocated tensors for all intermediate slots
// in the execution plan based on the slot shapes determined during compilation.
// After calling this method, RunInstructions will copy each Forward() result
// into the pre-allocated buffer, keeping memory addresses stable across runs.
//
// Frozen and input slots are excluded since they are managed externally.
func (p *ExecutionPlan[T]) PreallocateBuffers() {
	layout := ComputeBufferLayout(p.slotShapes, p.frozenIdx, p.inputIdx)
	if layout.TotalElements == 0 {
		return
	}

	// Allocate one contiguous backing buffer for all intermediate tensors.
	backing := make([]T, layout.TotalElements)

	preallocated := make([]*tensor.TensorNumeric[T], len(p.slotShapes))
	for i, shape := range p.slotShapes {
		if layout.Offsets[i] < 0 || layout.Sizes[i] == 0 {
			continue
		}
		region := backing[layout.Offsets[i] : layout.Offsets[i]+layout.Sizes[i]]
		t, err := tensor.New[T](shape, region)
		if err != nil {
			continue
		}
		preallocated[i] = t
	}

	p.bufferLayout = &layout
	p.preallocated = preallocated
}

// HasPreallocatedBuffers reports whether buffers have been pre-allocated.
func (p *ExecutionPlan[T]) HasPreallocatedBuffers() bool {
	return p.bufferLayout != nil
}

// BufferLayout returns the computed buffer layout, or nil if buffers have
// not been pre-allocated.
func (p *ExecutionPlan[T]) BufferLayout() *BufferLayout {
	return p.bufferLayout
}
