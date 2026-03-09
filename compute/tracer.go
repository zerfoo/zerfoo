package compute

import (
	"reflect"

	"github.com/zerfoo/zerfoo/tensor"
)

// TracedOp records a single engine operation with slot-based tensor identity.
type TracedOp struct {
	OpName    string
	InputIDs  []int          // slot indices for inputs
	OutputID  int            // slot index for output
	OutputIDs []int          // for multi-output ops like Split
	ExtraArgs map[string]any // axis, scalar value, shape, etc.
}

// Tracer records engine operations and tracks tensor identity by pointer.
// It assigns each unique tensor pointer a slot index, enabling later
// compilation into an ExecutionPlan.
type Tracer[T tensor.Numeric] struct {
	ops          []TracedOp
	tensorMap    map[uintptr]int  // tensor pointer -> slot index
	nextSlot     int
	frozen       map[uintptr]bool // known frozen tensors (weights)
	shapes       map[int][]int    // slot -> shape
	hasOpaqueOps bool
}

// NewTracer creates a Tracer and pre-registers frozen tensors (model weights)
// with slot indices.
func NewTracer[T tensor.Numeric](frozenTensors []*tensor.TensorNumeric[T]) *Tracer[T] {
	t := &Tracer[T]{
		tensorMap: make(map[uintptr]int),
		frozen:    make(map[uintptr]bool),
		shapes:    make(map[int][]int),
	}
	for _, ft := range frozenTensors {
		if ft == nil {
			continue
		}
		ptr := ptrOf(ft)
		slot := t.nextSlot
		t.nextSlot++
		t.tensorMap[ptr] = slot
		t.frozen[ptr] = true
		t.shapes[slot] = ft.Shape()
	}
	return t
}

// Record appends a TracedOp for a single-output operation.
func (t *Tracer[T]) Record(opName string, inputs []*tensor.TensorNumeric[T], output *tensor.TensorNumeric[T], extra map[string]any) {
	inputIDs := make([]int, len(inputs))
	for i, in := range inputs {
		inputIDs[i] = t.slotFor(in)
	}
	outID := t.slotFor(output)
	t.ops = append(t.ops, TracedOp{
		OpName:    opName,
		InputIDs:  inputIDs,
		OutputID:  outID,
		ExtraArgs: extra,
	})
}

// slotFor returns the existing slot for a tensor or assigns a new one.
func (t *Tracer[T]) slotFor(tn *tensor.TensorNumeric[T]) int {
	ptr := ptrOf(tn)
	if slot, ok := t.tensorMap[ptr]; ok {
		return slot
	}
	slot := t.nextSlot
	t.nextSlot++
	t.tensorMap[ptr] = slot
	if tn != nil {
		t.shapes[slot] = tn.Shape()
	}
	return slot
}

// TracedOps returns the recorded operations in order.
func (t *Tracer[T]) TracedOps() []TracedOp {
	return t.ops
}

// SlotFor returns the existing slot for a tensor or assigns a new one.
// This is the exported version of slotFor.
func (t *Tracer[T]) SlotFor(tn *tensor.TensorNumeric[T]) int {
	return t.slotFor(tn)
}

// NextSlot returns the next slot index that would be assigned.
// This indicates the total number of slots allocated.
func (t *Tracer[T]) NextSlot() int {
	return t.nextSlot
}

// SlotShapes returns the shape for each slot index.
func (t *Tracer[T]) SlotShapes() map[int][]int {
	return t.shapes
}

// FrozenSlots returns the slot indices of frozen (weight) tensors.
func (t *Tracer[T]) FrozenSlots() []int {
	var slots []int
	for ptr, slot := range t.tensorMap {
		if t.frozen[ptr] {
			slots = append(slots, slot)
		}
	}
	return slots
}

// HasOpaqueOps reports whether any opaque (non-traceable) operations
// were encountered during tracing.
func (t *Tracer[T]) HasOpaqueOps() bool {
	return t.hasOpaqueOps
}

// MarkOpaque marks the trace as containing opaque operations.
func (t *Tracer[T]) MarkOpaque() {
	t.hasOpaqueOps = true
}

// RecordMultiOutput appends a TracedOp for a multi-output operation (e.g., Split).
func (t *Tracer[T]) RecordMultiOutput(opName string, inputs []*tensor.TensorNumeric[T], outputs []*tensor.TensorNumeric[T], extra map[string]any) {
	inputIDs := make([]int, len(inputs))
	for i, in := range inputs {
		inputIDs[i] = t.slotFor(in)
	}
	outputIDs := make([]int, len(outputs))
	for i, out := range outputs {
		outputIDs[i] = t.slotFor(out)
	}
	outID := -1
	if len(outputIDs) > 0 {
		outID = outputIDs[0]
	}
	t.ops = append(t.ops, TracedOp{
		OpName:    opName,
		InputIDs:  inputIDs,
		OutputID:  outID,
		OutputIDs: outputIDs,
		ExtraArgs: extra,
	})
}

// RecordGather appends a TracedOp for Gather which uses int indices.
func (t *Tracer[T]) RecordGather(params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T], extra map[string]any) {
	paramsID := t.slotFor(params)
	indicesID := t.slotForIntTensor(indices)
	outID := t.slotFor(output)
	t.ops = append(t.ops, TracedOp{
		OpName:    "Gather",
		InputIDs:  []int{paramsID, indicesID},
		OutputID:  outID,
		ExtraArgs: extra,
	})
}

// slotForIntTensor returns the existing slot for an int tensor or assigns a new one.
func (t *Tracer[T]) slotForIntTensor(tn *tensor.TensorNumeric[int]) int {
	ptr := reflect.ValueOf(tn).Pointer()
	if slot, ok := t.tensorMap[ptr]; ok {
		return slot
	}
	slot := t.nextSlot
	t.nextSlot++
	t.tensorMap[ptr] = slot
	if tn != nil {
		t.shapes[slot] = tn.Shape()
	}
	return slot
}

// ptrOf returns the pointer identity of a tensor as a uintptr.
func ptrOf[T tensor.Numeric](t *tensor.TensorNumeric[T]) uintptr {
	return reflect.ValueOf(t).Pointer()
}
