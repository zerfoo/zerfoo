package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

func TestComputeBufferLayout(t *testing.T) {
	tests := []struct {
		name          string
		slotShapes    [][]int
		frozenIdx     []int
		inputIdx      []int
		wantOffsets   []int
		wantSizes     []int
		wantTotal     int
	}{
		{
			name:        "single intermediate slot",
			slotShapes:  [][]int{{1, 4}, {1, 4}},
			frozenIdx:   nil,
			inputIdx:    []int{0},
			wantOffsets: []int{-1, 0},
			wantSizes:   []int{0, 4},
			wantTotal:   4,
		},
		{
			name:        "frozen and input excluded",
			slotShapes:  [][]int{{1, 4}, {2, 3}, {1, 4}, {1, 2}},
			frozenIdx:   []int{1},
			inputIdx:    []int{0},
			wantOffsets: []int{-1, -1, 0, 4},
			wantSizes:   []int{0, 0, 4, 2},
			wantTotal:   6,
		},
		{
			name:        "all frozen or input",
			slotShapes:  [][]int{{1, 4}, {2, 3}},
			frozenIdx:   []int{1},
			inputIdx:    []int{0},
			wantOffsets: []int{-1, -1},
			wantSizes:   []int{0, 0},
			wantTotal:   0,
		},
		{
			name:        "nil shape slots skipped",
			slotShapes:  [][]int{{1, 4}, nil, {2, 3}},
			frozenIdx:   nil,
			inputIdx:    []int{0},
			wantOffsets: []int{-1, -1, 0},
			wantSizes:   []int{0, 0, 6},
			wantTotal:   6,
		},
		{
			name:        "multiple intermediates contiguous",
			slotShapes:  [][]int{{1, 4}, {1, 4}, {1, 8}, {1, 4}},
			frozenIdx:   nil,
			inputIdx:    []int{0},
			wantOffsets: []int{-1, 0, 4, 12},
			wantSizes:   []int{0, 4, 8, 4},
			wantTotal:   16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layout := ComputeBufferLayout(tt.slotShapes, tt.frozenIdx, tt.inputIdx)

			if layout.TotalElements != tt.wantTotal {
				t.Errorf("TotalElements = %d, want %d", layout.TotalElements, tt.wantTotal)
			}
			if len(layout.Offsets) != len(tt.wantOffsets) {
				t.Fatalf("Offsets length = %d, want %d", len(layout.Offsets), len(tt.wantOffsets))
			}
			for i := range tt.wantOffsets {
				if layout.Offsets[i] != tt.wantOffsets[i] {
					t.Errorf("Offsets[%d] = %d, want %d", i, layout.Offsets[i], tt.wantOffsets[i])
				}
			}
			for i := range tt.wantSizes {
				if layout.Sizes[i] != tt.wantSizes[i] {
					t.Errorf("Sizes[%d] = %d, want %d", i, layout.Sizes[i], tt.wantSizes[i])
				}
			}
		})
	}
}

func TestPreallocateBuffers(t *testing.T) {
	// Build a plan: input(slot 0) -> double(slot 1) -> add10(slot 2)
	double := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i := range in {
			out[i] = in[i] * 2
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}
	add10 := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i := range in {
			out[i] = in[i] + 10
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}

	plan := &ExecutionPlan[float32]{
		instructions: []Instruction[float32]{
			{Forward: double, InputIdx: []int{0}, OutputIdx: 1, OpName: "Double"},
			{Forward: add10, InputIdx: []int{1}, OutputIdx: 2, OpName: "Add10"},
		},
		slots:      make([]*tensor.TensorNumeric[float32], 3),
		slotShapes: [][]int{{1, 4}, {1, 4}, {1, 4}},
		inputIdx:   []int{0},
		outputIdx:  2,
	}

	plan.PreallocateBuffers()

	if !plan.HasPreallocatedBuffers() {
		t.Fatal("HasPreallocatedBuffers() = false after PreallocateBuffers()")
	}

	layout := plan.BufferLayout()
	if layout == nil {
		t.Fatal("BufferLayout() = nil after PreallocateBuffers()")
	}

	// Input slot should be excluded.
	if layout.Offsets[0] != -1 {
		t.Errorf("input slot offset = %d, want -1", layout.Offsets[0])
	}
	// Intermediate slots should have valid offsets.
	if layout.Offsets[1] < 0 {
		t.Errorf("slot 1 offset = %d, want >= 0", layout.Offsets[1])
	}
	if layout.Offsets[2] < 0 {
		t.Errorf("slot 2 offset = %d, want >= 0", layout.Offsets[2])
	}
	if layout.TotalElements != 8 { // 4 + 4
		t.Errorf("TotalElements = %d, want 8", layout.TotalElements)
	}
}

func TestPreallocatedRunMatchesDynamic(t *testing.T) {
	// Verify that pre-allocated execution produces the same results
	// as dynamic allocation.
	double := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i := range in {
			out[i] = in[i] * 2
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}
	add10 := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i := range in {
			out[i] = in[i] + 10
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}

	makePlan := func() *ExecutionPlan[float32] {
		return &ExecutionPlan[float32]{
			instructions: []Instruction[float32]{
				{Forward: double, InputIdx: []int{0}, OutputIdx: 1, OpName: "Double"},
				{Forward: add10, InputIdx: []int{1}, OutputIdx: 2, OpName: "Add10"},
			},
			slots:      make([]*tensor.TensorNumeric[float32], 3),
			slotShapes: [][]int{{1, 4}, {1, 4}, {1, 4}},
			inputIdx:   []int{0},
			outputIdx:  2,
		}
	}

	ctx := context.Background()

	tests := []struct {
		name  string
		input []float32
		want  []float32
	}{
		{
			name:  "basic",
			input: []float32{1, 2, 3, 4},
			want:  []float32{12, 14, 16, 18},
		},
		{
			name:  "zeros",
			input: []float32{0, 0, 0, 0},
			want:  []float32{10, 10, 10, 10},
		},
		{
			name:  "negative",
			input: []float32{-5, -3, -1, 0},
			want:  []float32{0, 4, 8, 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, _ := tensor.New[float32]([]int{1, 4}, tt.input)

			// Dynamic allocation.
			dynPlan := makePlan()
			dynResult, err := dynPlan.Run(ctx, input)
			if err != nil {
				t.Fatalf("dynamic Run: %v", err)
			}

			// Pre-allocated.
			prePlan := makePlan()
			prePlan.PreallocateBuffers()
			preResult, err := prePlan.Run(ctx, input)
			if err != nil {
				t.Fatalf("preallocated Run: %v", err)
			}

			dynData := dynResult.Data()
			preData := preResult.Data()
			if len(dynData) != len(preData) {
				t.Fatalf("length mismatch: dynamic=%d, preallocated=%d", len(dynData), len(preData))
			}
			for i := range tt.want {
				if dynData[i] != tt.want[i] {
					t.Errorf("dynamic[%d] = %v, want %v", i, dynData[i], tt.want[i])
				}
				if preData[i] != tt.want[i] {
					t.Errorf("preallocated[%d] = %v, want %v", i, preData[i], tt.want[i])
				}
			}
		})
	}
}

func TestPreallocatedStableAddresses(t *testing.T) {
	// Verify that pre-allocated buffers return the same tensor pointers
	// across multiple runs (stable addresses for CUDA graph capture).
	passthrough := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		copy(out, in)
		return tensor.New[float32](inputs[0].Shape(), out)
	}

	plan := &ExecutionPlan[float32]{
		instructions: []Instruction[float32]{
			{Forward: passthrough, InputIdx: []int{0}, OutputIdx: 1, OpName: "Copy"},
		},
		slots:      make([]*tensor.TensorNumeric[float32], 2),
		slotShapes: [][]int{{1, 4}, {1, 4}},
		inputIdx:   []int{0},
		outputIdx:  1,
	}

	plan.PreallocateBuffers()

	ctx := context.Background()
	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})

	// Run twice and verify the output tensor pointer is the same.
	result1, err := plan.Run(ctx, input)
	if err != nil {
		t.Fatalf("Run 1: %v", err)
	}
	ptr1 := result1

	input2, _ := tensor.New[float32]([]int{1, 4}, []float32{5, 6, 7, 8})
	result2, err := plan.Run(ctx, input2)
	if err != nil {
		t.Fatalf("Run 2: %v", err)
	}
	ptr2 := result2

	if ptr1 != ptr2 {
		t.Error("output tensor pointer changed between runs; addresses are not stable")
	}

	// Verify data was updated correctly.
	got := result2.Data()
	want := []float32{5, 6, 7, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result2[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestPreallocatedDiamondGraph(t *testing.T) {
	// Diamond: input -> branch1 (*2), input -> branch2 (*3), merge (add)
	mul2 := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i := range in {
			out[i] = in[i] * 2
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}
	mul3 := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		in := inputs[0].Data()
		out := make([]float32, len(in))
		for i := range in {
			out[i] = in[i] * 3
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}
	add := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		a := inputs[0].Data()
		b := inputs[1].Data()
		out := make([]float32, len(a))
		for i := range a {
			out[i] = a[i] + b[i]
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}

	plan := &ExecutionPlan[float32]{
		instructions: []Instruction[float32]{
			{Forward: mul2, InputIdx: []int{0}, OutputIdx: 1, OpName: "Mul2"},
			{Forward: mul3, InputIdx: []int{0}, OutputIdx: 2, OpName: "Mul3"},
			{Forward: add, InputIdx: []int{1, 2}, OutputIdx: 3, OpName: "Add"},
		},
		slots:      make([]*tensor.TensorNumeric[float32], 4),
		slotShapes: [][]int{{1, 3}, {1, 3}, {1, 3}, {1, 3}},
		inputIdx:   []int{0},
		outputIdx:  3,
	}

	plan.PreallocateBuffers()

	ctx := context.Background()
	input, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})
	result, err := plan.Run(ctx, input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := []float32{5, 10, 15} // x*2 + x*3 = x*5
	got := result.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestPreallocatedWithFrozenSlots(t *testing.T) {
	// Plan with a frozen (constant) slot that should not be pre-allocated.
	constData, _ := tensor.New[float32]([]int{1, 4}, []float32{10, 20, 30, 40})

	add := func(_ context.Context, inputs []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		a := inputs[0].Data()
		b := inputs[1].Data()
		out := make([]float32, len(a))
		for i := range a {
			out[i] = a[i] + b[i]
		}
		return tensor.New[float32](inputs[0].Shape(), out)
	}

	slots := make([]*tensor.TensorNumeric[float32], 3)
	slots[1] = constData // frozen

	plan := &ExecutionPlan[float32]{
		instructions: []Instruction[float32]{
			{Forward: add, InputIdx: []int{0, 1}, OutputIdx: 2, OpName: "Add"},
		},
		slots:      slots,
		slotShapes: [][]int{{1, 4}, {1, 4}, {1, 4}},
		inputIdx:   []int{0},
		outputIdx:  2,
		frozenIdx:  []int{1},
	}

	plan.PreallocateBuffers()

	layout := plan.BufferLayout()
	// Frozen slot 1 should be excluded.
	if layout.Offsets[1] != -1 {
		t.Errorf("frozen slot offset = %d, want -1", layout.Offsets[1])
	}
	// Only output slot 2 should be allocated.
	if layout.TotalElements != 4 {
		t.Errorf("TotalElements = %d, want 4", layout.TotalElements)
	}

	ctx := context.Background()
	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	result, err := plan.Run(ctx, input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := []float32{11, 22, 33, 44}
	got := result.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestPreallocatedNoShapes(t *testing.T) {
	// Plan with no slot shapes should be a no-op.
	plan := &ExecutionPlan[float32]{
		slots:    make([]*tensor.TensorNumeric[float32], 2),
		inputIdx: []int{0},
	}

	plan.PreallocateBuffers()

	if plan.HasPreallocatedBuffers() {
		t.Error("HasPreallocatedBuffers() = true when no shapes available")
	}
}
