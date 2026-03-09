package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// mockF32Node is a test node for float32 graphs.
type mockF32Node struct {
	name        string
	outputShape []int
	forwardFunc func(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
}

func (m *mockF32Node) OpType() string                     { return m.name }
func (m *mockF32Node) Attributes() map[string]any         { return nil }
func (m *mockF32Node) OutputShape() []int                 { return m.outputShape }
func (m *mockF32Node) Parameters() []*Parameter[float32]  { return nil }

func (m *mockF32Node) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if m.forwardFunc != nil {
		return m.forwardFunc(ctx, inputs...)
	}
	return inputs[0], nil
}

func (m *mockF32Node) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func TestExecutionPlanRun(t *testing.T) {
	// Build a simple 2-instruction plan: input -> double -> add10
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
			{Forward: double, InputIdx: []int{0}, OutputIdx: 1},
			{Forward: add10, InputIdx: []int{1}, OutputIdx: 2},
		},
		slots:     make([]*tensor.TensorNumeric[float32], 3),
		inputIdx:  []int{0},
		outputIdx: 2,
	}

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	result, err := plan.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := []float32{12, 14, 16, 18} // (x*2)+10
	got := result.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestExecutionPlanRunDiamond(t *testing.T) {
	// Diamond: input -> branch1 (*2), input -> branch2 (*3), merge (branch1 + branch2)
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
			{Forward: mul2, InputIdx: []int{0}, OutputIdx: 1},
			{Forward: mul3, InputIdx: []int{0}, OutputIdx: 2},
			{Forward: add, InputIdx: []int{1, 2}, OutputIdx: 3},
		},
		slots:     make([]*tensor.TensorNumeric[float32], 4),
		inputIdx:  []int{0},
		outputIdx: 3,
	}

	input, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})
	result, err := plan.Run(context.Background(), input)
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

func TestInstructionMeta(t *testing.T) {
	// Build graph: input -> Add(input, constant) -> output
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()
	b := NewBuilder[float32](engine)

	in := b.Input([]int{1, 4})

	constData, _ := tensor.New[float32]([]int{1, 4}, []float32{10, 20, 30, 40})
	constNode := &mockF32Node{
		name:        "Constant",
		outputShape: []int{1, 4},
		forwardFunc: func(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			return constData, nil
		},
	}
	cst := b.AddNode(constNode)

	addNode := &mockF32Node{
		name:        "Add",
		outputShape: []int{1, 4},
		forwardFunc: func(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			a := inputs[0].Data()
			bv := inputs[1].Data()
			out := make([]float32, len(a))
			for i := range a {
				out[i] = a[i] + bv[i]
			}
			return tensor.New[float32](inputs[0].Shape(), out)
		},
	}
	sum := b.AddNode(addNode, in, cst)

	g, err := b.Build(sum)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	ctx := context.Background()

	plan, err := g.Compile(ctx, input)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	// T91.1 acceptance: Instructions() returns correct metadata.
	metas := plan.Instructions()
	if len(metas) != 1 {
		t.Fatalf("Instructions: got %d, want 1", len(metas))
	}
	if metas[0].OpName != "Add" {
		t.Errorf("OpName = %q, want %q", metas[0].OpName, "Add")
	}
	if len(metas[0].InputIdx) != 2 {
		t.Errorf("InputIdx len = %d, want 2", len(metas[0].InputIdx))
	}

	// SlotShapes: all slots should have shapes from warmup.
	shapes := plan.SlotShapes()
	if len(shapes) == 0 {
		t.Fatal("SlotShapes: empty")
	}
	// The output slot should be [1, 4].
	outShape := shapes[metas[0].OutputIdx]
	if len(outShape) != 2 || outShape[0] != 1 || outShape[1] != 4 {
		t.Errorf("output slot shape = %v, want [1 4]", outShape)
	}

	// FrozenSlots: should include the constant.
	frozen := plan.FrozenSlots()
	if len(frozen) != 1 {
		t.Fatalf("FrozenSlots: got %d, want 1", len(frozen))
	}
	if frozen[0].Data == nil {
		t.Error("FrozenSlot data is nil")
	}

	// InputIdx and OutputIdx.
	if len(plan.InputSlots()) != 1 {
		t.Errorf("InputSlots: got %d, want 1", len(plan.InputSlots()))
	}
	if plan.OutputSlot() < 0 {
		t.Error("OutputSlot is negative")
	}
}

func TestExecutionPlanMegakernelFn(t *testing.T) {
	tests := []struct {
		name       string
		setFn      bool
		wantValues []float32
	}{
		{
			name:       "megakernel overrides instruction loop",
			setFn:      true,
			wantValues: []float32{99, 99, 99, 99},
		},
		{
			name:       "nil megakernel uses instruction loop",
			setFn:      false,
			wantValues: []float32{12, 14, 16, 18}, // (x*2)+10
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
				slots:     make([]*tensor.TensorNumeric[float32], 3),
				inputIdx:  []int{0},
				outputIdx: 2,
			}

			if tt.setFn {
				plan.SetMegakernelFn(func(_ context.Context, _ []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
					return tensor.New[float32]([]int{1, 4}, []float32{99, 99, 99, 99})
				})
			}

			input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
			result, err := plan.Run(context.Background(), input)
			if err != nil {
				t.Fatalf("Run: %v", err)
			}

			got := result.Data()
			for i, want := range tt.wantValues {
				if got[i] != want {
					t.Errorf("result[%d] = %v, want %v", i, got[i], want)
				}
			}
		})
	}
}

func TestGraphCompile(t *testing.T) {
	// Build graph: input -> double -> add3 -> output
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()
	b := NewBuilder[float32](engine)

	in := b.Input([]int{1, 4})

	doubleNode := &mockF32Node{
		name:        "Double",
		outputShape: []int{1, 4},
		forwardFunc: func(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			data := inputs[0].Data()
			out := make([]float32, len(data))
			for i, v := range data {
				out[i] = v * 2
			}
			return tensor.New[float32](inputs[0].Shape(), out)
		},
	}
	add3Node := &mockF32Node{
		name:        "Add3",
		outputShape: []int{1, 4},
		forwardFunc: func(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			data := inputs[0].Data()
			out := make([]float32, len(data))
			for i, v := range data {
				out[i] = v + 3
			}
			return tensor.New[float32](inputs[0].Shape(), out)
		},
	}

	doubled := b.AddNode(doubleNode, in)
	output := b.AddNode(add3Node, doubled)

	g, err := b.Build(output)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	ctx := context.Background()

	// Compile the graph.
	plan, err := g.Compile(ctx, input)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	// Verify compiled plan produces correct output.
	result, err := plan.Run(ctx, input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := []float32{5, 7, 9, 11} // (x*2)+3
	got := result.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Verify compiled and interpreted produce same output.
	interpResult, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	interpData := interpResult.Data()
	for i := range want {
		if interpData[i] != got[i] {
			t.Errorf("compiled vs interpreted mismatch at [%d]: %v vs %v", i, got[i], interpData[i])
		}
	}

	// Verify reuse with different input.
	input2, _ := tensor.New[float32]([]int{1, 4}, []float32{10, 20, 30, 40})
	result2, err := plan.Run(ctx, input2)
	if err != nil {
		t.Fatalf("Run2: %v", err)
	}
	want2 := []float32{23, 43, 63, 83}
	got2 := result2.Data()
	for i := range want2 {
		if got2[i] != want2[i] {
			t.Errorf("result2[%d] = %v, want %v", i, got2[i], want2[i])
		}
	}
}
