package graph

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- Graph getter methods ----------

func TestGraph_Inputs(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in1 := b.Input([]int{2})
	in2 := b.Input([]int{3})
	node := &mockNode{name: "n"}
	b.AddNode(node, in1, in2)

	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	inputs := g.Inputs()
	if len(inputs) != 2 {
		t.Fatalf("Inputs() len = %d, want 2", len(inputs))
	}
	if inputs[0] != in1 || inputs[1] != in2 {
		t.Error("Inputs() returned wrong nodes")
	}
}

func TestGraph_Output(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})
	node := &mockNode{name: "n"}
	b.AddNode(node, in)

	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	if g.Output() != node {
		t.Error("Output() returned wrong node")
	}
}

func TestGraph_Nodes(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})
	n1 := &mockNode{name: "n1"}
	b.AddNode(n1, in)

	g, err := b.Build(n1)
	if err != nil {
		t.Fatal(err)
	}

	nodes := g.Nodes()
	if len(nodes) != 2 { // inputNode + n1
		t.Errorf("Nodes() len = %d, want 2", len(nodes))
	}
}

func TestGraph_Dependencies(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})
	n1 := &mockNode{name: "n1"}
	b.AddNode(n1, in)

	g, err := b.Build(n1)
	if err != nil {
		t.Fatal(err)
	}

	deps := g.Dependencies(n1)
	if len(deps) != 1 || deps[0] != in {
		t.Errorf("Dependencies(n1) = %v, want [inputNode]", deps)
	}

	deps2 := g.Dependencies(in)
	if len(deps2) != 0 {
		t.Errorf("Dependencies(input) = %v, want []", deps2)
	}
}

func TestGraph_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})

	p1 := &Parameter[int]{Name: "w1"}
	p2 := &Parameter[int]{Name: "w2"}
	n1 := &mockNode{name: "n1", params: []*Parameter[int]{p1}}
	n2 := &mockNode{name: "n2", params: []*Parameter[int]{p2}}
	b.AddNode(n1, in)
	b.AddNode(n2, n1)

	g, err := b.Build(n2)
	if err != nil {
		t.Fatal(err)
	}

	params := g.Parameters()
	if len(params) != 2 {
		t.Errorf("Parameters() len = %d, want 2", len(params))
	}
}

// ---------- inputNode interface ----------

func TestInputNode_OpType(t *testing.T) {
	n := &inputNode[int]{shape: []int{2, 3}}
	if n.OpType() != "Input" {
		t.Errorf("OpType = %q, want %q", n.OpType(), "Input")
	}
}

func TestInputNode_Attributes(t *testing.T) {
	n := &inputNode[int]{shape: []int{2, 3}}
	attrs := n.Attributes()
	if attrs == nil || len(attrs) != 0 {
		t.Errorf("Attributes = %v, want empty map", attrs)
	}
}

// ---------- NoParameters ----------

func TestNoParameters(t *testing.T) {
	np := &NoParameters[float32]{}
	if np.Parameters() != nil {
		t.Error("NoParameters.Parameters() should return nil")
	}
}

// ---------- Forward error paths ----------

func TestGraph_Forward_InputCountMismatch(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	b.Input([]int{2})
	b.Input([]int{3})
	node := &mockNode{name: "n"}
	b.AddNode(node, b.nodes[0], b.nodes[1])

	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	input1, _ := tensor.New[int]([]int{2}, []int{1, 2})
	// Only 1 input when 2 expected
	_, err = g.Forward(context.Background(), input1)
	if err == nil {
		t.Error("expected error for input count mismatch")
	}
}

func TestGraph_Forward_NodeError(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})
	failNode := &mockNode{
		name: "fail",
		forwardFunc: func(_ context.Context, _ ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			return nil, fmt.Errorf("forward failed")
		},
	}
	b.AddNode(failNode, in)

	g, err := b.Build(failNode)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[int]([]int{2}, []int{1, 2})
	_, err = g.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected error from node Forward")
	}
}

// ---------- Backward error paths ----------

func TestGraph_Backward_NodeError(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})
	failNode := &mockNode{
		name: "fail",
		backwardFunc: func(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
			return nil, fmt.Errorf("backward failed")
		},
	}
	b.AddNode(failNode, in)

	g, err := b.Build(failNode)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[int]([]int{2}, []int{1, 2})
	_, err = g.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	grad, _ := tensor.New[int]([]int{2}, []int{1, 1})
	err = g.Backward(context.Background(), types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error from node Backward")
	}
}

// ---------- Gradient accumulation (diamond graph) ----------

func TestGraph_Backward_GradientAccumulation(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})

	// Two paths from input -> left/right -> merge
	left := &mockNode{name: "left"}
	right := &mockNode{name: "right"}
	merge := &mockNode{
		name: "merge",
		forwardFunc: func(_ context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
			// Return first input for simplicity
			return inputs[0], nil
		},
		backwardFunc: func(_ context.Context, _ types.BackwardMode, grad *tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
			// Pass gradient to both inputs
			return []*tensor.TensorNumeric[int]{grad, grad}, nil
		},
	}

	b.AddNode(left, in)
	b.AddNode(right, in)
	b.AddNode(merge, left, right)

	g, err := b.Build(merge)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[int]([]int{2}, []int{1, 2})
	_, err = g.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	grad, _ := tensor.New[int]([]int{2}, []int{1, 1})
	err = g.Backward(context.Background(), types.FullBackprop, grad)
	if err != nil {
		t.Fatal(err)
	}
}

// ---------- AddGradient for all numeric types ----------

func testAddGradientType[T tensor.Numeric](t *testing.T, typeName string, vals []T, grads []T) {
	t.Helper()

	value, err := tensor.New[T]([]int{len(vals)}, vals)
	if err != nil {
		t.Fatalf("%s: create value: %v", typeName, err)
	}

	param, err := NewParameter(typeName, value, tensor.New[T])
	if err != nil {
		t.Fatalf("%s: NewParameter: %v", typeName, err)
	}

	grad, err := tensor.New[T]([]int{len(grads)}, grads)
	if err != nil {
		t.Fatalf("%s: create grad: %v", typeName, err)
	}

	if err := param.AddGradient(grad); err != nil {
		t.Fatalf("%s: AddGradient: %v", typeName, err)
	}

	// Clear to verify ClearGradient works too
	param.ClearGradient()
	for _, v := range param.Gradient.Data() {
		if any(v) != any(*new(T)) {
			t.Errorf("%s: ClearGradient did not zero out: got %v", typeName, v)
		}
	}
}

func TestParameter_AddGradient_Float64(t *testing.T) {
	testAddGradientType(t, "f64", []float64{1.0, 2.0}, []float64{0.1, 0.2})
}

func TestParameter_AddGradient_Int(t *testing.T) {
	testAddGradientType(t, "int", []int{1, 2}, []int{3, 4})
}

func TestParameter_AddGradient_Int8(t *testing.T) {
	testAddGradientType(t, "int8", []int8{1, 2}, []int8{3, 4})
}

func TestParameter_AddGradient_Int16(t *testing.T) {
	testAddGradientType(t, "int16", []int16{1, 2}, []int16{3, 4})
}

func TestParameter_AddGradient_Int32(t *testing.T) {
	testAddGradientType(t, "int32", []int32{1, 2}, []int32{3, 4})
}

func TestParameter_AddGradient_Int64(t *testing.T) {
	testAddGradientType(t, "int64", []int64{1, 2}, []int64{3, 4})
}

func TestParameter_AddGradient_Uint(t *testing.T) {
	testAddGradientType(t, "uint", []uint{1, 2}, []uint{3, 4})
}

func TestParameter_AddGradient_Uint32(t *testing.T) {
	testAddGradientType(t, "uint32", []uint32{1, 2}, []uint32{3, 4})
}

func TestParameter_AddGradient_Uint64(t *testing.T) {
	testAddGradientType(t, "uint64", []uint64{1, 2}, []uint64{3, 4})
}
