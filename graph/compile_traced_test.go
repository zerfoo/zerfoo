package graph

import (
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// compositeNode wraps an engine and performs multiple primitive ops internally.
// This simulates a real composite layer like FFN or RMSNorm.
type compositeNode struct {
	engine      compute.Engine[float32]
	outputShape []int
}

// opaqueNode uses UnaryOp (opaque closure) which should trigger fallback.
type opaqueNode struct {
	engine      compute.Engine[float32]
	outputShape []int
}

func (n *compositeNode) OpType() string                     { return "FFN" }
func (n *compositeNode) Attributes() map[string]any         { return nil }
func (n *compositeNode) OutputShape() []int                 { return n.outputShape }
func (n *compositeNode) Parameters() []*Parameter[float32]  { return nil }

func (n *compositeNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Simulates FFN: output = (input * 2) + input  (two engine ops)
	doubled, err := n.engine.MulScalar(ctx, inputs[0], 2.0)
	if err != nil {
		return nil, err
	}
	return n.engine.Add(ctx, doubled, inputs[0])
}

func (n *compositeNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *opaqueNode) OpType() string                     { return "OpaqueLayer" }
func (n *opaqueNode) Attributes() map[string]any         { return nil }
func (n *opaqueNode) OutputShape() []int                 { return n.outputShape }
func (n *opaqueNode) Parameters() []*Parameter[float32]  { return nil }

func (n *opaqueNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Uses UnaryOp with an opaque closure -- should trigger opaque fallback.
	return n.engine.UnaryOp(ctx, inputs[0], func(v float32) float32 { return v * 2 })
}

func (n *opaqueNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func TestCompileTracedPrimitiveNode(t *testing.T) {
	// A graph with a single primitive Add node should produce the same
	// instructions from CompileTraced as from Compile.
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()

	proxy := compute.NewEngineProxy[float32](engine)
	b := NewBuilder[float32](proxy)

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
		forwardFunc: func(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			return proxy.Add(ctx, inputs[0], inputs[1])
		},
	}
	sum := b.AddNode(addNode, in, cst)

	g, err := b.Build(sum)
	if err != nil {
		t.Fatal(err)
	}
	g.SetEngineProxy(proxy)

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	ctx := context.Background()

	plan, err := g.CompileTraced(ctx, input)
	if err != nil {
		t.Fatalf("CompileTraced: %v", err)
	}

	// Should have exactly 1 instruction (the Add op).
	metas := plan.Instructions()
	if len(metas) != 1 {
		t.Fatalf("Instructions: got %d, want 1", len(metas))
	}
	if metas[0].OpName != "Add" {
		t.Errorf("OpName = %q, want Add", metas[0].OpName)
	}

	// Run the traced plan and verify output.
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

func TestCompileTracedCompositeNode(t *testing.T) {
	// A composite node (FFN) makes multiple engine calls internally.
	// CompileTraced should produce primitive ops (MulScalar, Add), not "FFN".
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()

	proxy := compute.NewEngineProxy[float32](engine)
	b := NewBuilder[float32](proxy)

	in := b.Input([]int{1, 4})

	ffn := &compositeNode{engine: proxy, outputShape: []int{1, 4}}
	out := b.AddNode(ffn, in)

	g, err := b.Build(out)
	if err != nil {
		t.Fatal(err)
	}
	g.SetEngineProxy(proxy)

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	ctx := context.Background()

	plan, err := g.CompileTraced(ctx, input)
	if err != nil {
		t.Fatalf("CompileTraced: %v", err)
	}

	// The composite FFN does MulScalar + Add = 2 primitive ops.
	metas := plan.Instructions()
	if len(metas) != 2 {
		t.Fatalf("Instructions: got %d, want 2", len(metas))
	}
	if metas[0].OpName != "MulScalar" {
		t.Errorf("metas[0].OpName = %q, want MulScalar", metas[0].OpName)
	}
	if metas[1].OpName != "Add" {
		t.Errorf("metas[1].OpName = %q, want Add", metas[1].OpName)
	}

	// Run the traced plan and verify output matches direct Forward.
	result, err := plan.Run(ctx, input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	want := []float32{3, 6, 9, 12} // input*2 + input = input*3
	got := result.Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Verify matches interpreted Forward.
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
}

func TestCompileTracedFrozenSlots(t *testing.T) {
	// Constant nodes should be frozen in the traced plan.
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()

	proxy := compute.NewEngineProxy[float32](engine)
	b := NewBuilder[float32](proxy)

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
		forwardFunc: func(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
			return proxy.Add(ctx, inputs[0], inputs[1])
		},
	}
	sum := b.AddNode(addNode, in, cst)

	g, err := b.Build(sum)
	if err != nil {
		t.Fatal(err)
	}
	g.SetEngineProxy(proxy)

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	plan, err := g.CompileTraced(context.Background(), input)
	if err != nil {
		t.Fatalf("CompileTraced: %v", err)
	}

	frozen := plan.FrozenSlots()
	if len(frozen) == 0 {
		t.Fatal("expected at least 1 frozen slot")
	}
	if frozen[0].Data == nil {
		t.Error("frozen slot data is nil")
	}
}

func TestCompileTracedNoEngineProxy(t *testing.T) {
	// CompileTraced should return an error if no EngineProxy is set.
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()

	b := NewBuilder[float32](engine)
	in := b.Input([]int{1, 4})
	node := &mockF32Node{name: "Pass", outputShape: []int{1, 4}}
	out := b.AddNode(node, in)
	g, err := b.Build(out)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	_, err = g.CompileTraced(context.Background(), input)
	if err == nil {
		t.Fatal("expected error when no EngineProxy is set")
	}
}

func TestCompileTracedReuseDifferentInput(t *testing.T) {
	// Verify the traced plan works correctly with a different input.
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()

	proxy := compute.NewEngineProxy[float32](engine)
	b := NewBuilder[float32](proxy)

	in := b.Input([]int{1, 4})
	ffn := &compositeNode{engine: proxy, outputShape: []int{1, 4}}
	out := b.AddNode(ffn, in)

	g, err := b.Build(out)
	if err != nil {
		t.Fatal(err)
	}
	g.SetEngineProxy(proxy)

	input1, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	ctx := context.Background()

	plan, err := g.CompileTraced(ctx, input1)
	if err != nil {
		t.Fatalf("CompileTraced: %v", err)
	}

	// Run with first input.
	result1, err := plan.Run(ctx, input1)
	if err != nil {
		t.Fatalf("Run1: %v", err)
	}
	want1 := []float32{3, 6, 9, 12}
	got1 := result1.Data()
	for i := range want1 {
		if got1[i] != want1[i] {
			t.Errorf("result1[%d] = %v, want %v", i, got1[i], want1[i])
		}
	}

	// Run with different input.
	input2, _ := tensor.New[float32]([]int{1, 4}, []float32{10, 20, 30, 40})
	result2, err := plan.Run(ctx, input2)
	if err != nil {
		t.Fatalf("Run2: %v", err)
	}
	want2 := []float32{30, 60, 90, 120}
	got2 := result2.Data()
	for i := range want2 {
		if got2[i] != want2[i] {
			t.Errorf("result2[%d] = %v, want %v", i, got2[i], want2[i])
		}
	}
}

func TestCompileTracedOpaqueOpFallback(t *testing.T) {
	// A graph containing a node that uses UnaryOp (opaque closure) should
	// cause CompileTraced to return an error indicating opaque ops detected.
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()

	proxy := compute.NewEngineProxy[float32](engine)
	b := NewBuilder[float32](proxy)

	in := b.Input([]int{1, 4})
	opaque := &opaqueNode{engine: proxy, outputShape: []int{1, 4}}
	out := b.AddNode(opaque, in)

	g, err := b.Build(out)
	if err != nil {
		t.Fatal(err)
	}
	g.SetEngineProxy(proxy)

	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	ctx := context.Background()

	_, err = g.CompileTraced(ctx, input)
	if err == nil {
		t.Fatal("expected error for opaque ops, got nil")
	}
	want := "opaque ops"
	if got := err.Error(); !strings.Contains(got, want) {
		t.Errorf("error = %q, want it to contain %q", got, want)
	}
}
