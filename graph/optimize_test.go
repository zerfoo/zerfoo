package graph

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// fakeTransposeEngine is a minimal Engine implementation for testing
// transpose folding. It only implements Transpose; other methods panic.
type fakeTransposeEngine struct{}

func (e *fakeTransposeEngine) Transpose(_ context.Context, a *tensor.TensorNumeric[float32], axes []int, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := a.Shape()
	data := a.Data()

	newShape := make([]int, len(axes))
	for i, ax := range axes {
		newShape[i] = shape[ax]
	}

	// Compute strides for the source tensor.
	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	size := 1
	for _, d := range newShape {
		size *= d
	}

	result := make([]float32, size)

	// New strides for the output.
	newStrides := make([]int, len(newShape))
	newStrides[len(newShape)-1] = 1
	for i := len(newShape) - 2; i >= 0; i-- {
		newStrides[i] = newStrides[i+1] * newShape[i+1]
	}

	for idx := range size {
		result[idx] = data[transposeIndex(idx, newShape, newStrides, strides, axes)]
	}

	return tensor.New(newShape, result)
}

// stubTransposeNode emulates a Transpose layer node for testing.
type stubTransposeNode struct {
	engine *fakeTransposeEngine
	perm   []int
}

func (n *stubTransposeNode) OpType() string { return "Transpose" }
func (n *stubTransposeNode) Attributes() map[string]interface{} {
	return map[string]interface{}{"perm": n.perm}
}
func (n *stubTransposeNode) OutputShape() []int { return nil }
func (n *stubTransposeNode) Parameters() []*Parameter[float32] { return nil }
func (n *stubTransposeNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return n.engine.Transpose(ctx, inputs[0], n.perm)
}
func (n *stubTransposeNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

// stubMatMulNode emulates a MatMul that just returns its second input (weight).
type stubMatMulNode struct{}

func (n *stubMatMulNode) OpType() string                  { return "MatMul" }
func (n *stubMatMulNode) Attributes() map[string]interface{} { return nil }
func (n *stubMatMulNode) OutputShape() []int               { return nil }
func (n *stubMatMulNode) Parameters() []*Parameter[float32] { return nil }
func (n *stubMatMulNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Simple: return second input (the weight) for shape/value inspection.
	if len(inputs) >= 2 {
		return inputs[1], nil
	}
	return inputs[0], nil
}
func (n *stubMatMulNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

// stubAddNode emulates an Add node that returns its first input.
type stubAddNode struct{}

func (n *stubAddNode) OpType() string                  { return "Add" }
func (n *stubAddNode) Attributes() map[string]interface{} { return nil }
func (n *stubAddNode) OutputShape() []int               { return nil }
func (n *stubAddNode) Parameters() []*Parameter[float32] { return nil }
func (n *stubAddNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return inputs[0], nil
}
func (n *stubAddNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

// paramConstNode wraps a tensor as a graph node (emulates parameterNode from model/).
type paramConstNode struct {
	value *tensor.TensorNumeric[float32]
}

func (n *paramConstNode) OpType() string                  { return "Parameter" }
func (n *paramConstNode) Attributes() map[string]interface{} { return nil }
func (n *paramConstNode) OutputShape() []int               { return n.value.Shape() }
func (n *paramConstNode) Parameters() []*Parameter[float32] { return nil }
func (n *paramConstNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return n.value, nil
}
func (n *paramConstNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func TestFoldConstantTransposes_ConstantInput(t *testing.T) {
	// Build: Parameter(2x3) -> Transpose([1,0]) -> MatMul
	// After folding: Parameter(3x2) -> MatMul  (Transpose removed)
	engine := &fakeTransposeEngine{}

	weight, err := tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}

	paramNode := &paramConstNode{value: weight}
	transposeNode := &stubTransposeNode{engine: engine, perm: []int{1, 0}}
	matmulNode := &stubMatMulNode{}

	builder := NewBuilder[float32](nil)
	inputNode := builder.Input([]int{1, 3})
	builder.AddNode(paramNode)
	builder.AddNode(transposeNode, paramNode)
	builder.AddNode(matmulNode, inputNode, transposeNode)

	g, err := builder.Build(matmulNode)
	if err != nil {
		t.Fatal(err)
	}

	// Count Transpose nodes before.
	countBefore := countTransposeNodes(g)
	if countBefore != 1 {
		t.Fatalf("expected 1 Transpose node before folding, got %d", countBefore)
	}

	// Fold.
	folded, err := FoldConstantTransposes(g, engine)
	if err != nil {
		t.Fatal(err)
	}

	// Count Transpose nodes after.
	countAfter := countTransposeNodes(folded)
	if countAfter != 0 {
		t.Fatalf("expected 0 Transpose nodes after folding, got %d", countAfter)
	}

	// Run forward pass and verify output matches unfolded graph.
	input, err := tensor.New([]int{1, 3}, []float32{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}

	// Original graph: the matmul just returns its weight input (the transposed param).
	origOut, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	foldedOut, err := folded.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	// Both should produce the same transposed weight: shape [3,2].
	if len(origOut.Shape()) != len(foldedOut.Shape()) {
		t.Fatalf("shape mismatch: orig %v, folded %v", origOut.Shape(), foldedOut.Shape())
	}
	for i := range origOut.Shape() {
		if origOut.Shape()[i] != foldedOut.Shape()[i] {
			t.Fatalf("shape mismatch at dim %d: orig %d, folded %d", i, origOut.Shape()[i], foldedOut.Shape()[i])
		}
	}

	origData := origOut.Data()
	foldedData := foldedOut.Data()
	for i := range origData {
		if math.Abs(float64(origData[i]-foldedData[i])) > 1e-6 {
			t.Fatalf("data mismatch at %d: orig %f, folded %f", i, origData[i], foldedData[i])
		}
	}
}

func TestFoldConstantTransposes_DynamicInput(t *testing.T) {
	// Build: Input -> Transpose([1,0]) -> Output
	// Dynamic input should NOT be folded.
	engine := &fakeTransposeEngine{}

	transposeNode := &stubTransposeNode{engine: engine, perm: []int{1, 0}}

	builder := NewBuilder[float32](nil)
	inputNode := builder.Input([]int{2, 3})
	builder.AddNode(transposeNode, inputNode)

	g, err := builder.Build(transposeNode)
	if err != nil {
		t.Fatal(err)
	}

	folded, err := FoldConstantTransposes(g, engine)
	if err != nil {
		t.Fatal(err)
	}

	// Transpose should still be present — input is dynamic.
	count := countTransposeNodes(folded)
	if count != 1 {
		t.Fatalf("expected 1 Transpose node (dynamic input not folded), got %d", count)
	}
}

func TestFoldConstantTransposes_OutputMatchesWithinTolerance(t *testing.T) {
	// End-to-end correctness: build a graph with a constant transpose,
	// fold it, and verify outputs match within 1e-6.
	engine := &fakeTransposeEngine{}

	// 3x4 weight matrix
	weightData := make([]float32, 12)
	for i := range weightData {
		weightData[i] = float32(i + 1)
	}
	weight, err := tensor.New([]int{3, 4}, weightData)
	if err != nil {
		t.Fatal(err)
	}

	paramNode := &paramConstNode{value: weight}
	transposeNode := &stubTransposeNode{engine: engine, perm: []int{1, 0}}
	matmulNode := &stubMatMulNode{}

	builder := NewBuilder[float32](nil)
	inputNode := builder.Input([]int{1, 4})
	builder.AddNode(paramNode)
	builder.AddNode(transposeNode, paramNode)
	builder.AddNode(matmulNode, inputNode, transposeNode)

	g, err := builder.Build(matmulNode)
	if err != nil {
		t.Fatal(err)
	}

	folded, err := FoldConstantTransposes(g, engine)
	if err != nil {
		t.Fatal(err)
	}

	input, err := tensor.New([]int{1, 4}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}

	origOut, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	foldedOut, err := folded.Forward(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	origData := origOut.Data()
	foldedData := foldedOut.Data()
	if len(origData) != len(foldedData) {
		t.Fatalf("length mismatch: %d vs %d", len(origData), len(foldedData))
	}
	for i := range origData {
		if math.Abs(float64(origData[i]-foldedData[i])) > 1e-6 {
			t.Errorf("data[%d] mismatch: orig=%f folded=%f", i, origData[i], foldedData[i])
		}
	}
}

func TestFoldConstantTransposes_MultipleConsumers(t *testing.T) {
	// Transpose feeds into two different nodes — both should be rewired.
	engine := &fakeTransposeEngine{}

	weight, err := tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}

	paramNode := &paramConstNode{value: weight}
	transposeNode := &stubTransposeNode{engine: engine, perm: []int{1, 0}}
	matmulNode := &stubMatMulNode{}
	addNode := &stubAddNode{}

	builder := NewBuilder[float32](nil)
	inputNode := builder.Input([]int{1, 3})
	builder.AddNode(paramNode)
	builder.AddNode(transposeNode, paramNode)
	builder.AddNode(matmulNode, inputNode, transposeNode)
	builder.AddNode(addNode, matmulNode, transposeNode)

	g, err := builder.Build(addNode)
	if err != nil {
		t.Fatal(err)
	}

	folded, err := FoldConstantTransposes(g, engine)
	if err != nil {
		t.Fatal(err)
	}

	count := countTransposeNodes(folded)
	if count != 0 {
		t.Fatalf("expected 0 Transpose nodes after folding, got %d", count)
	}

	// Both consumers should now reference the pre-transposed constant.
	input, err := tensor.New([]int{1, 3}, []float32{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}

	_, err = folded.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward pass failed after folding: %v", err)
	}
}

func TestFoldConstantTransposes_NoTransposeNodes(t *testing.T) {
	// Graph with no Transpose nodes should be returned unchanged.
	builder := NewBuilder[float32](nil)
	inputNode := builder.Input([]int{2, 3})
	matmulNode := &stubMatMulNode{}
	builder.AddNode(matmulNode, inputNode)

	g, err := builder.Build(matmulNode)
	if err != nil {
		t.Fatal(err)
	}

	folded, err := FoldConstantTransposes(g, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(folded.Nodes()) != len(g.Nodes()) {
		t.Fatalf("expected same node count, got %d vs %d", len(folded.Nodes()), len(g.Nodes()))
	}
}

func countTransposeNodes[T tensor.Numeric](g *Graph[T]) int {
	count := 0
	for _, n := range g.Nodes() {
		if n.OpType() == "Transpose" {
			count++
		}
	}
	return count
}

// transposeIndex maps an output-space flat index back to the source-space flat index.
func transposeIndex(idx int, newShape, newStrides, srcStrides, axes []int) int {
	srcIdx := 0
	rem := idx
	for d := range len(newShape) {
		coord := rem / newStrides[d]
		rem %= newStrides[d]
		srcIdx += coord * srcStrides[axes[d]]
	}
	return srcIdx
}
