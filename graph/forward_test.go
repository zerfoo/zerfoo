package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// matMulNode simulates a matrix multiplication node for graph forward tests.
// It uses the CPU engine to perform an actual MatMul.
type matMulNode[T tensor.Numeric] struct {
	NoParameters[T]
	engine      compute.Engine[T]
	outputShape []int
}

func (n *matMulNode[T]) OpType() string                        { return "MatMul" }
func (n *matMulNode[T]) OutputShape() []int                    { return n.outputShape }
func (n *matMulNode[T]) Attributes() map[string]interface{}    { return map[string]interface{}{} }
func (n *matMulNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return n.engine.MatMul(ctx, inputs[0], inputs[1])
}
func (n *matMulNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// addNode simulates element-wise addition using the CPU engine.
type addNode[T tensor.Numeric] struct {
	NoParameters[T]
	engine      compute.Engine[T]
	outputShape []int
}

func (n *addNode[T]) OpType() string                        { return "Add" }
func (n *addNode[T]) OutputShape() []int                    { return n.outputShape }
func (n *addNode[T]) Attributes() map[string]interface{}    { return map[string]interface{}{} }
func (n *addNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return n.engine.Add(ctx, inputs[0], inputs[1])
}
func (n *addNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// scaleNode multiplies every element by a fixed scalar. Simulates a simple
// activation or normalization step without needing a full engine op.
type scaleNode[T tensor.Numeric] struct {
	NoParameters[T]
	factor      T
	outputShape []int
}

func (n *scaleNode[T]) OpType() string                        { return "Scale" }
func (n *scaleNode[T]) OutputShape() []int                    { return n.outputShape }
func (n *scaleNode[T]) Attributes() map[string]interface{}    { return map[string]interface{}{"factor": n.factor} }
func (n *scaleNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	src := inputs[0].Data()
	out := make([]T, len(src))
	for i, v := range src {
		out[i] = v * n.factor // T is constrained to numeric, * is valid
	}
	return tensor.New[T](inputs[0].Shape(), out)
}
func (n *scaleNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// weightNode wraps a fixed tensor. Used as a weight matrix in forward tests.
type weightNode[T tensor.Numeric] struct {
	NoParameters[T]
	value *tensor.TensorNumeric[T]
}

func (n *weightNode[T]) OpType() string                     { return "Constant" }
func (n *weightNode[T]) OutputShape() []int                 { return n.value.Shape() }
func (n *weightNode[T]) Attributes() map[string]interface{} { return map[string]interface{}{} }
func (n *weightNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return n.value, nil
}
func (n *weightNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// TestForwardLargeMatMul builds a graph with a large final MatMul simulating
// an LM head projection (hidden -> vocab). It verifies correct output shape
// and spot-checks output values.
func TestForwardLargeMatMul(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := NewBuilder[float32](engine)

	// Simulate: batch=2, hidden=64, vocab=32768
	// Keep dimensions CPU-friendly while still exercising a large output.
	const (
		batch  = 2
		hidden = 64
		vocab  = 32768
	)

	input := b.Input([]int{batch, hidden})

	// Weight matrix: hidden x vocab
	weightData := make([]float32, hidden*vocab)
	// Fill with a simple pattern: w[i][j] = 1.0 for all i,j
	// so matmul(input, weight) = rowSum(input) broadcast across vocab cols.
	for i := range weightData {
		weightData[i] = 1.0
	}
	weightTensor, err := tensor.New[float32]([]int{hidden, vocab}, weightData)
	if err != nil {
		t.Fatalf("create weight tensor: %v", err)
	}
	weightNode := &weightNode[float32]{value: weightTensor}
	b.AddNode(weightNode)

	matmul := &matMulNode[float32]{engine: engine, outputShape: []int{batch, vocab}}
	b.AddNode(matmul, input, weightNode)

	g, err := b.Build(matmul)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Input: all ones -> each row sums to hidden=64, so output[i][j] = 64.0
	inputData := make([]float32, batch*hidden)
	for i := range inputData {
		inputData[i] = 1.0
	}
	inputTensor, err := tensor.New[float32]([]int{batch, hidden}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	out, err := g.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Verify shape
	shape := out.Shape()
	if len(shape) != 2 || shape[0] != batch || shape[1] != vocab {
		t.Fatalf("output shape = %v, want [%d, %d]", shape, batch, vocab)
	}

	// Spot-check values
	data := out.Data()
	expected := float32(hidden)
	for _, idx := range []int{0, vocab - 1, vocab, 2*vocab - 1} {
		if data[idx] != expected {
			t.Errorf("data[%d] = %g, want %g", idx, data[idx], expected)
		}
	}
}

// TestForwardMultiNodePropagation verifies that values propagate correctly
// through a chain of multiple nodes: Input -> Scale -> MatMul -> output.
func TestForwardMultiNodePropagation(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := NewBuilder[float32](engine)

	const (
		rows = 3
		cols = 4
		out  = 2
	)

	input := b.Input([]int{rows, cols})

	// Scale by 2
	scale := &scaleNode[float32]{factor: 2.0, outputShape: []int{rows, cols}}
	b.AddNode(scale, input)

	// Weight: cols x out, all 1.0
	wData := make([]float32, cols*out)
	for i := range wData {
		wData[i] = 1.0
	}
	wTensor, err := tensor.New[float32]([]int{cols, out}, wData)
	if err != nil {
		t.Fatalf("create weight: %v", err)
	}
	wNode := &weightNode[float32]{value: wTensor}
	b.AddNode(wNode)

	mm := &matMulNode[float32]{engine: engine, outputShape: []int{rows, out}}
	b.AddNode(mm, scale, wNode)

	g, err := b.Build(mm)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Input: row i has all values = i+1
	inputData := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			inputData[r*cols+c] = float32(r + 1)
		}
	}
	inputTensor, err := tensor.New[float32]([]int{rows, cols}, inputData)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	result, err := g.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// After scale by 2, row r has value 2*(r+1).
	// After matmul with all-1 weight [cols x out], each output element = 2*(r+1)*cols.
	data := result.Data()
	for r := 0; r < rows; r++ {
		expected := float32(2 * (r + 1) * cols)
		for c := 0; c < out; c++ {
			got := data[r*out+c]
			if got != expected {
				t.Errorf("row %d col %d: got %g, want %g", r, c, got, expected)
			}
		}
	}
}

// TestForwardTwoLayerTransformerLike builds a 2-layer transformer-like graph:
//
//	Input -> [Layer1: MatMul + Add bias + Scale(ReLU-like)] ->
//	         [Layer2: MatMul + Add bias + Scale(ReLU-like)] -> Output
//
// This verifies that a multi-layer forward pass completes and produces the
// correct output shape and reasonable values.
func TestForwardTwoLayerTransformerLike(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := NewBuilder[float32](engine)

	const (
		batch  = 4
		seqLen = 8
		dModel = 16
		dFF    = 32
	)

	// Flatten batch and sequence: [batch*seqLen, dModel]
	tokens := batch * seqLen
	input := b.Input([]int{tokens, dModel})

	// --- Layer 1: project dModel -> dFF ---
	w1Data := make([]float32, dModel*dFF)
	for i := range w1Data {
		w1Data[i] = 0.01 // small positive weights
	}
	w1Tensor, err := tensor.New[float32]([]int{dModel, dFF}, w1Data)
	if err != nil {
		t.Fatalf("w1: %v", err)
	}
	w1 := &weightNode[float32]{value: w1Tensor}
	b.AddNode(w1)

	mm1 := &matMulNode[float32]{engine: engine, outputShape: []int{tokens, dFF}}
	b.AddNode(mm1, input, w1)

	// Bias: 1-D broadcast via a custom forward that adds a constant
	bias1Data := make([]float32, tokens*dFF)
	for i := range bias1Data {
		bias1Data[i] = 0.1
	}
	bias1Tensor, err := tensor.New[float32]([]int{tokens, dFF}, bias1Data)
	if err != nil {
		t.Fatalf("bias1: %v", err)
	}
	bias1 := &weightNode[float32]{value: bias1Tensor}
	b.AddNode(bias1)

	add1 := &addNode[float32]{engine: engine, outputShape: []int{tokens, dFF}}
	b.AddNode(add1, mm1, bias1)

	// ReLU-like activation: scale by 1 (identity, but exercises the node)
	act1 := &scaleNode[float32]{factor: 1.0, outputShape: []int{tokens, dFF}}
	b.AddNode(act1, add1)

	// --- Layer 2: project dFF -> dModel ---
	w2Data := make([]float32, dFF*dModel)
	for i := range w2Data {
		w2Data[i] = 0.01
	}
	w2Tensor, err := tensor.New[float32]([]int{dFF, dModel}, w2Data)
	if err != nil {
		t.Fatalf("w2: %v", err)
	}
	w2 := &weightNode[float32]{value: w2Tensor}
	b.AddNode(w2)

	mm2 := &matMulNode[float32]{engine: engine, outputShape: []int{tokens, dModel}}
	b.AddNode(mm2, act1, w2)

	bias2Data := make([]float32, tokens*dModel)
	for i := range bias2Data {
		bias2Data[i] = 0.05
	}
	bias2Tensor, err := tensor.New[float32]([]int{tokens, dModel}, bias2Data)
	if err != nil {
		t.Fatalf("bias2: %v", err)
	}
	bias2 := &weightNode[float32]{value: bias2Tensor}
	b.AddNode(bias2)

	add2 := &addNode[float32]{engine: engine, outputShape: []int{tokens, dModel}}
	b.AddNode(add2, mm2, bias2)

	act2 := &scaleNode[float32]{factor: 1.0, outputShape: []int{tokens, dModel}}
	b.AddNode(act2, add2)

	g, err := b.Build(act2)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// Input: all ones
	inputData := make([]float32, tokens*dModel)
	for i := range inputData {
		inputData[i] = 1.0
	}
	inputTensor, err := tensor.New[float32]([]int{tokens, dModel}, inputData)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	result, err := g.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Verify output shape
	shape := result.Shape()
	if len(shape) != 2 || shape[0] != tokens || shape[1] != dModel {
		t.Fatalf("output shape = %v, want [%d, %d]", shape, tokens, dModel)
	}

	// With all-1 input and 0.01 weights:
	// Layer 1 matmul: each element = 0.01 * dModel = 0.16
	// + bias 0.1 -> 0.26
	// Layer 2 matmul: each element = 0.26 * 0.01 * dFF = 0.26 * 0.32 = 0.0832
	// + bias 0.05 -> 0.1332
	// Verify values are positive and in expected range.
	data := result.Data()
	for i, v := range data {
		if v < 0.10 || v > 0.20 {
			t.Errorf("data[%d] = %g, expected in [0.10, 0.20]", i, v)
			break
		}
	}
}

// TestForwardDiamondGraph verifies correct forward propagation through a
// diamond-shaped graph where two paths merge:
//
//	Input -> A -> C
//	      -> B -> C
func TestForwardDiamondGraph(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := NewBuilder[float32](engine)

	input := b.Input([]int{2, 3})

	// Path A: scale by 2
	nodeA := &scaleNode[float32]{factor: 2.0, outputShape: []int{2, 3}}
	b.AddNode(nodeA, input)

	// Path B: scale by 3
	nodeB := &scaleNode[float32]{factor: 3.0, outputShape: []int{2, 3}}
	b.AddNode(nodeB, input)

	// Merge: add A + B
	nodeC := &addNode[float32]{engine: engine, outputShape: []int{2, 3}}
	b.AddNode(nodeC, nodeA, nodeB)

	g, err := b.Build(nodeC)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	inputData := []float32{1, 2, 3, 4, 5, 6}
	inputTensor, err := tensor.New[float32]([]int{2, 3}, inputData)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := g.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Each element = input*2 + input*3 = input*5
	data := out.Data()
	for i, v := range data {
		expected := inputData[i] * 5
		if v != expected {
			t.Errorf("data[%d] = %g, want %g", i, v, expected)
		}
	}
}

// TestForwardRepeatedExecution verifies that a graph can be executed multiple
// times with different inputs and produces correct results each time (memo
// is properly reset between calls).
func TestForwardRepeatedExecution(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := NewBuilder[float32](engine)

	input := b.Input([]int{1, 4})
	scale := &scaleNode[float32]{factor: 3.0, outputShape: []int{1, 4}}
	b.AddNode(scale, input)

	g, err := b.Build(scale)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	for iter := 0; iter < 5; iter++ {
		val := float32(iter + 1)
		data := []float32{val, val, val, val}
		in, err := tensor.New[float32]([]int{1, 4}, data)
		if err != nil {
			t.Fatalf("iter %d: create input: %v", iter, err)
		}

		out, err := g.Forward(context.Background(), in)
		if err != nil {
			t.Fatalf("iter %d: Forward: %v", iter, err)
		}

		expected := val * 3.0
		for j, v := range out.Data() {
			if v != expected {
				t.Errorf("iter %d data[%d] = %g, want %g", iter, j, v, expected)
			}
		}
	}
}
