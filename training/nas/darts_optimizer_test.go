package nas

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// linearOp implements a simple learnable linear scaling: y = w * x.
// It has a single trainable weight parameter so we can test weight updates.
type linearOp[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	weight *graph.Parameter[T]
	shape  []int
}

func newLinearOp[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], initWeight T, shape []int) (*linearOp[T], error) {
	wVal, err := tensor.New[T]([]int{1}, []T{initWeight})
	if err != nil {
		return nil, err
	}
	wParam, err := graph.NewParameter[T]("w", wVal, tensor.New[T])
	if err != nil {
		return nil, err
	}
	return &linearOp[T]{engine: engine, ops: ops, weight: wParam, shape: shape}, nil
}

func (op *linearOp[T]) OpType() string                     { return "linear" }
func (op *linearOp[T]) Attributes() map[string]interface{} { return nil }
func (op *linearOp[T]) Parameters() []*graph.Parameter[T]  { return []*graph.Parameter[T]{op.weight} }
func (op *linearOp[T]) OutputShape() []int                 { return op.shape }

func (op *linearOp[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	w := op.weight.Value.Data()[0]
	return op.engine.MulScalar(ctx, inputs[0], w)
}

func (op *linearOp[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	w := op.weight.Value.Data()[0]

	// dInput = w * dOut
	dInput, err := op.engine.MulScalar(ctx, dOut, w)
	if err != nil {
		return nil, err
	}

	// dWeight = sum(dOut * input)
	input := inputs[0]
	dOutData := dOut.Data()
	inData := input.Data()
	var dw T
	for i := range dOutData {
		dw = op.ops.Add(dw, op.ops.Mul(dOutData[i], inData[i]))
	}
	dwTensor, err := tensor.New[T]([]int{1}, []T{dw})
	if err != nil {
		return nil, err
	}
	if err := op.weight.AddGradient(dwTensor); err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// mseLoss computes mean squared error loss and its gradient.
// loss = (1/n) * sum((pred - target)^2)
// dLoss/dPred = (2/n) * (pred - target)
func mseLoss[T tensor.Numeric](ops numeric.Arithmetic[T], pred, target *tensor.TensorNumeric[T]) (T, *tensor.TensorNumeric[T], error) {
	pData := pred.Data()
	tData := target.Data()
	n := len(pData)

	var loss T
	gradData := make([]T, n)
	nT := ops.FromFloat64(float64(n))

	for i := range pData {
		diff := ops.Sub(pData[i], tData[i])
		loss = ops.Add(loss, ops.Mul(diff, diff))
		two := ops.FromFloat64(2.0)
		gradData[i] = ops.Div(ops.Mul(two, diff), nT)
	}
	loss = ops.Div(loss, nT)

	grad, err := tensor.New[T](pred.Shape(), gradData)
	if err != nil {
		return loss, nil, err
	}
	return loss, grad, nil
}

func TestNewDARTSOptimizer(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	op1, err := newLinearOp[float32](engine, ops, 1.0, []int{4})
	if err != nil {
		t.Fatal(err)
	}
	op2, err := newLinearOp[float32](engine, ops, 2.0, []int{4})
	if err != nil {
		t.Fatal(err)
	}

	layer, err := NewDARTSLayer[float32](engine, ops, []graph.Node[float32]{op1, op2})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("valid config", func(t *testing.T) {
		cfg := DARTSOptimizerConfig[float32]{
			WeightLR: 0.01,
			AlphaLR:  0.01,
		}
		opt, err := NewDARTSOptimizer[float32](engine, ops, layer, cfg)
		if err != nil {
			t.Fatalf("NewDARTSOptimizer: %v", err)
		}
		if opt == nil {
			t.Fatal("optimizer is nil")
		}
	})

	t.Run("nil layer", func(t *testing.T) {
		cfg := DARTSOptimizerConfig[float32]{WeightLR: 0.01, AlphaLR: 0.01}
		_, err := NewDARTSOptimizer[float32](engine, ops, nil, cfg)
		if err == nil {
			t.Fatal("expected error for nil layer")
		}
	})

	t.Run("zero weight lr", func(t *testing.T) {
		cfg := DARTSOptimizerConfig[float32]{WeightLR: 0, AlphaLR: 0.01}
		_, err := NewDARTSOptimizer[float32](engine, ops, layer, cfg)
		if err == nil {
			t.Fatal("expected error for zero weight lr")
		}
	})

	t.Run("zero alpha lr", func(t *testing.T) {
		cfg := DARTSOptimizerConfig[float32]{WeightLR: 0.01, AlphaLR: 0}
		_, err := NewDARTSOptimizer[float32](engine, ops, layer, cfg)
		if err == nil {
			t.Fatal("expected error for zero alpha lr")
		}
	})
}

func TestDARTSOptimizerStep(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Two ops: scale by 1.0 and scale by 2.0.
	op1 := &identityOp[float32]{scale: 1.0, engine: engine, shape: []int{4}}
	op2 := &identityOp[float32]{scale: 2.0, engine: engine, shape: []int{4}}

	layer, err := NewDARTSLayer[float32](engine, ops, []graph.Node[float32]{op1, op2})
	if err != nil {
		t.Fatal(err)
	}

	cfg := DARTSOptimizerConfig[float32]{
		WeightLR: 0.01,
		AlphaLR:  0.01,
	}
	opt, err := NewDARTSOptimizer[float32](engine, ops, layer, cfg)
	if err != nil {
		t.Fatal(err)
	}

	trainInput, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	trainTarget, _ := tensor.New[float32]([]int{4}, []float32{2, 4, 6, 8})
	valInput, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	valTarget, _ := tensor.New[float32]([]int{4}, []float32{2, 2, 2, 2})

	// Record alpha before step.
	alphaBefore := make([]float32, len(layer.Parameters()[0].Value.Data()))
	copy(alphaBefore, layer.Parameters()[0].Value.Data())

	err = opt.Step(ctx, trainInput, trainTarget, valInput, valTarget)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}

	// Alpha should have been updated.
	alphaAfter := layer.Parameters()[0].Value.Data()
	changed := false
	for i := range alphaAfter {
		if alphaAfter[i] != alphaBefore[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("alpha was not updated after Step")
	}
}

func TestDARTSOptimizerAlternation(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Use learnable ops so weight updates are visible.
	op1, err := newLinearOp[float32](engine, ops, 0.5, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	op2, err := newLinearOp[float32](engine, ops, 1.5, []int{2})
	if err != nil {
		t.Fatal(err)
	}

	layer, err := NewDARTSLayer[float32](engine, ops, []graph.Node[float32]{op1, op2})
	if err != nil {
		t.Fatal(err)
	}

	cfg := DARTSOptimizerConfig[float32]{
		WeightLR: 0.1,
		AlphaLR:  0.1,
	}
	opt, err := NewDARTSOptimizer[float32](engine, ops, layer, cfg)
	if err != nil {
		t.Fatal(err)
	}

	trainInput, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	trainTarget, _ := tensor.New[float32]([]int{2}, []float32{3, 6})
	valInput, _ := tensor.New[float32]([]int{2}, []float32{1, 1})
	valTarget, _ := tensor.New[float32]([]int{2}, []float32{3, 3})

	w1Before := op1.weight.Value.Data()[0]
	w2Before := op2.weight.Value.Data()[0]

	err = opt.Step(ctx, trainInput, trainTarget, valInput, valTarget)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}

	w1After := op1.weight.Value.Data()[0]
	w2After := op2.weight.Value.Data()[0]

	// Weights should have changed from the inner (training) update.
	if w1After == w1Before && w2After == w2Before {
		t.Error("network weights were not updated (inner step failed)")
	}
}

func TestDARTSOptimizerConvergence(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Toy architecture search: two candidate ops (scale by 1.0 and scale by 3.0).
	// Target = 3*x, so the optimal architecture should heavily favor op2 (scale=3.0).
	op1 := &identityOp[float32]{scale: 1.0, engine: engine, shape: []int{4}}
	op2 := &identityOp[float32]{scale: 3.0, engine: engine, shape: []int{4}}

	layer, err := NewDARTSLayer[float32](engine, ops, []graph.Node[float32]{op1, op2})
	if err != nil {
		t.Fatal(err)
	}

	cfg := DARTSOptimizerConfig[float32]{
		WeightLR: 0.01,
		AlphaLR:  0.1,
	}
	opt, err := NewDARTSOptimizer[float32](engine, ops, layer, cfg)
	if err != nil {
		t.Fatal(err)
	}

	trainInput, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	trainTarget, _ := tensor.New[float32]([]int{4}, []float32{3, 6, 9, 12})
	valInput, _ := tensor.New[float32]([]int{4}, []float32{2, 3, 4, 5})
	valTarget, _ := tensor.New[float32]([]int{4}, []float32{6, 9, 12, 15})

	var lastLoss float32
	for step := range 200 {
		err := opt.Step(ctx, trainInput, trainTarget, valInput, valTarget)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}

		// Compute validation loss to track convergence.
		pred, err := layer.Forward(ctx, valInput)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		loss, _, err := mseLoss[float32](ops, pred, valTarget)
		if err != nil {
			t.Fatalf("loss: %v", err)
		}
		lastLoss = loss
	}

	// After 200 steps, loss should be very small.
	if lastLoss > 0.1 {
		t.Errorf("validation loss after 200 steps = %f, want < 0.1", lastLoss)
	}

	// Alpha should favor op2 (scale=3.0) since target = 3*x.
	weights := layer.Weights()
	if weights[1] < weights[0] {
		t.Errorf("expected alpha to favor op2 (scale=3.0), got weights %v", weights)
	}

	// The weight on op2 should be dominant.
	if weights[1] < 0.9 {
		t.Errorf("expected op2 weight > 0.9, got %f", weights[1])
	}
}

func TestDARTSOptimizerConvergenceWithLearnableOps(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Two learnable linear ops, both starting far from the target.
	// Target = 2*x. Op1 starts at w=0.1, op2 starts at w=0.5.
	// The optimizer should learn both which op to prefer AND update the op weights.
	op1, err := newLinearOp[float32](engine, ops, 0.1, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	op2, err := newLinearOp[float32](engine, ops, 0.5, []int{2})
	if err != nil {
		t.Fatal(err)
	}

	layer, err := NewDARTSLayer[float32](engine, ops, []graph.Node[float32]{op1, op2})
	if err != nil {
		t.Fatal(err)
	}

	cfg := DARTSOptimizerConfig[float32]{
		WeightLR: 0.05,
		AlphaLR:  0.05,
	}
	opt, err := NewDARTSOptimizer[float32](engine, ops, layer, cfg)
	if err != nil {
		t.Fatal(err)
	}

	trainInput, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	trainTarget, _ := tensor.New[float32]([]int{2}, []float32{2, 4})
	valInput, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	valTarget, _ := tensor.New[float32]([]int{2}, []float32{6, 8})

	var lastLoss float32
	for step := range 300 {
		err := opt.Step(ctx, trainInput, trainTarget, valInput, valTarget)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}

		pred, err := layer.Forward(ctx, valInput)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		loss, _, err := mseLoss[float32](ops, pred, valTarget)
		if err != nil {
			t.Fatalf("loss: %v", err)
		}
		lastLoss = loss
	}

	if lastLoss > 0.5 {
		t.Errorf("validation loss after 300 steps = %f, want < 0.5", lastLoss)
	}

	// At least one op weight should have moved toward 2.0.
	w1 := op1.weight.Value.Data()[0]
	w2 := op2.weight.Value.Data()[0]
	weights := layer.Weights()

	// The dominant op's weight should be closer to 2.0 than its initial value.
	dominantIdx := 0
	if weights[1] > weights[0] {
		dominantIdx = 1
	}
	var dominantW float32
	if dominantIdx == 0 {
		dominantW = w1
	} else {
		dominantW = w2
	}
	if math.Abs(float64(dominantW-2.0)) > 1.5 {
		t.Errorf("dominant op weight = %f, expected closer to 2.0", dominantW)
	}
}
