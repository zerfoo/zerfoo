package lora

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

// stubLinear is a minimal linear layer for testing that uses a known weight matrix.
type stubLinear[T tensor.Numeric] struct {
	weights *graph.Parameter[T]
	engine  compute.Engine[T]
}

func newStubLinear[T tensor.Numeric](engine compute.Engine[T], w *tensor.TensorNumeric[T]) (*stubLinear[T], error) {
	param, err := graph.NewParameter[T]("stub_weights", w, tensor.New[T])
	if err != nil {
		return nil, err
	}
	return &stubLinear[T]{weights: param, engine: engine}, nil
}

func (s *stubLinear[T]) OpType() string                     { return "StubLinear" }
func (s *stubLinear[T]) Attributes() map[string]interface{} { return nil }
func (s *stubLinear[T]) OutputShape() []int                 { return []int{-1, s.weights.Value.Shape()[1]} }
func (s *stubLinear[T]) Parameters() []*graph.Parameter[T]  { return []*graph.Parameter[T]{s.weights} }

func (s *stubLinear[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return s.engine.MatMul(ctx, inputs[0], s.weights.Value)
}

func (s *stubLinear[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	x := inputs[0]

	// dW = x^T @ grad
	xT, err := s.engine.Transpose(ctx, x, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dW, err := s.engine.MatMul(ctx, xT, outputGradient)
	if err != nil {
		return nil, err
	}
	if s.weights.Gradient == nil {
		s.weights.Gradient = dW
	} else {
		s.weights.Gradient, err = s.engine.Add(ctx, s.weights.Gradient, dW)
		if err != nil {
			return nil, err
		}
	}

	// dx = grad @ W^T
	wT, err := s.engine.Transpose(ctx, s.weights.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dx, err := s.engine.MatMul(ctx, outputGradient, wT)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{dx}, nil
}

func TestLoraLinear_Forward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	dIn, dOut, rank := 4, 3, 2
	alpha := float32(4.0)

	// Create base linear with known weights (dIn x dOut = 4x3)
	wData := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1,
	}
	wTensor, err := tensor.New[float32]([]int{dIn, dOut}, wData)
	if err != nil {
		t.Fatalf("failed to create W tensor: %v", err)
	}
	base, err := newStubLinear[float32](engine, wTensor)
	if err != nil {
		t.Fatalf("failed to create stub linear: %v", err)
	}

	lora, err := NewLoraLinear[float32]("test", base, rank, alpha, engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create LoraLinear: %v", err)
	}

	// Set known A and B values for deterministic test
	// A: (r=2, d_in=4)
	aData := []float32{
		1, 0, 1, 0,
		0, 1, 0, 1,
	}
	aTensor, err := tensor.New[float32]([]int{rank, dIn}, aData)
	if err != nil {
		t.Fatalf("failed to create A: %v", err)
	}
	lora.A.Value = aTensor

	// B: (d_out=3, r=2)
	bData := []float32{
		1, 0,
		0, 1,
		1, 1,
	}
	bTensor, err := tensor.New[float32]([]int{dOut, rank}, bData)
	if err != nil {
		t.Fatalf("failed to create B: %v", err)
	}
	lora.B.Value = bTensor

	// Input: (1, 4)
	x, err := tensor.New[float32]([]int{1, dIn}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	out, err := lora.Forward(ctx, x)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}

	// Manual computation:
	// base(x) = x @ W = [1,2,3,4] @ [[1,0,0],[0,1,0],[0,0,1],[1,1,1]] = [5, 6, 7]
	// A @ x^T: but we do x @ A^T => [1,2,3,4] @ [[1,0],[0,1],[1,0],[0,1]] = [1+3, 2+4] = [4, 6]
	// (x@A^T) @ B^T = [4,6] @ [[1,0,1],[0,1,1]] = [4, 6, 10]
	// scale = alpha/rank = 4/2 = 2
	// lora_out = 2 * [4, 6, 10] = [8, 12, 20]
	// total = [5+8, 6+12, 7+20] = [13, 18, 27]

	outData := out.Data()
	expected := []float32{13, 18, 27}
	for i, e := range expected {
		if math.Abs(float64(outData[i]-e)) > 1e-5 {
			t.Errorf("output[%d] = %f, want %f", i, outData[i], e)
		}
	}
}

func TestLoraLinear_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dIn, dOut, rank := 8, 4, 2

	wTensor, _ := tensor.New[float32]([]int{dIn, dOut}, make([]float32, dIn*dOut))
	base, _ := newStubLinear[float32](engine, wTensor)

	lora, err := NewLoraLinear[float32]("test", base, rank, 1.0, engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create LoraLinear: %v", err)
	}

	params := lora.Parameters()
	if len(params) != 2 {
		t.Fatalf("expected 2 parameters (A, B), got %d", len(params))
	}

	if params[0].Name != "test_lora_a" {
		t.Errorf("expected first param name 'test_lora_a', got %q", params[0].Name)
	}
	if params[1].Name != "test_lora_b" {
		t.Errorf("expected second param name 'test_lora_b', got %q", params[1].Name)
	}

	// Verify A shape: r x d_in
	aShape := params[0].Value.Shape()
	if aShape[0] != rank || aShape[1] != dIn {
		t.Errorf("A shape = %v, want [%d, %d]", aShape, rank, dIn)
	}

	// Verify B shape: d_out x r
	bShape := params[1].Value.Shape()
	if bShape[0] != dOut || bShape[1] != rank {
		t.Errorf("B shape = %v, want [%d, %d]", bShape, dOut, rank)
	}

	// Base parameters should NOT be returned
	for _, p := range params {
		if p.Name == "stub_weights" {
			t.Error("base layer weights should not be in LoRA parameters")
		}
	}
}

func TestLoraLinear_BZero(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	dIn, dOut, rank := 4, 3, 2

	wData := make([]float32, dIn*dOut)
	for i := range wData {
		wData[i] = float32(i+1) * 0.1
	}
	wTensor, _ := tensor.New[float32]([]int{dIn, dOut}, wData)
	base, _ := newStubLinear[float32](engine, wTensor)

	lora, err := NewLoraLinear[float32]("test", base, rank, 4.0, engine, dIn, dOut)
	if err != nil {
		t.Fatalf("failed to create LoraLinear: %v", err)
	}

	// B is initialized to zero by default, so LoRA output should equal base output
	x, _ := tensor.New[float32]([]int{1, dIn}, []float32{1, 2, 3, 4})

	baseOut, err := base.Forward(ctx, x)
	if err != nil {
		t.Fatalf("base forward failed: %v", err)
	}

	loraOut, err := lora.Forward(ctx, x)
	if err != nil {
		t.Fatalf("lora forward failed: %v", err)
	}

	baseData := baseOut.Data()
	loraData := loraOut.Data()
	for i := range baseData {
		if math.Abs(float64(loraData[i]-baseData[i])) > 1e-6 {
			t.Errorf("with B=0, output[%d] = %f, want %f (base output)", i, loraData[i], baseData[i])
		}
	}
}

func TestLoraLinear_Backward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	dIn, dOut, rank := 3, 2, 2
	alpha := float32(2.0)
	eps := float32(1e-3)

	// Create base with known weights
	wData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	wTensor, _ := tensor.New[float32]([]int{dIn, dOut}, wData)
	base, _ := newStubLinear[float32](engine, wTensor)

	lora, _ := NewLoraLinear[float32]("test", base, rank, alpha, engine, dIn, dOut)

	// Set known A and B
	aData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	aTensor, _ := tensor.New[float32]([]int{rank, dIn}, aData)
	lora.A.Value = aTensor

	bData := []float32{0.1, 0.2, 0.3, 0.4}
	bTensor, _ := tensor.New[float32]([]int{dOut, rank}, bData)
	lora.B.Value = bTensor

	xData := []float32{1.0, 2.0, 3.0}
	x, _ := tensor.New[float32]([]int{1, dIn}, xData)

	// Forward to populate caches
	_, err := lora.Forward(ctx, x)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}

	// Finite difference gradient check for A
	for i := 0; i < rank*dIn; i++ {
		// f(A + eps)
		aPlus := make([]float32, rank*dIn)
		copy(aPlus, aData)
		aPlus[i] += eps
		aPlusTensor, _ := tensor.New[float32]([]int{rank, dIn}, aPlus)
		lora.A.Value = aPlusTensor
		outPlus, _ := lora.Forward(ctx, x)
		lossPlus := sumAll(outPlus.Data())

		// f(A - eps)
		aMinus := make([]float32, rank*dIn)
		copy(aMinus, aData)
		aMinus[i] -= eps
		aMinusTensor, _ := tensor.New[float32]([]int{rank, dIn}, aMinus)
		lora.A.Value = aMinusTensor
		outMinus, _ := lora.Forward(ctx, x)
		lossMinus := sumAll(outMinus.Data())

		fdGrad := (lossPlus - lossMinus) / (2 * eps)

		// Analytical gradient: run backward with dL/dy = ones
		lora.A.Value = aTensor
		lora.A.ClearGradient()
		lora.B.ClearGradient()
		base.weights.ClearGradient()
		_, _ = lora.Forward(ctx, x)
		ones, _ := tensor.New[float32]([]int{1, dOut}, []float32{1, 1})
		_, err := lora.Backward(ctx, types.FullBackprop, ones, x)
		if err != nil {
			t.Fatalf("backward failed: %v", err)
		}
		analyticalGrad := lora.A.Gradient.Data()[i]

		if math.Abs(float64(fdGrad-analyticalGrad)) > 1e-2 {
			t.Errorf("A gradient[%d]: finite diff = %f, analytical = %f", i, fdGrad, analyticalGrad)
		}
	}

	// Finite difference gradient check for B
	for i := 0; i < dOut*rank; i++ {
		bPlus := make([]float32, dOut*rank)
		copy(bPlus, bData)
		bPlus[i] += eps
		bPlusTensor, _ := tensor.New[float32]([]int{dOut, rank}, bPlus)
		lora.B.Value = bPlusTensor
		lora.A.Value = aTensor
		outPlus, _ := lora.Forward(ctx, x)
		lossPlus := sumAll(outPlus.Data())

		bMinus := make([]float32, dOut*rank)
		copy(bMinus, bData)
		bMinus[i] -= eps
		bMinusTensor, _ := tensor.New[float32]([]int{dOut, rank}, bMinus)
		lora.B.Value = bMinusTensor
		outMinus, _ := lora.Forward(ctx, x)
		lossMinus := sumAll(outMinus.Data())

		fdGrad := (lossPlus - lossMinus) / (2 * eps)

		lora.B.Value = bTensor
		lora.A.ClearGradient()
		lora.B.ClearGradient()
		base.weights.ClearGradient()
		_, _ = lora.Forward(ctx, x)
		ones, _ := tensor.New[float32]([]int{1, dOut}, []float32{1, 1})
		_, err := lora.Backward(ctx, types.FullBackprop, ones, x)
		if err != nil {
			t.Fatalf("backward failed: %v", err)
		}
		analyticalGrad := lora.B.Gradient.Data()[i]

		if math.Abs(float64(fdGrad-analyticalGrad)) > 1e-2 {
			t.Errorf("B gradient[%d]: finite diff = %f, analytical = %f", i, fdGrad, analyticalGrad)
		}
	}
}

func TestLoraLinear_Backward_NilGradient(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	dIn, dOut, rank := 3, 2, 2
	alpha := float32(2.0)

	// Create base with known weights.
	wData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	wTensor, _ := tensor.New[float32]([]int{dIn, dOut}, wData)
	base, _ := newStubLinear[float32](engine, wTensor)

	lora, _ := NewLoraLinear[float32]("test", base, rank, alpha, engine, dIn, dOut)

	// Set known A and B values.
	aData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	aTensor, _ := tensor.New[float32]([]int{rank, dIn}, aData)
	lora.A.Value = aTensor

	bData := []float32{0.1, 0.2, 0.3, 0.4}
	bTensor, _ := tensor.New[float32]([]int{dOut, rank}, bData)
	lora.B.Value = bTensor

	x, _ := tensor.New[float32]([]int{1, dIn}, []float32{1.0, 2.0, 3.0})

	// Forward to populate caches.
	_, err := lora.Forward(ctx, x)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}

	// Set gradients to nil to simulate first backward call with no prior gradient.
	lora.A.Gradient = nil
	lora.B.Gradient = nil
	base.weights.Gradient = nil

	// Backward should not panic when gradients are nil.
	ones, _ := tensor.New[float32]([]int{1, dOut}, []float32{1, 1})
	grads, err := lora.Backward(ctx, types.FullBackprop, ones, x)
	if err != nil {
		t.Fatalf("backward with nil gradients failed: %v", err)
	}
	if len(grads) == 0 || grads[0] == nil {
		t.Fatal("expected non-nil input gradient from backward")
	}

	// Verify gradients were populated.
	if lora.A.Gradient == nil {
		t.Error("A.Gradient should not be nil after backward")
	}
	if lora.B.Gradient == nil {
		t.Error("B.Gradient should not be nil after backward")
	}
}

func sumAll(data []float32) float32 {
	var s float32
	for _, v := range data {
		s += v
	}
	return s
}
