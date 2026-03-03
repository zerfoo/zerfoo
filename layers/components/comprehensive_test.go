package components

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ---------- errEngine for injecting errors into Transpose/MatMul ----------

type errComponentEngine struct {
	compute.Engine[float32]
	calls   map[string]int
	failOn  map[string]int
	failErr error
}

func newErrComponentEngine(failOn map[string]int) *errComponentEngine {
	return &errComponentEngine{
		Engine:  compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		calls:   make(map[string]int),
		failOn:  failOn,
		failErr: errors.New("injected error"),
	}
}

func (e *errComponentEngine) check(op string) error {
	e.calls[op]++
	if n, ok := e.failOn[op]; ok && e.calls[op] >= n {
		return e.failErr
	}
	return nil
}

func (e *errComponentEngine) Transpose(ctx context.Context, a *tensor.TensorNumeric[float32], perm []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Transpose"); err != nil {
		return nil, err
	}
	return e.Engine.Transpose(ctx, a, perm, dst...)
}

func (e *errComponentEngine) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("MatMul"); err != nil {
		return nil, err
	}
	return e.Engine.MatMul(ctx, a, b, dst...)
}

// ---------- Constructor options coverage ----------

func TestNewLinearGradientComputer_WithOptions(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	dummyOpt := func(_ *LinearGradientComputerOptions[float32]) {}
	gc := NewLinearGradientComputer(engine, dummyOpt)
	if gc == nil {
		t.Fatal("expected non-nil LinearGradientComputer")
	}
}

func TestNewMatrixMultiplier_WithOptions(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	dummyOpt := func(_ *MatrixMultiplierOptions[float32]) {}
	mm := NewMatrixMultiplier(engine, dummyOpt)
	if mm == nil {
		t.Fatal("expected non-nil MatrixMultiplier")
	}
}

func TestNewXavierInitializer_WithOptions(t *testing.T) {
	ops := numeric.Float32Ops{}
	dummyOpt := func(_ *XavierInitializerOptions[float32]) {}
	xi := NewXavierInitializer(ops, dummyOpt)
	if xi == nil {
		t.Fatal("expected non-nil XavierInitializer")
	}
}

func TestNewHeInitializer_WithOptions(t *testing.T) {
	ops := numeric.Float32Ops{}
	dummyOpt := func(_ *HeInitializerOptions[float32]) {}
	hi := NewHeInitializer(ops, dummyOpt)
	if hi == nil {
		t.Fatal("expected non-nil HeInitializer")
	}
}

// ---------- MatrixMultiplier: MultiplyWithDestination and TransposeWithDestination ----------

func TestMatrixMultiplier_MultiplyWithDestination(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	mm := NewMatrixMultiplier[float32](engine)

	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	dst, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

	result, err := mm.MultiplyWithDestination(context.Background(), a, b, dst)
	if err != nil {
		t.Fatalf("MultiplyWithDestination error: %v", err)
	}
	if !equalIntSlices(result.Shape(), []int{2, 2}) {
		t.Errorf("shape = %v, want [2 2]", result.Shape())
	}
}

func TestMatrixMultiplier_TransposeWithDestination(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	mm := NewMatrixMultiplier[float32](engine)

	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	dst, _ := tensor.New[float32]([]int{3, 2}, make([]float32, 6))

	result, err := mm.TransposeWithDestination(context.Background(), a, dst)
	if err != nil {
		t.Fatalf("TransposeWithDestination error: %v", err)
	}
	if !equalIntSlices(result.Shape(), []int{3, 2}) {
		t.Errorf("shape = %v, want [3 2]", result.Shape())
	}
}

// ---------- LinearGradientComputer error paths ----------

func TestLinearGradientComputer_ComputeWeightGradient_TransposeError(t *testing.T) {
	eng := newErrComponentEngine(map[string]int{"Transpose": 1})
	gc := NewLinearGradientComputer[float32](eng)
	input, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	outputGrad, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

	_, err := gc.ComputeWeightGradient(context.Background(), input, outputGrad)
	if err == nil {
		t.Error("expected Transpose error")
	}
}

func TestLinearGradientComputer_ComputeWeightGradient_MatMulError(t *testing.T) {
	eng := newErrComponentEngine(map[string]int{"MatMul": 1})
	gc := NewLinearGradientComputer[float32](eng)
	input, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	outputGrad, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

	_, err := gc.ComputeWeightGradient(context.Background(), input, outputGrad)
	if err == nil {
		t.Error("expected MatMul error")
	}
}

func TestLinearGradientComputer_ComputeInputGradient_TransposeError(t *testing.T) {
	eng := newErrComponentEngine(map[string]int{"Transpose": 1})
	gc := NewLinearGradientComputer[float32](eng)
	weights, _ := tensor.New[float32]([]int{3, 2}, make([]float32, 6))
	outputGrad, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

	_, err := gc.ComputeInputGradient(context.Background(), weights, outputGrad)
	if err == nil {
		t.Error("expected Transpose error")
	}
}

func TestLinearGradientComputer_ComputeInputGradient_MatMulError(t *testing.T) {
	eng := newErrComponentEngine(map[string]int{"MatMul": 1})
	gc := NewLinearGradientComputer[float32](eng)
	weights, _ := tensor.New[float32]([]int{3, 2}, make([]float32, 6))
	outputGrad, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

	_, err := gc.ComputeInputGradient(context.Background(), weights, outputGrad)
	if err == nil {
		t.Error("expected MatMul error")
	}
}

func TestLinearGradientComputer_ComputeBothGradients_WeightError(t *testing.T) {
	eng := newErrComponentEngine(map[string]int{"Transpose": 1})
	gc := NewLinearGradientComputer[float32](eng)
	input, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	weights, _ := tensor.New[float32]([]int{3, 2}, make([]float32, 6))
	outputGrad, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

	_, _, err := gc.ComputeBothGradients(context.Background(), input, weights, outputGrad)
	if err == nil {
		t.Error("expected error from weight gradient in ComputeBothGradients")
	}
}

func TestLinearGradientComputer_ComputeBothGradients_InputError(t *testing.T) {
	// First Transpose (weight gradient) succeeds, second Transpose (input gradient) fails
	eng := newErrComponentEngine(map[string]int{"Transpose": 2})
	gc := NewLinearGradientComputer[float32](eng)
	input, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	weights, _ := tensor.New[float32]([]int{3, 2}, make([]float32, 6))
	outputGrad, _ := tensor.New[float32]([]int{2, 2}, make([]float32, 4))

	_, _, err := gc.ComputeBothGradients(context.Background(), input, weights, outputGrad)
	if err == nil {
		t.Error("expected error from input gradient in ComputeBothGradients")
	}
}
