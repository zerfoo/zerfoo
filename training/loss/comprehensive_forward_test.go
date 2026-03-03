package loss

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ---------- CrossEntropyLoss Forward continuation error paths ----------
// The existing errEngine in comprehensive_test.go covers Softmax, Log, OneHot, Sub, Mul.
// We add Gather, ReduceSum, MulScalar, DivScalar for the later Forward steps.
// Gather uses a custom implementation because the CPUEngine's embedding-style Gather
// rejects the shape that CrossEntropyLoss produces (output [N] vs expected [N,dim]).

type extErrEngine struct {
	errEngine
}

func newExtErrEngine(failOn map[string]int) *extErrEngine {
	return &extErrEngine{
		errEngine: errEngine{
			Engine:  newErrEngine(nil).Engine,
			calls:   make(map[string]int),
			failOn:  failOn,
			failErr: newErrEngine(nil).failErr,
		},
	}
}

// Gather extracts elements from src rows at indices, storing results in dst.
// src shape [N, C], indices shape [N], dst shape [N].
func (e *extErrEngine) Gather(_ context.Context, src *tensor.TensorNumeric[float32], indices *tensor.TensorNumeric[int], dst *tensor.TensorNumeric[float32]) error {
	if err := e.check("Gather"); err != nil {
		return err
	}
	srcData := src.Data()
	idxData := indices.Data()
	dstData := dst.Data()
	cols := src.Shape()[len(src.Shape())-1]
	for i, idx := range idxData {
		dstData[i] = srcData[i*cols+idx]
	}
	return nil
}

func (e *extErrEngine) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("ReduceSum"); err != nil {
		return nil, err
	}
	return e.Engine.ReduceSum(ctx, a, axis, keepDims, dst...)
}

func (e *extErrEngine) MulScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("MulScalar"); err != nil {
		return nil, err
	}
	return e.Engine.MulScalar(ctx, a, scalar, dst...)
}

func (e *extErrEngine) DivScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("DivScalar"); err != nil {
		return nil, err
	}
	return e.Engine.DivScalar(ctx, a, scalar, dst...)
}

func (e *extErrEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

// ---------- Tests for CrossEntropyLoss Forward error paths (post-Log) ----------

func TestCrossEntropyLoss_Forward_GatherError(t *testing.T) {
	eng := newExtErrEngine(map[string]int{"Gather": 1})
	cel := NewCrossEntropyLoss[float32](eng)

	preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[float32]([]int{2}, []float32{2.0, 0.0})

	_, err := cel.Forward(context.Background(), preds, targets)
	if err == nil {
		t.Error("expected Gather error")
	}
}

func TestCrossEntropyLoss_Forward_ReduceSumError(t *testing.T) {
	eng := newExtErrEngine(map[string]int{"ReduceSum": 1})
	cel := NewCrossEntropyLoss[float32](eng)

	preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[float32]([]int{2}, []float32{2.0, 0.0})

	_, err := cel.Forward(context.Background(), preds, targets)
	if err == nil {
		t.Error("expected ReduceSum error")
	}
}

func TestCrossEntropyLoss_Forward_MulScalarError(t *testing.T) {
	eng := newExtErrEngine(map[string]int{"MulScalar": 1})
	cel := NewCrossEntropyLoss[float32](eng)

	preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[float32]([]int{2}, []float32{2.0, 0.0})

	_, err := cel.Forward(context.Background(), preds, targets)
	if err == nil {
		t.Error("expected MulScalar error")
	}
}

func TestCrossEntropyLoss_Forward_DivScalarError(t *testing.T) {
	eng := newExtErrEngine(map[string]int{"DivScalar": 1})
	cel := NewCrossEntropyLoss[float32](eng)

	preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[float32]([]int{2}, []float32{2.0, 0.0})

	_, err := cel.Forward(context.Background(), preds, targets)
	if err == nil {
		t.Error("expected DivScalar error")
	}
}

// ---------- CrossEntropyLoss Forward success (full path) ----------

func TestCrossEntropyLoss_Forward_Success(t *testing.T) {
	eng := newExtErrEngine(nil) // No errors injected
	cel := NewCrossEntropyLoss[float32](eng)

	preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[float32]([]int{2}, []float32{2.0, 0.0})

	loss, err := cel.Forward(context.Background(), preds, targets)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if loss == nil {
		t.Fatal("expected non-nil loss tensor")
	}
	if len(loss.Shape()) != 1 || loss.Shape()[0] != 1 {
		t.Errorf("loss shape = %v, want [1]", loss.Shape())
	}
}
