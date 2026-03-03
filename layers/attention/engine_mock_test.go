package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// failingEngine wraps a real CPU engine and fails on specific operation calls.
// The failOn map specifies which operation should fail at which call number.
type failingEngine struct {
	compute.Engine[float32]
	failOn map[string]int
	calls  map[string]int
}

func newFailingEngine(failOn map[string]int) *failingEngine {
	return &failingEngine{
		Engine: compute.NewCPUEngine(numeric.Float32Ops{}),
		failOn: failOn,
		calls:  make(map[string]int),
	}
}

func (e *failingEngine) shouldFail(op string) error {
	e.calls[op]++
	if n, ok := e.failOn[op]; ok && e.calls[op] >= n {
		return fmt.Errorf("injected %s failure at call %d", op, e.calls[op])
	}
	return nil
}

func (e *failingEngine) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("MatMul"); err != nil {
		return nil, err
	}
	return e.Engine.MatMul(ctx, a, b, dst...)
}

func (e *failingEngine) Transpose(ctx context.Context, a *tensor.TensorNumeric[float32], perm []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Transpose"); err != nil {
		return nil, err
	}
	return e.Engine.Transpose(ctx, a, perm, dst...)
}

func (e *failingEngine) MulScalar(ctx context.Context, a *tensor.TensorNumeric[float32], s float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("MulScalar"); err != nil {
		return nil, err
	}
	return e.Engine.MulScalar(ctx, a, s, dst...)
}

func (e *failingEngine) Softmax(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Softmax"); err != nil {
		return nil, err
	}
	return e.Engine.Softmax(ctx, a, axis, dst...)
}

func (e *failingEngine) Reshape(ctx context.Context, a *tensor.TensorNumeric[float32], shape []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Reshape"); err != nil {
		return nil, err
	}
	return e.Engine.Reshape(ctx, a, shape, dst...)
}

func (e *failingEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *failingEngine) Mul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *failingEngine) Sub(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}

func (e *failingEngine) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("ReduceSum"); err != nil {
		return nil, err
	}
	return e.Engine.ReduceSum(ctx, a, axis, keepDims, dst...)
}

func (e *failingEngine) Repeat(ctx context.Context, a *tensor.TensorNumeric[float32], axis, count int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Repeat"); err != nil {
		return nil, err
	}
	return e.Engine.Repeat(ctx, a, axis, count, dst...)
}

func (e *failingEngine) Sum(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Sum"); err != nil {
		return nil, err
	}
	return e.Engine.Sum(ctx, a, axis, keepDims, dst...)
}
