package loss

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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

func (e *failingEngine) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("ReduceMean"); err != nil {
		return nil, err
	}
	return e.Engine.ReduceMean(ctx, a, axis, keepDims, dst...)
}

func (e *failingEngine) AddScalar(ctx context.Context, a *tensor.TensorNumeric[float32], s float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("AddScalar"); err != nil {
		return nil, err
	}
	return e.Engine.AddScalar(ctx, a, s, dst...)
}

func (e *failingEngine) Mul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *failingEngine) Sum(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Sum"); err != nil {
		return nil, err
	}
	return e.Engine.Sum(ctx, a, axis, keepDims, dst...)
}

func (e *failingEngine) MulScalar(ctx context.Context, a *tensor.TensorNumeric[float32], s float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("MulScalar"); err != nil {
		return nil, err
	}
	return e.Engine.MulScalar(ctx, a, s, dst...)
}

func (e *failingEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *failingEngine) Sub(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.shouldFail("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}
