package training

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

type mockModel[T tensor.Numeric] struct {
	params []*graph.Parameter[T]
}

func (m *mockModel[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return inputs[0], nil
}

func (m *mockModel[T]) Backward(ctx context.Context, grad *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	return []*tensor.Tensor[T]{grad}, nil
}

func (m *mockModel[T]) Parameters() []*graph.Parameter[T] {
	return m.params
}

type mockOptimizer[T tensor.Numeric] struct{}

func (o *mockOptimizer[T]) Step(_ context.Context, _ []*graph.Parameter[T]) error            { return nil }
func (o *mockOptimizer[T]) Clip(_ context.Context, _ []*graph.Parameter[T], _ float32) {}

type mockLoss[T tensor.Numeric] struct{}

func (l *mockLoss[T]) Forward(ctx context.Context, predictions, _ *tensor.Tensor[T]) (T, *tensor.Tensor[T], error) {
	var lossValue T

	return lossValue, predictions, nil
}

func TestTrainer(t *testing.T) {
	model := &mockModel[float32]{}
	optimizer := &mockOptimizer[float32]{}
	lossFn := &mockLoss[float32]{}
	trainer := NewTrainer[float32](model, optimizer, lossFn)

	inputs, _ := tensor.New[float32]([]int{1, 1}, []float32{1})
	targets, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	lossValue, err := trainer.Train(context.Background(), inputs, targets)
	testutils.AssertNoError(t, err, "expected no error, got %v")
	testutils.AssertNotNil(t, lossValue, "expected lossValue to not be nil")
}