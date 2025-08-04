package training

import (
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

type mockModel[T tensor.Numeric] struct {
	params []*graph.Parameter[T]
}

func (m *mockModel[T]) Forward(inputs ...*tensor.Tensor[T]) *tensor.Tensor[T] {
	return inputs[0]
}

func (m *mockModel[T]) Backward(grad *tensor.Tensor[T]) []*tensor.Tensor[T] {
	return []*tensor.Tensor[T]{grad}
}

func (m *mockModel[T]) Parameters() []*graph.Parameter[T] {
	return m.params
}

type mockOptimizer[T tensor.Numeric] struct{}

func (o *mockOptimizer[T]) Step(_ []*graph.Parameter[T])                    {}
func (o *mockOptimizer[T]) Clip(_ []*graph.Parameter[T], threshold float32) {}

type mockLoss[T tensor.Numeric] struct{}

func (l *mockLoss[T]) Forward(predictions, _ *tensor.Tensor[T]) (T, *tensor.Tensor[T]) {
	var lossValue T
	return lossValue, predictions
}

func TestTrainer(t *testing.T) {
	model := &mockModel[float32]{}
	optimizer := &mockOptimizer[float32]{}
	lossFn := &mockLoss[float32]{}
	trainer := NewTrainer[float32](model, optimizer, lossFn)

	inputs, _ := tensor.New[float32]([]int{1, 1}, []float32{1})
	targets, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	lossValue, err := trainer.Train(inputs, targets)
	testutils.AssertNoError(t, err, "expected no error, got %v")
	testutils.AssertNotNil(t, lossValue, "expected lossValue to not be nil")
}
