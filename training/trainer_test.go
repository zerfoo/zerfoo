package training

import (
	"context"
	"errors"
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

func TestNewTrainer(t *testing.T) {
	model := &mockModel[float32]{}
	optimizer := &mockOptimizer[float32]{}
	lossFn := &mockLoss[float32]{}
	
	trainer := NewTrainer[float32](model, optimizer, lossFn)
	
	testutils.AssertNotNil(t, trainer, "trainer should not be nil")
	// Check that the trainer fields are set (can't directly compare interfaces)
	testutils.AssertNotNil(t, trainer.model, "model should be set")
	testutils.AssertNotNil(t, trainer.optimizer, "optimizer should be set")
	testutils.AssertNotNil(t, trainer.lossFn, "loss function should be set")
}

// Error mocks for testing error handling
type errorModel[T tensor.Numeric] struct {
	forwardErr  bool
	backwardErr bool
	params      []*graph.Parameter[T]
}

func (m *errorModel[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if m.forwardErr {
		return nil, errors.New("forward error")
	}
	return inputs[0], nil
}

func (m *errorModel[T]) Backward(ctx context.Context, grad *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	if m.backwardErr {
		return nil, errors.New("backward error")
	}
	return []*tensor.Tensor[T]{grad}, nil
}

func (m *errorModel[T]) Parameters() []*graph.Parameter[T] {
	return m.params
}

type errorOptimizer[T tensor.Numeric] struct {
	stepErr bool
}

func (o *errorOptimizer[T]) Step(_ context.Context, _ []*graph.Parameter[T]) error {
	if o.stepErr {
		return errors.New("optimizer step error")
	}
	return nil
}

func (o *errorOptimizer[T]) Clip(_ context.Context, _ []*graph.Parameter[T], _ float32) {}

type errorLoss[T tensor.Numeric] struct {
	forwardErr bool
}

func (l *errorLoss[T]) Forward(ctx context.Context, predictions, _ *tensor.Tensor[T]) (T, *tensor.Tensor[T], error) {
	if l.forwardErr {
		var zero T
		return zero, nil, errors.New("loss forward error")
	}
	var lossValue T
	return lossValue, predictions, nil
}

func TestTrainer_Train_ErrorHandling(t *testing.T) {
	inputs, _ := tensor.New[float32]([]int{1, 1}, []float32{1})
	targets, _ := tensor.New[float32]([]int{1, 1}, []float32{1})
	ctx := context.Background()

	// Test model forward error
	t.Run("model forward error", func(t *testing.T) {
		model := &errorModel[float32]{forwardErr: true}
		optimizer := &mockOptimizer[float32]{}
		lossFn := &mockLoss[float32]{}
		trainer := NewTrainer[float32](model, optimizer, lossFn)

		_, err := trainer.Train(ctx, inputs, targets)
		testutils.AssertError(t, err, "expected forward error")
	})

	// Test loss forward error
	t.Run("loss forward error", func(t *testing.T) {
		model := &mockModel[float32]{}
		optimizer := &mockOptimizer[float32]{}
		lossFn := &errorLoss[float32]{forwardErr: true}
		trainer := NewTrainer[float32](model, optimizer, lossFn)

		_, err := trainer.Train(ctx, inputs, targets)
		testutils.AssertError(t, err, "expected loss forward error")
	})

	// Test model backward error
	t.Run("model backward error", func(t *testing.T) {
		model := &errorModel[float32]{backwardErr: true}
		optimizer := &mockOptimizer[float32]{}
		lossFn := &mockLoss[float32]{}
		trainer := NewTrainer[float32](model, optimizer, lossFn)

		_, err := trainer.Train(ctx, inputs, targets)
		testutils.AssertError(t, err, "expected backward error")
	})

	// Test optimizer step error
	t.Run("optimizer step error", func(t *testing.T) {
		model := &mockModel[float32]{}
		optimizer := &errorOptimizer[float32]{stepErr: true}
		lossFn := &mockLoss[float32]{}
		trainer := NewTrainer[float32](model, optimizer, lossFn)

		_, err := trainer.Train(ctx, inputs, targets)
		testutils.AssertError(t, err, "expected optimizer step error")
	})
}

func TestTrainer_Train_WithParameters(t *testing.T) {
	// Create a parameter for the model
	value, err := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	testutils.AssertNoError(t, err, "Failed to create parameter value")
	
	param, err := graph.NewParameter("test_param", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")
	
	model := &mockModel[float32]{params: []*graph.Parameter[float32]{param}}
	optimizer := &mockOptimizer[float32]{}
	lossFn := &mockLoss[float32]{}
	trainer := NewTrainer[float32](model, optimizer, lossFn)

	inputs, _ := tensor.New[float32]([]int{1, 1}, []float32{1})
	targets, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	lossValue, err := trainer.Train(context.Background(), inputs, targets)
	testutils.AssertNoError(t, err, "training should not error with parameters")
	testutils.AssertNotNil(t, lossValue, "loss value should not be nil")
}

func TestTrainer_Train_DifferentTensorShapes(t *testing.T) {
	model := &mockModel[float32]{}
	optimizer := &mockOptimizer[float32]{}
	lossFn := &mockLoss[float32]{}
	trainer := NewTrainer[float32](model, optimizer, lossFn)

	// Test with different tensor shapes
	inputs, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	targets, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	lossValue, err := trainer.Train(context.Background(), inputs, targets)
	testutils.AssertNoError(t, err, "training should work with different tensor shapes")
	testutils.AssertNotNil(t, lossValue, "loss value should not be nil")
}