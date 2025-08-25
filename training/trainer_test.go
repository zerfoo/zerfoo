// Package training_test contains tests for the training package.
package training_test

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/zerfoo/types"
)

type mockNode[T tensor.Numeric] struct {
	forwardErr  bool
	backwardErr bool
	params      []*graph.Parameter[T]
	outputShape []int
}

func (m *mockNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.forwardErr {
		return nil, errors.New("forward error")
	}

	return tensor.New[T](m.outputShape, nil)
}

func (m *mockNode[T]) Backward(_ context.Context, _ types.BackwardMode, grad *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if m.backwardErr {
		return nil, errors.New("backward error")
	}

	return []*tensor.TensorNumeric[T]{grad}, nil
}

func (m *mockNode[T]) Parameters() []*graph.Parameter[T] {
	return m.params
}

func (m *mockNode[T]) OutputShape() []int {
	return m.outputShape
}

func (m *mockNode[T]) OpType() string {
	return "MockNode"
}

func (m *mockNode[T]) Attributes() map[string]any {
	return nil
}

type mockOptimizer[T tensor.Numeric] struct {
	stepErr bool
}

func (o *mockOptimizer[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	if o.stepErr {
		return errors.New("optimizer step error")
	}

	return nil
}

func TestDefaultTrainer(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	builder := graph.NewBuilder[float32](engine)

	inputNode := builder.Input([]int{1, 1})
	mockModelNode := &mockNode[float32]{outputShape: []int{1, 1}}
	builder.AddNode(mockModelNode, inputNode)
	modelGraph, err := builder.Build(mockModelNode)
	testutils.AssertNoError(t, err, "graph build failed")

	optimizer := &mockOptimizer[float32]{}
	lossFn := &mockNode[float32]{outputShape: []int{1}}
	trainer := training.NewDefaultTrainer[float32](modelGraph, lossFn, optimizer, nil)

	inputTensor, _ := tensor.New[float32]([]int{1, 1}, []float32{0})
	inputs := map[graph.Node[float32]]*tensor.TensorNumeric[float32]{
		inputNode: inputTensor,
	}
	targets, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	_, err = trainer.TrainStep(context.Background(), modelGraph, optimizer, inputs, targets)
	testutils.AssertNoError(t, err, "expected no error, got %v")
}
