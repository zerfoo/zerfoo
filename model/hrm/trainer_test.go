// Package hrm_test contains tests for the HRM model.
package hrm_test

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/model/hrm"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
)

type mockOptimizer[T tensor.Numeric] struct {
	stepErr bool
}

func (o *mockOptimizer[T]) Step(ctx context.Context, _ []*graph.Parameter[T]) error {
	if o.stepErr {
		return errors.New("optimizer step error")
	}

	return nil
}

type mockLoss[T tensor.Numeric] struct{}

func (l *mockLoss[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return tensor.New[T]([]int{1}, nil)
}

func (l *mockLoss[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	grad, _ := tensor.New[T]([]int{1}, nil)
	return []*tensor.TensorNumeric[T]{grad}, nil
}

func (l *mockLoss[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

func (l *mockLoss[T]) OutputShape() []int {
	return []int{1}
}

func (l *mockLoss[T]) OpType() string {
	return "MockLoss"
}

func (l *mockLoss[T]) Attributes() map[string]any {
	return nil
}

func TestHRMTrainer(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 32
	inputDim := 16
	outputDim := 8
	N := 2
	T := 3
	numHeads := 2

	hAttention, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create H-attention: %v", err)
	}

	lAttention, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create L-attention: %v", err)
	}

	model, err := hrm.NewHRM[float32](engine, ops, modelDim, ffnDim, inputDim, outputDim, hAttention, lAttention)
	if err != nil {
		t.Fatalf("failed to create HRM model: %v", err)
	}

	lossFn := &mockLoss[float32]{}
	trainer := hrm.NewHRMTrainer[float32](model, lossFn)
	optimizer := &mockOptimizer[float32]{}

	inputTensor, _ := tensor.New[float32]([]int{1, inputDim}, nil)
	inputs := map[graph.Node[float32]]*tensor.TensorNumeric[float32]{
		model.InputNet: inputTensor,
	}
	targets, _ := tensor.New[float32]([]int{1, outputDim}, nil)

	_, err = trainer.TrainStep(context.Background(), optimizer, inputs, targets, N, T)
	testutils.AssertNoError(t, err, "expected no error from TrainStep")
}
