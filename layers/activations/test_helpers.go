package activations

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

// ActivationLayer defines the interface for activation layers used in tests.
type ActivationLayer[T tensor.Numeric] interface {
	Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)
	Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error)
}

// testActivationForward is a common helper for testing activation forward passes.
func testActivationForward[T tensor.Numeric](t *testing.T, activation ActivationLayer[T]) {
	input, _ := tensor.New[T]([]int{1, 2}, []T{T(1), T(2)})
	output, err := activation.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Shape()[0] != 1 || output.Shape()[1] != 2 {
		t.Errorf("expected shape [1 2], got %v", output.Shape())
	}
}

// testActivationBackward is a common helper for testing activation backward passes.
func testActivationBackward[T tensor.Numeric](t *testing.T, activation ActivationLayer[T]) {
	input, _ := tensor.New[T]([]int{1, 2}, []T{T(1), T(2)})
	_, err := activation.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	outputGradient, _ := tensor.New[T]([]int{1, 2}, []T{T(1), T(2)})
	inputGrads, err := activation.Backward(context.Background(), outputGradient)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if inputGrads[0].Shape()[0] != 1 || inputGrads[0].Shape()[1] != 2 {
		t.Errorf("expected shape [1 2], got %v", inputGrads[0].Shape())
	}
}
