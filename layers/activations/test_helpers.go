package activations

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

// ActivationLayer defines the interface for activation layers used in tests.
type ActivationLayer[T tensor.Numeric] interface {
	Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error)
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
func testActivationBackward[T tensor.Numeric](t *testing.T, activation ActivationLayer[T], inputData, expectedGrad []T) {
	ctx := context.Background()
	input, _ := tensor.New[T]([]int{len(inputData)}, inputData)
	output, _ := activation.Forward(ctx, input)
	outputGrad, _ := tensor.New[T](output.Shape(), tensor.Ones[T](len(inputData)))

	inputGrad, err := activation.Backward(ctx, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	expectedTensor, _ := tensor.New[T]([]int{len(expectedGrad)}, expectedGrad)
	tensor.AssertClose(t, expectedTensor, inputGrad[0], 1e-6)
}

// testActivationCoverage is a common helper for testing activation coverage scenarios.
//
//nolint:unused // helper retained for future coverage tests
func testActivationCoverage[T tensor.Numeric](t *testing.T, newActivationFunc func() ActivationLayer[T]) {
	// Test activation creation (Error test)
	_ = newActivationFunc()

	// Test forward error
	activation := newActivationFunc()
	input, _ := tensor.New[T]([]int{1, 1}, []T{T(1)})

	_, err := activation.Forward(context.Background(), input)
	if err != nil {
		// Expected for mock engine with error
		return
	}

	// Test output shape
	input2, _ := tensor.New[T]([]int{1, 5}, []T{T(1), T(2), T(3), T(4), T(5)})
	activation2 := newActivationFunc()
	_, _ = activation2.Forward(context.Background(), input2)
	// OutputShape test would be activation-specific

	// Test parameters
	activation3 := newActivationFunc()

	params := activation3.(*BaseActivation[T]).Parameters()
	if len(params) != 0 {
		t.Errorf("expected parameters to be empty, got %v", len(params))
	}
}
