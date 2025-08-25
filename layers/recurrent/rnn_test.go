package recurrent

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestSimpleRNN_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	inputDim := 10
	hiddenDim := 20

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, inputDim}, make([]float32, inputDim))

	output, err := rnn.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if !reflect.DeepEqual([]int{1, hiddenDim}, output.Shape()) {
		t.Errorf("unexpected output shape: got %v want %v", output.Shape(), []int{1, hiddenDim})
	}
}

func TestSimpleRNN_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	inputDim := 10
	hiddenDim := 20

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, inputDim}, make([]float32, inputDim))

	// Forward pass to populate intermediate values
	_, err = rnn.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outputGradient, _ := tensor.New[float32]([]int{1, hiddenDim}, make([]float32, hiddenDim))

	inputGradients, err := rnn.Backward(ctx, types.OneStepApproximation, outputGradient, input)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if len(inputGradients) != 2 {
		t.Fatalf("expected 2 input gradients, got %d", len(inputGradients))
	}
	if !reflect.DeepEqual([]int{1, inputDim}, inputGradients[0].Shape()) {
		t.Errorf("unexpected dInput shape: got %v want %v", inputGradients[0].Shape(), []int{1, inputDim})
	}
	if !reflect.DeepEqual([]int{1, hiddenDim}, inputGradients[1].Shape()) {
		t.Errorf("unexpected dHidden shape: got %v want %v", inputGradients[1].Shape(), []int{1, hiddenDim})
	}

	// Check that parameter gradients are populated
	params := rnn.Parameters()
	for _, p := range params {
		if p.Gradient == nil {
			t.Errorf("expected parameter gradient to be populated")
		}
	}
}

func TestSimpleRNN_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	inputDim := 10
	hiddenDim := 20

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	params := rnn.Parameters()
	// input weights + hidden weights + bias
	if len(params) != 3 {
		t.Fatalf("expected 3 params, got %d", len(params))
	}
}
