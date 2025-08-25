package features

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestSpectralFingerprint_Forward(t *testing.T) {
	ctx := context.Background()
	outputDim := 8

	layer := NewSpectralFingerprint[float32](outputDim)

	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = float32(i)
	}
	input, _ := tensor.New[float32]([]int{1, 16}, inputData)

	output, err := layer.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if !reflect.DeepEqual([]int{1, outputDim}, output.Shape()) {
		t.Errorf("unexpected output shape: got %v want %v", output.Shape(), []int{1, outputDim})
	}
}

func TestSpectralFingerprint_Backward(t *testing.T) {
	ctx := context.Background()
	outputDim := 8

	layer := NewSpectralFingerprint[float32](outputDim)

	inputData := make([]float32, 16)
	input, _ := tensor.New[float32]([]int{1, 16}, inputData)

	outputGradient, _ := tensor.New[float32]([]int{1, outputDim}, make([]float32, outputDim))

	inputGradients, err := layer.Backward(ctx, types.FullBackprop, outputGradient, input)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if len(inputGradients) != 1 {
		t.Fatalf("expected 1 input gradient, got %d", len(inputGradients))
	}
	if !reflect.DeepEqual(input.Shape(), inputGradients[0].Shape()) {
		t.Errorf("unexpected gradient shape: got %v want %v", inputGradients[0].Shape(), input.Shape())
	}

	// Check if the gradient is all zeros
	for _, v := range inputGradients[0].Data() {
		if v != 0 {
			t.Errorf("expected zero gradient, got %v", v)
		}
	}
}

func TestSpectralFingerprint_Parameters(t *testing.T) {
	layer := NewSpectralFingerprint[float32](8)
	if len(layer.Parameters()) != 0 {
		t.Fatalf("expected no parameters")
	}
}
