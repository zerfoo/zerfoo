package activations

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestTanh_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	tanh := NewTanh[int](engine, numeric.IntOps{})
	input, _ := tensor.New[int]([]int{1, 2}, []int{1, 2})
	output, err := tanh.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Shape()[0] != 1 || output.Shape()[1] != 2 {
		t.Errorf("expected shape [1 2], got %v", output.Shape())
	}
}

func TestTanh_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	tanh := NewTanh[int](engine, numeric.IntOps{})
	input, _ := tensor.New[int]([]int{1, 2}, []int{1, 2})
	_, err := tanh.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	outputGradient, _ := tensor.New[int]([]int{1, 2}, []int{1, 2})
	inputGrads, err := tanh.Backward(context.Background(), outputGradient)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if inputGrads[0].Shape()[0] != 1 || inputGrads[0].Shape()[1] != 2 {
		t.Errorf("expected shape [1 2], got %v", inputGrads[0].Shape())
	}
}
