package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestConv1D_ForwardShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	conv, err := NewConv1D[float32]("test_conv", engine, ops, 2, 3, 3)
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	// Input: [batch=1, channels=2, length=5]
	input, _ := tensor.New[float32]([]int{1, 2, 5}, make([]float32, 10))
	output, err := conv.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Output: [1, 3, 3] (length = (5-3)/1 + 1 = 3)
	want := []int{1, 3, 3}
	got := output.Shape()
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Errorf("shape = %v, want %v", got, want)
	}
}

func TestConv1D_ForwardWithPadding(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	conv, err := NewConv1D[float32]("test_conv", engine, ops, 2, 3, 3, Conv1DPadding(1))
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	// Input: [batch=1, channels=2, length=5]
	input, _ := tensor.New[float32]([]int{1, 2, 5}, make([]float32, 10))
	output, err := conv.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Output: [1, 3, 5] (length = (5+2-3)/1 + 1 = 5)
	want := []int{1, 3, 5}
	got := output.Shape()
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Errorf("shape = %v, want %v", got, want)
	}
}

func TestConv1D_ForwardValues(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	conv, err := NewConv1D[float32]("test_conv", engine, ops, 1, 1, 3, Conv1DWithoutBias())
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	// Set weight to [1, 1, 1] (sum kernel)
	conv.weight.Value.Data()[0] = 1.0
	conv.weight.Value.Data()[1] = 1.0
	conv.weight.Value.Data()[2] = 1.0
	conv.weight.Value.SetData(conv.weight.Value.Data())

	// Input: [1, 1, 5] = [1, 2, 3, 4, 5]
	input, _ := tensor.New[float32]([]int{1, 1, 5}, []float32{1, 2, 3, 4, 5})
	output, err := conv.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Expected: [6, 9, 12] (sliding window sum of 3)
	data := output.Data()
	expected := []float32{6, 9, 12}
	for i, want := range expected {
		if data[i] != want {
			t.Errorf("output[%d] = %f, want %f", i, data[i], want)
		}
	}
}

func TestConv1D_BackwardShapes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	conv, err := NewConv1D[float32]("test_conv", engine, ops, 2, 3, 3)
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	input, _ := tensor.New[float32]([]int{2, 2, 5}, make([]float32, 20))
	output, err := conv.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outputGrad, _ := tensor.New[float32](output.Shape(), make([]float32, output.Size()))
	for i := range outputGrad.Data() {
		outputGrad.Data()[i] = 1.0
	}
	outputGrad.SetData(outputGrad.Data())

	inputGrads, err := conv.Backward(context.Background(), types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(inputGrads) != 1 {
		t.Fatalf("expected 1 input grad, got %d", len(inputGrads))
	}
	if inputGrads[0].Shape()[0] != 2 || inputGrads[0].Shape()[1] != 2 || inputGrads[0].Shape()[2] != 5 {
		t.Errorf("input grad shape = %v, want [2, 2, 5]", inputGrads[0].Shape())
	}

	// Check weight gradient exists and has correct shape
	wGrad := conv.weight.Gradient
	if wGrad == nil {
		t.Fatal("weight gradient is nil")
	}
	wShape := wGrad.Shape()
	if wShape[0] != 3 || wShape[1] != 2 || wShape[2] != 3 {
		t.Errorf("weight grad shape = %v, want [3, 2, 3]", wShape)
	}
}

func TestConv1D_NoBias(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	conv, err := NewConv1D[float32]("test_conv", engine, ops, 1, 1, 3, Conv1DWithoutBias())
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	params := conv.Parameters()
	if len(params) != 1 {
		t.Errorf("expected 1 parameter (weight only), got %d", len(params))
	}
}

func TestConv1D_ParameterCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	conv, err := NewConv1D[float32]("test_conv", engine, ops, 4, 8, 3)
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	params := conv.Parameters()
	if len(params) != 2 { // weight + bias
		t.Fatalf("expected 2 parameters, got %d", len(params))
	}

	// Weight: 8 * 4 * 3 = 96
	if params[0].Value.Size() != 96 {
		t.Errorf("weight size = %d, want 96", params[0].Value.Size())
	}
	// Bias: 8
	if params[1].Value.Size() != 8 {
		t.Errorf("bias size = %d, want 8", params[1].Value.Size())
	}
}

func TestConv1D_WithStride(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	conv, err := NewConv1D[float32]("test_conv", engine, ops, 1, 1, 3, Conv1DStride(2))
	if err != nil {
		t.Fatalf("NewConv1D: %v", err)
	}

	// Input: [1, 1, 7]
	input, _ := tensor.New[float32]([]int{1, 1, 7}, make([]float32, 7))
	output, err := conv.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Output length = (7-3)/2 + 1 = 3
	if output.Shape()[2] != 3 {
		t.Errorf("output length = %d, want 3", output.Shape()[2])
	}
}
