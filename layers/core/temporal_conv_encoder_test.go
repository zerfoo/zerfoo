package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestTemporalConvEncoder_ForwardShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	enc, err := NewTemporalConvEncoder[float32]("test_tce", engine, ops, 6, 16, 32)
	if err != nil {
		t.Fatalf("NewTemporalConvEncoder: %v", err)
	}

	// [batch=2, numStats=6, windowSize=10]
	input, _ := tensor.New[float32]([]int{2, 6, 10}, make([]float32, 2*6*10))
	output, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	want := []int{2, 32}
	got := output.Shape()
	if got[0] != want[0] || got[1] != want[1] {
		t.Errorf("shape = %v, want %v", got, want)
	}
}

func TestTemporalConvEncoder_BackwardNonZeroGradients(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	enc, err := NewTemporalConvEncoder[float32]("test_tce", engine, ops, 4, 8, 16)
	if err != nil {
		t.Fatalf("NewTemporalConvEncoder: %v", err)
	}

	data := make([]float32, 2*4*10)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	input, _ := tensor.New[float32]([]int{2, 4, 10}, data)

	output, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	gradData := make([]float32, output.Size())
	for i := range gradData {
		gradData[i] = 1.0
	}
	outputGrad, _ := tensor.New[float32](output.Shape(), gradData)

	inputGrads, err := enc.Backward(context.Background(), types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(inputGrads) != 1 {
		t.Fatalf("expected 1 input grad, got %d", len(inputGrads))
	}

	// Check input gradient has non-zero values
	hasNonZero := false
	for _, v := range inputGrads[0].Data() {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("input gradient is all zeros")
	}

	// Check parameter gradients are non-zero
	for _, p := range enc.Parameters() {
		if p.Gradient == nil {
			t.Errorf("parameter %s has nil gradient", p.Name)
			continue
		}
		hasNonZero = false
		for _, v := range p.Gradient.Data() {
			if v != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Errorf("parameter %s gradient is all zeros", p.Name)
		}
	}
}

func TestTemporalConvEncoder_ParameterCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	numStats, hidden, outputDim := 6, 16, 32
	enc, err := NewTemporalConvEncoder[float32]("test_tce", engine, ops, numStats, hidden, outputDim)
	if err != nil {
		t.Fatalf("NewTemporalConvEncoder: %v", err)
	}

	params := enc.Parameters()
	// conv1: weight[hidden, numStats, 3] + bias[hidden]
	// conv2: weight[hidden, hidden, 3] + bias[hidden]
	// linear: weight[hidden, outputDim]
	// Total: 5 parameters
	totalSize := 0
	for _, p := range params {
		totalSize += p.Value.Size()
	}

	wantConv1W := hidden * numStats * 3
	wantConv1B := hidden
	wantConv2W := hidden * hidden * 3
	wantConv2B := hidden
	wantLinearW := hidden * outputDim
	wantTotal := wantConv1W + wantConv1B + wantConv2W + wantConv2B + wantLinearW
	if totalSize != wantTotal {
		t.Errorf("total param size = %d, want %d", totalSize, wantTotal)
	}
}

func TestTemporalConvEncoder_DifferentWindowSizes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	enc, err := NewTemporalConvEncoder[float32]("test_tce", engine, ops, 4, 8, 16)
	if err != nil {
		t.Fatalf("NewTemporalConvEncoder: %v", err)
	}

	for _, windowSize := range []int{10, 50, 100} {
		data := make([]float32, 2*4*windowSize)
		for i := range data {
			data[i] = float32(i) * 0.001
		}
		input, _ := tensor.New[float32]([]int{2, 4, windowSize}, data)
		output, err := enc.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("Forward (window=%d): %v", windowSize, err)
		}
		if output.Shape()[0] != 2 || output.Shape()[1] != 16 {
			t.Errorf("window=%d: shape = %v, want [2, 16]", windowSize, output.Shape())
		}
	}
}

func TestTemporalConvEncoder_DifferentNumStats(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	for _, numStats := range []int{2, 8, 20} {
		enc, err := NewTemporalConvEncoder[float32]("test_tce", engine, ops, numStats, 8, 16)
		if err != nil {
			t.Fatalf("NewTemporalConvEncoder (numStats=%d): %v", numStats, err)
		}
		data := make([]float32, 3*numStats*10)
		for i := range data {
			data[i] = float32(i) * 0.01
		}
		input, _ := tensor.New[float32]([]int{3, numStats, 10}, data)
		output, err := enc.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("Forward (numStats=%d): %v", numStats, err)
		}
		if output.Shape()[0] != 3 || output.Shape()[1] != 16 {
			t.Errorf("numStats=%d: shape = %v, want [3, 16]", numStats, output.Shape())
		}
	}
}
