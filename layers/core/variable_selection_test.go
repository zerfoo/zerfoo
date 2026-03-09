package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestVariableSelection_ForwardShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	vsn, err := NewVariableSelection[float32]("test_vsn", engine, ops, 10, 8)
	if err != nil {
		t.Fatalf("NewVariableSelection: %v", err)
	}

	input, _ := tensor.New[float32]([]int{4, 10}, make([]float32, 40))
	output, err := vsn.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	want := []int{4, 10}
	got := output.Shape()
	if got[0] != want[0] || got[1] != want[1] {
		t.Errorf("shape = %v, want %v", got, want)
	}
}

func TestVariableSelection_GateWeightsSumToOne(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	vsn, err := NewVariableSelection[float32]("test_vsn", engine, ops, 5, 4)
	if err != nil {
		t.Fatalf("NewVariableSelection: %v", err)
	}

	input, _ := tensor.New[float32]([]int{2, 5}, []float32{1, 2, 3, 4, 5, 5, 4, 3, 2, 1})
	_, err = vsn.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Check that gate weights (from softmax) sum to 1 per sample
	weights := vsn.lastWeights.Data()
	for b := 0; b < 2; b++ {
		sum := float64(0)
		for f := 0; f < 5; f++ {
			sum += float64(weights[b*5+f])
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Errorf("batch %d: weights sum = %f, want 1.0", b, sum)
		}
	}
}

func TestVariableSelection_BackwardNonZeroGradients(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	vsn, err := NewVariableSelection[float32]("test_vsn", engine, ops, 5, 4)
	if err != nil {
		t.Fatalf("NewVariableSelection: %v", err)
	}

	input, _ := tensor.New[float32]([]int{2, 5}, []float32{1, 2, 3, 4, 5, 5, 4, 3, 2, 1})
	output, err := vsn.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create output gradient of all ones
	gradData := make([]float32, output.Size())
	for i := range gradData {
		gradData[i] = 1.0
	}
	outputGrad, _ := tensor.New[float32](output.Shape(), gradData)

	inputGrads, err := vsn.Backward(context.Background(), types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(inputGrads) != 1 {
		t.Fatalf("expected 1 input grad, got %d", len(inputGrads))
	}

	// Check input gradient is non-zero
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
	for _, p := range vsn.Parameters() {
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

func TestVariableSelection_ParameterCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	vsn, err := NewVariableSelection[float32]("test_vsn", engine, ops, 10, 8)
	if err != nil {
		t.Fatalf("NewVariableSelection: %v", err)
	}

	params := vsn.Parameters()
	if len(params) != 4 {
		t.Fatalf("expected 4 parameters (w1, b1, w2, b2), got %d", len(params))
	}

	// w1: [10, 8] = 80
	if params[0].Value.Size() != 80 {
		t.Errorf("w1 size = %d, want 80", params[0].Value.Size())
	}
	// b1: [8] = 8
	if params[1].Value.Size() != 8 {
		t.Errorf("b1 size = %d, want 8", params[1].Value.Size())
	}
	// w2: [8, 10] = 80
	if params[2].Value.Size() != 80 {
		t.Errorf("w2 size = %d, want 80", params[2].Value.Size())
	}
	// b2: [10] = 10
	if params[3].Value.Size() != 10 {
		t.Errorf("b2 size = %d, want 10", params[3].Value.Size())
	}
}

func TestVariableSelection_DifferentBatchSizes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	vsn, err := NewVariableSelection[float32]("test_vsn", engine, ops, 5, 4)
	if err != nil {
		t.Fatalf("NewVariableSelection: %v", err)
	}

	for _, bs := range []int{1, 4, 16} {
		data := make([]float32, bs*5)
		for i := range data {
			data[i] = float32(i) * 0.1
		}
		input, _ := tensor.New[float32]([]int{bs, 5}, data)
		output, err := vsn.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("Forward (bs=%d): %v", bs, err)
		}
		if output.Shape()[0] != bs || output.Shape()[1] != 5 {
			t.Errorf("bs=%d: shape = %v, want [%d, 5]", bs, output.Shape(), bs)
		}
	}
}
