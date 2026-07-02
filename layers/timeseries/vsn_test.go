package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestNewGRN_Valid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	grn, err := NewGRN[float32]("test_grn", engine, ops, 8, 16, 8)
	if err != nil {
		t.Fatalf("NewGRN: %v", err)
	}
	params := grn.Parameters()
	if len(params) != 7 {
		t.Errorf("expected 7 parameters, got %d", len(params))
	}
}

func TestNewGRN_Invalid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name      string
		layerName string
		inputDim  int
		hiddenDim int
		outputDim int
	}{
		{"empty name", "", 8, 16, 8},
		{"zero inputDim", "grn", 0, 16, 8},
		{"negative hiddenDim", "grn", 8, -1, 8},
		{"zero outputDim", "grn", 8, 16, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewGRN[float32](tt.layerName, engine, ops, tt.inputDim, tt.hiddenDim, tt.outputDim)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestGRN_Forward(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch := 2
	inputDim := 8
	hiddenDim := 16
	outputDim := 8

	grn, err := NewGRN[float32]("grn", engine, ops, inputDim, hiddenDim, outputDim)
	if err != nil {
		t.Fatalf("NewGRN: %v", err)
	}

	data := make([]float32, batch*inputDim)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, inputDim}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	out, err := grn.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != batch || shape[1] != outputDim {
		t.Errorf("output shape = %v, want [%d, %d]", shape, batch, outputDim)
	}
}

func TestNewVSN_Valid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	vsn, err := NewVSN[float32]("test_vsn", engine, ops, 4, 3, 8)
	if err != nil {
		t.Fatalf("NewVSN: %v", err)
	}
	if vsn.OpType() != "VSN" {
		t.Errorf("OpType = %q, want %q", vsn.OpType(), "VSN")
	}
	if vsn.Name() != "test_vsn" {
		t.Errorf("Name = %q, want %q", vsn.Name(), "test_vsn")
	}
}

func TestNewVSN_Invalid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name        string
		layerName   string
		numVars     int
		varInputDim int
		dModel      int
	}{
		{"empty name", "", 4, 3, 8},
		{"zero numVars", "vsn", 0, 3, 8},
		{"negative varInputDim", "vsn", 4, -1, 8},
		{"zero dModel", "vsn", 4, 3, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewVSN[float32](tt.layerName, engine, ops, tt.numVars, tt.varInputDim, tt.dModel)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestVSN_Forward(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	numVars := 4
	varInputDim := 3
	dModel := 8
	batch := 2

	vsn, err := NewVSN[float32]("vsn", engine, ops, numVars, varInputDim, dModel)
	if err != nil {
		t.Fatalf("NewVSN: %v", err)
	}

	// Create synthetic inputs: 4 variables, each [batch, varInputDim].
	inputs := make([]*tensor.TensorNumeric[float32], numVars)
	for i := 0; i < numVars; i++ {
		data := make([]float32, batch*varInputDim)
		for j := range data {
			data[j] = float32(i*10+j) * 0.01
		}
		inp, err := tensor.New[float32]([]int{batch, varInputDim}, data)
		if err != nil {
			t.Fatalf("tensor.New input[%d]: %v", i, err)
		}
		inputs[i] = inp
	}

	ctx := context.Background()
	output, weights, err := vsn.Forward(ctx, inputs)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Check output shape: [batch, dModel].
	outShape := output.Shape()
	if len(outShape) != 2 || outShape[0] != batch || outShape[1] != dModel {
		t.Errorf("output shape = %v, want [%d, %d]", outShape, batch, dModel)
	}

	// Check importance weights: should have numVars elements.
	if len(weights) != numVars {
		t.Fatalf("importance weights length = %d, want %d", len(weights), numVars)
	}

	// Check importance weights sum to 1.0.
	var weightSum float64
	for _, w := range weights {
		weightSum += float64(w)
		if w < 0 {
			t.Errorf("importance weight %f is negative", w)
		}
	}
	if math.Abs(weightSum-1.0) > 1e-4 {
		t.Errorf("importance weights sum = %f, want 1.0", weightSum)
	}
}

func TestVSN_ForwardErrors(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	vsn, err := NewVSN[float32]("vsn", engine, ops, 4, 3, 8)
	if err != nil {
		t.Fatalf("NewVSN: %v", err)
	}

	ctx := context.Background()

	t.Run("wrong number of inputs", func(t *testing.T) {
		inp, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
		_, _, err := vsn.Forward(ctx, []*tensor.TensorNumeric[float32]{inp, inp})
		if err == nil {
			t.Error("expected error for wrong number of inputs")
		}
	})

	t.Run("wrong input shape", func(t *testing.T) {
		inputs := make([]*tensor.TensorNumeric[float32], 4)
		for i := range inputs {
			// Wrong varInputDim (5 instead of 3).
			inputs[i], _ = tensor.New[float32]([]int{2, 5}, make([]float32, 10))
		}
		_, _, err := vsn.Forward(ctx, inputs)
		if err == nil {
			t.Error("expected error for wrong input shape")
		}
	})
}

func TestVSN_Backward(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	numVars := 4
	varInputDim := 3
	dModel := 8
	batch := 2

	vsn, err := NewVSN[float32]("vsn", engine, ops, numVars, varInputDim, dModel)
	if err != nil {
		t.Fatalf("NewVSN: %v", err)
	}

	inputs := make([]*tensor.TensorNumeric[float32], numVars)
	for i := 0; i < numVars; i++ {
		data := make([]float32, batch*varInputDim)
		for j := range data {
			data[j] = float32(i*10+j) * 0.01
		}
		inp, err := tensor.New[float32]([]int{batch, varInputDim}, data)
		if err != nil {
			t.Fatalf("tensor.New input[%d]: %v", i, err)
		}
		inputs[i] = inp
	}

	ctx := context.Background()

	// Forward pass.
	output, _, err := vsn.Forward(ctx, inputs)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	_ = output

	// Create output gradient (ones).
	gradData := make([]float32, batch*dModel)
	for i := range gradData {
		gradData[i] = 1.0
	}
	outputGrad, err := tensor.New[float32]([]int{batch, dModel}, gradData)
	if err != nil {
		t.Fatalf("tensor.New outputGrad: %v", err)
	}

	grads, err := vsn.Backward(ctx, 0, outputGrad, inputs...)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) != numVars {
		t.Fatalf("expected %d gradients, got %d", numVars, len(grads))
	}

	// Check each gradient shape matches input shape.
	for i, g := range grads {
		gShape := g.Shape()
		if gShape[0] != batch || gShape[1] != varInputDim {
			t.Errorf("grad[%d] shape = %v, want [%d, %d]", i, gShape, batch, varInputDim)
		}
	}

	// Verify at least one projection weight gradient was accumulated.
	hasGrad := false
	for _, p := range vsn.varProj {
		if p.Gradient != nil {
			hasGrad = true
			break
		}
	}
	if !hasGrad {
		t.Error("no projection weight gradients accumulated")
	}
}

func TestVSN_Attributes(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vsn, err := NewVSN[float32]("vsn", engine, ops, 4, 3, 8)
	if err != nil {
		t.Fatalf("NewVSN: %v", err)
	}
	attrs := vsn.Attributes()
	if attrs["num_vars"] != 4 {
		t.Errorf("num_vars = %v, want 4", attrs["num_vars"])
	}
	if attrs["var_input_dim"] != 3 {
		t.Errorf("var_input_dim = %v, want 3", attrs["var_input_dim"])
	}
	if attrs["d_model"] != 8 {
		t.Errorf("d_model = %v, want 8", attrs["d_model"])
	}
}

func TestVSN_Parameters(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vsn, err := NewVSN[float32]("vsn", engine, ops, 4, 3, 8)
	if err != nil {
		t.Fatalf("NewVSN: %v", err)
	}
	params := vsn.Parameters()
	// 4 variable projections + 7 GRN params (5 weights + 2 LayerNorm) = 11.
	if len(params) != 11 {
		t.Errorf("expected 11 parameters, got %d", len(params))
	}
}

func TestVSN_SetName(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vsn, err := NewVSN[float32]("old", engine, ops, 2, 3, 8)
	if err != nil {
		t.Fatalf("NewVSN: %v", err)
	}
	vsn.SetName("new")
	if vsn.Name() != "new" {
		t.Errorf("Name = %q, want %q", vsn.Name(), "new")
	}
	if vsn.varProj[0].Name != "new_var0_proj" {
		t.Errorf("varProj[0].Name = %q, want %q", vsn.varProj[0].Name, "new_var0_proj")
	}
}
