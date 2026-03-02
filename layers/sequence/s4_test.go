package sequence

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func makeEngine() compute.Engine[float32] {
	return compute.NewCPUEngine(numeric.Float32Ops{})
}

func TestNewS4_ValidParams(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}
	if s4.OpType() != "S4" {
		t.Errorf("OpType = %q, want %q", s4.OpType(), "S4")
	}
}

func TestNewS4_InvalidParams(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name     string
		layerName string
		inputDim int
		stateDim int
	}{
		{"empty name", "", 4, 8},
		{"zero input dim", "s4", 0, 8},
		{"negative input dim", "s4", -1, 8},
		{"zero state dim", "s4", 4, 0},
		{"negative state dim", "s4", 4, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewS4[float32](tt.layerName, engine, ops, tt.inputDim, tt.stateDim)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestS4_Attributes(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	attrs := s4.Attributes()
	if attrs["input_dim"] != 4 {
		t.Errorf("input_dim = %v, want 4", attrs["input_dim"])
	}
	if attrs["state_dim"] != 8 {
		t.Errorf("state_dim = %v, want 8", attrs["state_dim"])
	}
}

func TestS4_OutputShape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	shape := s4.OutputShape()
	if len(shape) != 3 {
		t.Fatalf("OutputShape len = %d, want 3", len(shape))
	}
	// [-1, -1, 4] where -1 = dynamic batch and seq dims
	if shape[2] != 4 {
		t.Errorf("OutputShape[2] = %d, want 4 (input_dim)", shape[2])
	}
}

func TestS4_Parameters(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	params := s4.Parameters()
	if len(params) != 4 {
		t.Fatalf("Parameters len = %d, want 4 (aLog, b, c, d)", len(params))
	}

	// Check parameter names and shapes.
	wantNames := []string{"test_s4_a_log", "test_s4_b", "test_s4_c", "test_s4_d"}
	wantShapes := [][]int{{4, 8}, {4, 8}, {4, 8}, {4}}
	for i, p := range params {
		if p.Name != wantNames[i] {
			t.Errorf("param[%d].Name = %q, want %q", i, p.Name, wantNames[i])
		}
		shape := p.Value.Shape()
		if len(shape) != len(wantShapes[i]) {
			t.Errorf("param[%d].Shape = %v, want %v", i, shape, wantShapes[i])
			continue
		}
		for j := range shape {
			if shape[j] != wantShapes[i][j] {
				t.Errorf("param[%d].Shape[%d] = %d, want %d", i, j, shape[j], wantShapes[i][j])
			}
		}
	}
}

func TestS4_Forward_Shape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Input: [batch=2, seq=5, dim=4]
	input, err := tensor.New[float32]([]int{2, 5, 4}, make([]float32, 2*5*4))
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 5 || shape[2] != 4 {
		t.Errorf("output shape = %v, want [2, 5, 4]", shape)
	}
}

func TestS4_Forward_NonZeroOutput(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Non-zero input should produce non-zero output (due to skip connection D).
	inputData := make([]float32, 1*3*2)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	input, err := tensor.New[float32]([]int{1, 3, 2}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	hasNonZero := false
	for _, v := range output.Data() {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("expected non-zero output for non-zero input")
	}
}

func TestS4_Forward_FiniteOutput(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	inputData := make([]float32, 2*10*4)
	for i := range inputData {
		inputData[i] = float32(i%7) * 0.1
	}
	input, err := tensor.New[float32]([]int{2, 10, 4}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] = %f (not finite)", i, v)
		}
	}
}

func TestS4_Forward_WrongInputCount(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	ctx := context.Background()
	_, err = s4.Forward(ctx)
	if err == nil {
		t.Error("expected error for no inputs")
	}
}

func TestS4_Forward_WrongInputRank(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// 2D input instead of 3D
	input, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))
	ctx := context.Background()
	_, err = s4.Forward(ctx, input)
	if err == nil {
		t.Error("expected error for 2D input")
	}
}

func TestS4_Forward_EdgeCase_SingleStep(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// seq_len=1, batch=1
	input, _ := tensor.New[float32]([]int{1, 1, 2}, []float32{1, 2})
	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if output.Shape()[0] != 1 || output.Shape()[1] != 1 || output.Shape()[2] != 2 {
		t.Errorf("shape = %v, want [1, 1, 2]", output.Shape())
	}
}

func TestS4_Forward_EdgeCase_Dim1(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 1, 1)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 1}, []float32{1, 2, 3})
	ctx := context.Background()
	output, err := s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if output.Shape()[2] != 1 {
		t.Errorf("output dim = %d, want 1", output.Shape()[2])
	}
}

func TestS4_Backward_Shape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Run forward first.
	inputData := make([]float32, 2*5*4)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	input, _ := tensor.New[float32]([]int{2, 5, 4}, inputData)
	ctx := context.Background()
	_, err = s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Backward with output gradient.
	gradData := make([]float32, 2*5*4)
	for i := range gradData {
		gradData[i] = 1.0
	}
	grad, _ := tensor.New[float32]([]int{2, 5, 4}, gradData)
	inputGrads, err := s4.Backward(ctx, types.OneStepApproximation, grad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(inputGrads) != 1 {
		t.Fatalf("Backward returned %d gradients, want 1", len(inputGrads))
	}
	gradShape := inputGrads[0].Shape()
	if len(gradShape) != 3 || gradShape[0] != 2 || gradShape[1] != 5 || gradShape[2] != 4 {
		t.Errorf("input gradient shape = %v, want [2, 5, 4]", gradShape)
	}
}

func TestS4_Backward_FiniteGradients(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	s4, err := NewS4[float32]("test_s4", engine, ops, 2, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	inputData := make([]float32, 1*3*2)
	for i := range inputData {
		inputData[i] = float32(i) * 0.1
	}
	input, _ := tensor.New[float32]([]int{1, 3, 2}, inputData)
	ctx := context.Background()
	_, err = s4.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	gradData := make([]float32, 1*3*2)
	for i := range gradData {
		gradData[i] = 1.0
	}
	grad, _ := tensor.New[float32]([]int{1, 3, 2}, gradData)
	inputGrads, err := s4.Backward(ctx, types.OneStepApproximation, grad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for i, v := range inputGrads[0].Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("input gradient[%d] = %f (not finite)", i, v)
		}
	}

	// Check parameter gradients are finite.
	for _, p := range s4.Parameters() {
		for i, v := range p.Gradient.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("param %s gradient[%d] = %f (not finite)", p.Name, i, v)
			}
		}
	}
}

// Compile-time interface check.
var _ = (*S4[float32])(nil)
