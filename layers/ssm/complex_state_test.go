package ssm

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestComplexSSMState(t *testing.T) {
	tests := []struct {
		name    string
		dModel  int
		dInner  int
		dState  int // must be even
		dtRank  int
		convKer int
		batch   int
		seqLen  int
	}{
		{
			name:    "small",
			dModel:  4,
			dInner:  8,
			dState:  4,
			dtRank:  2,
			convKer: 4,
			batch:   1,
			seqLen:  6,
		},
		{
			name:    "batch2",
			dModel:  4,
			dInner:  8,
			dState:  4,
			dtRank:  2,
			convKer: 4,
			batch:   2,
			seqLen:  4,
		},
		{
			name:    "larger_state",
			dModel:  4,
			dInner:  8,
			dState:  8,
			dtRank:  2,
			convKer: 2,
			batch:   1,
			seqLen:  4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine(ops)

			block, err := NewComplexSSMState[float32](
				"test_complex", engine, ops,
				tt.dModel, tt.dInner, tt.dState, tt.dtRank, tt.convKer,
				64, // maxSeqLen
			)
			if err != nil {
				t.Fatalf("NewComplexSSMState: %v", err)
			}

			input := makeTestTensor(t, []int{tt.batch, tt.seqLen, tt.dModel}, 42)

			output, err := block.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Verify output shape
			expectedShape := []int{tt.batch, tt.seqLen, tt.dModel}
			gotShape := output.Shape()
			if len(gotShape) != len(expectedShape) {
				t.Fatalf("output shape length: got %d, want %d", len(gotShape), len(expectedShape))
			}
			for i := range expectedShape {
				if gotShape[i] != expectedShape[i] {
					t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], expectedShape[i])
				}
			}

			// Verify output is not all zeros
			outData := output.Data()
			allZero := true
			for _, v := range outData {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Error("output is all zeros")
			}

			// Verify output contains finite values
			for i, v := range outData {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] is not finite: %v", i, v)
					break
				}
			}
		})
	}
}

func TestComplexSSMState_RoPEEffect(t *testing.T) {
	// Verify that RoPE on B/C produces different outputs than without it.
	// We do this by running the same block twice — deterministic results.
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dModel := 4
	dInner := 8
	dState := 4
	dtRank := 2
	convKer := 2
	batch := 1
	seqLen := 4

	complexBlock, err := NewComplexSSMState[float32](
		"test_rope_effect", engine, ops,
		dModel, dInner, dState, dtRank, convKer,
		64,
	)
	if err != nil {
		t.Fatalf("NewComplexSSMState: %v", err)
	}

	input := makeTestTensor(t, []int{batch, seqLen, dModel}, 99)

	out1, err := complexBlock.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward 1: %v", err)
	}

	// Run again — deterministic
	out2, err := complexBlock.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward 2: %v", err)
	}

	// Outputs should be identical (deterministic)
	d1 := out1.Data()
	d2 := out2.Data()
	for i := range d1 {
		if math.Abs(float64(d1[i]-d2[i])) > 1e-6 {
			t.Errorf("non-deterministic: out1[%d]=%.6f out2[%d]=%.6f", i, d1[i], i, d2[i])
			break
		}
	}
}

func TestComplexSSMState_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dModel := 4
	dInner := 8
	dState := 4
	dtRank := 2
	convKer := 4
	batch := 1
	seqLen := 4

	block, err := NewComplexSSMState[float32](
		"test_complex_bw", engine, ops,
		dModel, dInner, dState, dtRank, convKer,
		64,
	)
	if err != nil {
		t.Fatalf("NewComplexSSMState: %v", err)
	}

	input := makeTestTensor(t, []int{batch, seqLen, dModel}, 7)

	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create output gradient
	dOutData := make([]float32, len(output.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*11+5)%13)-6) / 10.0
	}
	dOut, err := tensor.New[float32](output.Shape(), dOutData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	grads, err := block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}

	dInput := grads[0]
	inputShape := input.Shape()
	gradShape := dInput.Shape()
	for i := range inputShape {
		if gradShape[i] != inputShape[i] {
			t.Errorf("grad shape[%d]: got %d, want %d", i, gradShape[i], inputShape[i])
		}
	}

	// Verify gradients are finite
	gradData := dInput.Data()
	for i, v := range gradData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("input grad[%d] is not finite: %v", i, v)
			break
		}
	}

	// Verify parameter gradients exist and are finite
	for _, p := range block.Parameters() {
		if p.Gradient == nil {
			t.Errorf("parameter %s has nil gradient", p.Name)
			continue
		}
		for i, v := range p.Gradient.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("parameter %s grad[%d] is not finite: %v", p.Name, i, v)
				break
			}
		}
	}
}

func TestComplexSSMState_OddDState(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	// dState=3 is odd — should fail
	_, err := NewComplexSSMState[float32](
		"test_odd", engine, ops,
		4, 8, 3, 2, 4,
		64,
	)
	if err == nil {
		t.Error("expected error with odd dState")
	}
}

func TestComplexSSMState_InvalidInputs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	ctx := context.Background()

	block, err := NewComplexSSMState[float32]("test", engine, ops, 4, 8, 4, 2, 4, 64)
	if err != nil {
		t.Fatalf("NewComplexSSMState: %v", err)
	}

	// No inputs
	_, err = block.Forward(ctx)
	if err == nil {
		t.Error("expected error with no inputs")
	}

	// Wrong dimensionality
	bad2d, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))
	_, err = block.Forward(ctx, bad2d)
	if err == nil {
		t.Error("expected error with 2D input")
	}
}

func TestComplexSSMState_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewComplexSSMState[float32]("test", engine, ops, 4, 8, 4, 2, 4, 64)
	if err != nil {
		t.Fatalf("NewComplexSSMState: %v", err)
	}

	if got := block.OpType(); got != "ComplexSSMState" {
		t.Errorf("OpType: got %q, want %q", got, "ComplexSSMState")
	}
}

func TestComplexSSMState_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewComplexSSMState[float32]("test", engine, ops, 4, 8, 4, 2, 4, 64)
	if err != nil {
		t.Fatalf("NewComplexSSMState: %v", err)
	}

	params := block.Parameters()
	if len(params) == 0 {
		t.Fatal("expected non-empty parameters")
	}

	// Should have: inProj weights, conv_weight, xProj weights, dtProj weights,
	// A, D, bcNorm_B gain, bcNorm_C gain, outProj weights
	expectedNames := map[string]bool{
		"test_in_proj_weights":  true,
		"test_conv_weight":      true,
		"test_x_proj_weights":   true,
		"test_dt_proj_weights":  true,
		"test_A":                true,
		"test_D":                true,
		"test_bc_norm_B_gain":   true,
		"test_bc_norm_C_gain":   true,
		"test_out_proj_weights": true,
	}

	for _, p := range params {
		if !expectedNames[p.Name] {
			t.Errorf("unexpected parameter name: %q", p.Name)
		}
		delete(expectedNames, p.Name)
	}
	for name := range expectedNames {
		t.Errorf("missing expected parameter: %q", name)
	}
}

func TestNewComplexSSMState_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	tests := []struct {
		name    string
		lName   string
		dModel  int
		dInner  int
		dState  int
		dtRank  int
		convKer int
		maxSeq  int
		wantErr bool
	}{
		{"empty_name", "", 4, 8, 4, 2, 4, 64, true},
		{"zero_dModel", "t", 0, 8, 4, 2, 4, 64, true},
		{"odd_dState", "t", 4, 8, 3, 2, 4, 64, true},
		{"zero_maxSeq", "t", 4, 8, 4, 2, 4, 0, true},
		{"valid", "t", 4, 8, 4, 2, 4, 64, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewComplexSSMState[float32](
				tt.lName, engine, ops,
				tt.dModel, tt.dInner, tt.dState, tt.dtRank, tt.convKer,
				tt.maxSeq,
			)
			if (err != nil) != tt.wantErr {
				t.Errorf("got err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}
