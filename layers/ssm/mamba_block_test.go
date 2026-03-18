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

func makeTestTensor(t *testing.T, shape []int, seed int) *tensor.TensorNumeric[float32] {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(((i*7+seed)%19)-9) / 20.0
	}
	out, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("makeTestTensor: %v", err)
	}
	return out
}

func TestMambaBlock(t *testing.T) {
	tests := []struct {
		name    string
		dModel  int
		dInner  int
		dState  int
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
			name:    "tiny",
			dModel:  2,
			dInner:  4,
			dState:  2,
			dtRank:  1,
			convKer: 2,
			batch:   1,
			seqLen:  3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine(ops)

			block, err := NewMambaBlock[float32](
				"test_mamba", engine, ops,
				tt.dModel, tt.dInner, tt.dState, tt.dtRank, tt.convKer,
			)
			if err != nil {
				t.Fatalf("NewMambaBlock: %v", err)
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

func TestMambaBlock_Backward(t *testing.T) {
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

	block, err := NewMambaBlock[float32](
		"test_mamba_bw", engine, ops,
		dModel, dInner, dState, dtRank, convKer,
	)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
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
	// Verify gradient shape matches input shape
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

func TestMambaBlock_BackwardFiniteDiff(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dModel := 2
	dInner := 4
	dState := 2
	dtRank := 1
	convKer := 2
	batch := 1
	seqLen := 3

	block, err := NewMambaBlock[float32](
		"test_mamba_fd", engine, ops,
		dModel, dInner, dState, dtRank, convKer,
	)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	input := makeTestTensor(t, []int{batch, seqLen, dModel}, 13)

	// Forward + backward
	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	dOutData := make([]float32, len(output.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*3+1)%7)-3) / 10.0
	}
	dOut, err := tensor.New[float32](output.Shape(), dOutData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	grads, err := block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	dInput := grads[0]

	// Finite difference check on input gradient
	eps := float32(1e-3)
	tol := float32(5e-3)

	analyticalGrad := make([]float32, len(dInput.Data()))
	copy(analyticalGrad, dInput.Data())

	inputData := input.Data()
	numFailed := 0
	for i := range inputData {
		orig := inputData[i]

		inputData[i] = orig + eps
		oPlus, err := block.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward(+eps): %v", err)
		}
		lPlus := dotProduct(oPlus.Data(), dOutData)

		inputData[i] = orig - eps
		oMinus, err := block.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward(-eps): %v", err)
		}
		lMinus := dotProduct(oMinus.Data(), dOutData)

		inputData[i] = orig

		numerical := (lPlus - lMinus) / (2 * eps)
		a := analyticalGrad[i]
		diff := float32(math.Abs(float64(a - numerical)))
		denom := float32(math.Max(1.0, math.Max(math.Abs(float64(a)), math.Abs(float64(numerical)))))

		if diff/denom > tol {
			numFailed++
			if numFailed <= 5 {
				t.Errorf("input grad[%d]: analytical=%.6f numerical=%.6f relErr=%.4f",
					i, a, numerical, diff/denom)
			}
		}
	}
	if numFailed > 0 {
		t.Errorf("input gradient: %d/%d exceeded tol=%.4f", numFailed, len(inputData), tol)
	} else {
		t.Logf("input gradient: %d elements passed finite-diff check", len(inputData))
	}
}

func TestMambaBlock_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewMambaBlock[float32]("test", engine, ops, 4, 8, 4, 2, 4)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	if got := block.OpType(); got != "MambaBlock" {
		t.Errorf("OpType: got %q, want %q", got, "MambaBlock")
	}
}

func TestMambaBlock_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewMambaBlock[float32]("test", engine, ops, 4, 8, 4, 2, 4)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	params := block.Parameters()
	if len(params) == 0 {
		t.Fatal("expected non-empty parameters")
	}

	// Should have: inProj weights, conv_weight, xProj weights, dtProj weights, A, D, outProj weights
	expectedNames := map[string]bool{
		"test_in_proj_weights":  true,
		"test_conv_weight":      true,
		"test_x_proj_weights":   true,
		"test_dt_proj_weights":  true,
		"test_A":                true,
		"test_D":                true,
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

func TestMambaBlock_InvalidInputs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	ctx := context.Background()

	block, err := NewMambaBlock[float32]("test", engine, ops, 4, 8, 4, 2, 4)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
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

func TestNewMambaBlock_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	_, err := NewMambaBlock[float32]("", engine, ops, 4, 8, 4, 2, 4)
	if err == nil {
		t.Error("expected error with empty name")
	}

	_, err = NewMambaBlock[float32]("test", engine, ops, 0, 8, 4, 2, 4)
	if err == nil {
		t.Error("expected error with zero dModel")
	}

	_, err = NewMambaBlock[float32]("test", engine, ops, 4, -1, 4, 2, 4)
	if err == nil {
		t.Error("expected error with negative dInner")
	}
}

func dotProduct(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
