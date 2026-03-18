package ssm

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestBCNorm(t *testing.T) {
	tests := []struct {
		name   string
		dim    int
		shape  []int
		seed   int
	}{
		{"1d_small", 4, []int{1, 3, 4}, 42},
		{"1d_batch2", 4, []int{2, 3, 4}, 7},
		{"larger_dim", 8, []int{1, 4, 8}, 13},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine(ops)

			bn, err := NewBCNorm[float32]("test_bcnorm", engine, ops, tt.dim)
			if err != nil {
				t.Fatalf("NewBCNorm: %v", err)
			}

			input := makeTestTensor(t, tt.shape, tt.seed)
			output, err := bn.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Verify output shape matches input
			if got, want := output.Shape(), input.Shape(); len(got) != len(want) {
				t.Fatalf("shape length: got %d, want %d", len(got), len(want))
			}
			for i := range output.Shape() {
				if output.Shape()[i] != input.Shape()[i] {
					t.Errorf("shape[%d]: got %d, want %d", i, output.Shape()[i], input.Shape()[i])
				}
			}

			// Verify output is finite
			for i, v := range output.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] is not finite: %v", i, v)
					break
				}
			}

			// Verify normalization: each vector should have RMS ≈ 1.0
			// (since gain is initialized to 1.0)
			outData := output.Data()
			numVectors := len(outData) / tt.dim
			for v := 0; v < numVectors; v++ {
				off := v * tt.dim
				var sumSq float64
				for i := 0; i < tt.dim; i++ {
					val := float64(outData[off+i])
					sumSq += val * val
				}
				rms := math.Sqrt(sumSq / float64(tt.dim))
				// RMS should be approximately 1.0
				if math.Abs(rms-1.0) > 0.1 {
					t.Errorf("vector %d: RMS=%.4f, want ≈1.0", v, rms)
				}
			}
		})
	}
}

func TestBCNorm_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dim := 4
	bn, err := NewBCNorm[float32]("test_bcnorm_bw", engine, ops, dim)
	if err != nil {
		t.Fatalf("NewBCNorm: %v", err)
	}

	input := makeTestTensor(t, []int{1, 3, dim}, 42)
	output, err := bn.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create gradient
	dOutData := make([]float32, len(output.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*3+1)%7)-3) / 10.0
	}
	dOut, err := tensor.New[float32](output.Shape(), dOutData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	dInput, err := bn.Backward(ctx, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Verify gradient shape
	if got, want := dInput.Shape(), input.Shape(); len(got) != len(want) {
		t.Fatalf("grad shape length: got %d, want %d", len(got), len(want))
	}
	for i := range dInput.Shape() {
		if dInput.Shape()[i] != input.Shape()[i] {
			t.Errorf("grad shape[%d]: got %d, want %d", i, dInput.Shape()[i], input.Shape()[i])
		}
	}

	// Verify gradients are finite
	for i, v := range dInput.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("grad[%d] is not finite: %v", i, v)
			break
		}
	}

	// Verify gain gradient exists
	if bn.gain.Gradient == nil {
		t.Error("gain gradient is nil after backward")
	} else {
		for i, v := range bn.gain.Gradient.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("gain grad[%d] is not finite: %v", i, v)
				break
			}
		}
	}
}

func TestBCNorm_FiniteDiff(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dim := 4
	bn, err := NewBCNorm[float32]("test_bcnorm_fd", engine, ops, dim)
	if err != nil {
		t.Fatalf("NewBCNorm: %v", err)
	}

	input := makeTestTensor(t, []int{1, 2, dim}, 13)
	output, err := bn.Forward(ctx, input)
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

	dInput, err := bn.Backward(ctx, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	analyticalGrad := make([]float32, len(dInput.Data()))
	copy(analyticalGrad, dInput.Data())

	eps := float32(1e-3)
	tol := float32(5e-2)
	inputData := input.Data()
	numFailed := 0

	for i := range inputData {
		orig := inputData[i]

		inputData[i] = orig + eps
		// Reset gain gradient
		bn.gain.Gradient = nil
		oPlus, _ := bn.Forward(ctx, input)
		lPlus := dotProduct(oPlus.Data(), dOutData)

		inputData[i] = orig - eps
		bn.gain.Gradient = nil
		oMinus, _ := bn.Forward(ctx, input)
		lMinus := dotProduct(oMinus.Data(), dOutData)

		inputData[i] = orig

		numerical := (lPlus - lMinus) / (2 * eps)
		a := analyticalGrad[i]
		diff := float32(math.Abs(float64(a - numerical)))
		denom := float32(math.Max(1.0, math.Max(math.Abs(float64(a)), math.Abs(float64(numerical)))))

		if diff/denom > tol {
			numFailed++
			if numFailed <= 5 {
				t.Errorf("grad[%d]: analytical=%.6f numerical=%.6f relErr=%.4f",
					i, a, numerical, diff/denom)
			}
		}
	}
	if numFailed > 0 {
		t.Errorf("BCNorm gradient: %d/%d exceeded tol=%.4f", numFailed, len(inputData), tol)
	} else {
		t.Logf("BCNorm gradient: %d elements passed finite-diff check", len(inputData))
	}
}

func TestBCNorm_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	_, err := NewBCNorm[float32]("", engine, ops, 4)
	if err == nil {
		t.Error("expected error with empty name")
	}

	_, err = NewBCNorm[float32]("test", engine, ops, 0)
	if err == nil {
		t.Error("expected error with zero dim")
	}

	_, err = NewBCNorm[float32]("test", engine, ops, -1)
	if err == nil {
		t.Error("expected error with negative dim")
	}
}

func TestBCNorm_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	bn, err := NewBCNorm[float32]("test_bn", engine, ops, 4)
	if err != nil {
		t.Fatalf("NewBCNorm: %v", err)
	}

	params := bn.Parameters()
	if len(params) != 1 {
		t.Fatalf("expected 1 parameter, got %d", len(params))
	}
	if params[0].Name != "test_bn_gain" {
		t.Errorf("parameter name: got %q, want %q", params[0].Name, "test_bn_gain")
	}

	// Gain should be initialized to ones
	for i, v := range params[0].Value.Data() {
		if v != 1.0 {
			t.Errorf("gain[%d]: got %v, want 1.0", i, v)
		}
	}
}

func TestBCNorm_ZeroInput(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dim := 4
	bn, err := NewBCNorm[float32]("test_zero", engine, ops, dim)
	if err != nil {
		t.Fatalf("NewBCNorm: %v", err)
	}

	// All-zero input: should not produce NaN due to epsilon
	zeroData := make([]float32, 1*2*dim)
	zeroInput, _ := tensor.New[float32]([]int{1, 2, dim}, zeroData)

	output, err := bn.Forward(ctx, zeroInput)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("zero input: output[%d] is not finite: %v", i, v)
			break
		}
	}
}
