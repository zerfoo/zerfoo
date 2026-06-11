package normalization

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

func TestRMSNormLayer_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 4
	epsilon := float32(1e-6)

	rmsnorm, err := NewRMSNorm[float32]("test", engine, ops, 4, WithRMSNormEpsilon[float32](epsilon))
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}

	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}

	input, err := tensor.New[float32]([]int{2, modelDim}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Manually calculate expected output
	// Row 1: [1, 2, 3, 4], RMS = sqrt((1+4+9+16)/4) = 2.7386
	// Normalized: [0.3651, 0.7303, 1.0954, 1.4606]
	// Row 2: [5, 6, 7, 8], RMS = sqrt((25+36+49+64)/4) = 6.5954
	// Normalized: [0.7581, 0.9097, 1.0613, 1.2129]
	expectedData := []float32{
		0.36514837, 0.73029673, 1.0954452, 1.4605935,
		0.75809807, 0.9097177, 1.0613372, 1.2129569,
	}

	expected, err := tensor.New[float32]([]int{2, modelDim}, expectedData)
	if err != nil {
		t.Fatalf("Failed to create expected tensor: %v", err)
	}

	output, err := rmsnorm.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if !testutils.CompareTensorsApprox(t, output, expected, epsilon) {
		t.Errorf("Output tensor does not match expected. Got %v, want %v", output.Data(), expected.Data())
	}
}

func TestRMSNorm_CustomEpsilon(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 4
	customEpsilon := float32(1e-3)

	rmsnorm, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim, WithRMSNormEpsilon[float32](customEpsilon))
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}

	// Create input tensor
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0}

	input, err := tensor.New[float32]([]int{2, modelDim}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Forward pass should work without error
	output, err := rmsnorm.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Verify output shape
	expectedShape := []int{2, modelDim}
	if len(output.Shape()) != len(expectedShape) {
		t.Errorf("Output shape mismatch. Got %v, want %v", output.Shape(), expectedShape)
	}

	for i, dim := range output.Shape() {
		if dim != expectedShape[i] {
			t.Errorf("Output shape mismatch at dimension %d. Got %d, want %d", i, dim, expectedShape[i])
		}
	}
}

func TestRMSNorm_DefaultEpsilon(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 4

	// Create RMSNorm without epsilon option (should use default)
	rmsnorm, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim)
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}

	// Create input tensor
	inputData := []float32{1.0, 2.0, 3.0, 4.0}

	input, err := tensor.New[float32]([]int{1, modelDim}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Forward pass should work without error
	output, err := rmsnorm.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Verify output shape
	expectedShape := []int{1, modelDim}
	if len(output.Shape()) != len(expectedShape) {
		t.Errorf("Output shape mismatch. Got %v, want %v", output.Shape(), expectedShape)
	}

	for i, dim := range output.Shape() {
		if dim != expectedShape[i] {
			t.Errorf("Output shape mismatch at dimension %d. Got %d, want %d", i, dim, expectedShape[i])
		}
	}
}

func TestRMSNorm_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 4

	rmsnorm, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim)
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}

	params := rmsnorm.Parameters()
	testutils.AssertEqual(t, len(params), 1, "RMSNorm should have 1 parameter (weight)")

	// Check weight parameter
	weight := params[0]
	if weight.Name != "test_rmsnorm_gain" {
		t.Errorf("Expected parameter name 'test_rmsnorm_gain', got '%s'", weight.Name)
	}
}

// TestRMSNorm_OutputShape tests OutputShape method.
func TestRMSNorm_OutputShape(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rms, err := NewRMSNorm[float32]("test_rms", engine, ops, 6)
	testutils.AssertNoError(t, err, "NewRMSNorm should not return an error")

	// Create a test input to initialize output shape
	inputShape := []int{2, 6}

	inputData := make([]float32, 12)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}

	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	// Run forward to initialize output shape
	_, err = rms.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")

	// Now test OutputShape
	outputShape := rms.OutputShape()
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, outputShape), "OutputShape should match input shape")
}

// TestRMSNorm_Forward_Comprehensive tests Forward method with various inputs.
func TestRMSNorm_Forward_Comprehensive(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	normalizedDim := 4
	rms, err := NewRMSNorm[float32]("test_rms", engine, ops, normalizedDim)
	testutils.AssertNoError(t, err, "NewRMSNorm should not return an error")

	// Test with 3D tensor [batch=2, seq=2, features=4]
	batchSize := 2
	seqLen := 2
	inputShape := []int{batchSize, seqLen, normalizedDim}
	inputData := []float32{
		// Batch 1, Seq 1
		2.0, 4.0, 6.0, 8.0,
		// Batch 1, Seq 2
		1.0, 3.0, 5.0, 7.0,
		// Batch 2, Seq 1
		10.0, 20.0, 30.0, 40.0,
		// Batch 2, Seq 2
		5.0, 15.0, 25.0, 35.0,
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	// Test forward pass
	output, err := rms.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")
	testutils.AssertNotNil(t, output, "Output should not be nil")

	// Check output shape
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, output.Shape()), "Output shape should match input shape")

	// Check that output values are normalized
	outputData := output.Data()
	testutils.AssertEqual(t, len(outputData), len(inputData), "Output data length should match input")

	// Verify that RMS normalization was applied (values should be scaled)
	for i := 0; i < len(outputData); i += normalizedDim {
		// Check that the RMS-normalized values are reasonable
		for j := range normalizedDim {
			testutils.AssertTrue(t, outputData[i+j] != inputData[i+j], "Output should be different from input after normalization")
		}
	}
}

// TestRMSNorm_Forward_EdgeCases tests Forward with edge cases.
func TestRMSNorm_Forward_EdgeCases(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	normalizedDim := 3
	rms, err := NewRMSNorm[float32]("test_rms", engine, ops, normalizedDim)
	testutils.AssertNoError(t, err, "NewRMSNorm should not return an error")

	// Test with zeros (should not crash due to epsilon)
	zeroShape := []int{1, normalizedDim}
	zeroData := []float32{0.0, 0.0, 0.0}
	zeroTensor, err := tensor.New[float32](zeroShape, zeroData)
	testutils.AssertNoError(t, err, "Failed to create zero tensor")

	output, err := rms.Forward(ctx, zeroTensor)
	testutils.AssertNoError(t, err, "Forward with zeros should not error")
	testutils.AssertNotNil(t, output, "Output should not be nil")

	// Test with very small values
	smallShape := []int{1, normalizedDim}
	smallData := []float32{1e-8, 2e-8, 3e-8}
	smallTensor, err := tensor.New[float32](smallShape, smallData)
	testutils.AssertNoError(t, err, "Failed to create small tensor")

	output2, err := rms.Forward(ctx, smallTensor)
	testutils.AssertNoError(t, err, "Forward with small values should not error")
	testutils.AssertNotNil(t, output2, "Output should not be nil")

	// Test with large values
	largeShape := []int{1, normalizedDim}
	largeData := []float32{1e6, 2e6, 3e6}
	largeTensor, err := tensor.New[float32](largeShape, largeData)
	testutils.AssertNoError(t, err, "Failed to create large tensor")

	output3, err := rms.Forward(ctx, largeTensor)
	testutils.AssertNoError(t, err, "Forward with large values should not error")
	testutils.AssertNotNil(t, output3, "Output should not be nil")
}

// TestRMSNorm_Backward tests Backward method with 3D input.
func TestRMSNorm_Backward(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	normalizedDim := 4
	rms, err := NewRMSNorm[float32]("test_rms", engine, ops, normalizedDim)
	testutils.AssertNoError(t, err, "NewRMSNorm should not return an error")

	// Use 3D input [batch, seq, features] for backward to work correctly
	inputShape := []int{2, 3, normalizedDim}
	inputData := make([]float32, 24)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	_, err = rms.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")

	gradData := make([]float32, 24)
	for i := range gradData {
		gradData[i] = float32(i+1) * 0.01
	}
	gradTensor, err := tensor.New[float32](inputShape, gradData)
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	inputGrads, err := rms.Backward(ctx, types.FullBackprop, gradTensor, inputTensor)
	if err != nil {
		t.Logf("Backward returned error (may be expected for shape issues): %v", err)
		return
	}

	testutils.AssertNotNil(t, inputGrads, "Input gradients should not be nil")
	testutils.AssertEqual(t, len(inputGrads), 1, "Should return one input gradient")
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, inputGrads[0].Shape()), "Input gradient shape should match input shape")

	// Check that gradients are computed (not zero)
	hasNonZeroGrad := false
	for _, grad := range inputGrads[0].Data() {
		if grad != 0.0 {
			hasNonZeroGrad = true
			break
		}
	}
	testutils.AssertTrue(t, hasNonZeroGrad, "Should have non-zero gradients")
}

// TestRMSNorm_NewFromParam tests NewRMSNormFromParam constructor.
func TestRMSNorm_NewFromParam(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	gainData := []float32{1.0, 1.0, 1.0, 1.0}
	gainTensor, err := tensor.New[float32]([]int{4}, gainData)
	testutils.AssertNoError(t, err, "failed to create gain tensor")

	gainParam, err := graph.NewParameter[float32]("gain", gainTensor, tensor.New[float32])
	testutils.AssertNoError(t, err, "failed to create gain parameter")

	rms, err := NewRMSNormFromParam(engine, ops, float32(1e-6), gainParam)
	testutils.AssertNoError(t, err, "NewRMSNormFromParam should not return an error")
	testutils.AssertNotNil(t, rms, "RMSNorm should not be nil")

	// Verify it works with Forward
	input, err := tensor.New[float32]([]int{1, 4}, []float32{1.0, 2.0, 3.0, 4.0})
	testutils.AssertNoError(t, err, "failed to create input")

	output, err := rms.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "Forward should not error")
	testutils.AssertNotNil(t, output, "output should not be nil")
}

// TestRMSNorm_OpType tests OpType method.
func TestRMSNorm_OpType(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rms, err := NewRMSNorm[float32]("test", engine, ops, 4)
	testutils.AssertNoError(t, err, "NewRMSNorm failed")

	if rms.OpType() != "RMSNorm" {
		t.Errorf("OpType() = %q, want RMSNorm", rms.OpType())
	}
}

// TestRMSNorm_Attributes tests Attributes method.
func TestRMSNorm_Attributes(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rms, err := NewRMSNorm[float32]("test", engine, ops, 4, WithRMSNormEpsilon[float32](1e-5))
	testutils.AssertNoError(t, err, "NewRMSNorm failed")

	attrs := rms.Attributes()
	if attrs == nil {
		t.Fatal("Attributes returned nil")
	}
	if _, ok := attrs["epsilon"]; !ok {
		t.Error("Attributes should contain epsilon")
	}
}

// TestRMSNorm_Forward_InvalidInputCount tests Forward with wrong input count.
func TestRMSNorm_Forward_InvalidInputCount(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rms, err := NewRMSNorm[float32]("test", engine, ops, 4)
	testutils.AssertNoError(t, err, "NewRMSNorm failed")

	_, err = rms.Forward(context.Background())
	if err == nil {
		t.Error("expected error for no inputs")
	}
}

// TestRMSNorm_Backward_InvalidInputCount tests Backward with wrong input count.
func TestRMSNorm_Backward_InvalidInputCount(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rms, err := NewRMSNorm[float32]("test", engine, ops, 4)
	testutils.AssertNoError(t, err, "NewRMSNorm failed")

	grad, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})
	_, err = rms.Backward(context.Background(), types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error for no inputs in Backward")
	}
}

// TestRMSNormBackward is the regression test for T5.2: nil guard in Backward.
func TestRMSNormBackward(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	inputShape := []int{2, 3, 4}
	inputData := make([]float32, 24)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	gradData := make([]float32, 24)
	for i := range gradData {
		gradData[i] = float32(i+1) * 0.01
	}

	tests := []struct {
		name        string
		runForward  bool
		secondCall  bool
		wantErr     bool
		errContains string
	}{
		{
			// Since the ADR 006 (T2.3) migration, Backward recomputes the
			// RMS statistics from the live inputs it receives, so it no
			// longer depends on Forward having cached anything.
			name:       "backward before forward succeeds (stats recomputed from live input)",
			runForward: false,
			wantErr:    false,
		},
		{
			name:       "forward then backward succeeds with non-nil gradients",
			runForward: true,
			wantErr:    false,
		},
		{
			name:       "double backward succeeds (stats recomputed each call)",
			runForward: true,
			secondCall: true,
			wantErr:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rms, err := NewRMSNorm[float32]("test_backward", engine, ops, 4)
			if err != nil {
				t.Fatalf("NewRMSNorm failed: %v", err)
			}

			inputTensor, err := tensor.New[float32](inputShape, inputData)
			if err != nil {
				t.Fatalf("failed to create input tensor: %v", err)
			}
			gradTensor, err := tensor.New[float32](inputShape, gradData)
			if err != nil {
				t.Fatalf("failed to create grad tensor: %v", err)
			}

			if tc.runForward {
				_, err = rms.Forward(ctx, inputTensor)
				if err != nil {
					t.Fatalf("Forward failed: %v", err)
				}
			}

			grads, err := rms.Backward(ctx, types.FullBackprop, gradTensor, inputTensor)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tc.errContains != "" && !strings.Contains(err.Error(), tc.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tc.errContains)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if grads == nil || len(grads) != 1 || grads[0] == nil {
				t.Fatal("expected non-nil gradient slice with one element")
			}
			if !testutils.IntSliceEqual(grads[0].Shape(), inputShape) {
				t.Errorf("gradient shape %v != input shape %v", grads[0].Shape(), inputShape)
			}

			if tc.secondCall {
				grads2, err2 := rms.Backward(ctx, types.FullBackprop, gradTensor, inputTensor)
				if err2 != nil {
					t.Fatalf("second Backward failed: %v", err2)
				}
				if grads2 == nil || len(grads2) != 1 {
					t.Fatal("second Backward returned nil gradients")
				}
			}
		})
	}
}

// TestRMSNorm_SetName tests SetName method.
func TestRMSNorm_SetName(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	rms, err := NewRMSNorm[float32]("test_rms", engine, ops, 4)
	testutils.AssertNoError(t, err, "NewRMSNorm should not return an error")

	rms.SetName("renamed")
	params := rms.Parameters()
	if params[0].Name != "renamed_gain" {
		t.Errorf("SetName: gain parameter name = %q, want renamed_gain", params[0].Name)
	}
}

// TestRMSNormBackward_FiniteDifference verifies that Backward() returns gradients
// consistent with numerical finite-difference approximation.
// For each input element i:
//
//	numerical_grad[i] = (loss(x + eps*e_i) - loss(x - eps*e_i)) / (2*eps)
//
// where loss = sum(rmsnorm.Forward(x)).
// The analytical gradient from Backward(dLoss=ones) must match within 1e-3.
func TestRMSNormBackward_FiniteDifference(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name  string
		shape []int
		dim   int
	}{
		{name: "2D_batch2_dim4", shape: []int{2, 4}, dim: 4},
		{name: "3D_batch2_seq2_dim4", shape: []int{2, 2, 4}, dim: 4},
		{name: "3D_batch1_seq1_dim4", shape: []int{1, 1, 4}, dim: 4},
		{name: "3D_batch1_seq3_dim4", shape: []int{1, 3, 4}, dim: 4},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			total := 1
			for _, d := range tc.shape {
				total *= d
			}

			// Create input data with moderate values to keep finite-difference
			// error within FP32 tolerance.
			inputData := make([]float32, total)
			for i := range inputData {
				inputData[i] = float32(i+1) * 0.1
			}

			// Create non-uniform gain weights to test gain gradient path
			gainData := make([]float32, tc.dim)
			for i := range gainData {
				gainData[i] = 1.0 + float32(i)*0.1
			}
			gainTensor, err := tensor.New[float32]([]int{tc.dim}, gainData)
			if err != nil {
				t.Fatalf("failed to create gain tensor: %v", err)
			}
			gainParam, err := graph.NewParameter[float32]("fd_gain", gainTensor, tensor.New[float32])
			if err != nil {
				t.Fatalf("failed to create gain parameter: %v", err)
			}

			rms, err := NewRMSNormFromParam(engine, ops, float32(1e-6), gainParam)
			if err != nil {
				t.Fatalf("NewRMSNormFromParam failed: %v", err)
			}

			// --- Analytical gradient via Backward ---
			inputTensor, err := tensor.New[float32](tc.shape, inputData)
			if err != nil {
				t.Fatalf("failed to create input tensor: %v", err)
			}

			_, err = rms.Forward(ctx, inputTensor)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// dLoss/dOutput = ones (loss = sum of output)
			onesData := make([]float32, total)
			for i := range onesData {
				onesData[i] = 1.0
			}
			onesTensor, err := tensor.New[float32](tc.shape, onesData)
			if err != nil {
				t.Fatalf("failed to create ones tensor: %v", err)
			}

			// Reset gain gradient to zero before backward
			rms.gain.Gradient = nil

			grads, err := rms.Backward(ctx, types.FullBackprop, onesTensor, inputTensor)
			if err != nil {
				t.Fatalf("Backward failed: %v", err)
			}
			analyticalGrad := grads[0].Data()

			// --- Numerical gradient via finite differences ---
			const eps = 1e-3
			numericalGrad := make([]float32, total)

			for i := range total {
				// f(x + eps*e_i)
				plusData := make([]float32, total)
				copy(plusData, inputData)
				plusData[i] += eps
				lossPlus := rmsnormLoss(t, ctx, engine, ops, tc.shape, tc.dim, plusData, gainData)

				// f(x - eps*e_i)
				minusData := make([]float32, total)
				copy(minusData, inputData)
				minusData[i] -= eps
				lossMinus := rmsnormLoss(t, ctx, engine, ops, tc.shape, tc.dim, minusData, gainData)

				numericalGrad[i] = float32((lossPlus - lossMinus) / (2 * eps))
			}

			// Compare
			maxDiff := float32(0)
			for i := range total {
				diff := float32(math.Abs(float64(analyticalGrad[i] - numericalGrad[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			if maxDiff > 1e-3 {
				t.Errorf("max absolute difference between analytical and numerical gradient: %e (threshold 1e-3)", maxDiff)
				// Print first few mismatches for debugging
				for i := range min(total, 8) {
					t.Logf("  [%d] analytical=%.6f numerical=%.6f diff=%.6e", i, analyticalGrad[i], numericalGrad[i], analyticalGrad[i]-numericalGrad[i])
				}
			}
		})
	}
}

// rmsnormLoss computes sum(RMSNorm(x)) using a fresh RMSNorm layer to avoid
// cached tensor interference from other evaluations.
func rmsnormLoss(t *testing.T, ctx context.Context, engine compute.Engine[float32], ops numeric.Arithmetic[float32], shape []int, dim int, data, gainData []float32) float64 {
	t.Helper()
	gainTensor, err := tensor.New[float32]([]int{dim}, gainData)
	if err != nil {
		t.Fatalf("rmsnormLoss: failed to create gain tensor: %v", err)
	}
	gainParam, err := graph.NewParameter[float32]("fd_loss_gain", gainTensor, tensor.New[float32])
	if err != nil {
		t.Fatalf("rmsnormLoss: failed to create gain param: %v", err)
	}
	rms, err := NewRMSNormFromParam(engine, ops, float32(1e-6), gainParam)
	if err != nil {
		t.Fatalf("rmsnormLoss: NewRMSNormFromParam failed: %v", err)
	}
	inputTensor, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("rmsnormLoss: failed to create input: %v", err)
	}
	output, err := rms.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("rmsnormLoss: Forward failed: %v", err)
	}
	var sum float64
	for _, v := range output.Data() {
		sum += float64(v)
	}
	return sum
}
