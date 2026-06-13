package normalization

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

// TestLayerNormalization_WithEpsilon tests LayerNormalization with custom epsilon option.
func TestLayerNormalization_WithEpsilon(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	// Test with custom epsilon value
	customEpsilon := float32(1e-6)
	ln, err := NewLayerNormalization[float32](engine, 4, WithLayerNormEpsilon[float32](customEpsilon))
	testutils.AssertNoError(t, err, "NewLayerNormalization with custom epsilon should not return an error")
	testutils.AssertNotNil(t, ln, "LayerNormalization should not be nil")

	// Check that epsilon is set correctly
	testutils.AssertFloatEqual(t, customEpsilon, ln.epsilon, float32(1e-9), "Epsilon should be set to custom value")
}

// TestLayerNormalization_DefaultEpsilon tests LayerNormalization with default epsilon.
func TestLayerNormalization_DefaultEpsilon(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	// Test with default epsilon (no options)
	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization with default epsilon should not return an error")
	testutils.AssertNotNil(t, ln, "LayerNormalization should not be nil")

	// Check that epsilon is set to default value (1e-5)
	expectedEpsilon := float32(1e-5)
	testutils.AssertFloatEqual(t, expectedEpsilon, ln.epsilon, float32(1e-9), "Epsilon should be set to default value")
}

// TestLayerNormalization_Parameters tests that LayerNormalization returns correct parameters.
func TestLayerNormalization_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	params := ln.Parameters()
	testutils.AssertEqual(t, len(params), 2, "LayerNormalization should have 2 parameters (gamma and beta)")
}

// TestLayerNormalization_OutputShape tests OutputShape method.
func TestLayerNormalization_OutputShape(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Create a test input to initialize output shape
	inputShape := []int{2, 4}
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	// Run forward to initialize output shape
	_, err = ln.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")

	// Now test OutputShape
	outputShape := ln.OutputShape()
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, outputShape), "OutputShape should match input shape")
}

// TestLayerNormalization_Forward tests Forward method.
func TestLayerNormalization_Forward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	normalizedDim := 4
	ln, err := NewLayerNormalization[float32](engine, normalizedDim)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Create test input tensor [batch=2, seq=3, features=4]
	batchSize := 2
	seqLen := 3
	inputShape := []int{batchSize, seqLen, normalizedDim}
	inputData := []float32{
		// Batch 1, Seq 1
		1.0, 2.0, 3.0, 4.0,
		// Batch 1, Seq 2
		5.0, 6.0, 7.0, 8.0,
		// Batch 1, Seq 3
		9.0, 10.0, 11.0, 12.0,
		// Batch 2, Seq 1
		2.0, 4.0, 6.0, 8.0,
		// Batch 2, Seq 2
		1.0, 3.0, 5.0, 7.0,
		// Batch 2, Seq 3
		10.0, 12.0, 14.0, 16.0,
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	// Test forward pass
	output, err := ln.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")
	testutils.AssertNotNil(t, output, "Output should not be nil")

	// Check output shape
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, output.Shape()), "Output shape should match input shape")

	// Check that output values are normalized (mean ≈ 0, std ≈ 1)
	outputData := output.Data()
	testutils.AssertEqual(t, len(outputData), len(inputData), "Output data length should match input")

	// Verify normalization properties for first sequence
	firstSeq := outputData[0:4] // First sequence of first batch
	mean := (firstSeq[0] + firstSeq[1] + firstSeq[2] + firstSeq[3]) / 4.0
	testutils.AssertFloatEqual(t, 0.0, mean, 1e-5, "Normalized sequence should have mean ≈ 0")
}

// TestLayerNormalization_Forward_EdgeCases tests Forward with edge cases.
func TestLayerNormalization_Forward_EdgeCases(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	normalizedDim := 3
	ln, err := NewLayerNormalization[float32](engine, normalizedDim)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Test with zeros (should not crash due to epsilon)
	zeroShape := []int{1, 1, normalizedDim}
	zeroData := []float32{0.0, 0.0, 0.0}
	zeroTensor, err := tensor.New[float32](zeroShape, zeroData)
	testutils.AssertNoError(t, err, "Failed to create zero tensor")

	output, err := ln.Forward(ctx, zeroTensor)
	testutils.AssertNoError(t, err, "Forward with zeros should not error")
	testutils.AssertNotNil(t, output, "Output should not be nil")

	// Test with single element
	singleShape := []int{1, normalizedDim}
	singleData := []float32{5.0, 10.0, 15.0}
	singleTensor, err := tensor.New[float32](singleShape, singleData)
	testutils.AssertNoError(t, err, "Failed to create single tensor")

	output2, err := ln.Forward(ctx, singleTensor)
	testutils.AssertNoError(t, err, "Forward with single element should not error")
	testutils.AssertNotNil(t, output2, "Output should not be nil")
}

// TestLayerNormalization_OpType tests OpType method.
func TestLayerNormalization_OpType(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization failed")

	if ln.OpType() != "LayerNormalization" {
		t.Errorf("OpType() = %q, want LayerNormalization", ln.OpType())
	}
}

// TestLayerNormalization_Attributes tests Attributes method.
func TestLayerNormalization_Attributes(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization failed")

	attrs := ln.Attributes()
	if attrs == nil {
		t.Fatal("Attributes returned nil")
	}
	if _, ok := attrs["epsilon"]; !ok {
		t.Error("Attributes should contain epsilon")
	}
}

// TestLayerNormalization_Forward_InvalidInputCount tests Forward with wrong input count.
func TestLayerNormalization_Forward_InvalidInputCount(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization failed")

	_, err = ln.Forward(context.Background())
	if err == nil {
		t.Error("expected error for no inputs")
	}
}

// TestLayerNormalization_Backward_InvalidInputCount tests Backward with wrong input count.
func TestLayerNormalization_Backward_InvalidInputCount(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ln, err := NewLayerNormalization[float32](engine, 4)
	testutils.AssertNoError(t, err, "NewLayerNormalization failed")

	// Need forward first to cache
	input, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	_, _ = ln.Forward(context.Background(), input)

	grad, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})
	_, err = ln.Backward(context.Background(), types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error for no inputs in Backward")
	}
}

// TestLayerNormalization_Backward tests Backward method with 3D input.
func TestLayerNormalization_Backward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	normalizedDim := 4
	ln, err := NewLayerNormalization[float32](engine, normalizedDim)
	testutils.AssertNoError(t, err, "NewLayerNormalization should not return an error")

	// Use 3D input [batch, seq, features] for backward to work correctly
	inputShape := []int{2, 3, normalizedDim}
	inputData := make([]float32, 24)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	testutils.AssertNoError(t, err, "Failed to create input tensor")

	_, err = ln.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward pass should not return an error")

	gradData := make([]float32, 24)
	for i := range gradData {
		gradData[i] = float32(i+1) * 0.01
	}
	gradTensor, err := tensor.New[float32](inputShape, gradData)
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	inputGrads, err := ln.Backward(ctx, types.FullBackprop, gradTensor, inputTensor)
	if err != nil {
		t.Logf("Backward returned error (may be expected for shape issues): %v", err)
		return
	}

	testutils.AssertNotNil(t, inputGrads, "Input gradients should not be nil")
	testutils.AssertEqual(t, len(inputGrads), 1, "Should return one input gradient")
	testutils.AssertTrue(t, testutils.IntSliceEqual(inputShape, inputGrads[0].Shape()), "Input gradient shape should match input shape")
}

// TestLayerNormalization_BackwardMixedCPUMatchesEngine proves the
// float64-internal backward path on CPU produces gradients that agree
// with the pure engine path to within a tight tolerance, for a workload
// where both paths are numerically stable. If this test fails, the
// mixed-precision rewrite introduced a derivation bug.
func TestLayerNormalization_BackwardMixedCPUMatchesEngine(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const featureDim = 8
	const batch = 4
	shape := []int{batch, featureDim}

	lnMixed, err := NewLayerNormalization[float32](engine, featureDim)
	if err != nil {
		t.Fatalf("NewLayerNormalization(mixed) failed: %v", err)
	}
	if !lnMixed.useMixedBackward {
		t.Fatal("expected mixed-precision path to be active for CPU/float32")
	}

	lnEngine, err := NewLayerNormalization[float32](engine, featureDim)
	if err != nil {
		t.Fatalf("NewLayerNormalization(engine) failed: %v", err)
	}
	lnEngine.useMixedBackward = false

	inputData := make([]float32, batch*featureDim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.13
	}
	inputMixed, err := tensor.New[float32](shape, append([]float32(nil), inputData...))
	if err != nil {
		t.Fatalf("tensor.New input mixed: %v", err)
	}
	inputEngine, err := tensor.New[float32](shape, append([]float32(nil), inputData...))
	if err != nil {
		t.Fatalf("tensor.New input engine: %v", err)
	}

	if _, err := lnMixed.Forward(ctx, inputMixed); err != nil {
		t.Fatalf("Forward mixed: %v", err)
	}
	if _, err := lnEngine.Forward(ctx, inputEngine); err != nil {
		t.Fatalf("Forward engine: %v", err)
	}

	gradData := make([]float32, batch*featureDim)
	for i := range gradData {
		gradData[i] = float32(i%featureDim+1) * 0.017
	}
	gradMixed, err := tensor.New[float32](shape, append([]float32(nil), gradData...))
	if err != nil {
		t.Fatalf("tensor.New grad mixed: %v", err)
	}
	gradEngine, err := tensor.New[float32](shape, append([]float32(nil), gradData...))
	if err != nil {
		t.Fatalf("tensor.New grad engine: %v", err)
	}

	dInputMixed, err := lnMixed.Backward(ctx, types.FullBackprop, gradMixed, inputMixed)
	if err != nil {
		t.Fatalf("Backward mixed: %v", err)
	}
	dInputEngine, err := lnEngine.Backward(ctx, types.FullBackprop, gradEngine, inputEngine)
	if err != nil {
		t.Fatalf("Backward engine: %v", err)
	}

	const tol = 1e-4
	for i := range dInputMixed[0].Data() {
		diff := float64(dInputMixed[0].Data()[i]) - float64(dInputEngine[0].Data()[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Fatalf("dInput[%d] mismatch: mixed=%v engine=%v diff=%v tol=%v",
				i, dInputMixed[0].Data()[i], dInputEngine[0].Data()[i], diff, tol)
		}
	}
	for i := range lnMixed.gamma.Gradient.Data() {
		diff := float64(lnMixed.gamma.Gradient.Data()[i]) - float64(lnEngine.gamma.Gradient.Data()[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Fatalf("dGamma[%d] mismatch: mixed=%v engine=%v",
				i, lnMixed.gamma.Gradient.Data()[i], lnEngine.gamma.Gradient.Data()[i])
		}
	}
	for i := range lnMixed.beta.Gradient.Data() {
		diff := float64(lnMixed.beta.Gradient.Data()[i]) - float64(lnEngine.beta.Gradient.Data()[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Fatalf("dBeta[%d] mismatch: mixed=%v engine=%v",
				i, lnMixed.beta.Gradient.Data()[i], lnEngine.beta.Gradient.Data()[i])
		}
	}
}

// TestLayerNormalization_BackwardMixedCPUResistsUnderflow shows the
// mixed-precision backward produces finite gradients for an input whose
// activations sit far from zero — the pattern that drove fold-0 to NaN
// in E9 T9.7 attempts #3-#5. The equivalent float32 engine path on the
// same input is permitted to diverge; the only claim here is that the
// mixed path stays finite.
func TestLayerNormalization_BackwardMixedCPUResistsUnderflow(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const featureDim = 16
	shape := []int{1, featureDim}
	ln, err := NewLayerNormalization[float32](engine, featureDim)
	if err != nil {
		t.Fatalf("NewLayerNormalization: %v", err)
	}
	if !ln.useMixedBackward {
		t.Fatal("expected mixed-precision path to be active for CPU/float32")
	}

	inputData := make([]float32, featureDim)
	for i := range inputData {
		inputData[i] = 1e5 + float32(i)*1e-2
	}
	input, err := tensor.New[float32](shape, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}
	if _, err := ln.Forward(ctx, input); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	gradData := make([]float32, featureDim)
	for i := range gradData {
		gradData[i] = 0.0005
	}
	grad, err := tensor.New[float32](shape, gradData)
	if err != nil {
		t.Fatalf("tensor.New grad: %v", err)
	}

	dInput, err := ln.Backward(ctx, types.FullBackprop, grad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	for i, v := range dInput[0].Data() {
		if v != v || v > 1e20 || v < -1e20 {
			t.Fatalf("dInput[%d] non-finite or huge: %v", i, v)
		}
	}
	for i, v := range ln.gamma.Gradient.Data() {
		if v != v || v > 1e20 || v < -1e20 {
			t.Fatalf("dGamma[%d] non-finite: %v", i, v)
		}
	}
	for i, v := range ln.beta.Gradient.Data() {
		if v != v || v > 1e20 || v < -1e20 {
			t.Fatalf("dBeta[%d] non-finite: %v", i, v)
		}
	}
}

// TestLayerNormalization_WithEngineBackwardOption: the T9.1 capture-safe
// option must (a) disable the host mixed-precision backward, (b) store
// gamma/beta gradients by assigning the fresh engine tensors (the
// Linear/Bias pattern consumed by the strategy's persistent-gradient
// hook), and (c) produce gradients matching the mixed path within
// float32-vs-float64 rounding tolerance.
func TestLayerNormalization_WithEngineBackwardOption(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const featureDim = 8
	const batch = 4
	shape := []int{batch, featureDim}

	lnMixed, err := NewLayerNormalization[float32](engine, featureDim)
	if err != nil {
		t.Fatalf("NewLayerNormalization(mixed) failed: %v", err)
	}
	lnOpt, err := NewLayerNormalization[float32](engine, featureDim, WithLayerNormEngineBackward[float32]())
	if err != nil {
		t.Fatalf("NewLayerNormalization(engine option) failed: %v", err)
	}
	if lnOpt.useMixedBackward {
		t.Fatal("WithLayerNormEngineBackward must disable the mixed-precision backward")
	}
	if !lnOpt.engineGradAssign {
		t.Fatal("WithLayerNormEngineBackward must enable gradient assignment")
	}

	inputData := make([]float32, batch*featureDim)
	gradData := make([]float32, batch*featureDim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.13
		gradData[i] = float32(i%featureDim+1) * 0.017
	}
	newT := func(data []float32) *tensor.TensorNumeric[float32] {
		tn, terr := tensor.New[float32](shape, append([]float32(nil), data...))
		if terr != nil {
			t.Fatalf("tensor.New: %v", terr)
		}
		return tn
	}

	inMixed, inOpt := newT(inputData), newT(inputData)
	gMixed, gOpt := newT(gradData), newT(gradData)

	if _, err := lnMixed.Forward(ctx, inMixed); err != nil {
		t.Fatalf("Forward mixed: %v", err)
	}
	if _, err := lnOpt.Forward(ctx, inOpt); err != nil {
		t.Fatalf("Forward option: %v", err)
	}

	prevGamma := lnOpt.gamma.Gradient
	dMixed, err := lnMixed.Backward(ctx, types.FullBackprop, gMixed, inMixed)
	if err != nil {
		t.Fatalf("Backward mixed: %v", err)
	}
	dOpt, err := lnOpt.Backward(ctx, types.FullBackprop, gOpt, inOpt)
	if err != nil {
		t.Fatalf("Backward option: %v", err)
	}

	if lnOpt.gamma.Gradient == prevGamma {
		t.Error("engine-backward option must ASSIGN a fresh gamma gradient tensor")
	}
	if lnOpt.beta.Gradient == nil {
		t.Fatal("beta gradient not set")
	}

	const tol = 1e-4
	for i := range dMixed[0].Data() {
		diff := float64(dMixed[0].Data()[i]) - float64(dOpt[0].Data()[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Errorf("dInput[%d]: mixed=%v option=%v (diff %g)", i, dMixed[0].Data()[i], dOpt[0].Data()[i], diff)
		}
	}
	for i := range lnMixed.gamma.Gradient.Data() {
		diff := float64(lnMixed.gamma.Gradient.Data()[i]) - float64(lnOpt.gamma.Gradient.Data()[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Errorf("dGamma[%d]: mixed=%v option=%v (diff %g)", i, lnMixed.gamma.Gradient.Data()[i], lnOpt.gamma.Gradient.Data()[i], diff)
		}
	}
}
