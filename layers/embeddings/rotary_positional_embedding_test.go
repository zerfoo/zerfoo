package embeddings

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

func TestRotaryPositionalEmbedding_NewRotaryPositionalEmbedding(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	tests := []struct {
		name      string
		headDim   int
		seqLen    int
		expectErr bool
	}{
		{"Valid RoPE", 4, 10, false},
		{"Odd Head Dim", 3, 10, true},
		{"Zero Seq Len", 4, 0, false},
		{"Zero Head Dim", 0, 10, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, tt.headDim, tt.seqLen, WithRotaryBase(10000.0))

			if tt.expectErr {
				testutils.AssertError(t, err, "expected error")

				if rpe != nil {
					t.Errorf("expected nil rpe, got %v", rpe)
				}
			} else {
				testutils.AssertNoError(t, err, "unexpected error")
				testutils.AssertNotNil(t, rpe, "expected non-nil rpe")
				testutils.AssertEqual(t, rpe.headDim, tt.headDim, "headDim mismatch")
				testutils.AssertNotNil(t, rpe.cosAngles, "cosAngles should be precomputed")
				testutils.AssertNotNil(t, rpe.sinAngles, "sinAngles should be precomputed")

				// Verify shapes of precomputed angles
				expectedAngleShape := []int{tt.seqLen, tt.headDim / 2}
				dummyAngleTensor, _ := tensor.New[float64](expectedAngleShape, nil)
				testutils.AssertTrue(t, rpe.cosAngles.ShapeEquals(dummyAngleTensor), "cosAngles shape mismatch")
				testutils.AssertTrue(t, rpe.sinAngles.ShapeEquals(dummyAngleTensor), "sinAngles shape mismatch")

				// Verify some values (e.g., first and last)
				if tt.seqLen > 0 && tt.headDim > 0 {
					// For position 0, invFreqsData[0] = 1.0, so anglesData[0] = 0.0
					// cos(0) = 1, sin(0) = 0
					testutils.AssertFloatEqual(t, 1.0, rpe.cosAngles.Data()[0], 1e-6, "cosAngles[0] mismatch")
					testutils.AssertFloatEqual(t, 0.0, rpe.sinAngles.Data()[0], 1e-6, "sinAngles[0] mismatch")

					// For last position, last invFreqsData
					if tt.seqLen > 1 && tt.headDim/2 > 1 {
						lastPos := tt.seqLen - 1
						lastInvFreqIdx := tt.headDim/2 - 1
						expectedAngle := float64(lastPos) * (1.0 / math.Pow(10000.0, float64(2*lastInvFreqIdx)/float64(tt.headDim)))
						testutils.AssertFloatEqual(t, math.Cos(expectedAngle), rpe.cosAngles.Data()[len(rpe.cosAngles.Data())-1], 1e-6, "last cosAngles mismatch")
						testutils.AssertFloatEqual(t, math.Sin(expectedAngle), rpe.sinAngles.Data()[len(rpe.sinAngles.Data())-1], 1e-6, "last sinAngles mismatch")
					}
				} else if tt.headDim == 0 || tt.seqLen == 0 {
					testutils.AssertEqual(t, 0, len(rpe.cosAngles.Data()), "cosAngles data should be empty for zero headDim/seqLen")
					testutils.AssertEqual(t, 0, len(rpe.sinAngles.Data()), "sinAngles data should be empty for zero headDim/seqLen")
				}
			}
		})
	}
}

func TestRotaryPositionalEmbedding_OutputShape(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	rpe, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, 4, 10, WithRotaryBase(10000.0))

	tests := []struct {
		name        string
		inputShapes [][]int
		expected    []int
		expectErr   bool
	}{
		{"Valid 3D Input", [][]int{{1, 10, 4}}, []int{1, 10, 4}, false},
		{"Invalid Input Count", [][]int{{1, 10}, {1, 5}}, nil, true},
		{"Invalid Input Dim (1D)", [][]int{{10}}, nil, true},
		{"Invalid Input Dim (3D)", [][]int{{1, 10, 5}}, nil, true},
		{"Head Dim Mismatch", [][]int{{1, 10, 2}}, nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a dummy input tensor to set the output shape
			input, _ := tensor.New[float64](tt.inputShapes[0], nil)

			_, err := rpe.Forward(context.Background(), input)
			if err != nil && !tt.expectErr {
				t.Fatalf("unexpected error during forward pass: %v", err)
			}

			shape := rpe.OutputShape()
			if !tt.expectErr {
				testutils.AssertTrue(t, reflect.DeepEqual(shape, tt.expected), "output shape mismatch")
			}
		})
	}
}

func TestRotaryPositionalEmbedding_Parameters(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	rpe, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, 4, 10, WithRotaryBase(10000.0))

	params := rpe.Parameters()
	if params != nil {
		t.Errorf("RoPE should have no trainable parameters, got %v", params)
	}
}

func TestRotaryPositionalEmbedding_Forward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	rpe, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, 4, 2, WithRotaryBase(10000.0)) // headDim=4, seqLen=2

	inputData := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	inputTensor, _ := tensor.New[float64]([]int{1, 2, 4}, inputData)

	output, err := rpe.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward should not return an error")
	testutils.AssertNotNil(t, output, "Output tensor should not be nil")

	dummyOutput, _ := tensor.New[float64]([]int{1, 2, 4}, nil)
	testutils.AssertTrue(t, output.ShapeEquals(dummyOutput), "Output shape mismatch")

	// Dynamically calculate expected output
	expectedOutputData := []float64{1, 2, 3, 4, -3.1887853643145765, 5.919701335826659, 7.9894710651164615, 8.059599003338322}

	for i := range expectedOutputData {
		testutils.AssertFloatEqual(t, expectedOutputData[i], output.Data()[i], 1e-6, fmt.Sprintf("Output data mismatch at index %d", i))
	}
}

func TestRotaryPositionalEmbedding_Backward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	rpe, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, 4, 2, WithRotaryBase(10000.0)) // headDim=4, seqLen=2

	// Simulate cached inputs from Forward pass
	inputData := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	inputTensor, _ := tensor.New[float64]([]int{1, 2, 4}, inputData)

	_, err := rpe.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOutData := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	dOutTensor, _ := tensor.New[float64]([]int{1, 2, 4}, dOutData)

	expectedDInputData := []float64{0.1, 0.2, 0.3, 0.4, 0.8591808422995973, 0.6079698669173325, -0.04252387829625043, 0.7939601003328323}

	dInputs, err := rpe.Backward(ctx, types.FullBackprop, dOutTensor)
	testutils.AssertNoError(t, err, "Backward should not return an error")
	testutils.AssertNotNil(t, dInputs, "dInputs should not be nil")
	testutils.AssertEqual(t, len(dInputs), 1, "expected 1 input gradient")

	dummyDInput, _ := tensor.New[float64]([]int{1, 2, 4}, nil)
	testutils.AssertTrue(t, dInputs[0].ShapeEquals(dummyDInput), "dInput shape mismatch")

	for i := range expectedDInputData {
		testutils.AssertFloatEqual(t, expectedDInputData[i], dInputs[0].Data()[i], 1e-6, fmt.Sprintf("dInput data mismatch at index %d", i))
	}
}

func TestRotaryPositionalEmbedding_SimpleCase(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 2
	seqLen := 1
	rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen, WithRotaryBase(10000.0))
	testutils.AssertNoError(t, err, "NewRotaryPositionalEmbedding should not return an error")
	testutils.AssertNotNil(t, rpe, "expected non-nil rpe")

	// Input tensor: batch=1, seq_len=1, head_dim=2
	inputData := []float64{10, 20}
	inputTensor, _ := tensor.New[float64]([]int{1, 1, 2}, inputData)

	// --- Forward Pass ---
	output, err := rpe.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward should not return an error")
	testutils.AssertNotNil(t, output, "Output tensor should not be nil")

	dummyOutput, _ := tensor.New[float64]([]int{1, 1, 2}, nil)
	testutils.AssertTrue(t, output.ShapeEquals(dummyOutput), "Output shape mismatch")

	// Expected output for pos=0, headDim=2
	// invFreqs = [1.0]
	// angles = [0.0]
	// cos(0) = 1, sin(0) = 0
	// x0 = 10, x1 = 20
	// y0 = x0*cos(0) - x1*sin(0) = 10*1 - 20*0 = 10
	// y1 = x1*cos(0) + x0*sin(0) = 20*1 + 10*0 = 20
	expectedOutputData := []float64{10, 20}
	for i := range expectedOutputData {
		testutils.AssertFloatEqual(t, expectedOutputData[i], output.Data()[i], 1e-6, fmt.Sprintf("Forward output mismatch at index %d", i))
	}

	// --- Backward Pass ---
	dOutData := []float64{0.5, 0.6}
	dOutTensor, _ := tensor.New[float64]([]int{1, 1, 2}, dOutData)

	dInputs, err := rpe.Backward(ctx, types.FullBackprop, dOutTensor)
	testutils.AssertNoError(t, err, "Backward should not return an error")
	testutils.AssertNotNil(t, dInputs, "dInputs should not be nil")
	testutils.AssertEqual(t, len(dInputs), 1, "expected 1 input gradient")

	dummyDInput, _ := tensor.New[float64]([]int{1, 1, 2}, nil)
	testutils.AssertTrue(t, dInputs[0].ShapeEquals(dummyDInput), "dInput shape mismatch")

	// Expected dInput for pos=0, headDim=2
	// d_y0 = 0.5, d_y1 = 0.6
	// d_x0 = d_y0*cos(0) + d_y1*sin(0) = 0.5*1 + 0.6*0 = 0.5
	// d_x1 = d_y1*cos(0) + d_y0*sin(0) = 0.6*1 + 0.5*0 = 0.6
	expectedDInputData := []float64{
		0.5, 0.6,
	}
	for i := range expectedDInputData {
		testutils.AssertFloatEqual(t, expectedDInputData[i], dInputs[0].Data()[i], 1e-6, fmt.Sprintf("Backward dInput mismatch at index %d", i))
	}
}

// TestRotaryPositionalEmbedding_WithBase tests RotaryPositionalEmbedding with custom base option.
func TestRotaryPositionalEmbedding_WithBase(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	// Test with custom base value
	customBase := 5000.0
	rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, 4, 2, WithRotaryBase(customBase))
	testutils.AssertNoError(t, err, "NewRotaryPositionalEmbedding with custom base should not return an error")
	testutils.AssertNotNil(t, rpe, "RotaryPositionalEmbedding should not be nil")

	// Test forward pass works with custom base
	inputData := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	inputTensor, _ := tensor.New[float64]([]int{1, 2, 4}, inputData)

	output, err := rpe.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward should not return an error")
	testutils.AssertNotNil(t, output, "Output tensor should not be nil")

	// Output should have same shape as input
	testutils.AssertTrue(t, output.ShapeEquals(inputTensor), "Output shape should match input shape")
}

// TestRotaryPositionalEmbedding_DefaultBase tests RotaryPositionalEmbedding with default base.
func TestRotaryPositionalEmbedding_DefaultBase(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	// Test with default base (no options)
	rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, 4, 2)
	testutils.AssertNoError(t, err, "NewRotaryPositionalEmbedding with default base should not return an error")
	testutils.AssertNotNil(t, rpe, "RotaryPositionalEmbedding should not be nil")

	// Test forward pass works with default base
	inputData := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	inputTensor, _ := tensor.New[float64]([]int{1, 2, 4}, inputData)

	output, err := rpe.Forward(ctx, inputTensor)
	testutils.AssertNoError(t, err, "Forward should not return an error")
	testutils.AssertNotNil(t, output, "Output tensor should not be nil")
}

func TestRotaryPositionalEmbedding_YaRN_DefaultUnchanged(t *testing.T) {
	// Without YaRN, behavior should be identical to default RoPE.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	headDim := 8
	seqLen := 16
	base := 10000.0

	rpeDefault, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, seqLen, WithRotaryBase(base))
	if err != nil {
		t.Fatalf("default RoPE failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, seqLen, headDim}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.1
	}

	outDefault, err := rpeDefault.Forward(ctx, input)
	if err != nil {
		t.Fatalf("default Forward failed: %v", err)
	}

	// Sanity: output is non-nil and has same shape
	if !reflect.DeepEqual(outDefault.Shape(), input.Shape()) {
		t.Errorf("default output shape = %v, want %v", outDefault.Shape(), input.Shape())
	}
}

func TestRotaryPositionalEmbedding_YaRN_FrequenciesDiffer(t *testing.T) {
	// With YaRN scaling, cos/sin tables should differ from default RoPE.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	headDim := 8
	seqLen := 32
	base := 10000.0
	factor := 4.0
	origMaxLen := 64 // Small enough that some wavelengths exceed origMaxLen

	rpeDefault, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, seqLen, WithRotaryBase(base))
	if err != nil {
		t.Fatalf("default RoPE failed: %v", err)
	}

	rpeYaRN, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, seqLen,
		WithRotaryBase(base),
		WithYaRNScaling(factor, origMaxLen),
	)
	if err != nil {
		t.Fatalf("YaRN RoPE failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, seqLen, headDim}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%5+1) * 0.1
	}

	outDefault, err := rpeDefault.Forward(ctx, input)
	if err != nil {
		t.Fatalf("default Forward failed: %v", err)
	}

	outYaRN, err := rpeYaRN.Forward(ctx, input)
	if err != nil {
		t.Fatalf("YaRN Forward failed: %v", err)
	}

	// Outputs should differ
	defaultData := outDefault.Data()
	yarnData := outYaRN.Data()
	allSame := true
	for i := range defaultData {
		if defaultData[i] != yarnData[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("YaRN output should differ from default RoPE output")
	}
}

func TestRotaryPositionalEmbedding_YaRN_AttentionScaleFactor(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	factor := 4.0
	origMaxLen := 8192

	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, engine, 8, 16,
		WithRotaryBase(10000.0),
		WithYaRNScaling(factor, origMaxLen),
	)
	if err != nil {
		t.Fatalf("YaRN RoPE failed: %v", err)
	}

	// Expected: sqrt(1 + ln(factor) / ln(origMaxLen))
	expected := math.Sqrt(1 + math.Log(factor)/math.Log(float64(origMaxLen)))
	got := rpe.AttentionScaleFactor()

	if math.Abs(got-expected) > 1e-6 {
		t.Errorf("AttentionScaleFactor() = %f, want %f", got, expected)
	}
}

func TestRotaryPositionalEmbedding_NoYaRN_AttentionScaleFactor(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, engine, 8, 16,
		WithRotaryBase(10000.0),
	)
	if err != nil {
		t.Fatalf("default RoPE failed: %v", err)
	}

	// Without YaRN, attention scale factor should be 1.0
	if got := rpe.AttentionScaleFactor(); got != 1.0 {
		t.Errorf("AttentionScaleFactor() = %f, want 1.0", got)
	}
}

func TestRotaryPositionalEmbedding_YaRN_ForwardBackward(t *testing.T) {
	// YaRN RoPE should still produce valid backward gradients.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, engine, 8, 16,
		WithRotaryBase(10000.0),
		WithYaRNScaling(4.0, 8192),
	)
	if err != nil {
		t.Fatalf("YaRN RoPE failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{2, 8, 8}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%5+1) * 0.01
	}

	out, err := rpe.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := rpe.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Errorf("expected 1 gradient, got %d", len(grads))
	}
	if !reflect.DeepEqual(grads[0].Shape(), input.Shape()) {
		t.Errorf("gradient shape = %v, want %v", grads[0].Shape(), input.Shape())
	}
}

func TestRotaryPositionalEmbedding_PartialRotation_Default(t *testing.T) {
	// Default fraction (1.0) should produce same output as explicit 1.0.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	headDim := 8
	seqLen := 4
	base := 10000.0

	rpeDefault, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, seqLen, WithRotaryBase(base))
	if err != nil {
		t.Fatalf("default RoPE failed: %v", err)
	}

	rpeFull, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, seqLen,
		WithRotaryBase(base),
		WithRotaryDimFraction(1.0),
	)
	if err != nil {
		t.Fatalf("fraction=1.0 RoPE failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, seqLen, headDim}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.1
	}

	outDefault, err := rpeDefault.Forward(ctx, input)
	if err != nil {
		t.Fatalf("default Forward failed: %v", err)
	}

	outFull, err := rpeFull.Forward(ctx, input)
	if err != nil {
		t.Fatalf("fraction=1.0 Forward failed: %v", err)
	}

	for i, v := range outDefault.Data() {
		if v != outFull.Data()[i] {
			t.Errorf("default vs fraction=1.0 differ at index %d: %f != %f", i, v, outFull.Data()[i])
			break
		}
	}
}

func TestRotaryPositionalEmbedding_PartialRotation_Half(t *testing.T) {
	// fraction=0.5 with headDim=8: 4 dims rotated, 4 dims unchanged.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	headDim := 8
	seqLen := 4
	base := 10000.0

	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, seqLen,
		WithRotaryBase(base),
		WithRotaryDimFraction(0.5),
	)
	if err != nil {
		t.Fatalf("fraction=0.5 RoPE failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, seqLen, headDim}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.1
	}

	output, err := rpe.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if !reflect.DeepEqual(output.Shape(), input.Shape()) {
		t.Fatalf("output shape = %v, want %v", output.Shape(), input.Shape())
	}

	// The last 4 dims of each position should be unchanged (pass-through).
	rotDim := 4 // headDim * 0.5
	for pos := 0; pos < seqLen; pos++ {
		for d := rotDim; d < headDim; d++ {
			idx := pos*headDim + d
			if output.Data()[idx] != input.Data()[idx] {
				t.Errorf("pos=%d dim=%d: output=%f should equal input=%f (unrotated region)",
					pos, d, output.Data()[idx], input.Data()[idx])
			}
		}
	}

	// The first 4 dims should differ (at least for pos > 0).
	anyDiffer := false
	for pos := 1; pos < seqLen; pos++ {
		for d := 0; d < rotDim; d++ {
			idx := pos*headDim + d
			if output.Data()[idx] != input.Data()[idx] {
				anyDiffer = true
				break
			}
		}
		if anyDiffer {
			break
		}
	}
	if !anyDiffer {
		t.Error("rotated region should differ from input for pos > 0")
	}
}

func TestRotaryPositionalEmbedding_PositionOffset(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	const (
		headDim = 8
		maxSeq  = 32
		batch   = 2
	)

	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, maxSeq)
	if err != nil {
		t.Fatal(err)
	}

	// Create a full 4-position input and run RoPE without offset.
	fullData := make([]float32, batch*4*headDim)
	for i := range fullData {
		fullData[i] = float32(i%7) * 0.1
	}
	fullInput, err := tensor.New([]int{batch, 4, headDim}, fullData)
	if err != nil {
		t.Fatal(err)
	}

	rpe.SetPositionOffset(0)
	fullOut, err := rpe.Forward(ctx, fullInput)
	if err != nil {
		t.Fatalf("full forward: %v", err)
	}
	fullOutData := fullOut.Data()

	// Run single-position inputs with offsets and verify against full output.
	tests := []struct {
		name   string
		offset int
	}{
		{"position 0", 0},
		{"position 1", 1},
		{"position 2", 2},
		{"position 3", 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			posData := make([]float32, batch*1*headDim)
			for b := range batch {
				srcOff := b*4*headDim + tt.offset*headDim
				dstOff := b * headDim
				copy(posData[dstOff:dstOff+headDim], fullData[srcOff:srcOff+headDim])
			}
			posInput, err := tensor.New([]int{batch, 1, headDim}, posData)
			if err != nil {
				t.Fatal(err)
			}

			rpe.SetPositionOffset(tt.offset)
			posOut, err := rpe.Forward(ctx, posInput)
			if err != nil {
				t.Fatalf("offset forward: %v", err)
			}
			posOutData := posOut.Data()

			for b := range batch {
				srcOff := b*4*headDim + tt.offset*headDim
				dstOff := b * headDim
				for d := range headDim {
					got := posOutData[dstOff+d]
					want := fullOutData[srcOff+d]
					if diff := got - want; diff < -1e-5 || diff > 1e-5 {
						t.Errorf("batch %d dim %d: got %f, want %f", b, d, got, want)
					}
				}
			}
		})
	}

	// Verify offset resets to 0.
	t.Run("reset to 0", func(t *testing.T) {
		rpe.SetPositionOffset(0)
		singleData := make([]float32, batch*1*headDim)
		for b := range batch {
			copy(singleData[b*headDim:(b+1)*headDim], fullData[b*4*headDim:b*4*headDim+headDim])
		}
		singleInput, err := tensor.New([]int{batch, 1, headDim}, singleData)
		if err != nil {
			t.Fatal(err)
		}
		out, err := rpe.Forward(ctx, singleInput)
		if err != nil {
			t.Fatal(err)
		}
		for b := range batch {
			for d := range headDim {
				got := out.Data()[b*headDim+d]
				want := fullOutData[b*4*headDim+d]
				if diff := got - want; diff < -1e-5 || diff > 1e-5 {
					t.Errorf("reset batch %d dim %d: got %f, want %f", b, d, got, want)
				}
			}
		}
	})
}

func TestRoPEBackward(t *testing.T) {
	// Finite difference verification of the RoPE backward pass.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	const (
		batch   = 2
		seqLen  = 4
		headDim = 8
		eps     = 1e-4
		tol     = 1e-3
	)

	rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen, WithRotaryBase(10000.0))
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding failed: %v", err)
	}

	// Create input with non-trivial values.
	inputData := make([]float64, batch*seqLen*headDim)
	for i := range inputData {
		inputData[i] = float64(i%11-5) * 0.1
	}
	input, err := tensor.New[float64]([]int{batch, seqLen, headDim}, inputData)
	if err != nil {
		t.Fatalf("input tensor: %v", err)
	}

	// Forward pass.
	out, err := rpe.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Loss = sum(output). dLoss/dOutput = ones.
	dOut, err := tensor.New[float64](out.Shape(), nil)
	if err != nil {
		t.Fatalf("dOut tensor: %v", err)
	}
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	// Backward pass.
	grads, err := rpe.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}
	analyticGrad := grads[0].Data()

	// Finite difference: for each input element, compute
	// (f(x+eps) - f(x-eps)) / (2*eps) and compare with analytic gradient.
	n := len(inputData)
	for i := 0; i < n; i++ {
		origVal := inputData[i]

		// f(x + eps)
		inputData[i] = origVal + eps
		inPlus, _ := tensor.New[float64]([]int{batch, seqLen, headDim}, inputData)
		// Must create a fresh RoPE to avoid stale cached slices.
		rpePlus, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen, WithRotaryBase(10000.0))
		outPlus, err := rpePlus.Forward(ctx, inPlus)
		if err != nil {
			t.Fatalf("Forward(+eps) failed at i=%d: %v", i, err)
		}
		sumPlus := 0.0
		for _, v := range outPlus.Data() {
			sumPlus += v
		}

		// f(x - eps)
		inputData[i] = origVal - eps
		inMinus, _ := tensor.New[float64]([]int{batch, seqLen, headDim}, inputData)
		rpeMinus, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen, WithRotaryBase(10000.0))
		outMinus, err := rpeMinus.Forward(ctx, inMinus)
		if err != nil {
			t.Fatalf("Forward(-eps) failed at i=%d: %v", i, err)
		}
		sumMinus := 0.0
		for _, v := range outMinus.Data() {
			sumMinus += v
		}

		// Restore
		inputData[i] = origVal

		numericGrad := (sumPlus - sumMinus) / (2 * eps)
		diff := math.Abs(numericGrad - analyticGrad[i])
		if diff > tol {
			t.Errorf("gradient mismatch at index %d: numeric=%.6f analytic=%.6f diff=%.6f",
				i, numericGrad, analyticGrad[i], diff)
		}
	}
}

func TestRotaryPositionalEmbedding_PartialRotation_ForwardBackward(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, engine, 8, 16,
		WithRotaryBase(10000.0),
		WithRotaryDimFraction(0.75),
	)
	if err != nil {
		t.Fatalf("partial RoPE failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{2, 8, 8}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%5+1) * 0.01
	}

	out, err := rpe.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := rpe.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Errorf("expected 1 gradient, got %d", len(grads))
	}
	if !reflect.DeepEqual(grads[0].Shape(), input.Shape()) {
		t.Errorf("gradient shape = %v, want %v", grads[0].Shape(), input.Shape())
	}
}

func TestDocumentWiseRoPE_PositionReset(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	seqLen := 6

	tests := []struct {
		name         string
		boundaries   []int
		wantLocalPos []int
	}{
		{
			name:         "single boundary mid-sequence",
			boundaries:   []int{3},
			wantLocalPos: []int{0, 1, 2, 0, 1, 2},
		},
		{
			name:         "two boundaries",
			boundaries:   []int{2, 4},
			wantLocalPos: []int{0, 1, 0, 1, 0, 1},
		},
		{
			name:         "boundary at start",
			boundaries:   []int{0},
			wantLocalPos: []int{0, 1, 2, 3, 4, 5},
		},
		{
			name:         "no boundaries",
			boundaries:   nil,
			wantLocalPos: []int{0, 1, 2, 3, 4, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
			if err != nil {
				t.Fatalf("NewRotaryPositionalEmbedding: %v", err)
			}

			inputData := make([]float64, seqLen*headDim)
			for i := range inputData {
				inputData[i] = 1.0
			}
			input, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)

			rpe.SetDocumentBoundaries(tt.boundaries)
			outDoc, err := rpe.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward with boundaries: %v", err)
			}

			if !reflect.DeepEqual(outDoc.Shape(), input.Shape()) {
				t.Errorf("output shape = %v, want %v", outDoc.Shape(), input.Shape())
			}

			halfRotary := headDim / 2
			cosData := rpe.cosAngles.Data()
			sinData := rpe.sinAngles.Data()
			docData := outDoc.Data()

			for pos := 0; pos < seqLen; pos++ {
				localP := tt.wantLocalPos[pos]
				for j := 0; j < halfRotary; j++ {
					cosVal := cosData[localP*halfRotary+j]
					sinVal := sinData[localP*halfRotary+j]
					wantR0 := cosVal - sinVal
					wantR1 := cosVal + sinVal
					gotR0 := docData[pos*headDim+j]
					gotR1 := docData[pos*headDim+halfRotary+j]
					if math.Abs(float64(gotR0)-float64(wantR0)) > 1e-10 {
						t.Errorf("pos=%d dim=%d: rotated0 = %v, want %v", pos, j, gotR0, wantR0)
					}
					if math.Abs(float64(gotR1)-float64(wantR1)) > 1e-10 {
						t.Errorf("pos=%d dim=%d: rotated1 = %v, want %v", pos, j, gotR1, wantR1)
					}
				}
			}
		})
	}
}

func TestDocumentWiseRoPE_OutputMatchesStandard_NoBoundaries(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	seqLen := 8
	inputData := make([]float64, seqLen*headDim)
	for i := range inputData {
		inputData[i] = float64(i+1) * 0.1
	}

	rpeStd, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
	input1, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outStd, err := rpeStd.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("standard Forward: %v", err)
	}

	rpeDoc, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
	rpeDoc.SetDocumentBoundaries(nil)
	input2, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outDoc, err := rpeDoc.Forward(ctx, input2)
	if err != nil {
		t.Fatalf("doc-wise Forward: %v", err)
	}

	stdData := outStd.Data()
	docData := outDoc.Data()
	for i := range stdData {
		if math.Abs(float64(stdData[i])-float64(docData[i])) > 1e-10 {
			t.Errorf("index %d: standard=%v doc-wise=%v", i, stdData[i], docData[i])
		}
	}
}

func TestDocumentWiseRoPE_ClearBoundaries(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	seqLen := 4
	inputData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

	rpe, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)

	rpe.SetDocumentBoundaries([]int{2})
	input1, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outWithBoundary, err := rpe.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("Forward with boundary: %v", err)
	}

	rpe.SetDocumentBoundaries(nil)
	input2, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outNoBoundary, err := rpe.Forward(ctx, input2)
	if err != nil {
		t.Fatalf("Forward without boundary: %v", err)
	}

	rpeStd, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
	input3, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outStd, err := rpeStd.Forward(ctx, input3)
	if err != nil {
		t.Fatalf("standard Forward: %v", err)
	}

	for i := range outStd.Data() {
		if math.Abs(float64(outNoBoundary.Data()[i])-float64(outStd.Data()[i])) > 1e-10 {
			t.Errorf("index %d: cleared=%v standard=%v", i, outNoBoundary.Data()[i], outStd.Data()[i])
		}
	}

	allSame := true
	for i := range outStd.Data() {
		if outWithBoundary.Data()[i] != outStd.Data()[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("document-wise output should differ from standard when boundaries are set")
	}
}

func TestDocumentWiseRoPE_PositionResetAtBoundaries(t *testing.T) {
	// Verify that position IDs reset to 0 at each document boundary by
	// comparing the document-wise output against manually constructed
	// per-document RoPE outputs.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	maxSeq := 10

	tests := []struct {
		name       string
		seqLen     int
		boundaries []int
		// wantLocalPos is the expected local position for each sequence position.
		wantLocalPos []int
	}{
		{
			name:         "single boundary at position 3",
			seqLen:       6,
			boundaries:   []int{3},
			wantLocalPos: []int{0, 1, 2, 0, 1, 2},
		},
		{
			name:         "three documents",
			seqLen:       9,
			boundaries:   []int{3, 6},
			wantLocalPos: []int{0, 1, 2, 0, 1, 2, 0, 1, 2},
		},
		{
			name:         "boundary at every position",
			seqLen:       4,
			boundaries:   []int{0, 1, 2, 3},
			wantLocalPos: []int{0, 0, 0, 0},
		},
		{
			name:         "boundary only at start",
			seqLen:       5,
			boundaries:   []int{0},
			wantLocalPos: []int{0, 1, 2, 3, 4},
		},
		{
			name:         "boundary at last position",
			seqLen:       4,
			boundaries:   []int{3},
			wantLocalPos: []int{0, 1, 2, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, maxSeq)
			if err != nil {
				t.Fatalf("NewRotaryPositionalEmbedding: %v", err)
			}

			// Use constant input so rotation is the only source of variation.
			inputData := make([]float64, tt.seqLen*headDim)
			for i := range inputData {
				inputData[i] = 1.0
			}
			input, _ := tensor.New[float64]([]int{1, tt.seqLen, headDim}, inputData)

			rpe.SetDocumentBoundaries(tt.boundaries)
			outDoc, err := rpe.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Build expected output by looking up cos/sin at the local position.
			halfRotary := headDim / 2
			cosData := rpe.cosAngles.Data()
			sinData := rpe.sinAngles.Data()
			docData := outDoc.Data()

			for pos := 0; pos < tt.seqLen; pos++ {
				localP := tt.wantLocalPos[pos]
				for j := 0; j < halfRotary; j++ {
					cosVal := cosData[localP*halfRotary+j]
					sinVal := sinData[localP*halfRotary+j]
					// input is all 1s: rotated0 = 1*cos - 1*sin, rotated1 = 1*cos + 1*sin
					wantR0 := cosVal - sinVal
					wantR1 := cosVal + sinVal
					gotR0 := docData[pos*headDim+j]
					gotR1 := docData[pos*headDim+halfRotary+j]
					if math.Abs(gotR0-wantR0) > 1e-10 {
						t.Errorf("pos=%d dim=%d: rotated0=%v, want %v (localPos=%d)",
							pos, j, gotR0, wantR0, localP)
					}
					if math.Abs(gotR1-wantR1) > 1e-10 {
						t.Errorf("pos=%d dim=%d: rotated1=%v, want %v (localPos=%d)",
							pos, j, gotR1, wantR1, localP)
					}
				}
			}
		})
	}
}

func TestDocumentWiseRoPE_GlobalOffsetIgnoredWhenBoundariesSet(t *testing.T) {
	// When document boundaries are set, gatherDocumentWiseAngles computes
	// local positions from the boundary list. The posOffset (global offset)
	// is only used in the non-document-wise path. Verify that setting
	// posOffset does not change the output when boundaries are active.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	seqLen := 6
	maxSeq := 16

	tests := []struct {
		name       string
		offset     int
		boundaries []int
	}{
		{"offset=0 with boundary", 0, []int{3}},
		{"offset=5 with boundary", 5, []int{3}},
		{"offset=10 with boundary", 10, []int{3}},
	}

	// Compute reference output with offset=0.
	rpeRef, err := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, maxSeq)
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding: %v", err)
	}

	inputData := make([]float64, seqLen*headDim)
	for i := range inputData {
		inputData[i] = float64(i+1) * 0.1
	}
	inputRef, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)

	rpeRef.SetDocumentBoundaries([]int{3})
	rpeRef.SetPositionOffset(0)
	outRef, err := rpeRef.Forward(ctx, inputRef)
	if err != nil {
		t.Fatalf("reference Forward: %v", err)
	}
	refData := outRef.Data()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rpe, err := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, maxSeq)
			if err != nil {
				t.Fatalf("NewRotaryPositionalEmbedding: %v", err)
			}

			input, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
			rpe.SetDocumentBoundaries(tt.boundaries)
			rpe.SetPositionOffset(tt.offset)

			out, err := rpe.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			for i, v := range out.Data() {
				if math.Abs(v-refData[i]) > 1e-10 {
					t.Errorf("index %d: got %v, want %v (offset=%d should not affect document-wise)",
						i, v, refData[i], tt.offset)
				}
			}
		})
	}
}

func TestDocumentWiseRoPE_DiffersFromStandardAtBoundaries(t *testing.T) {
	// Standard and document-wise RoPE must produce different outputs when
	// a boundary resets positions mid-sequence.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	seqLen := 8

	tests := []struct {
		name       string
		boundaries []int
	}{
		{"single boundary at 4", []int{4}},
		{"two boundaries", []int{2, 5}},
		{"boundary at 1", []int{1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputData := make([]float64, seqLen*headDim)
			for i := range inputData {
				inputData[i] = float64(i+1) * 0.1
			}

			// Standard RoPE
			rpeStd, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
			inputStd, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
			outStd, err := rpeStd.Forward(ctx, inputStd)
			if err != nil {
				t.Fatalf("standard Forward: %v", err)
			}

			// Document-wise RoPE
			rpeDoc, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
			rpeDoc.SetDocumentBoundaries(tt.boundaries)
			inputDoc, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
			outDoc, err := rpeDoc.Forward(ctx, inputDoc)
			if err != nil {
				t.Fatalf("document-wise Forward: %v", err)
			}

			// Outputs must differ at positions after the first boundary.
			firstBoundary := tt.boundaries[0]
			anyDiffer := false
			for pos := firstBoundary; pos < seqLen; pos++ {
				for d := 0; d < headDim; d++ {
					idx := pos*headDim + d
					if outStd.Data()[idx] != outDoc.Data()[idx] {
						anyDiffer = true
						break
					}
				}
				if anyDiffer {
					break
				}
			}
			if !anyDiffer {
				t.Error("document-wise output should differ from standard at/after boundary positions")
			}
		})
	}
}

func TestDocumentWiseRoPE_SingleDocumentMatchesStandard(t *testing.T) {
	// A single document (no boundaries, or boundary only at position 0)
	// must produce the same output as standard RoPE.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	seqLen := 8

	tests := []struct {
		name       string
		boundaries []int
	}{
		{"nil boundaries", nil},
		{"empty slice", []int{}},
		{"boundary at 0 only", []int{0}},
	}

	inputData := make([]float64, seqLen*headDim)
	for i := range inputData {
		inputData[i] = float64(i+1) * 0.1
	}

	// Reference: standard RoPE
	rpeStd, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
	inputStd, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outStd, err := rpeStd.Forward(ctx, inputStd)
	if err != nil {
		t.Fatalf("standard Forward: %v", err)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rpe, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
			rpe.SetDocumentBoundaries(tt.boundaries)
			input, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)

			out, err := rpe.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			for i, v := range out.Data() {
				if math.Abs(v-outStd.Data()[i]) > 1e-10 {
					t.Errorf("index %d: got %v, want %v", i, v, outStd.Data()[i])
				}
			}
		})
	}
}

func TestDocumentWiseRoPE_EmptyBoundaryList(t *testing.T) {
	// An empty boundary list should fall through to the standard RoPE path
	// (no gatherDocumentWiseAngles call) and produce identical results.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	headDim := 4
	seqLen := 6

	inputData := make([]float64, seqLen*headDim)
	for i := range inputData {
		inputData[i] = float64(i+1) * 0.5
	}

	// Standard path (no SetDocumentBoundaries call at all)
	rpeStd, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
	inputStd, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outStd, err := rpeStd.Forward(ctx, inputStd)
	if err != nil {
		t.Fatalf("standard Forward: %v", err)
	}

	// Explicitly set empty boundary list
	rpeEmpty, _ := NewRotaryPositionalEmbedding[float64](ctx, engine, headDim, seqLen)
	rpeEmpty.SetDocumentBoundaries([]int{})
	inputEmpty, _ := tensor.New[float64]([]int{1, seqLen, headDim}, inputData)
	outEmpty, err := rpeEmpty.Forward(ctx, inputEmpty)
	if err != nil {
		t.Fatalf("empty boundaries Forward: %v", err)
	}

	for i, v := range outEmpty.Data() {
		if math.Abs(v-outStd.Data()[i]) > 1e-10 {
			t.Errorf("index %d: empty boundaries=%v, standard=%v", i, v, outStd.Data()[i])
		}
	}
}
