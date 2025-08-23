package embeddings

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
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

	dInputs, err := rpe.Backward(ctx, dOutTensor)
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

	dInputs, err := rpe.Backward(ctx, dOutTensor)
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
