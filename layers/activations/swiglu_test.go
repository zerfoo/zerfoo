package activations

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

func TestNewSwiGLU_FunctionalOptions(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test with functional options (even if no specific options are defined yet)
	swiglu := NewSwiGLU(
		engine,
		ops,
		// No specific options to pass yet, but demonstrating the pattern
	)

	testutils.AssertNotNil(t, swiglu, "expected SwiGLU to not be nil")

	// Test forward pass
	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	input, err := tensor.New[float32]([]int{1, 4}, inputData)
	testutils.AssertNoError(t, err, "failed to create input tensor")

	output, err := swiglu.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward pass failed")
	testutils.AssertNotNil(t, output, "expected output to not be nil")

	// Expected output for SwiGLU(x1, x2) = silu(x1) * x2
	// x1 = [1.0, 2.0], x2 = [3.0, 4.0]
	// silu(x) = x * sigmoid(x)
	// silu(1.0) = 1.0 * sigmoid(1.0) = 1.0 * 0.7310586 = 0.7310586
	// silu(2.0) = 2.0 * sigmoid(2.0) = 2.0 * 0.8807970 = 1.7615941
	// output = [0.7310586 * 3.0, 1.7615941 * 4.0] = [2.1931758, 7.0463762]
	expectedOutputData := []float32{2.1931758, 7.0463762}
	testutils.AssertFloat32SliceApproxEqual(t, expectedOutputData, output.Data(), 1e-6, "forward output mismatch")

	// Test backward pass (simplified check)
	outputGradData := []float32{1.0, 1.0}
	outputGrad, err := tensor.New[float32]([]int{1, 2}, outputGradData)
	testutils.AssertNoError(t, err, "failed to create output gradient tensor")

	inputGrads, err := swiglu.Backward(context.Background(), types.FullBackprop, outputGrad, input)
	testutils.AssertNoError(t, err, "backward pass failed")
	testutils.AssertTrue(t, len(inputGrads) == 1, "expected 1 input gradient")
	testutils.AssertNotNil(t, inputGrads[0], "expected non-nil input gradient")

	// The exact backward gradient calculation is complex, so we'll just check for non-nil for now.
	// A more thorough test would involve numerical gradient checking.
}

// sigmoid computes the sigmoid function for float64.
func sigmoid64(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// silu computes silu(x) = x * sigmoid(x).
func silu64(x float64) float64 {
	return x * sigmoid64(x)
}

// siluPrime computes silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
func siluPrime64(x float64) float64 {
	s := sigmoid64(x)
	return s * (1.0 + x*(1.0-s))
}

func TestSiluBackward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name  string
		shape []int
		data  []float32
	}{
		{
			name:  "small_1x4",
			shape: []int{1, 4},
			data:  []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:  "negative_values",
			shape: []int{1, 4},
			data:  []float32{-1.0, -2.0, 0.5, 1.5},
		},
		{
			name:  "zeros",
			shape: []int{1, 4},
			data:  []float32{0.0, 0.0, 0.0, 0.0},
		},
		{
			name:  "batch_2x4",
			shape: []int{2, 4},
			data:  []float32{0.5, -0.3, 0.8, 1.2, -0.5, 0.3, -0.8, -1.2},
		},
		{
			name:  "3d_input",
			shape: []int{1, 2, 4},
			data:  []float32{0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4},
		},
		{
			name:  "large_values",
			shape: []int{1, 4},
			data:  []float32{5.0, -5.0, 3.0, -3.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			swiglu := NewSwiGLU(engine, ops)
			input, err := tensor.New[float32](tt.shape, tt.data)
			testutils.AssertNoError(t, err, "failed to create input")

			output, err := swiglu.Forward(ctx, input)
			testutils.AssertNoError(t, err, "forward failed")

			// Compute expected gradients analytically in float64
			halfDim := tt.shape[len(tt.shape)-1] / 2
			totalElements := 1
			for _, d := range tt.shape {
				totalElements *= d
			}
			halfTotal := totalElements / 2

			// Extract x1 and x2 from input data
			// The split is along the last dimension
			x1Data := make([]float64, halfTotal)
			x2Data := make([]float64, halfTotal)

			batchSize := totalElements / tt.shape[len(tt.shape)-1]
			for b := 0; b < batchSize; b++ {
				for j := 0; j < halfDim; j++ {
					x1Data[b*halfDim+j] = float64(tt.data[b*tt.shape[len(tt.shape)-1]+j])
					x2Data[b*halfDim+j] = float64(tt.data[b*tt.shape[len(tt.shape)-1]+halfDim+j])
				}
			}

			// dOut = ones (loss = sum(output))
			dOutData := make([]float32, halfTotal)
			for i := range dOutData {
				dOutData[i] = 1.0
			}

			outShape := make([]int, len(tt.shape))
			copy(outShape, tt.shape)
			outShape[len(outShape)-1] = halfDim

			dOut, err := tensor.New[float32](outShape, dOutData)
			testutils.AssertNoError(t, err, "failed to create dOut")

			grads, err := swiglu.Backward(ctx, types.FullBackprop, dOut, input)
			testutils.AssertNoError(t, err, "backward failed")
			testutils.AssertTrue(t, len(grads) == 1, "expected 1 gradient tensor")

			gradData := grads[0].Data()

			// Compute expected gradients
			// dL/dx1 = dOut * x2 * silu'(x1) = 1 * x2 * silu'(x1)
			// dL/dx2 = dOut * silu(x1)        = 1 * silu(x1)
			expectedGrad := make([]float32, totalElements)
			for b := 0; b < batchSize; b++ {
				for j := 0; j < halfDim; j++ {
					idx := b*halfDim + j
					expectedGrad[b*tt.shape[len(tt.shape)-1]+j] = float32(x2Data[idx] * siluPrime64(x1Data[idx]))
					expectedGrad[b*tt.shape[len(tt.shape)-1]+halfDim+j] = float32(silu64(x1Data[idx]))
				}
			}

			testutils.AssertFloat32SliceApproxEqual(t, expectedGrad, gradData, 1e-5, "backward gradient mismatch")

			// Verify output shape is correct
			testutils.AssertTrue(t, len(output.Data()) == halfTotal, "output size mismatch")
		})
	}
}

func TestSiluBackwardFiniteDiff(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	const eps = 1e-3

	tests := []struct {
		name  string
		shape []int
		data  []float32
	}{
		{
			name:  "basic",
			shape: []int{1, 4},
			data:  []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:  "negative",
			shape: []int{1, 4},
			data:  []float32{-1.0, 0.5, 1.5, -0.5},
		},
		{
			name:  "batch",
			shape: []int{2, 4},
			data:  []float32{0.3, -0.7, 0.9, 1.1, -0.4, 0.6, -0.8, 1.3},
		},
		{
			name:  "near_zero",
			shape: []int{1, 4},
			data:  []float32{0.01, -0.01, 0.02, -0.02},
		},
		{
			name:  "3d",
			shape: []int{1, 2, 4},
			data:  []float32{0.5, -0.3, 0.8, 1.2, -0.5, 0.3, -0.8, -1.2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			totalElements := len(tt.data)
			halfDim := tt.shape[len(tt.shape)-1] / 2
			halfTotal := totalElements / 2

			// Compute analytical gradients
			swiglu := NewSwiGLU(engine, ops)
			input, err := tensor.New[float32](tt.shape, tt.data)
			testutils.AssertNoError(t, err, "failed to create input")

			_, err = swiglu.Forward(ctx, input)
			testutils.AssertNoError(t, err, "forward failed")

			outShape := make([]int, len(tt.shape))
			copy(outShape, tt.shape)
			outShape[len(outShape)-1] = halfDim

			dOutData := make([]float32, halfTotal)
			for i := range dOutData {
				dOutData[i] = 1.0
			}
			dOut, err := tensor.New[float32](outShape, dOutData)
			testutils.AssertNoError(t, err, "failed to create dOut")

			grads, err := swiglu.Backward(ctx, types.FullBackprop, dOut, input)
			testutils.AssertNoError(t, err, "backward failed")
			analyticGrad := grads[0].Data()

			// Compute numerical gradients via finite differences
			// loss = sum(SwiGLU(input))
			computeLoss := func(data []float32) float64 {
				s := NewSwiGLU(engine, ops)
				in, err := tensor.New[float32](tt.shape, data)
				if err != nil {
					t.Fatalf("tensor.New: %v", err)
				}
				out, err := s.Forward(ctx, in)
				if err != nil {
					t.Fatalf("Forward: %v", err)
				}
				var sum float64
				for _, v := range out.Data() {
					sum += float64(v)
				}
				return sum
			}

			maxDiff := float64(0)
			for i := 0; i < totalElements; i++ {
				// f(x + eps)
				plusData := make([]float32, totalElements)
				copy(plusData, tt.data)
				plusData[i] += eps

				// f(x - eps)
				minusData := make([]float32, totalElements)
				copy(minusData, tt.data)
				minusData[i] -= eps

				numericalGrad := (computeLoss(plusData) - computeLoss(minusData)) / (2 * eps)
				diff := math.Abs(float64(analyticGrad[i]) - numericalGrad)
				if diff > maxDiff {
					maxDiff = diff
				}

				if diff > 1e-3 {
					t.Errorf("element %d: analytic=%.6f, numerical=%.6f, diff=%.6f",
						i, analyticGrad[i], numericalGrad, diff)
				}
			}

			t.Logf("max finite-diff error: %.6e", maxDiff)
		})
	}
}
