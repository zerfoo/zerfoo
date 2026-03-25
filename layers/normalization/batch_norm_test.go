// Package normalization_test tests the normalization layers.
package normalization_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TestBatchNormalization_ZeroMean: when mean equals each channel value the output
// after scale=1, bias=0 is zero.
func TestBatchNormalization_ZeroMean(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// X: [1,2,1,1] two channels, values 3 and 7.
	X, _ := tensor.New[float32]([]int{1, 2, 1, 1}, []float32{3, 7})
	scale, _ := tensor.New[float32]([]int{2}, []float32{1, 1})
	B, _ := tensor.New[float32]([]int{2}, []float32{0, 0})
	mean, _ := tensor.New[float32]([]int{2}, []float32{3, 7})
	variance, _ := tensor.New[float32]([]int{2}, []float32{1, 1})

	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))
	out, err := layer.Forward(ctx, X, scale, B, mean, variance)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	wantShape := []int{1, 2, 1, 1}
	if !intSliceEq(out.Shape(), wantShape) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), wantShape)
	}
	for i, v := range out.Data() {
		if math.Abs(float64(v)) > 1e-4 {
			t.Errorf("out[%d] = %v, want ~0 (x==mean)", i, v)
		}
	}
}

// TestBatchNormalization_ScaleAndBias: scale=2, bias=1, mean=0, var=1 -> y = 2*x + 1.
func TestBatchNormalization_ScaleAndBias(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// X: [1,2,1,1] values 2 and 4.
	X, _ := tensor.New[float32]([]int{1, 2, 1, 1}, []float32{2, 4})
	scale, _ := tensor.New[float32]([]int{2}, []float32{2, 1})
	B, _ := tensor.New[float32]([]int{2}, []float32{1, 0})
	mean, _ := tensor.New[float32]([]int{2}, []float32{0, 0})
	variance, _ := tensor.New[float32]([]int{2}, []float32{1, 4})
	const eps = float32(1e-7)

	layer := normalization.NewBatchNormalization[float32](engine, &ops, eps)
	out, err := layer.Forward(ctx, X, scale, B, mean, variance)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	got := out.Data()
	const tol = float32(1e-4)

	// ch0: 2 * (2 - 0) / sqrt(1 + eps) + 1 = 4/~1 + 1 ≈ 5.0
	if math.Abs(float64(got[0]-5.0)) > float64(tol) {
		t.Errorf("ch0 = %v, want ~5.0", got[0])
	}
	// ch1: 1 * (4 - 0) / sqrt(4 + eps) + 0 = 4/2 = 2.0
	if math.Abs(float64(got[1]-2.0)) > float64(tol) {
		t.Errorf("ch1 = %v, want ~2.0", got[1])
	}
}

// TestBatchNormalization_Spatial: X has spatial dims [1,2,2,2].
func TestBatchNormalization_Spatial(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// Two channels, each 2x2; channel 0: all 1, channel 1: all 2.
	data := []float32{1, 1, 1, 1, 2, 2, 2, 2}
	X, _ := tensor.New[float32]([]int{1, 2, 2, 2}, data)
	scale, _ := tensor.New[float32]([]int{2}, []float32{1, 1})
	B, _ := tensor.New[float32]([]int{2}, []float32{0, 0})
	mean, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	variance, _ := tensor.New[float32]([]int{2}, []float32{1, 1})

	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))
	out, err := layer.Forward(ctx, X, scale, B, mean, variance)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	wantShape := []int{1, 2, 2, 2}
	if !intSliceEq(out.Shape(), wantShape) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), wantShape)
	}
	// x==mean for every element, so all outputs should be ~0.
	for i, v := range out.Data() {
		if math.Abs(float64(v)) > 1e-4 {
			t.Errorf("out[%d] = %v, want ~0", i, v)
		}
	}
}

func TestBatchNormalization_InvalidInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))

	X, _ := tensor.New[float32]([]int{1, 1, 1, 1}, []float32{1})
	_, err := layer.Forward(context.Background(), X)
	if err == nil {
		t.Fatal("expected error for 1 input")
	}
}

func TestBatchNormalization_OpTypeAndMeta(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))

	if layer.OpType() != "BatchNormalization" {
		t.Errorf("OpType = %q, want BatchNormalization", layer.OpType())
	}
	// Inference-only mode: no learnable parameters.
	if layer.Parameters() != nil {
		t.Error("Parameters should be nil for inference-only mode")
	}
}

func TestBuildBatchNormalization_DefaultEpsilon(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	node, err := normalization.BuildBatchNormalization[float32](engine, &ops, "bn", nil, map[string]interface{}{})
	if err != nil {
		t.Fatalf("BuildBatchNormalization failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildBatchNormalization returned nil")
	}
	if node.OpType() != "BatchNormalization" {
		t.Errorf("OpType = %q, want BatchNormalization", node.OpType())
	}
}

func TestBuildBatchNormalization_WithEpsilon(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	attrs := map[string]interface{}{"epsilon": float32(1e-3)}
	node, err := normalization.BuildBatchNormalization[float32](engine, &ops, "bn", nil, attrs)
	if err != nil {
		t.Fatalf("BuildBatchNormalization failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildBatchNormalization returned nil")
	}
}

func TestBuildBatchNormalization_WithFloat64Epsilon(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	attrs := map[string]interface{}{"epsilon": float64(1e-3)}
	node, err := normalization.BuildBatchNormalization[float32](engine, &ops, "bn", nil, attrs)
	if err != nil {
		t.Fatalf("BuildBatchNormalization failed: %v", err)
	}
	_ = node
}

// TestBatchNormalization_Attributes tests that Attributes returns epsilon.
func TestBatchNormalization_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	const eps = float32(1e-5)
	layer := normalization.NewBatchNormalization[float32](engine, &ops, eps)
	attrs := layer.Attributes()
	if attrs == nil {
		t.Fatal("Attributes returned nil")
	}
	if _, ok := attrs["epsilon"]; !ok {
		t.Error("missing 'epsilon' in Attributes")
	}
}

// TestBatchNormalization_OutputShape verifies OutputShape is populated after Forward.
func TestBatchNormalization_OutputShape(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	X, _ := tensor.New[float32]([]int{2, 3, 4, 4}, nil)
	data := make([]float32, 2*3*4*4)
	X.SetData(data)
	scale, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})
	B, _ := tensor.New[float32]([]int{3}, []float32{0, 0, 0})
	mean, _ := tensor.New[float32]([]int{3}, []float32{0, 0, 0})
	variance, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})

	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))
	if _, err := layer.Forward(ctx, X, scale, B, mean, variance); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	want := []int{2, 3, 4, 4}
	if !intSliceEq(layer.OutputShape(), want) {
		t.Errorf("OutputShape = %v, want %v", layer.OutputShape(), want)
	}
}

// Intentionally avoid importing graph in test; use a local helper.
func intSliceEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Compile-time check that BatchNormalization satisfies graph.Node.
var _ graph.Node[float32] = normalization.NewBatchNormalization[float32](nil, nil, 0)

// --- Backward pass tests ---

// makeBNWithParams creates a BatchNormalization layer with learnable parameters
// and fixed mean/variance tensors for testing.
func makeBNWithParams(t *testing.T, channels int, scaleData, biasData, meanData, varData []float32) (
	*normalization.BatchNormalization[float32],
	*graph.Parameter[float32],
	*graph.Parameter[float32],
	*tensor.TensorNumeric[float32],
	*tensor.TensorNumeric[float32],
) {
	t.Helper()
	ops := numeric.Float32Ops{}

	scaleTensor, err := tensor.New[float32]([]int{channels}, scaleData)
	if err != nil {
		t.Fatalf("tensor.New scale: %v", err)
	}
	scaleParam, err := graph.NewParameter[float32]("test_scale", scaleTensor, tensor.New[float32])
	if err != nil {
		t.Fatalf("graph.NewParameter scale: %v", err)
	}

	biasTensor, err := tensor.New[float32]([]int{channels}, biasData)
	if err != nil {
		t.Fatalf("tensor.New bias: %v", err)
	}
	biasParam, err := graph.NewParameter[float32]("test_bias", biasTensor, tensor.New[float32])
	if err != nil {
		t.Fatalf("graph.NewParameter bias: %v", err)
	}

	engine := compute.NewCPUEngine[float32](&ops)
	bn := normalization.NewBatchNormalizationWithParams[float32](engine, &ops, float32(1e-5), scaleParam, biasParam)

	mean, err := tensor.New[float32]([]int{channels}, meanData)
	if err != nil {
		t.Fatalf("tensor.New mean: %v", err)
	}
	variance, err := tensor.New[float32]([]int{channels}, varData)
	if err != nil {
		t.Fatalf("tensor.New variance: %v", err)
	}

	return bn, scaleParam, biasParam, mean, variance
}

// batchnormLoss computes sum(BatchNorm(X)) for finite-difference testing.
func batchnormLoss(t *testing.T, ctx context.Context, xShape []int, xData, scaleData, biasData, meanData, varData []float32) float64 {
	t.Helper()
	channels := xShape[1]
	bn, _, _, mean, variance := makeBNWithParams(t, channels, scaleData, biasData, meanData, varData)

	X, err := tensor.New[float32](xShape, xData)
	if err != nil {
		t.Fatalf("tensor.New X: %v", err)
	}
	scaleTensor, _ := tensor.New[float32]([]int{channels}, scaleData)
	biasTensor, _ := tensor.New[float32]([]int{channels}, biasData)

	out, err := bn.Forward(ctx, X, scaleTensor, biasTensor, mean, variance)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	var sum float64
	for _, v := range out.Data() {
		sum += float64(v)
	}
	return sum
}

// TestBatchNormalization_BackwardRoundTrip tests that Forward followed by Backward
// produces valid non-zero gradients for input, scale, and bias.
func TestBatchNormalization_BackwardRoundTrip(t *testing.T) {
	tests := []struct {
		name      string
		xShape    []int
		channels  int
		scaleData []float32
		biasData  []float32
		meanData  []float32
		varData   []float32
	}{
		{
			name:      "2D_batch2_ch4",
			xShape:    []int{2, 4},
			channels:  4,
			scaleData: []float32{1.0, 1.5, 0.5, 2.0},
			biasData:  []float32{0.0, 0.1, -0.1, 0.5},
			meanData:  []float32{0.0, 0.0, 0.0, 0.0},
			varData:   []float32{1.0, 2.0, 0.5, 1.0},
		},
		{
			name:      "4D_batch1_ch2_hw2x2",
			xShape:    []int{1, 2, 2, 2},
			channels:  2,
			scaleData: []float32{1.0, 2.0},
			biasData:  []float32{0.0, 0.0},
			meanData:  []float32{0.5, 1.0},
			varData:   []float32{1.0, 4.0},
		},
		{
			name:      "2D_batch1_ch1",
			xShape:    []int{1, 1},
			channels:  1,
			scaleData: []float32{3.0},
			biasData:  []float32{1.0},
			meanData:  []float32{0.0},
			varData:   []float32{1.0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](&ops)

			scaleTensor, _ := tensor.New[float32]([]int{tc.channels}, tc.scaleData)
			scaleParam, _ := graph.NewParameter[float32]("scale", scaleTensor, tensor.New[float32])
			biasTensor, _ := tensor.New[float32]([]int{tc.channels}, tc.biasData)
			biasParam, _ := graph.NewParameter[float32]("bias", biasTensor, tensor.New[float32])

			bn := normalization.NewBatchNormalizationWithParams[float32](engine, &ops, float32(1e-5), scaleParam, biasParam)

			// Create input data.
			total := 1
			for _, d := range tc.xShape {
				total *= d
			}
			xData := make([]float32, total)
			for i := range xData {
				xData[i] = float32(i+1) * 0.1
			}
			X, _ := tensor.New[float32](tc.xShape, xData)
			mean, _ := tensor.New[float32]([]int{tc.channels}, tc.meanData)
			variance, _ := tensor.New[float32]([]int{tc.channels}, tc.varData)

			// Forward.
			out, err := bn.Forward(ctx, X, scaleParam.Value, biasParam.Value, mean, variance)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// dOut = ones (loss = sum(output)).
			onesData := make([]float32, out.Size())
			for i := range onesData {
				onesData[i] = 1.0
			}
			dOut, _ := tensor.New[float32](out.Shape(), onesData)

			// Backward.
			grads, err := bn.Backward(ctx, types.FullBackprop, dOut, X)
			if err != nil {
				t.Fatalf("Backward: %v", err)
			}

			if len(grads) != 1 || grads[0] == nil {
				t.Fatal("expected exactly 1 non-nil input gradient")
			}
			if !intSliceEq(grads[0].Shape(), tc.xShape) {
				t.Errorf("gradient shape %v != input shape %v", grads[0].Shape(), tc.xShape)
			}

			// Check non-zero input gradient.
			hasNonZero := false
			for _, v := range grads[0].Data() {
				if v != 0 {
					hasNonZero = true
					break
				}
			}
			if !hasNonZero {
				t.Error("input gradient is all zeros")
			}

			// Check parameter gradients.
			for _, p := range bn.Parameters() {
				if p.Gradient == nil {
					t.Errorf("parameter %q has nil gradient", p.Name)
					continue
				}
				nonZero := false
				for _, v := range p.Gradient.Data() {
					if v != 0 {
						nonZero = true
						break
					}
				}
				if !nonZero {
					t.Errorf("parameter %q has all-zero gradient", p.Name)
				}
			}
		})
	}
}

// TestBatchNormalization_BackwardFiniteDifference validates analytical gradients
// against numerical finite-difference approximation for input, scale, and bias.
func TestBatchNormalization_BackwardFiniteDifference(t *testing.T) {
	tests := []struct {
		name      string
		xShape    []int
		channels  int
		scaleData []float32
		biasData  []float32
		meanData  []float32
		varData   []float32
	}{
		{
			name:      "2D_batch2_ch4",
			xShape:    []int{2, 4},
			channels:  4,
			scaleData: []float32{1.0, 1.5, 0.5, 2.0},
			biasData:  []float32{0.0, 0.1, -0.1, 0.5},
			meanData:  []float32{0.0, 0.0, 0.0, 0.0},
			varData:   []float32{1.0, 2.0, 0.5, 1.0},
		},
		{
			name:      "4D_batch1_ch2_hw2x2",
			xShape:    []int{1, 2, 2, 2},
			channels:  2,
			scaleData: []float32{1.0, 2.0},
			biasData:  []float32{0.0, 0.0},
			meanData:  []float32{0.5, 1.0},
			varData:   []float32{1.0, 4.0},
		},
		{
			name:      "2D_batch3_ch2",
			xShape:    []int{3, 2},
			channels:  2,
			scaleData: []float32{1.5, 0.8},
			biasData:  []float32{0.2, -0.3},
			meanData:  []float32{1.0, 2.0},
			varData:   []float32{0.5, 3.0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](&ops)

			total := 1
			for _, d := range tc.xShape {
				total *= d
			}
			xData := make([]float32, total)
			for i := range xData {
				xData[i] = float32(i+1) * 0.1
			}

			// --- Analytical gradient via Backward ---
			scaleTensor, _ := tensor.New[float32]([]int{tc.channels}, tc.scaleData)
			scaleParam, _ := graph.NewParameter[float32]("scale", scaleTensor, tensor.New[float32])
			biasTensor, _ := tensor.New[float32]([]int{tc.channels}, tc.biasData)
			biasParam, _ := graph.NewParameter[float32]("bias", biasTensor, tensor.New[float32])

			bn := normalization.NewBatchNormalizationWithParams[float32](engine, &ops, float32(1e-5), scaleParam, biasParam)

			X, _ := tensor.New[float32](tc.xShape, xData)
			mean, _ := tensor.New[float32]([]int{tc.channels}, tc.meanData)
			variance, _ := tensor.New[float32]([]int{tc.channels}, tc.varData)

			_, err := bn.Forward(ctx, X, scaleParam.Value, biasParam.Value, mean, variance)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			onesData := make([]float32, total)
			for i := range onesData {
				onesData[i] = 1.0
			}
			dOut, _ := tensor.New[float32](tc.xShape, onesData)

			scaleParam.Gradient = nil
			biasParam.Gradient = nil

			grads, err := bn.Backward(ctx, types.FullBackprop, dOut, X)
			if err != nil {
				t.Fatalf("Backward: %v", err)
			}
			analyticalDX := grads[0].Data()
			analyticalDScale := scaleParam.Gradient.Data()
			analyticalDBias := biasParam.Gradient.Data()

			const eps = 1e-3

			// --- Numerical gradient for input X ---
			t.Run("dX", func(t *testing.T) {
				numericalDX := make([]float32, total)
				for i := range total {
					plusData := make([]float32, total)
					copy(plusData, xData)
					plusData[i] += eps
					lossPlus := batchnormLoss(t, ctx, tc.xShape, plusData, tc.scaleData, tc.biasData, tc.meanData, tc.varData)

					minusData := make([]float32, total)
					copy(minusData, xData)
					minusData[i] -= eps
					lossMinus := batchnormLoss(t, ctx, tc.xShape, minusData, tc.scaleData, tc.biasData, tc.meanData, tc.varData)

					numericalDX[i] = float32((lossPlus - lossMinus) / (2 * eps))
				}

				maxDiff := float32(0)
				for i := range total {
					diff := float32(math.Abs(float64(analyticalDX[i] - numericalDX[i])))
					if diff > maxDiff {
						maxDiff = diff
					}
				}
				if maxDiff > 1e-3 {
					t.Errorf("max dX diff: %e (threshold 1e-3)", maxDiff)
					for i := range min(total, 8) {
						t.Logf("  [%d] analytical=%.6f numerical=%.6f", i, analyticalDX[i], numericalDX[i])
					}
				}
			})

			// --- Numerical gradient for scale ---
			t.Run("dScale", func(t *testing.T) {
				numericalDScale := make([]float32, tc.channels)
				for i := range tc.channels {
					plusScale := make([]float32, tc.channels)
					copy(plusScale, tc.scaleData)
					plusScale[i] += eps
					lossPlus := batchnormLoss(t, ctx, tc.xShape, xData, plusScale, tc.biasData, tc.meanData, tc.varData)

					minusScale := make([]float32, tc.channels)
					copy(minusScale, tc.scaleData)
					minusScale[i] -= eps
					lossMinus := batchnormLoss(t, ctx, tc.xShape, xData, minusScale, tc.biasData, tc.meanData, tc.varData)

					numericalDScale[i] = float32((lossPlus - lossMinus) / (2 * eps))
				}

				maxDiff := float32(0)
				for i := range tc.channels {
					diff := float32(math.Abs(float64(analyticalDScale[i] - numericalDScale[i])))
					if diff > maxDiff {
						maxDiff = diff
					}
				}
				if maxDiff > 1e-3 {
					t.Errorf("max dScale diff: %e (threshold 1e-3)", maxDiff)
					for i := range tc.channels {
						t.Logf("  [%d] analytical=%.6f numerical=%.6f", i, analyticalDScale[i], numericalDScale[i])
					}
				}
			})

			// --- Numerical gradient for bias ---
			t.Run("dBias", func(t *testing.T) {
				numericalDBias := make([]float32, tc.channels)
				for i := range tc.channels {
					plusBias := make([]float32, tc.channels)
					copy(plusBias, tc.biasData)
					plusBias[i] += eps
					lossPlus := batchnormLoss(t, ctx, tc.xShape, xData, tc.scaleData, plusBias, tc.meanData, tc.varData)

					minusBias := make([]float32, tc.channels)
					copy(minusBias, tc.biasData)
					minusBias[i] -= eps
					lossMinus := batchnormLoss(t, ctx, tc.xShape, xData, tc.scaleData, minusBias, tc.meanData, tc.varData)

					numericalDBias[i] = float32((lossPlus - lossMinus) / (2 * eps))
				}

				maxDiff := float32(0)
				for i := range tc.channels {
					diff := float32(math.Abs(float64(analyticalDBias[i] - numericalDBias[i])))
					if diff > maxDiff {
						maxDiff = diff
					}
				}
				if maxDiff > 1e-3 {
					t.Errorf("max dBias diff: %e (threshold 1e-3)", maxDiff)
					for i := range tc.channels {
						t.Logf("  [%d] analytical=%.6f numerical=%.6f", i, analyticalDBias[i], numericalDBias[i])
					}
				}
			})
		})
	}
}

// TestBatchNormalization_BackwardOptimizerStep verifies that an SGD step using
// BatchNorm gradients reduces the MSE loss.
func TestBatchNormalization_BackwardOptimizerStep(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	const (
		batch    = 2
		channels = 4
	)

	scaleData := []float32{1.0, 1.0, 1.0, 1.0}
	biasData := []float32{0.0, 0.0, 0.0, 0.0}
	meanData := []float32{0.0, 0.0, 0.0, 0.0}
	varData := []float32{1.0, 1.0, 1.0, 1.0}

	scaleTensor, _ := tensor.New[float32]([]int{channels}, scaleData)
	scaleParam, _ := graph.NewParameter[float32]("scale", scaleTensor, tensor.New[float32])
	biasTensor, _ := tensor.New[float32]([]int{channels}, biasData)
	biasParam, _ := graph.NewParameter[float32]("bias", biasTensor, tensor.New[float32])

	bn := normalization.NewBatchNormalizationWithParams[float32](engine, &ops, float32(1e-5), scaleParam, biasParam)

	xData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	X, _ := tensor.New[float32]([]int{batch, channels}, xData)
	mean, _ := tensor.New[float32]([]int{channels}, meanData)
	variance, _ := tensor.New[float32]([]int{channels}, varData)

	target := []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}

	// Step 1: Forward pass and compute initial MSE loss.
	out1, err := bn.Forward(ctx, X, scaleParam.Value, biasParam.Value, mean, variance)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	outData1 := out1.Data()
	var loss1 float32
	for i := range outData1 {
		d := outData1[i] - target[i]
		loss1 += d * d
	}
	loss1 /= float32(len(outData1))

	// Step 2: Compute MSE gradient: dL/dOut = 2*(out - target) / N.
	gradData := make([]float32, len(outData1))
	n := float32(len(outData1))
	for i := range gradData {
		gradData[i] = 2 * (outData1[i] - target[i]) / n
	}
	dOut, _ := tensor.New[float32](out1.Shape(), gradData)

	// Step 3: Backward.
	scaleParam.Gradient = nil
	biasParam.Gradient = nil
	_, err = bn.Backward(ctx, types.FullBackprop, dOut, X)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Step 4: SGD step.
	lr := float32(0.01)
	for _, p := range bn.Parameters() {
		if p.Gradient == nil {
			continue
		}
		pData := p.Value.Data()
		gData := p.Gradient.Data()
		for j := range pData {
			pData[j] -= lr * gData[j]
		}
	}

	// Step 5: Forward again with updated parameters.
	out2, err := bn.Forward(ctx, X, scaleParam.Value, biasParam.Value, mean, variance)
	if err != nil {
		t.Fatalf("Forward after update: %v", err)
	}
	outData2 := out2.Data()
	var loss2 float32
	for i := range outData2 {
		d := outData2[i] - target[i]
		loss2 += d * d
	}
	loss2 /= float32(len(outData2))

	t.Logf("loss before: %f, after: %f", loss1, loss2)
	if loss2 >= loss1 {
		t.Errorf("loss did not decrease: before=%f after=%f", loss1, loss2)
	}
}

// TestBatchNormalization_BackwardErrors tests error conditions in Backward.
func TestBatchNormalization_BackwardErrors(t *testing.T) {
	tests := []struct {
		name        string
		runForward  bool
		inputCount  int
		wantErr     bool
		errContains string
	}{
		{
			name:        "backward before forward",
			runForward:  false,
			inputCount:  1,
			wantErr:     true,
			errContains: "backward called before forward",
		},
		{
			name:        "wrong input count zero",
			runForward:  true,
			inputCount:  0,
			wantErr:     true,
			errContains: "invalid number of inputs",
		},
		{
			name:        "wrong input count two",
			runForward:  true,
			inputCount:  2,
			wantErr:     true,
			errContains: "invalid number of inputs",
		},
		{
			name:       "correct usage",
			runForward: true,
			inputCount: 1,
			wantErr:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](&ops)

			const channels = 2
			scaleTensor, _ := tensor.New[float32]([]int{channels}, []float32{1, 1})
			scaleParam, _ := graph.NewParameter[float32]("s", scaleTensor, tensor.New[float32])
			biasTensor, _ := tensor.New[float32]([]int{channels}, []float32{0, 0})
			biasParam, _ := graph.NewParameter[float32]("b", biasTensor, tensor.New[float32])

			bn := normalization.NewBatchNormalizationWithParams[float32](engine, &ops, float32(1e-5), scaleParam, biasParam)

			X, _ := tensor.New[float32]([]int{1, channels}, []float32{1, 2})
			mean, _ := tensor.New[float32]([]int{channels}, []float32{0, 0})
			variance, _ := tensor.New[float32]([]int{channels}, []float32{1, 1})
			dOut, _ := tensor.New[float32]([]int{1, channels}, []float32{1, 1})

			if tc.runForward {
				_, err := bn.Forward(ctx, X, scaleParam.Value, biasParam.Value, mean, variance)
				if err != nil {
					t.Fatalf("Forward: %v", err)
				}
			}

			var inputs []*tensor.TensorNumeric[float32]
			for range tc.inputCount {
				inputs = append(inputs, X)
			}

			_, err := bn.Backward(ctx, types.FullBackprop, dOut, inputs...)
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
		})
	}
}

// TestBatchNormalization_ParametersWithParams verifies Parameters() returns
// scale and bias when created with NewBatchNormalizationWithParams.
func TestBatchNormalization_ParametersWithParams(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	scaleTensor, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})
	scaleParam, _ := graph.NewParameter[float32]("scale", scaleTensor, tensor.New[float32])
	biasTensor, _ := tensor.New[float32]([]int{3}, []float32{0, 0, 0})
	biasParam, _ := graph.NewParameter[float32]("bias", biasTensor, tensor.New[float32])

	bn := normalization.NewBatchNormalizationWithParams[float32](engine, &ops, float32(1e-5), scaleParam, biasParam)

	params := bn.Parameters()
	if len(params) != 2 {
		t.Fatalf("expected 2 parameters, got %d", len(params))
	}
	if params[0].Name != "scale" {
		t.Errorf("params[0].Name = %q, want scale", params[0].Name)
	}
	if params[1].Name != "bias" {
		t.Errorf("params[1].Name = %q, want bias", params[1].Name)
	}
}
