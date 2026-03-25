package normalization

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// computeMSE computes the mean squared error between output and target slices.
func computeMSE(output, target []float32) float32 {
	var sum float32
	for i := range output {
		d := output[i] - target[i]
		sum += d * d
	}
	return sum / float32(len(output))
}

// mseGradient computes d(MSE)/d(output) = 2*(output - target) / n.
func mseGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	n := float32(len(output))
	for i := range grad {
		grad[i] = 2 * (output[i] - target[i]) / n
	}
	return grad
}

// applyGradientDescent performs a single manual SGD step: param -= lr * grad.
func applyGradientDescent(params []*graph.Parameter[float32], lr float32) {
	for _, p := range params {
		if p.Gradient == nil {
			continue
		}
		pData := p.Value.Data()
		gData := p.Gradient.Data()
		for j := range pData {
			pData[j] -= lr * gData[j]
		}
	}
}

// hasNonZeroGradient returns true if at least one element in the gradient is non-zero.
func hasNonZeroGradient(p *graph.Parameter[float32]) bool {
	if p.Gradient == nil {
		return false
	}
	for _, v := range p.Gradient.Data() {
		if v != 0 {
			return true
		}
	}
	return false
}

// normLayerTestCase defines a table-driven test case for normalization learning verification.
type normLayerTestCase struct {
	name string
	// setup returns: forward func, backward func, parameters, input tensor, target data.
	setup func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
		forward func(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
		backward func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], input *tensor.TensorNumeric[float32]) error,
		params []*graph.Parameter[float32],
		input *tensor.TensorNumeric[float32],
		target []float32,
	)
}

func buildNormTestCases() []normLayerTestCase {
	return []normLayerTestCase{
		{
			name: "RMSNorm",
			setup: func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
				func(context.Context, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
				func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) error,
				[]*graph.Parameter[float32],
				*tensor.TensorNumeric[float32],
				[]float32,
			) {
				const (
					batch    = 2
					modelDim = 8
				)

				rmsnorm, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim)
				if err != nil {
					t.Fatalf("NewRMSNorm: %v", err)
				}

				inputData := make([]float32, batch*modelDim)
				for i := range inputData {
					inputData[i] = float32(rng.NormFloat64()) * 0.5
				}
				input, err := tensor.New[float32]([]int{batch, modelDim}, inputData)
				if err != nil {
					t.Fatalf("tensor.New input: %v", err)
				}

				targetData := make([]float32, batch*modelDim)
				for i := range targetData {
					targetData[i] = float32(rng.NormFloat64()) * 0.1
				}

				params := rmsnorm.Parameters()

				forward := func(ctx context.Context, in *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
					return rmsnorm.Forward(ctx, in)
				}
				backward := func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], in *tensor.TensorNumeric[float32]) error {
					_, err := rmsnorm.Backward(ctx, types.FullBackprop, outputGrad, in)
					return err
				}

				return forward, backward, params, input, targetData
			},
		},
		{
			name: "BatchNorm",
			setup: func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
				func(context.Context, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
				func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) error,
				[]*graph.Parameter[float32],
				*tensor.TensorNumeric[float32],
				[]float32,
			) {
				const (
					batch    = 2
					channels = 8
				)

				// Create learnable scale and bias parameters.
				scaleData := make([]float32, channels)
				for i := range scaleData {
					scaleData[i] = 1.0
				}
				scaleTensor, err := tensor.New[float32]([]int{channels}, scaleData)
				if err != nil {
					t.Fatalf("tensor.New scale: %v", err)
				}
				scaleParam, err := graph.NewParameter[float32]("bn_scale", scaleTensor, tensor.New[float32])
				if err != nil {
					t.Fatalf("graph.NewParameter scale: %v", err)
				}

				biasData := make([]float32, channels)
				biasTensor, err := tensor.New[float32]([]int{channels}, biasData)
				if err != nil {
					t.Fatalf("tensor.New bias: %v", err)
				}
				biasParam, err := graph.NewParameter[float32]("bn_bias", biasTensor, tensor.New[float32])
				if err != nil {
					t.Fatalf("graph.NewParameter bias: %v", err)
				}

				bn := NewBatchNormalizationWithParams[float32](engine, ops, float32(1e-5), scaleParam, biasParam)

				// Pre-computed running mean and variance.
				meanData := make([]float32, channels)
				mean, err := tensor.New[float32]([]int{channels}, meanData)
				if err != nil {
					t.Fatalf("tensor.New mean: %v", err)
				}
				varData := make([]float32, channels)
				for i := range varData {
					varData[i] = 1.0
				}
				variance, err := tensor.New[float32]([]int{channels}, varData)
				if err != nil {
					t.Fatalf("tensor.New variance: %v", err)
				}

				inputData := make([]float32, batch*channels)
				for i := range inputData {
					inputData[i] = float32(rng.NormFloat64()) * 0.5
				}
				input, err := tensor.New[float32]([]int{batch, channels}, inputData)
				if err != nil {
					t.Fatalf("tensor.New input: %v", err)
				}

				targetData := make([]float32, batch*channels)
				for i := range targetData {
					targetData[i] = float32(rng.NormFloat64()) * 0.1
				}

				params := bn.Parameters()

				forward := func(ctx context.Context, in *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
					return bn.Forward(ctx, in, scaleParam.Value, biasParam.Value, mean, variance)
				}
				backward := func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], in *tensor.TensorNumeric[float32]) error {
					_, err := bn.Backward(ctx, types.FullBackprop, outputGrad, in)
					return err
				}

				return forward, backward, params, input, targetData
			},
		},
	}
}

// TestNormLayers_LossDecreases verifies that normalization layers with learnable
// parameters can learn: after Forward -> MSE -> Backward -> gradient step, the
// loss decreases.
func TestNormLayers_LossDecreases(t *testing.T) {
	for _, tc := range buildNormTestCases() {
		t.Run(tc.name, func(t *testing.T) {
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
			ops := numeric.Float32Ops{}
			rng := rand.New(rand.NewPCG(42, 0))
			ctx := context.Background()

			forward, backward, params, input, target := tc.setup(t, engine, ops, rng)

			// Step 1: Forward pass.
			output, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			outputData := output.Data()
			if len(outputData) != len(target) {
				t.Fatalf("output size %d != target size %d", len(outputData), len(target))
			}

			// Step 2: Compute initial MSE loss.
			loss1 := computeMSE(outputData, target)
			if math.IsNaN(float64(loss1)) || math.IsInf(float64(loss1), 0) {
				t.Fatalf("initial loss is NaN or Inf: %v", loss1)
			}
			t.Logf("initial loss: %f", loss1)

			// Step 3: Compute output gradient (dL/dOutput for MSE).
			gradData := mseGradient(outputData, target)
			outputGrad, err := tensor.New[float32](output.Shape(), gradData)
			if err != nil {
				t.Fatalf("tensor.New outputGrad: %v", err)
			}

			// Step 4: Backward pass.
			if err := backward(ctx, outputGrad, input); err != nil {
				t.Fatalf("Backward: %v", err)
			}

			// Step 5: Verify all parameters have non-zero gradients.
			for _, p := range params {
				if p.Gradient == nil {
					t.Errorf("parameter %q has nil gradient", p.Name)
					continue
				}
				if !hasNonZeroGradient(p) {
					t.Errorf("parameter %q has all-zero gradient", p.Name)
				}
			}

			// Step 6: Apply gradient descent.
			lr := float32(0.01)
			applyGradientDescent(params, lr)

			// Step 7: Forward pass again with updated parameters.
			output2, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward (after update): %v", err)
			}

			// Step 8: Compute new loss.
			loss2 := computeMSE(output2.Data(), target)
			if math.IsNaN(float64(loss2)) || math.IsInf(float64(loss2), 0) {
				t.Fatalf("updated loss is NaN or Inf: %v", loss2)
			}
			t.Logf("updated loss: %f", loss2)

			// Step 9: Verify loss decreased.
			if loss2 >= loss1 {
				t.Errorf("loss did not decrease: before=%f, after=%f", loss1, loss2)
			}
		})
	}
}

// TestBatchNorm_Backward tests that BatchNormalization backward produces
// non-nil, non-zero gradients for input and accumulates parameter gradients.
func TestBatchNorm_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const (
		batch    = 2
		channels = 8
	)
	epsilon := float32(1e-5)

	// Create learnable scale and bias parameters.
	scaleData := make([]float32, channels)
	for i := range scaleData {
		scaleData[i] = 1.0
	}
	scaleTensor, err := tensor.New[float32]([]int{channels}, scaleData)
	if err != nil {
		t.Fatalf("tensor.New scale: %v", err)
	}
	scaleParam, err := graph.NewParameter[float32]("bn_scale", scaleTensor, tensor.New[float32])
	if err != nil {
		t.Fatalf("graph.NewParameter scale: %v", err)
	}

	biasData := make([]float32, channels)
	biasTensor, err := tensor.New[float32]([]int{channels}, biasData)
	if err != nil {
		t.Fatalf("tensor.New bias: %v", err)
	}
	biasParam, err := graph.NewParameter[float32]("bn_bias", biasTensor, tensor.New[float32])
	if err != nil {
		t.Fatalf("graph.NewParameter bias: %v", err)
	}

	bn := NewBatchNormalizationWithParams[float32](engine, ops, epsilon, scaleParam, biasParam)

	rng := rand.New(rand.NewPCG(42, 0))

	// Input: [batch, channels]
	inputData := make([]float32, batch*channels)
	for i := range inputData {
		inputData[i] = float32(rng.NormFloat64()) * 0.5
	}
	input, err := tensor.New[float32]([]int{batch, channels}, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	// Running mean: zeros
	meanData := make([]float32, channels)
	mean, err := tensor.New[float32]([]int{channels}, meanData)
	if err != nil {
		t.Fatalf("tensor.New mean: %v", err)
	}

	// Running variance: ones
	varData := make([]float32, channels)
	for i := range varData {
		varData[i] = 1.0
	}
	variance, err := tensor.New[float32]([]int{channels}, varData)
	if err != nil {
		t.Fatalf("tensor.New variance: %v", err)
	}

	// Forward pass.
	output, err := bn.Forward(ctx, input, scaleParam.Value, biasTensor, mean, variance)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create output gradient for backward.
	gradData := make([]float32, batch*channels)
	for i := range gradData {
		gradData[i] = float32(rng.NormFloat64()) * 0.1
	}
	outputGrad, err := tensor.New[float32](output.Shape(), gradData)
	if err != nil {
		t.Fatalf("tensor.New outputGrad: %v", err)
	}

	// Backward pass.
	inputGrads, err := bn.Backward(ctx, types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward returned unexpected error: %v", err)
	}

	// Input gradients must be non-nil and non-zero.
	if inputGrads == nil || len(inputGrads) != 1 || inputGrads[0] == nil {
		t.Fatal("expected non-nil gradient slice with one element")
	}

	hasNonZero := false
	for _, v := range inputGrads[0].Data() {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("input gradient is all zeros")
	}

	// Parameter gradients must be non-nil and non-zero.
	params := bn.Parameters()
	if len(params) != 2 {
		t.Fatalf("expected 2 parameters, got %d", len(params))
	}
	for _, p := range params {
		if !hasNonZeroGradient(p) {
			t.Errorf("parameter %q has zero gradient", p.Name)
		}
	}
}
