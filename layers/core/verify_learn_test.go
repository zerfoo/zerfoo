package core

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

// layerTestCase defines a table-driven test case for learning verification.
type layerTestCase struct {
	name string
	// setup returns: layer forward func, layer backward func, parameters, input tensor, target data
	setup func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
		forward func(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
		backward func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], input *tensor.TensorNumeric[float32]) error,
		params []*graph.Parameter[float32],
		input *tensor.TensorNumeric[float32],
		target []float32,
	)
}

func buildTestCases() []layerTestCase {
	return []layerTestCase{
		{
			name: "Dense",
			setup: func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
				func(context.Context, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
				func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) error,
				[]*graph.Parameter[float32],
				*tensor.TensorNumeric[float32],
				[]float32,
			) {
				const (
					batch     = 4
					inputDim  = 8
					outputDim = 4
				)
				dense, err := NewDense[float32]("test_dense", engine, ops, inputDim, outputDim)
				if err != nil {
					t.Fatalf("NewDense: %v", err)
				}

				inputData := make([]float32, batch*inputDim)
				for i := range inputData {
					inputData[i] = float32(rng.NormFloat64()) * 0.5
				}
				input, err := tensor.New[float32]([]int{batch, inputDim}, inputData)
				if err != nil {
					t.Fatalf("tensor.New input: %v", err)
				}

				targetData := make([]float32, batch*outputDim)
				for i := range targetData {
					targetData[i] = float32(rng.NormFloat64()) * 0.1
				}

				params := dense.Parameters()

				forward := func(ctx context.Context, in *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
					return dense.Forward(ctx, in)
				}
				backward := func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], in *tensor.TensorNumeric[float32]) error {
					_, err := dense.Backward(ctx, types.FullBackprop, outputGrad, in)
					return err
				}

				return forward, backward, params, input, targetData
			},
		},
		{
			name: "Linear",
			setup: func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
				func(context.Context, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
				func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) error,
				[]*graph.Parameter[float32],
				*tensor.TensorNumeric[float32],
				[]float32,
			) {
				const (
					batch     = 4
					inputDim  = 8
					outputDim = 4
				)
				linear, err := NewLinear[float32]("test_linear", engine, ops, inputDim, outputDim)
				if err != nil {
					t.Fatalf("NewLinear: %v", err)
				}

				inputData := make([]float32, batch*inputDim)
				for i := range inputData {
					inputData[i] = float32(rng.NormFloat64()) * 0.5
				}
				input, err := tensor.New[float32]([]int{batch, inputDim}, inputData)
				if err != nil {
					t.Fatalf("tensor.New input: %v", err)
				}

				targetData := make([]float32, batch*outputDim)
				for i := range targetData {
					targetData[i] = float32(rng.NormFloat64()) * 0.1
				}

				params := linear.Parameters()

				forward := func(ctx context.Context, in *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
					return linear.Forward(ctx, in)
				}
				backward := func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], in *tensor.TensorNumeric[float32]) error {
					_, err := linear.Backward(ctx, types.FullBackprop, outputGrad, in)
					return err
				}

				return forward, backward, params, input, targetData
			},
		},
		{
			name: "FFN",
			setup: func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
				func(context.Context, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
				func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) error,
				[]*graph.Parameter[float32],
				*tensor.TensorNumeric[float32],
				[]float32,
			) {
				const (
					batch     = 4
					modelDim  = 8
					ffnDim    = 16
					outputDim = 8 // FFN outputDim == modelDim (w2 maps back)
				)
				ffn, err := NewFFN[float32]("test_ffn", engine, ops, modelDim, ffnDim, outputDim)
				if err != nil {
					t.Fatalf("NewFFN: %v", err)
				}

				inputData := make([]float32, batch*modelDim)
				for i := range inputData {
					inputData[i] = float32(rng.NormFloat64()) * 0.5
				}
				input, err := tensor.New[float32]([]int{batch, modelDim}, inputData)
				if err != nil {
					t.Fatalf("tensor.New input: %v", err)
				}

				targetData := make([]float32, batch*outputDim)
				for i := range targetData {
					targetData[i] = float32(rng.NormFloat64()) * 0.1
				}

				params := ffn.Parameters()

				forward := func(ctx context.Context, in *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
					return ffn.Forward(ctx, in)
				}
				backward := func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], in *tensor.TensorNumeric[float32]) error {
					// FFN backward reads the live input (ztensor ADR 006).
					_, err := ffn.Backward(ctx, types.FullBackprop, outputGrad, in)
					return err
				}

				return forward, backward, params, input, targetData
			},
		},
		{
			name: "Conv1D",
			setup: func(t *testing.T, engine compute.Engine[float32], ops numeric.Arithmetic[float32], rng *rand.Rand) (
				func(context.Context, *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error),
				func(context.Context, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) error,
				[]*graph.Parameter[float32],
				*tensor.TensorNumeric[float32],
				[]float32,
			) {
				const (
					batch       = 2
					inChannels  = 3
					outChannels = 4
					kernelSize  = 3
					seqLen      = 8
					// output_length = (seqLen + 2*0 - kernelSize) / 1 + 1 = 6
					outLen = 6
				)
				conv, err := NewConv1D[float32]("test_conv1d", engine, ops, inChannels, outChannels, kernelSize)
				if err != nil {
					t.Fatalf("NewConv1D: %v", err)
				}

				// Input shape: [batch, inChannels, seqLen]
				inputData := make([]float32, batch*inChannels*seqLen)
				for i := range inputData {
					inputData[i] = float32(rng.NormFloat64()) * 0.5
				}
				input, err := tensor.New[float32]([]int{batch, inChannels, seqLen}, inputData)
				if err != nil {
					t.Fatalf("tensor.New input: %v", err)
				}

				// Target shape: [batch, outChannels, outLen]
				targetData := make([]float32, batch*outChannels*outLen)
				for i := range targetData {
					targetData[i] = float32(rng.NormFloat64()) * 0.1
				}

				params := conv.Parameters()

				forward := func(ctx context.Context, in *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
					return conv.Forward(ctx, in)
				}
				backward := func(ctx context.Context, outputGrad *tensor.TensorNumeric[float32], in *tensor.TensorNumeric[float32]) error {
					_, err := conv.Backward(ctx, types.FullBackprop, outputGrad, in)
					return err
				}

				return forward, backward, params, input, targetData
			},
		},
	}
}

// TestCoreLayers_LossDecreases verifies that each core layer can learn:
// after Forward -> Loss -> Backward -> GradientDescent, the loss decreases.
func TestCoreLayers_LossDecreases(t *testing.T) {
	for _, tc := range buildTestCases() {
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

			// Step 5: Apply gradient descent.
			lr := float32(0.01)
			applyGradientDescent(params, lr)

			// Step 6: Forward pass again with updated parameters.
			output2, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward (after update): %v", err)
			}

			// Step 7: Compute new loss.
			loss2 := computeMSE(output2.Data(), target)
			if math.IsNaN(float64(loss2)) || math.IsInf(float64(loss2), 0) {
				t.Fatalf("updated loss is NaN or Inf: %v", loss2)
			}
			t.Logf("updated loss: %f", loss2)

			// Step 8: Verify loss decreased.
			if loss2 >= loss1 {
				t.Errorf("loss did not decrease: before=%f, after=%f", loss1, loss2)
			}
		})
	}
}

// TestCoreLayers_GradientsNonZero verifies that after a backward pass,
// all parameter gradients have at least one non-zero element.
func TestCoreLayers_GradientsNonZero(t *testing.T) {
	for _, tc := range buildTestCases() {
		t.Run(tc.name, func(t *testing.T) {
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
			ops := numeric.Float32Ops{}
			rng := rand.New(rand.NewPCG(42, 0))
			ctx := context.Background()

			forward, backward, params, input, target := tc.setup(t, engine, ops, rng)

			// Forward pass to produce output.
			output, err := forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Compute MSE gradient as the output gradient.
			outputData := output.Data()
			gradData := mseGradient(outputData, target)
			outputGrad, err := tensor.New[float32](output.Shape(), gradData)
			if err != nil {
				t.Fatalf("tensor.New outputGrad: %v", err)
			}

			// Backward pass.
			if err := backward(ctx, outputGrad, input); err != nil {
				t.Fatalf("Backward: %v", err)
			}

			// Verify each parameter has a non-zero gradient.
			for _, p := range params {
				if p.Gradient == nil {
					t.Errorf("parameter %q has nil gradient", p.Name)
					continue
				}
				if !hasNonZeroGradient(p) {
					t.Errorf("parameter %q has all-zero gradient", p.Name)
				}
			}
		})
	}
}
