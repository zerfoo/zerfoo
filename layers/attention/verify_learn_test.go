package attention

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

// TestAttentionHead_LossDecreases verifies that an AttentionHead can learn:
// Forward -> MSE loss -> Backward -> gradient step -> loss decreases.
func TestAttentionHead_LossDecreases(t *testing.T) {
	const (
		batch    = 2
		seqLen   = 3
		modelDim = 8
		headDim  = 4
		lr       = float32(0.01)
	)

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rng := rand.New(rand.NewPCG(42, 0))
	ctx := context.Background()

	head, err := NewAttentionHead[float32](engine, modelDim, headDim,
		WithBidirectionalAttention[float32]())
	if err != nil {
		t.Fatalf("NewAttentionHead: %v", err)
	}

	// Create input tensor: [batch, seqLen, modelDim].
	inputData := make([]float32, batch*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(rng.NormFloat64()) * 0.5
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, modelDim}, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	// Create target data: [batch, seqLen, headDim].
	targetData := make([]float32, batch*seqLen*headDim)
	for i := range targetData {
		targetData[i] = float32(rng.NormFloat64()) * 0.1
	}

	// Step 1: Forward pass.
	output, err := head.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outputData := output.Data()
	if len(outputData) != len(targetData) {
		t.Fatalf("output size %d != target size %d", len(outputData), len(targetData))
	}

	// Step 2: Compute initial MSE loss.
	loss1 := computeMSE(outputData, targetData)
	if math.IsNaN(float64(loss1)) || math.IsInf(float64(loss1), 0) {
		t.Fatalf("initial loss is NaN or Inf: %v", loss1)
	}
	t.Logf("initial loss: %f", loss1)

	// Step 3: Compute output gradient (dL/dOutput for MSE).
	gradData := mseGradient(outputData, targetData)
	outputGrad, err := tensor.New[float32](output.Shape(), gradData)
	if err != nil {
		t.Fatalf("tensor.New outputGrad: %v", err)
	}

	// Step 4: Backward pass.
	_, err = head.Backward(ctx, types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Step 5: Apply gradient descent.
	params := head.Parameters()
	applyGradientDescent(params, lr)

	// Step 6: Forward pass again with updated parameters.
	output2, err := head.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward (after update): %v", err)
	}

	// Step 7: Compute new loss.
	loss2 := computeMSE(output2.Data(), targetData)
	if math.IsNaN(float64(loss2)) || math.IsInf(float64(loss2), 0) {
		t.Fatalf("updated loss is NaN or Inf: %v", loss2)
	}
	t.Logf("updated loss: %f", loss2)

	// Step 8: Verify loss decreased.
	if loss2 >= loss1 {
		t.Errorf("loss did not decrease: before=%f, after=%f", loss1, loss2)
	}
}

// TestAttentionHead_ParameterGradientsNonZero verifies that after a backward
// pass, all parameter gradients have at least one non-zero element.
func TestAttentionHead_ParameterGradientsNonZero(t *testing.T) {
	const (
		batch    = 2
		seqLen   = 3
		modelDim = 8
		headDim  = 4
	)

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rng := rand.New(rand.NewPCG(42, 0))
	ctx := context.Background()

	head, err := NewAttentionHead[float32](engine, modelDim, headDim,
		WithBidirectionalAttention[float32]())
	if err != nil {
		t.Fatalf("NewAttentionHead: %v", err)
	}

	// Create input tensor: [batch, seqLen, modelDim].
	inputData := make([]float32, batch*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(rng.NormFloat64()) * 0.5
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, modelDim}, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	// Create target data: [batch, seqLen, headDim].
	targetData := make([]float32, batch*seqLen*headDim)
	for i := range targetData {
		targetData[i] = float32(rng.NormFloat64()) * 0.1
	}

	// Forward pass.
	output, err := head.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Compute MSE gradient as the output gradient.
	gradData := mseGradient(output.Data(), targetData)
	outputGrad, err := tensor.New[float32](output.Shape(), gradData)
	if err != nil {
		t.Fatalf("tensor.New outputGrad: %v", err)
	}

	// Backward pass.
	_, err = head.Backward(ctx, types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Verify each parameter has a non-zero gradient.
	params := head.Parameters()
	if len(params) == 0 {
		t.Fatal("AttentionHead has no parameters")
	}
	for _, p := range params {
		if p.Gradient == nil {
			t.Errorf("parameter %q has nil gradient", p.Name)
			continue
		}
		if !hasNonZeroGradient(p) {
			t.Errorf("parameter %q has all-zero gradient", p.Name)
		}
	}
}
