package transformer

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
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

// zeroGradients resets all parameter gradients to zero.
func zeroGradients(params []*graph.Parameter[float32]) {
	for _, p := range params {
		if p.Gradient == nil {
			continue
		}
		p.ClearGradient()
	}
}

// newTransformerBlockForTest creates a TransformerBlock with small dimensions
// suitable for gradient checking.
func newTransformerBlockForTest(t *testing.T) *Block[float32] {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const (
		modelDim = 16
		ffnDim   = 32
		numHeads = 2
	)

	attn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("NewGlobalAttention: %v", err)
	}

	block, err := NewTransformerBlock[float32](engine, ops, modelDim, ffnDim, attn)
	if err != nil {
		t.Fatalf("NewTransformerBlock: %v", err)
	}

	// Scale parameters to moderate range for numerical stability.
	for _, p := range block.Parameters() {
		d := p.Value.Data()
		for i := range d {
			d[i] *= 0.1
		}
	}

	return block
}

func TestTransformerBlock_LossDecreases(t *testing.T) {
	ctx := context.Background()
	block := newTransformerBlockForTest(t)

	const (
		batch    = 1
		seqLen   = 4
		modelDim = 16
	)

	inputData := make([]float32, batch*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, modelDim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	target := make([]float32, batch*seqLen*modelDim)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 30.0
	}

	params := block.Parameters()
	const lr = float32(0.01)
	const steps = 20

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		zeroGradients(params)

		output, fErr := block.Forward(ctx, input)
		if fErr != nil {
			t.Fatalf("step %d: Forward: %v", step, fErr)
		}

		loss := computeMSE(output.Data(), target)
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d: loss is NaN or Inf", step)
		}

		if step == 0 {
			initialLoss = loss
			t.Logf("step 0: loss=%.6f", loss)
		}
		if step == steps-1 {
			finalLoss = loss
			t.Logf("step %d: loss=%.6f", step, loss)
		}

		gradData := mseGradient(output.Data(), target)
		outputGrad, gErr := tensor.New[float32](output.Shape(), gradData)
		if gErr != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, gErr)
		}

		_, bErr := block.Backward(ctx, types.FullBackprop, outputGrad, input)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		applyGradientDescent(params, lr)
	}

	if finalLoss >= initialLoss {
		t.Errorf("TransformerBlock loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("TransformerBlock loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestTransformerBlock_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	block := newTransformerBlockForTest(t)

	const (
		batch    = 1
		seqLen   = 4
		modelDim = 16
	)

	inputData := make([]float32, batch*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, modelDim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	onesData := make([]float32, len(output.Data()))
	for i := range onesData {
		onesData[i] = 1.0
	}
	dOut, err := tensor.New[float32](output.Shape(), onesData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	_, err = block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for _, p := range block.Parameters() {
		if !hasNonZeroGradient(p) {
			t.Errorf("TransformerBlock parameter %q has all-zero gradient", p.Name)
		}
	}
}
