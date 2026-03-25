package embeddings

import (
	"context"
	"math"
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

// zeroGradients resets all parameter gradients to zero.
func zeroGradients(params []*graph.Parameter[float32]) {
	for _, p := range params {
		if p.Gradient == nil {
			continue
		}
		p.ClearGradient()
	}
}

// ---------------------------------------------------------------------------
// TokenEmbedding verify-learn tests
// ---------------------------------------------------------------------------

func TestTokenEmbedding_LossDecreases(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const (
		vocabSize    = 8
		embeddingDim = 4
		seqLen       = 3
	)

	emb, err := NewTokenEmbedding[float32](engine, vocabSize, embeddingDim)
	if err != nil {
		t.Fatalf("NewTokenEmbedding: %v", err)
	}

	// Token IDs as float32 (the layer converts internally).
	idsData := []float32{0, 2, 5}
	ids, err := tensor.New[float32]([]int{seqLen}, idsData)
	if err != nil {
		t.Fatalf("make ids: %v", err)
	}

	// Target output: [seqLen, embeddingDim].
	target := make([]float32, seqLen*embeddingDim)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 30.0
	}

	params := emb.Parameters()
	const lr = float32(0.1)
	const steps = 20

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		zeroGradients(params)

		output, fErr := emb.Forward(ctx, ids)
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

		_, bErr := emb.Backward(ctx, types.FullBackprop, outputGrad)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		applyGradientDescent(params, lr)
	}

	if finalLoss >= initialLoss {
		t.Errorf("TokenEmbedding loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("TokenEmbedding loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestTokenEmbedding_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	emb, err := NewTokenEmbedding[float32](engine, 8, 4)
	if err != nil {
		t.Fatalf("NewTokenEmbedding: %v", err)
	}

	idsData := []float32{0, 2, 5}
	ids, err := tensor.New[float32]([]int{3}, idsData)
	if err != nil {
		t.Fatalf("make ids: %v", err)
	}

	output, err := emb.Forward(ctx, ids)
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

	_, err = emb.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for _, p := range emb.Parameters() {
		if !hasNonZeroGradient(p) {
			t.Errorf("TokenEmbedding parameter %q has all-zero gradient", p.Name)
		}
	}
}

// ---------------------------------------------------------------------------
// RotaryPositionalEmbedding verify-learn tests
//
// RoPE has no trainable parameters (Parameters() returns nil), so we verify
// that gradients flow through correctly: the backward pass should produce
// non-zero input gradients.
// ---------------------------------------------------------------------------

func TestRotaryPositionalEmbedding_GradientPassthrough(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const (
		batch   = 1
		seqLen  = 4
		headDim = 8
	)

	rope, err := NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, seqLen)
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding: %v", err)
	}

	inputData := make([]float32, batch*seqLen*headDim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, headDim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	output, err := rope.Forward(ctx, input)
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

	grads, err := rope.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) == 0 || grads[0] == nil {
		t.Fatal("RoPE backward returned no input gradient")
	}

	hasNonZero := false
	for _, v := range grads[0].Data() {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("RoPE backward produced all-zero input gradient")
	}
}
