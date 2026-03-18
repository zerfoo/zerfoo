// Package training_test contains end-to-end training loop tests.
package training_test

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
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// squeezeBatchNode removes the leading batch dimension from [1, seqLen, dim]
// to produce [seqLen, dim], so that CrossEntropyLoss can use Gather on 2D input.
type squeezeBatchNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (s *squeezeBatchNode[T]) OpType() string                    { return "SqueezeBatch" }
func (s *squeezeBatchNode[T]) Attributes() map[string]any         { return nil }
func (s *squeezeBatchNode[T]) OutputShape() []int                 { return nil }
func (s *squeezeBatchNode[T]) Parameters() []*graph.Parameter[T]  { return nil }

func (s *squeezeBatchNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := inputs[0].Shape()
	// [1, seqLen, dim] -> [seqLen, dim]
	return s.engine.Reshape(ctx, inputs[0], shape[1:])
}

func (s *squeezeBatchNode[T]) Backward(_ context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Unsqueeze: [seqLen, dim] -> [1, seqLen, dim]
	shape := dOut.Shape()
	newShape := make([]int, len(shape)+1)
	newShape[0] = 1
	copy(newShape[1:], shape)
	reshaped, err := tensor.New[T](newShape, dOut.Data())
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{reshaped}, nil
}

// softmaxCrossEntropyNode computes cross-entropy loss from logits and integer
// targets directly on CPU without relying on engine.Gather (which is designed
// for embedding lookup, not single-element selection).
//
// Forward: logits [N, C], targets [N] -> scalar loss
// Backward: dL/dlogits = softmax(logits) - one_hot(targets), scaled by 1/N
type softmaxCrossEntropyNode struct {
	softmax []float32 // cached for backward [N*C]
	targets []int     // cached for backward [N]
	n, c    int       // dimensions
}

func (s *softmaxCrossEntropyNode) OpType() string                          { return "SoftmaxCrossEntropy" }
func (s *softmaxCrossEntropyNode) Attributes() map[string]any               { return nil }
func (s *softmaxCrossEntropyNode) OutputShape() []int                       { return []int{1} }
func (s *softmaxCrossEntropyNode) Parameters() []*graph.Parameter[float32]  { return nil }

func (s *softmaxCrossEntropyNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	logits := inputs[0] // [N, C]
	targetsT := inputs[1] // [N]

	shape := logits.Shape()
	n, c := shape[0], shape[1]
	s.n = n
	s.c = c

	logitData := logits.Data()
	targetData := targetsT.Data()

	// Convert targets to int
	s.targets = make([]int, n)
	for i := range n {
		s.targets[i] = int(targetData[i])
	}

	// Compute softmax per row and cross-entropy loss
	s.softmax = make([]float32, n*c)
	var totalLoss float64
	for i := range n {
		rowStart := i * c
		// Find max for numerical stability
		maxVal := logitData[rowStart]
		for j := 1; j < c; j++ {
			if logitData[rowStart+j] > maxVal {
				maxVal = logitData[rowStart+j]
			}
		}
		// Compute exp and sum
		var expSum float64
		for j := range c {
			s.softmax[rowStart+j] = float32(math.Exp(float64(logitData[rowStart+j] - maxVal)))
			expSum += float64(s.softmax[rowStart+j])
		}
		// Normalize
		for j := range c {
			s.softmax[rowStart+j] /= float32(expSum)
		}
		// Cross-entropy: -log(softmax[target])
		targetIdx := s.targets[i]
		prob := float64(s.softmax[rowStart+targetIdx])
		if prob < 1e-10 {
			prob = 1e-10
		}
		totalLoss -= math.Log(prob)
	}

	avgLoss := float32(totalLoss / float64(n))
	return tensor.New[float32]([]int{1}, []float32{avgLoss})
}

func (s *softmaxCrossEntropyNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	// dL/dlogits = (softmax - one_hot) / N
	grad := make([]float32, s.n*s.c)
	invN := float32(1.0 / float64(s.n))
	copy(grad, s.softmax)
	for i := range s.n {
		grad[i*s.c+s.targets[i]] -= 1.0
	}
	for i := range grad {
		grad[i] *= invN
	}

	gradTensor, err := tensor.New[float32]([]int{s.n, s.c}, grad)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{gradTensor, nil}, nil
}

// TestLlama3Training builds a tiny Llama 3 transformer (2 layers, d_model=64,
// n_heads=2, vocab=256) and trains it for 100 AdamW steps on synthetic token
// classification data, asserting monotonic loss decrease and no NaN gradients.
func TestLlama3Training(t *testing.T) {
	const (
		numLayers  = 2
		modelDim   = 64
		numHeads   = 2
		numKVHeads = 2
		ffnDim     = 128 // intermediate size
		vocabSize  = 256
		seqLen     = 8
		batchSize  = 1
		maxSeqLen  = 64
		ropeTheta  = 500000.0 // Llama 3 RoPE theta
		numSteps   = 100
	)

	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	builder := graph.NewBuilder[float32](engine)

	// Input: [batch, seqLen, modelDim] — pre-embedded hidden states.
	input := builder.Input([]int{batchSize, seqLen, modelDim})

	hidden := input
	for layer := 0; layer < numLayers; layer++ {
		// Pre-attention RMSNorm
		preAttnNorm, err := normalization.NewRMSNorm[float32](
			"llama3_pre_attn_norm_"+itoa(layer), engine, ops, modelDim,
			normalization.WithRMSNormEpsilon[float32](1e-5),
		)
		if err != nil {
			t.Fatalf("layer %d: pre-attn norm: %v", layer, err)
		}
		normed := builder.AddNode(preAttnNorm, hidden)

		// Grouped Query Attention with Llama 3 RoPE theta
		gqa, err := attention.NewGroupedQueryAttention[float32](
			engine, ops, modelDim, numHeads, numKVHeads,
			attention.WithRopeBase[float32](ropeTheta),
			attention.WithMaxSeqLen[float32](maxSeqLen),
		)
		if err != nil {
			t.Fatalf("layer %d: gqa: %v", layer, err)
		}
		attnOut := builder.AddNode(gqa, normed)

		// Residual add: hidden + attnOut
		add1 := core.NewAdd[float32](engine)
		hidden = builder.AddNode(add1, hidden, attnOut)

		// Pre-FFN RMSNorm
		preFfnNorm, err := normalization.NewRMSNorm[float32](
			"llama3_pre_ffn_norm_"+itoa(layer), engine, ops, modelDim,
			normalization.WithRMSNormEpsilon[float32](1e-5),
		)
		if err != nil {
			t.Fatalf("layer %d: pre-ffn norm: %v", layer, err)
		}
		normed2 := builder.AddNode(preFfnNorm, hidden)

		// FFN (SwiGLU, no bias — Llama style)
		ffn, err := core.NewFFN[float32](
			"llama3_ffn_"+itoa(layer), engine, ops,
			modelDim, ffnDim, modelDim,
			core.WithSwiGLU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			t.Fatalf("layer %d: ffn: %v", layer, err)
		}
		ffnOut := builder.AddNode(ffn, normed2)

		// Residual add: hidden + ffnOut
		add2 := core.NewAdd[float32](engine)
		hidden = builder.AddNode(add2, hidden, ffnOut)
	}

	// Final RMSNorm
	finalNorm, err := normalization.NewRMSNorm[float32](
		"llama3_final_norm", engine, ops, modelDim,
		normalization.WithRMSNormEpsilon[float32](1e-5),
	)
	if err != nil {
		t.Fatalf("final norm: %v", err)
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// LM Head: project to vocab size (no bias, Llama style)
	lmHead, err := core.NewDense[float32](
		"llama3_lm_head", engine, ops, modelDim, vocabSize,
		core.WithoutBias[float32](),
	)
	if err != nil {
		t.Fatalf("lm head: %v", err)
	}
	lmOut := builder.AddNode(lmHead, normedFinal)

	// Squeeze batch dimension for CrossEntropyLoss compatibility:
	// [1, seqLen, vocabSize] -> [seqLen, vocabSize]
	squeeze := &squeezeBatchNode[float32]{engine: engine}
	output := builder.AddNode(squeeze, lmOut)

	// Build computation graph
	g, err := builder.Build(output)
	if err != nil {
		t.Fatalf("graph build: %v", err)
	}

	// Initialize weights with Xavier uniform for stable training
	rng := rand.New(rand.NewPCG(42, 0))
	for _, p := range g.Parameters() {
		data := p.Value.Data()
		fan := len(data)
		scale := float32(math.Sqrt(2.0 / float64(fan)))
		for i := range data {
			data[i] = float32(rng.Float64()*2-1) * scale
		}
	}

	// Loss and optimizer
	crossEntropy := &softmaxCrossEntropyNode{}
	adamw := optimizer.NewAdamW[float32](engine, 1e-3, 0.9, 0.999, 1e-8, 0.01)
	trainer := training.NewDefaultTrainer[float32](g, crossEntropy, adamw, nil)

	// Generate synthetic data: random inputs, random target token IDs
	inputData := make([]float32, batchSize*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(rng.Float64()*2 - 1) * 0.1
	}
	inputTensor, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, inputData)
	if err != nil {
		t.Fatalf("input tensor: %v", err)
	}

	targetData := make([]float32, seqLen)
	for i := range targetData {
		targetData[i] = float32(rng.IntN(vocabSize))
	}
	targets, err := tensor.New[float32]([]int{seqLen}, targetData)
	if err != nil {
		t.Fatalf("target tensor: %v", err)
	}

	inputs := map[graph.Node[float32]]*tensor.TensorNumeric[float32]{
		input: inputTensor,
	}

	// Train for numSteps steps
	var prevLoss float32
	for step := 0; step < numSteps; step++ {
		lossVal, err := trainer.TrainStep(ctx, g, adamw, inputs, targets)
		if err != nil {
			t.Fatalf("step %d: train step failed: %v", step, err)
		}

		// Check for NaN loss
		if math.IsNaN(float64(lossVal)) || math.IsInf(float64(lossVal), 0) {
			t.Fatalf("step %d: loss is NaN or Inf: %v", step, lossVal)
		}

		// Check for NaN gradients
		for _, p := range g.Parameters() {
			if p.Gradient == nil {
				continue
			}
			for j, v := range p.Gradient.Data() {
				if math.IsNaN(float64(v)) {
					t.Fatalf("step %d: NaN gradient in param %q at index %d", step, p.Name, j)
				}
			}
		}

		// Check monotonic decrease (allow small tolerance for numerical noise)
		if step > 0 && lossVal > prevLoss+1e-4 {
			t.Errorf("step %d: loss increased: %.6f -> %.6f (delta=%.6f)",
				step, prevLoss, lossVal, lossVal-prevLoss)
		}

		if step%20 == 0 {
			t.Logf("step %3d: loss=%.6f", step, lossVal)
		}

		prevLoss = lossVal
	}

	t.Logf("final loss after %d steps: %.6f", numSteps, prevLoss)
}

// itoa converts an int to a string without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}
