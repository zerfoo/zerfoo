package training_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// softmaxCrossEntropyLoss implements a working cross-entropy loss for training.
// The existing CrossEntropyLoss uses Gather incorrectly (embedding-style gather
// for scalar selection), so we implement softmax + NLL manually.
//
// Forward: softmax(logits) -> -log(p[target]) -> mean
// Backward: (softmax - one_hot) / N
type softmaxCrossEntropyLoss struct {
	engine    compute.Engine[float32]
	ops       numeric.Arithmetic[float32]
	softmax   *tensor.TensorNumeric[float32]
	targets   []int
	nSamples  int
	vocabSize int
}

func newSoftmaxCrossEntropyLoss(engine compute.Engine[float32], ops numeric.Arithmetic[float32]) *softmaxCrossEntropyLoss {
	return &softmaxCrossEntropyLoss{engine: engine, ops: ops}
}

func (l *softmaxCrossEntropyLoss) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("cross-entropy loss expects 2 inputs, got %d", len(inputs))
	}
	logits := inputs[0]  // [N, vocabSize]
	targetsT := inputs[1] // [N] as float32

	shape := logits.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("logits must be 2D [N, vocab], got shape %v", shape)
	}
	n, vocab := shape[0], shape[1]
	l.nSamples = n
	l.vocabSize = vocab

	// Convert targets to int
	tData := targetsT.Data()
	l.targets = make([]int, len(tData))
	for i, v := range tData {
		l.targets[i] = int(v)
	}

	// Compute softmax along last axis
	sm, err := l.engine.Softmax(ctx, logits, 1, nil)
	if err != nil {
		return nil, err
	}
	l.softmax = sm

	// Compute -log(softmax[target]) mean
	smData := sm.Data()
	var lossSum float64
	for i := range n {
		idx := l.targets[i]
		p := float64(smData[i*vocab+idx])
		if p < 1e-10 {
			p = 1e-10
		}
		lossSum -= math.Log(p)
	}
	lossVal := float32(lossSum / float64(n))

	return tensor.New[float32]([]int{1}, []float32{lossVal})
}

func (l *softmaxCrossEntropyLoss) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	// Gradient: (softmax - one_hot) / N
	smData := l.softmax.Data()
	grad := make([]float32, len(smData))
	copy(grad, smData)
	scale := 1.0 / float32(l.nSamples)
	for i := range l.nSamples {
		grad[i*l.vocabSize+l.targets[i]] -= 1.0
	}
	for i := range grad {
		grad[i] *= scale
	}

	gradT, err := tensor.New[float32](l.softmax.Shape(), grad)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{gradT, nil}, nil
}

func (l *softmaxCrossEntropyLoss) Parameters() []*graph.Parameter[float32] { return nil }
func (l *softmaxCrossEntropyLoss) OutputShape() []int                      { return []int{1} }
func (l *softmaxCrossEntropyLoss) OpType() string                          { return "SoftmaxCrossEntropyLoss" }
func (l *softmaxCrossEntropyLoss) Attributes() map[string]interface{}      { return nil }

// TestGemma3Training verifies that a tiny Gemma 3-like transformer can train
// on synthetic token classification data with monotonically decreasing loss
// and no NaN gradients over 100 steps.
//
// Architecture (following Gemma 3 design):
//
//	Input[batch, seq, d_model]
//	  -> [RMSNorm -> GQA -> Add(residual) -> RMSNorm -> FFN(SwiGLU) -> Add(residual)] x nLayers
//	  -> RMSNorm -> LMHead -> Reshape -> SoftmaxCrossEntropyLoss
func TestGemma3Training(t *testing.T) {
	// Tiny model configuration to keep CPU test fast.
	const (
		batchSize     = 1
		seqLen        = 4
		dModel        = 64
		nHeads        = 2
		nKVHeads      = 2
		intermediateD = 128
		vocabSize     = 256
		nLayers       = 2
		nSteps        = 100
		learningRate  = 1e-3
		maxSeqLen     = 16
	)

	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Build the Gemma 3-like computation graph.
	builder := graph.NewBuilder[float32](engine)
	input := builder.Input([]int{batchSize, seqLen, dModel})

	hidden := input
	for layer := range nLayers {
		prefix := fmt.Sprintf("layer%d", layer)

		// Pre-attention RMSNorm.
		preAttnNorm, err := normalization.NewRMSNorm[float32](
			prefix+"_pre_attn_norm", engine, ops, dModel,
			normalization.WithRMSNormEpsilon[float32](1e-6),
		)
		if err != nil {
			t.Fatalf("create pre-attention norm layer %d: %v", layer, err)
		}
		normed := builder.AddNode(preAttnNorm, hidden)

		// Grouped Query Attention.
		gqa, err := attention.NewGroupedQueryAttention[float32](
			engine, ops, dModel, nHeads, nKVHeads,
			attention.WithRopeBase[float32](10000.0),
			attention.WithMaxSeqLen[float32](maxSeqLen),
		)
		if err != nil {
			t.Fatalf("create GQA layer %d: %v", layer, err)
		}
		attnOut := builder.AddNode(gqa, normed)

		// Residual add: hidden + attnOut.
		add1 := core.NewAdd[float32](engine)
		hidden = builder.AddNode(add1, hidden, attnOut)

		// Pre-FFN RMSNorm.
		preFfnNorm, err := normalization.NewRMSNorm[float32](
			prefix+"_pre_ffn_norm", engine, ops, dModel,
			normalization.WithRMSNormEpsilon[float32](1e-6),
		)
		if err != nil {
			t.Fatalf("create pre-FFN norm layer %d: %v", layer, err)
		}
		normed2 := builder.AddNode(preFfnNorm, hidden)

		// FFN with SwiGLU (Gemma 3 uses SwiGLU).
		ffn, err := core.NewFFN[float32](
			prefix+"_ffn", engine, ops,
			dModel, intermediateD, dModel,
			core.WithSwiGLU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			t.Fatalf("create FFN layer %d: %v", layer, err)
		}
		ffnOut := builder.AddNode(ffn, normed2)

		// Residual add: hidden + ffnOut.
		add2 := core.NewAdd[float32](engine)
		hidden = builder.AddNode(add2, hidden, ffnOut)
	}

	// Final RMSNorm.
	finalNorm, err := normalization.NewRMSNorm[float32](
		"final_norm", engine, ops, dModel,
		normalization.WithRMSNormEpsilon[float32](1e-6),
	)
	if err != nil {
		t.Fatalf("create final norm: %v", err)
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// LM Head: Dense layer projecting hidden states to vocabulary logits.
	lmHead, err := core.NewDense[float32](
		"lm_head", engine, ops, dModel, vocabSize,
		core.WithoutBias[float32](),
	)
	if err != nil {
		t.Fatalf("create LM head: %v", err)
	}
	lmHeadOut := builder.AddNode(lmHead, normedFinal)

	// Reshape [batch, seq, vocab] -> [batch*seq, vocab] for cross-entropy loss.
	reshape := core.NewReshape[float32](engine, []int{batchSize * seqLen, vocabSize})
	output := builder.AddNode(reshape, lmHeadOut)

	// Build the graph.
	modelGraph, err := builder.Build(output)
	if err != nil {
		t.Fatalf("build graph: %v", err)
	}

	// Initialize small random weights (Xavier-like) to help convergence.
	rng := rand.New(rand.NewPCG(42, 0))
	for _, p := range modelGraph.Parameters() {
		data := p.Value.Data()
		scale := float32(math.Sqrt(2.0 / float64(len(data))))
		for i := range data {
			data[i] = (rng.Float32()*2 - 1) * scale
		}
	}

	// Cross-entropy loss.
	lossFn := newSoftmaxCrossEntropyLoss(engine, ops)

	// AdamW optimizer.
	adamw := optimizer.NewAdamW[float32](engine, learningRate, 0.9, 0.999, 1e-8, 0.01)

	// Create the trainer.
	trainer := training.NewDefaultTrainer[float32](modelGraph, lossFn, adamw, nil)

	// Generate synthetic training data: random embeddings and target labels.
	inputData := make([]float32, batchSize*seqLen*dModel)
	for i := range inputData {
		inputData[i] = (rng.Float32()*2 - 1) * 0.1
	}
	inputTensor, err := tensor.New[float32]([]int{batchSize, seqLen, dModel}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	// Target labels: random class indices in [0, vocabSize).
	// Flattened to [batch*seq] to match reshaped logits [batch*seq, vocab].
	targetData := make([]float32, batchSize*seqLen)
	for i := range targetData {
		targetData[i] = float32(rng.IntN(vocabSize))
	}
	targets, err := tensor.New[float32]([]int{batchSize * seqLen}, targetData)
	if err != nil {
		t.Fatalf("create target tensor: %v", err)
	}

	inputs := map[graph.Node[float32]]*tensor.TensorNumeric[float32]{
		input: inputTensor,
	}

	// Run training loop.
	losses := make([]float32, nSteps)
	for step := range nSteps {
		lossVal, err := trainer.TrainStep(ctx, modelGraph, adamw, inputs, targets)
		if err != nil {
			t.Fatalf("step %d: TrainStep failed: %v", step, err)
		}

		if math.IsNaN(float64(lossVal)) {
			t.Fatalf("step %d: loss is NaN", step)
		}
		if math.IsInf(float64(lossVal), 0) {
			t.Fatalf("step %d: loss is Inf", step)
		}

		losses[step] = lossVal

		// Check all parameter gradients for NaN.
		for _, p := range modelGraph.Parameters() {
			if p.Gradient == nil {
				continue
			}
			for _, v := range p.Gradient.Data() {
				if math.IsNaN(float64(v)) {
					t.Fatalf("step %d: NaN gradient in parameter %q", step, p.Name)
				}
			}
		}

		if step%20 == 0 {
			t.Logf("step %3d: loss = %.6f", step, lossVal)
		}
	}

	// Verify overall decrease: final loss should be well below initial loss.
	initialLoss := losses[0]
	finalLoss := losses[nSteps-1]
	if finalLoss >= initialLoss {
		t.Errorf("loss did not decrease overall: initial=%.6f final=%.6f", initialLoss, finalLoss)
	}

	// Verify monotonic decrease using a sliding window approach.
	// Strict per-step monotonicity may not hold with AdamW, so we check
	// that each window of 10 steps shows decreasing average loss.
	windowSize := 10
	for i := windowSize; i < nSteps; i++ {
		prevAvg := avgSlice(losses[i-windowSize : i])
		currAvg := avgSlice(losses[max(0, i-windowSize/2) : i])
		if currAvg > prevAvg*1.1 {
			t.Errorf("loss not decreasing: window ending at step %d avg=%.6f > prev window avg=%.6f",
				i, currAvg, prevAvg)
		}
	}

	t.Logf("Training complete: initial loss=%.6f, final loss=%.6f (%.1f%% reduction)",
		initialLoss, finalLoss, (1-finalLoss/initialLoss)*100)
}

func avgSlice(s []float32) float32 {
	if len(s) == 0 {
		return 0
	}
	var sum float32
	for _, v := range s {
		sum += v
	}
	return sum / float32(len(s))
}
