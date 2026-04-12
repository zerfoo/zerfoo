package crossasset

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// cpuEngine is a package-level float32 CPU computation engine used by the
// cross-asset model's forward pass helpers.
var cpuEngine = compute.NewCPUEngine[float32](numeric.Float32Ops{})

// cpuOps provides float32 arithmetic for activation functions.
var cpuOps = numeric.Float32Ops{}

// Direction represents a trading signal direction.
type Direction int

const (
	// Long indicates a buy signal.
	Long Direction = iota
	// Short indicates a sell signal.
	Short
	// Flat indicates no signal.
	Flat
)

// Config holds the configuration for a cross-asset attention model.
type Config struct {
	NSources          int
	FeaturesPerSource int
	DModel            int
	NHeads            int
	NLayers           int
	DropoutRate       float64
	LearningRate      float64
}

// TrainConfig holds training hyperparameters.
type TrainConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
}

// layer holds weights for one cross-attention transformer layer.
type layer struct {
	// Cross-attention: Q from self, K/V from all sources.
	qW, kW, vW []float32 // [DModel, DModel]
	outW        []float32 // [DModel, DModel]
	lnGamma     []float32 // [DModel]
	lnBeta      []float32 // [DModel]

	// Feed-forward network.
	ffnW1     []float32 // [DModel, 4*DModel]
	ffnB1     []float32 // [4*DModel]
	ffnW2     []float32 // [4*DModel, DModel]
	ffnB2     []float32 // [DModel]
	ffnGamma  []float32 // [DModel]
	ffnBeta   []float32 // [DModel]
}

// Model implements a cross-attention model for multi-source features.
type Model struct {
	config Config
	layers []layer

	// engine is the compute engine used for forward/backward passes.
	// Defaults to the package-level cpuEngine.
	engine compute.Engine[float32]

	// Input projection: features_per_source -> d_model, per source.
	inputW [][]float32 // [NSources][FeaturesPerSource * DModel]
	inputB [][]float32 // [NSources][DModel]

	// Classification head: d_model -> 3.
	headW []float32 // [DModel * 3]
	headB []float32 // [3]
}

// NewModel creates a new cross-asset attention model with the given configuration.
func NewModel(config Config) *Model {
	m := &Model{config: config, engine: cpuEngine}

	// Input projections per source.
	m.inputW = make([][]float32, config.NSources)
	m.inputB = make([][]float32, config.NSources)
	for s := 0; s < config.NSources; s++ {
		m.inputW[s] = heInit(config.FeaturesPerSource, config.DModel)
		m.inputB[s] = make([]float32, config.DModel)
	}

	// Transformer layers.
	m.layers = make([]layer, config.NLayers)
	for i := 0; i < config.NLayers; i++ {
		m.layers[i] = newLayer(config.DModel)
	}

	// Classification head.
	m.headW = heInit(config.DModel, 3)
	m.headB = make([]float32, 3)

	return m
}

// SetEngine sets the compute engine used for forward and backward passes.
// When a GPU engine is provided, all computation happens on the GPU.
func (m *Model) SetEngine(engine compute.Engine[float32]) {
	m.engine = engine
}

// Forward processes features through the cross-attention model.
// features shape: [n_sources][features_per_source].
// Returns: [n_sources][d_model].
func (m *Model) Forward(features [][]float32) ([][]float32, error) {
	if len(features) != m.config.NSources {
		return nil, fmt.Errorf("crossasset: expected %d sources, got %d", m.config.NSources, len(features))
	}
	for i, f := range features {
		if len(f) != m.config.FeaturesPerSource {
			return nil, fmt.Errorf("crossasset: source %d: expected %d features, got %d", i, m.config.FeaturesPerSource, len(f))
		}
	}

	ns := m.config.NSources
	dm := m.config.DModel
	fps := m.config.FeaturesPerSource
	ctx := context.Background()
	eng := m.engine

	// Project each source: [1, fps] @ [fps, dm] + bias → [1, dm].
	// Each source has its own weight matrix.
	projected := make([][]float32, ns)
	for s := 0; s < ns; s++ {
		srcT, err := tensor.New[float32]([]int{1, fps}, features[s])
		panicOnErr("Forward: src tensor", err)
		wT, err := tensor.New[float32]([]int{fps, dm}, m.inputW[s])
		panicOnErr("Forward: input weight tensor", err)
		bT, err := tensor.New[float32]([]int{1, dm}, m.inputB[s])
		panicOnErr("Forward: input bias tensor", err)

		result, err := eng.MatMul(ctx, srcT, wT)
		panicOnErr("Forward: input matmul", err)
		result, err = eng.Add(ctx, result, bT)
		panicOnErr("Forward: input add bias", err)
		projected[s] = make([]float32, dm)
		copy(projected[s], result.Data())
	}

	// Apply cross-attention layers.
	x := slicesToTensor(projected, ns, dm)
	for _, l := range m.layers {
		var err error
		x, err = m.forwardLayerEngine(ctx, eng, x, l)
		if err != nil {
			return nil, err
		}
	}

	return tensorToSlices(x, ns, dm), nil
}

// AttentionWeights computes the attention weight matrix showing how much each
// source attends to each other source. Returns [n_sources][n_sources] where
// result[i][j] is how much source i attends to source j. Weights sum to 1
// across the attended (j) dimension.
func (m *Model) AttentionWeights(features [][]float32) ([][]float32, error) {
	if len(features) != m.config.NSources {
		return nil, fmt.Errorf("crossasset: expected %d sources, got %d", m.config.NSources, len(features))
	}
	for i, f := range features {
		if len(f) != m.config.FeaturesPerSource {
			return nil, fmt.Errorf("crossasset: source %d: expected %d features, got %d", i, m.config.FeaturesPerSource, len(f))
		}
	}

	ns := m.config.NSources
	dm := m.config.DModel
	fps := m.config.FeaturesPerSource
	nHeads := m.config.NHeads
	headDim := dm / nHeads
	ctx := context.Background()
	eng := m.engine

	// Project inputs via engine.
	projected := make([][]float32, ns)
	for s := 0; s < ns; s++ {
		srcT, err := tensor.New[float32]([]int{1, fps}, features[s])
		panicOnErr("AttentionWeights: src tensor", err)
		wT, err := tensor.New[float32]([]int{fps, dm}, m.inputW[s])
		panicOnErr("AttentionWeights: weight tensor", err)
		bT, err := tensor.New[float32]([]int{1, dm}, m.inputB[s])
		panicOnErr("AttentionWeights: bias tensor", err)
		result, err := eng.MatMul(ctx, srcT, wT)
		panicOnErr("AttentionWeights: matmul", err)
		result, err = eng.Add(ctx, result, bT)
		panicOnErr("AttentionWeights: add bias", err)
		projected[s] = make([]float32, dm)
		copy(projected[s], result.Data())
	}

	x := slicesToTensor(projected, ns, dm)

	// Use only the first layer for attention weights.
	l := m.layers[0]

	qW, err := tensor.New[float32]([]int{dm, dm}, l.qW)
	panicOnErr("AttentionWeights: qW tensor", err)
	kW, err := tensor.New[float32]([]int{dm, dm}, l.kW)
	panicOnErr("AttentionWeights: kW tensor", err)

	// Q, K projections: [ns, dm] @ [dm, dm] = [ns, dm].
	q, err := eng.MatMul(ctx, x, qW)
	panicOnErr("AttentionWeights: Q matmul", err)
	k, err := eng.MatMul(ctx, x, kW)
	panicOnErr("AttentionWeights: K matmul", err)

	// Reshape and transpose for multi-head: [ns, dm] → [nHeads, ns, headDim].
	q, err = eng.Reshape(ctx, q, []int{ns, nHeads, headDim}, nil)
	panicOnErr("AttentionWeights: Q reshape", err)
	q, err = eng.Transpose(ctx, q, []int{1, 0, 2})
	panicOnErr("AttentionWeights: Q transpose", err)
	k, err = eng.Reshape(ctx, k, []int{ns, nHeads, headDim}, nil)
	panicOnErr("AttentionWeights: K reshape", err)
	k, err = eng.Transpose(ctx, k, []int{1, 0, 2})
	panicOnErr("AttentionWeights: K transpose", err)

	// scores = Q @ K^T / sqrt(headDim), shape [nHeads, ns, ns].
	kT, err := eng.Transpose(ctx, k, []int{0, 2, 1})
	panicOnErr("AttentionWeights: K^T", err)
	scores, err := eng.MatMul(ctx, q, kT)
	panicOnErr("AttentionWeights: scores matmul", err)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores, err = eng.MulScalar(ctx, scores, scale)
	panicOnErr("AttentionWeights: scale", err)

	// Softmax over last dim → [nHeads, ns, ns].
	weights, err := functional.Softmax(ctx, eng, scores, -1)
	panicOnErr("AttentionWeights: softmax", err)

	// Average over heads.
	// ReduceSum over axis 0 → [ns, ns], then divide by nHeads.
	sumW, err := eng.ReduceSum(ctx, weights, 0, false)
	panicOnErr("AttentionWeights: reduce sum", err)
	avgW, err := eng.DivScalar(ctx, sumW, float32(nHeads))
	panicOnErr("AttentionWeights: div scalar", err)

	return tensorToSlices(avgW, ns, ns), nil
}

// Train trains the model on the given data.
// data shape: [n_samples][n_sources][features_per_source].
// labels shape: [n_samples][n_sources] with values in {0=Long, 1=Short, 2=Flat}.
func (m *Model) Train(data [][][]float32, labels [][]int, tc TrainConfig) error {
	if len(data) == 0 {
		return fmt.Errorf("crossasset: train: no data provided")
	}
	if len(data) != len(labels) {
		return fmt.Errorf("crossasset: train: data length %d != labels length %d", len(data), len(labels))
	}
	if tc.Epochs <= 0 {
		return fmt.Errorf("crossasset: train: Epochs must be positive")
	}
	if tc.LearningRate <= 0 {
		tc.LearningRate = m.config.LearningRate
	}
	if tc.LearningRate <= 0 {
		tc.LearningRate = 0.001
	}

	ns := m.config.NSources
	dm := m.config.DModel
	lr := tc.LearningRate

	// Initialize AdamW state.
	adamState, err := newAdamState(m, lr)
	if err != nil {
		return err
	}

	for epoch := 0; epoch < tc.Epochs; epoch++ {
		perm := rand.Perm(len(data))

		for batchStart := 0; batchStart < len(data); batchStart += tc.BatchSize {
			batchEnd := batchStart + tc.BatchSize
			if batchEnd > len(data) {
				batchEnd = len(data)
			}
			batchSize := batchEnd - batchStart

			// Zero all gradients.
			dHeadW := make([]float32, len(m.headW))
			dHeadB := make([]float32, len(m.headB))
			dLayers := make([]layer, len(m.layers))
			for li := range dLayers {
				dLayers[li] = zeroLayer(dm)
			}
			dInputW := make([][]float32, ns)
			dInputB := make([][]float32, ns)
			for s := range ns {
				dInputW[s] = make([]float32, len(m.inputW[s]))
				dInputB[s] = make([]float32, len(m.inputB[s]))
			}

			for bi := 0; bi < batchSize; bi++ {
				idx := perm[batchStart+bi]
				sample := data[idx]
				sampleLabels := labels[idx]

				// Forward with caching: input projection via engine.
				fps := m.config.FeaturesPerSource
				x := make([][]float32, ns)
				for s := range ns {
					x[s] = make([]float32, dm)
					matVecMulEngine(m.engine, x[s], m.inputW[s], sample[s], fps, dm)
					vecAddEngine(m.engine, x[s], m.inputB[s])
				}
				// Forward through layers with caches.
				layerCaches := make([]*cpuLayerCache, len(m.layers))
				for li := range m.layers {
					var cache *cpuLayerCache
					x, cache = m.forwardLayerCached(m.engine, x, m.layers[li])
					layerCaches[li] = cache
				}

				// Head forward + loss gradient.
				scaleFactor := float32(1.0 / float64(batchSize*ns))
				dx := make([][]float32, ns)
				for s := range ns {
					logits := make([]float32, 3)
					matVecMulEngine(m.engine, logits, m.headW, x[s], dm, 3)
					vecAddEngine(m.engine, logits, m.headB)

					probs := softmaxEngine(m.engine, logits)

					dLogits := make([]float32, 3)
					copy(dLogits, probs)
					if sampleLabels[s] >= 0 && sampleLabels[s] < 3 {
						dLogits[sampleLabels[s]] -= 1.0
					}
					for j := range dLogits {
						dLogits[j] *= scaleFactor
					}

					// Head weight gradients.
					for d := range dm {
						for c := range 3 {
							dHeadW[d*3+c] += x[s][d] * dLogits[c]
						}
					}
					for c := range 3 {
						dHeadB[c] += dLogits[c]
					}

					// dx from head: dLogits @ headW^T.
					dx[s] = make([]float32, dm)
					for d := range dm {
						for c := range 3 {
							dx[s][d] += dLogits[c] * m.headW[d*3+c]
						}
					}
				}

				// Backward through layers in reverse.
				for li := len(m.layers) - 1; li >= 0; li-- {
					dx = m.backwardLayer(m.engine, dx, layerCaches[li], &m.layers[li], &dLayers[li])
				}

				// Input projection backward.
				for s := range ns {
					for d := range fps {
						for c := range dm {
							dInputW[s][d*dm+c] += sample[s][d] * dx[s][c]
						}
					}
					for c := range dm {
						dInputB[s][c] += dx[s][c]
					}
				}
			}

			// AdamW update: collect all param/grad pairs and update.
			if err := adamWUpdateAll(m, dHeadW, dHeadB, dLayers, dInputW, dInputB, adamState); err != nil {
				return err
			}
		}
	}

	return nil
}

// Predict returns per-source direction and confidence.
// features shape: [n_sources][features_per_source].
// Returns: directions [n_sources], confidences [n_sources].
func (m *Model) Predict(features [][]float32) ([]int, []float32, error) {
	outputs, err := m.Forward(features)
	if err != nil {
		return nil, nil, err
	}

	ns := m.config.NSources
	dirs := make([]int, ns)
	confs := make([]float32, ns)

	for s := 0; s < ns; s++ {
		logits := make([]float32, 3)
		matVecMulEngine(m.engine, logits, m.headW, outputs[s], m.config.DModel, 3)
		vecAddEngine(m.engine, logits, m.headB)

		probs := softmaxEngine(m.engine, logits)
		bestIdx := 0
		bestProb := probs[0]
		for j := 1; j < 3; j++ {
			if probs[j] > bestProb {
				bestIdx = j
				bestProb = probs[j]
			}
		}
		dirs[s] = bestIdx
		confs[s] = bestProb
	}

	return dirs, confs, nil
}

// forwardLayerEngine applies one cross-attention layer using Engine[T] ops.
// x is a tensor [ns, dm].
func (m *Model) forwardLayerEngine(
	ctx context.Context,
	eng compute.Engine[float32],
	x *tensor.TensorNumeric[float32],
	l layer,
) (*tensor.TensorNumeric[float32], error) {
	ns := m.config.NSources
	dm := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dm / nHeads
	ffnDim := dm * 4

	// Weight tensors.
	qW, err := tensor.New[float32]([]int{dm, dm}, l.qW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: qW tensor: %w", err)
	}
	kW, err := tensor.New[float32]([]int{dm, dm}, l.kW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: kW tensor: %w", err)
	}
	vW, err := tensor.New[float32]([]int{dm, dm}, l.vW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: vW tensor: %w", err)
	}

	// Q, K, V projections: [ns, dm] @ [dm, dm] = [ns, dm].
	q, err := eng.MatMul(ctx, x, qW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: Q matmul: %w", err)
	}
	k, err := eng.MatMul(ctx, x, kW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: K matmul: %w", err)
	}
	v, err := eng.MatMul(ctx, x, vW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: V matmul: %w", err)
	}

	// Reshape for multi-head attention: [ns, dm] → [ns, nHeads, headDim].
	q, err = eng.Reshape(ctx, q, []int{ns, nHeads, headDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("crossasset: Q reshape: %w", err)
	}
	k, err = eng.Reshape(ctx, k, []int{ns, nHeads, headDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("crossasset: K reshape: %w", err)
	}
	v, err = eng.Reshape(ctx, v, []int{ns, nHeads, headDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("crossasset: V reshape: %w", err)
	}

	// Transpose to [nHeads, ns, headDim] for batched attention.
	q, err = eng.Transpose(ctx, q, []int{1, 0, 2})
	if err != nil {
		return nil, fmt.Errorf("crossasset: Q transpose: %w", err)
	}
	k, err = eng.Transpose(ctx, k, []int{1, 0, 2})
	if err != nil {
		return nil, fmt.Errorf("crossasset: K transpose: %w", err)
	}
	v, err = eng.Transpose(ctx, v, []int{1, 0, 2})
	if err != nil {
		return nil, fmt.Errorf("crossasset: V transpose: %w", err)
	}

	// Scaled dot-product attention: scores = Q @ K^T / sqrt(headDim).
	kT, err := eng.Transpose(ctx, k, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("crossasset: K^T transpose: %w", err)
	}
	scores, err := eng.MatMul(ctx, q, kT)
	if err != nil {
		return nil, fmt.Errorf("crossasset: attention scores: %w", err)
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores, err = eng.MulScalar(ctx, scores, scale)
	if err != nil {
		return nil, fmt.Errorf("crossasset: scale scores: %w", err)
	}

	// Softmax over last dimension.
	weights, err := functional.Softmax(ctx, eng, scores, -1)
	if err != nil {
		return nil, fmt.Errorf("crossasset: attention softmax: %w", err)
	}

	// Weighted sum: attn = weights @ V, shape [nHeads, ns, headDim].
	attnOut, err := eng.MatMul(ctx, weights, v)
	if err != nil {
		return nil, fmt.Errorf("crossasset: attention matmul: %w", err)
	}

	// Transpose back: [nHeads, ns, headDim] → [ns, nHeads, headDim].
	attnOut, err = eng.Transpose(ctx, attnOut, []int{1, 0, 2})
	if err != nil {
		return nil, fmt.Errorf("crossasset: attn transpose back: %w", err)
	}

	// Reshape to [ns, dm].
	attnOut, err = eng.Reshape(ctx, attnOut, []int{ns, dm}, nil)
	if err != nil {
		return nil, fmt.Errorf("crossasset: attn reshape: %w", err)
	}

	// Output projection: [ns, dm] @ [dm, dm] = [ns, dm].
	outW, err := tensor.New[float32]([]int{dm, dm}, l.outW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: outW tensor: %w", err)
	}
	projOut, err := eng.MatMul(ctx, attnOut, outW)
	if err != nil {
		return nil, fmt.Errorf("crossasset: output projection: %w", err)
	}

	// Residual + LayerNorm.
	res1, err := eng.Add(ctx, x, projOut)
	if err != nil {
		return nil, fmt.Errorf("crossasset: residual1: %w", err)
	}
	gT1, err := tensor.New[float32]([]int{1, dm}, l.lnGamma)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ln gamma tensor: %w", err)
	}
	bT1, err := tensor.New[float32]([]int{1, dm}, l.lnBeta)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ln beta tensor: %w", err)
	}
	normed, err := functional.LayerNorm(ctx, eng, res1, gT1, bT1, 1e-5)
	if err != nil {
		return nil, fmt.Errorf("crossasset: layernorm1: %w", err)
	}

	// FFN: Linear1 + bias + GELU.
	ffnW1, err := tensor.New[float32]([]int{dm, ffnDim}, l.ffnW1)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffnW1 tensor: %w", err)
	}
	ffnB1, err := tensor.New[float32]([]int{1, ffnDim}, l.ffnB1)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffnB1 tensor: %w", err)
	}
	hidden, err := eng.MatMul(ctx, normed, ffnW1)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffn1 matmul: %w", err)
	}
	hidden, err = eng.Add(ctx, hidden, ffnB1)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffn1 add bias: %w", err)
	}
	hidden, err = functional.GELU(ctx, eng, cpuOps, hidden)
	if err != nil {
		return nil, fmt.Errorf("crossasset: gelu: %w", err)
	}

	// FFN: Linear2 + bias.
	ffnW2, err := tensor.New[float32]([]int{ffnDim, dm}, l.ffnW2)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffnW2 tensor: %w", err)
	}
	ffnB2, err := tensor.New[float32]([]int{1, dm}, l.ffnB2)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffnB2 tensor: %w", err)
	}
	ffnOut, err := eng.MatMul(ctx, hidden, ffnW2)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffn2 matmul: %w", err)
	}
	ffnOut, err = eng.Add(ctx, ffnOut, ffnB2)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffn2 add bias: %w", err)
	}

	// Residual + LayerNorm.
	res2, err := eng.Add(ctx, normed, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("crossasset: residual2: %w", err)
	}
	gT2, err := tensor.New[float32]([]int{1, dm}, l.ffnGamma)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffn gamma tensor: %w", err)
	}
	bT2, err := tensor.New[float32]([]int{1, dm}, l.ffnBeta)
	if err != nil {
		return nil, fmt.Errorf("crossasset: ffn beta tensor: %w", err)
	}
	out, err := functional.LayerNorm(ctx, eng, res2, gT2, bT2, 1e-5)
	if err != nil {
		return nil, fmt.Errorf("crossasset: layernorm2: %w", err)
	}

	return out, nil
}

// newLayer initializes weights for one cross-attention layer.
func newLayer(dModel int) layer {
	ffnHidden := dModel * 4
	return layer{
		qW:       heInit(dModel, dModel),
		kW:       heInit(dModel, dModel),
		vW:       heInit(dModel, dModel),
		outW:     heInit(dModel, dModel),
		lnGamma:  ones(dModel),
		lnBeta:   make([]float32, dModel),
		ffnW1:    heInit(dModel, ffnHidden),
		ffnB1:    make([]float32, ffnHidden),
		ffnW2:    heInit(ffnHidden, dModel),
		ffnB2:    make([]float32, dModel),
		ffnGamma: ones(dModel),
		ffnBeta:  make([]float32, dModel),
	}
}

// heInit creates a flat weight matrix [in*out] with He/Kaiming initialization.
func heInit(in, out int) []float32 {
	scale := math.Sqrt(2.0 / float64(in))
	w := make([]float32, in*out)
	for i := range w {
		w[i] = float32(rand.NormFloat64() * scale)
	}
	return w
}

// ones returns a slice of length n filled with 1.0.
func ones(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1.0
	}
	return s
}

// matVecMulEngine computes dst = W @ src where W is [in, out] stored row-major.
// dst must have length out, src must have length in.
// Arithmetic is routed through the provided Engine[float32] via MatMul.
func matVecMulEngine(eng compute.Engine[float32], dst, w, src []float32, in, out int) {
	ctx := context.Background()

	srcT, err := tensor.New[float32]([]int{1, in}, src)
	panicOnErr("matVecMulEngine: src tensor", err)

	wT, err := tensor.New[float32]([]int{in, out}, w)
	panicOnErr("matVecMulEngine: weight tensor", err)

	result, err := eng.MatMul(ctx, srcT, wT)
	panicOnErr("matVecMulEngine: matmul", err)

	copy(dst, result.Data())
}

// vecAddEngine computes dst[i] += src[i] via the provided engine.
func vecAddEngine(eng compute.Engine[float32], dst, src []float32) {
	ctx := context.Background()

	dstT, err := tensor.New[float32]([]int{len(dst)}, dst)
	panicOnErr("vecAddEngine: dst tensor", err)
	srcT, err := tensor.New[float32]([]int{len(src)}, src)
	panicOnErr("vecAddEngine: src tensor", err)

	result, err := eng.Add(ctx, dstT, srcT)
	panicOnErr("vecAddEngine: add", err)

	copy(dst, result.Data())
}

// softmaxEngine computes softmax over a slice via the provided engine.
func softmaxEngine(eng compute.Engine[float32], x []float32) []float32 {
	ctx := context.Background()

	t, err := tensor.New[float32]([]int{len(x)}, x)
	panicOnErr("softmaxEngine: tensor", err)

	result, err := functional.Softmax(ctx, eng, t, 0)
	panicOnErr("softmaxEngine: softmax", err)

	out := make([]float32, len(x))
	copy(out, result.Data())
	return out
}
