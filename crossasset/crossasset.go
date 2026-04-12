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

	// Input projection: features_per_source -> d_model, per source.
	inputW [][]float32 // [NSources][FeaturesPerSource * DModel]
	inputB [][]float32 // [NSources][DModel]

	// Classification head: d_model -> 3.
	headW []float32 // [DModel * 3]
	headB []float32 // [3]
}

// NewModel creates a new cross-asset attention model with the given configuration.
func NewModel(config Config) *Model {
	m := &Model{config: config}

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

	// Project each source's features to d_model.
	x := make([][]float32, ns)
	for s := 0; s < ns; s++ {
		x[s] = make([]float32, dm)
		matVecMul(x[s], m.inputW[s], features[s], m.config.FeaturesPerSource, dm)
		vecAdd(x[s], m.inputB[s])
	}

	// Apply cross-attention layers.
	for _, l := range m.layers {
		var err error
		x, err = m.forwardLayer(x, l)
		if err != nil {
			return nil, err
		}
	}

	return x, nil
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
	nHeads := m.config.NHeads
	headDim := dm / nHeads

	// Project inputs.
	x := make([][]float32, ns)
	for s := 0; s < ns; s++ {
		x[s] = make([]float32, dm)
		matVecMul(x[s], m.inputW[s], features[s], m.config.FeaturesPerSource, dm)
		vecAdd(x[s], m.inputB[s])
	}

	// Use only the first layer for attention weights.
	l := m.layers[0]

	// Compute Q, K for all sources.
	qs := make([][]float32, ns)
	ks := make([][]float32, ns)
	for s := 0; s < ns; s++ {
		qs[s] = make([]float32, dm)
		ks[s] = make([]float32, dm)
		matVecMul(qs[s], l.qW, x[s], dm, dm)
		matVecMul(ks[s], l.kW, x[s], dm, dm)
	}

	// Average attention weights across all heads.
	attn := make([][]float32, ns)
	for i := 0; i < ns; i++ {
		attn[i] = make([]float32, ns)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := 0; h < nHeads; h++ {
		hStart := h * headDim
		hEnd := hStart + headDim

		for i := 0; i < ns; i++ {
			scores := make([]float32, ns)
			for j := 0; j < ns; j++ {
				dot := float32(0)
				for d := hStart; d < hEnd; d++ {
					dot += qs[i][d] * ks[j][d]
				}
				scores[j] = dot * scale
			}
			probs := softmax(scores)
			for j := 0; j < ns; j++ {
				attn[i][j] += probs[j] / float32(nHeads)
			}
		}
	}

	return attn, nil
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

				// Forward with caching: input projection.
				x := make([][]float32, ns)
				for s := range ns {
					x[s] = make([]float32, dm)
					matVecMul(x[s], m.inputW[s], sample[s], m.config.FeaturesPerSource, dm)
					vecAdd(x[s], m.inputB[s])
				}
				// Forward through layers with caches.
				layerCaches := make([]*cpuLayerCache, len(m.layers))
				for li := range m.layers {
					var cache *cpuLayerCache
					x, cache = m.forwardLayerCached(x, m.layers[li])
					layerCaches[li] = cache
				}

				// Head forward + loss gradient.
				scaleFactor := float32(1.0 / float64(batchSize*ns))
				dx := make([][]float32, ns)
				for s := range ns {
					logits := make([]float32, 3)
					matVecMul(logits, m.headW, x[s], dm, 3)
					vecAdd(logits, m.headB)

					probs := softmax(logits)

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
					dx = m.backwardLayer(dx, layerCaches[li], &m.layers[li], &dLayers[li])
				}

				// Input projection backward.
				fps := m.config.FeaturesPerSource
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
		matVecMul(logits, m.headW, outputs[s], m.config.DModel, 3)
		vecAdd(logits, m.headB)

		probs := softmax(logits)
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

// forwardLayer applies one cross-attention layer.
func (m *Model) forwardLayer(x [][]float32, l layer) ([][]float32, error) {
	ns := m.config.NSources
	dm := m.config.DModel
	nHeads := m.config.NHeads
	headDim := dm / nHeads

	// Compute Q, K, V for all sources.
	qs := make([][]float32, ns)
	ks := make([][]float32, ns)
	vs := make([][]float32, ns)
	for s := 0; s < ns; s++ {
		qs[s] = make([]float32, dm)
		ks[s] = make([]float32, dm)
		vs[s] = make([]float32, dm)
		matVecMul(qs[s], l.qW, x[s], dm, dm)
		matVecMul(ks[s], l.kW, x[s], dm, dm)
		matVecMul(vs[s], l.vW, x[s], dm, dm)
	}

	// Cross-attention: each source attends to ALL sources.
	attnOut := make([][]float32, ns)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for i := 0; i < ns; i++ {
		// Concatenated head outputs.
		concat := make([]float32, dm)

		for h := 0; h < nHeads; h++ {
			hStart := h * headDim
			hEnd := hStart + headDim

			// Compute attention scores for source i attending to all sources.
			scores := make([]float32, ns)
			for j := 0; j < ns; j++ {
				dot := float32(0)
				for d := hStart; d < hEnd; d++ {
					dot += qs[i][d] * ks[j][d]
				}
				scores[j] = dot * scale
			}

			weights := softmax(scores)

			// Weighted sum of values.
			for d := hStart; d < hEnd; d++ {
				val := float32(0)
				for j := 0; j < ns; j++ {
					val += weights[j] * vs[j][d]
				}
				concat[d] = val
			}
		}

		// Output projection.
		attnOut[i] = make([]float32, dm)
		matVecMul(attnOut[i], l.outW, concat, dm, dm)
	}

	// Residual + LayerNorm.
	normed := make([][]float32, ns)
	for i := 0; i < ns; i++ {
		res := make([]float32, dm)
		for d := 0; d < dm; d++ {
			res[d] = x[i][d] + attnOut[i][d]
		}
		normed[i] = layerNorm(res, l.lnGamma, l.lnBeta)
	}

	// FFN + residual + LayerNorm.
	out := make([][]float32, ns)
	ffnHidden := dm * 4
	for i := 0; i < ns; i++ {
		// First linear + GELU.
		hidden := make([]float32, ffnHidden)
		matVecMul(hidden, l.ffnW1, normed[i], dm, ffnHidden)
		vecAdd(hidden, l.ffnB1)
		for d := range hidden {
			hidden[d] = gelu(hidden[d])
		}

		// Second linear.
		ffnOut := make([]float32, dm)
		matVecMul(ffnOut, l.ffnW2, hidden, ffnHidden, dm)
		vecAdd(ffnOut, l.ffnB2)

		// Residual + LayerNorm.
		res := make([]float32, dm)
		for d := 0; d < dm; d++ {
			res[d] = normed[i][d] + ffnOut[d]
		}
		out[i] = layerNorm(res, l.ffnGamma, l.ffnBeta)
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

// matVecMul computes dst = W @ src where W is [in, out] stored row-major.
// dst must have length out, src must have length in.
// Arithmetic is routed through the CPU Engine[float32] via MatMul.
func matVecMul(dst, w, src []float32, in, out int) {
	ctx := context.Background()

	// src as [1, in] row vector.
	srcT, err := tensor.New[float32]([]int{1, in}, src)
	if err != nil {
		panic(fmt.Sprintf("crossasset.matVecMul: src tensor: %v", err))
	}

	// w as [in, out] matrix.
	wT, err := tensor.New[float32]([]int{in, out}, w)
	if err != nil {
		panic(fmt.Sprintf("crossasset.matVecMul: weight tensor: %v", err))
	}

	// result = src @ W, shape [1, out].
	result, err := cpuEngine.MatMul(ctx, srcT, wT)
	if err != nil {
		panic(fmt.Sprintf("crossasset.matVecMul: matmul: %v", err))
	}

	copy(dst, result.Data())
}

// vecAdd computes dst[i] += src[i].
func vecAdd(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// softmax computes softmax over a slice.
// Arithmetic is routed through the CPU Engine[float32] via functional.Softmax.
func softmax(x []float32) []float32 {
	ctx := context.Background()

	t, err := tensor.New[float32]([]int{len(x)}, x)
	if err != nil {
		panic(fmt.Sprintf("crossasset.softmax: tensor: %v", err))
	}

	result, err := functional.Softmax(ctx, cpuEngine, t, 0)
	if err != nil {
		panic(fmt.Sprintf("crossasset.softmax: softmax: %v", err))
	}

	out := make([]float32, len(x))
	copy(out, result.Data())
	return out
}

// layerNorm applies layer normalization.
// Arithmetic is routed through the CPU Engine[float32] via functional.LayerNorm.
func layerNorm(x, gamma, beta []float32) []float32 {
	ctx := context.Background()
	n := len(x)

	xT, err := tensor.New[float32]([]int{1, n}, x)
	if err != nil {
		panic(fmt.Sprintf("crossasset.layerNorm: x tensor: %v", err))
	}

	gT, err := tensor.New[float32]([]int{1, n}, gamma)
	if err != nil {
		panic(fmt.Sprintf("crossasset.layerNorm: gamma tensor: %v", err))
	}

	bT, err := tensor.New[float32]([]int{1, n}, beta)
	if err != nil {
		panic(fmt.Sprintf("crossasset.layerNorm: beta tensor: %v", err))
	}

	result, err := functional.LayerNorm(ctx, cpuEngine, xT, gT, bT, 1e-5)
	if err != nil {
		panic(fmt.Sprintf("crossasset.layerNorm: layernorm: %v", err))
	}

	out := make([]float32, n)
	copy(out, result.Data())
	return out
}

// gelu computes the GELU activation function.
// Arithmetic is routed through the CPU Engine[float32] via functional.GELU.
func gelu(x float32) float32 {
	ctx := context.Background()

	t, err := tensor.New[float32]([]int{1}, []float32{x})
	if err != nil {
		panic(fmt.Sprintf("crossasset.gelu: tensor: %v", err))
	}

	result, err := functional.GELU(ctx, cpuEngine, cpuOps, t)
	if err != nil {
		panic(fmt.Sprintf("crossasset.gelu: gelu: %v", err))
	}

	return result.Data()[0]
}
