package tabular

import (
	"context"
	"fmt"
	"sort"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TabNetConfig holds the configuration for a TabNet model.
type TabNetConfig struct {
	InputDim             int
	OutputDim            int
	NSteps               int
	RelaxationFactor     float64
	SparsityCoefficient  float64
	FeatureTransformerDim int // hidden dim for feature transformer blocks
}

// attentiveTransformer holds weights for one attention step.
type attentiveTransformer struct {
	fc mlpLayer // linear: featureTransformerDim -> inputDim
}

// featureTransformer holds shared + step-specific FC layers with GLU.
type featureTransformer struct {
	shared      mlpLayer // linear: inputDim -> featureTransformerDim*2 (for GLU)
	stepSpecific mlpLayer // linear: featureTransformerDim -> featureTransformerDim*2 (for GLU)
}

// TabNet implements the TabNet architecture with sequential attention and sparsemax.
type TabNet struct {
	config             TabNetConfig
	engine             compute.Engine[float32]
	ops                numeric.Arithmetic[float32]
	initialBN          mlpLayer // batch norm approximated as learnable scale+bias
	attentiveSteps     []attentiveTransformer
	featureTransformers []featureTransformer
	outputHead         mlpLayer
	lastAttentionMasks []*tensor.TensorNumeric[float32]
}

// NewTabNet creates a new TabNet model with the given configuration.
func NewTabNet(config TabNetConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*TabNet, error) {
	if config.InputDim <= 0 {
		return nil, fmt.Errorf("tabular: InputDim must be positive, got %d", config.InputDim)
	}
	if config.OutputDim <= 0 {
		return nil, fmt.Errorf("tabular: OutputDim must be positive, got %d", config.OutputDim)
	}
	if config.NSteps <= 0 {
		return nil, fmt.Errorf("tabular: NSteps must be positive, got %d", config.NSteps)
	}
	if config.RelaxationFactor <= 0 {
		return nil, fmt.Errorf("tabular: RelaxationFactor must be positive, got %f", config.RelaxationFactor)
	}
	if config.SparsityCoefficient < 0 {
		return nil, fmt.Errorf("tabular: SparsityCoefficient must be non-negative, got %f", config.SparsityCoefficient)
	}
	if config.FeatureTransformerDim <= 0 {
		return nil, fmt.Errorf("tabular: FeatureTransformerDim must be positive, got %d", config.FeatureTransformerDim)
	}

	t := &TabNet{
		config: config,
		engine: engine,
		ops:    ops,
	}

	// Initial BN: learnable scale (ones) + bias (zeros) per feature.
	bnScale := make([]float32, config.InputDim)
	for i := range bnScale {
		bnScale[i] = 1.0
	}
	scaleT, err := tensor.New[float32]([]int{1, config.InputDim}, bnScale)
	if err != nil {
		return nil, fmt.Errorf("tabular: bn scale: %w", err)
	}
	biasT, err := tensor.New[float32]([]int{1, config.InputDim}, make([]float32, config.InputDim))
	if err != nil {
		return nil, fmt.Errorf("tabular: bn bias: %w", err)
	}
	t.initialBN = mlpLayer{weights: scaleT, biases: biasT}

	ftDim := config.FeatureTransformerDim

	// Build per-step components.
	t.attentiveSteps = make([]attentiveTransformer, config.NSteps)
	t.featureTransformers = make([]featureTransformer, config.NSteps)

	for i := 0; i < config.NSteps; i++ {
		// Attentive transformer: maps from ftDim -> inputDim (to produce attention mask).
		// For step 0, input to attentive transformer is the BN'd features projected to ftDim,
		// but we simplify: attentive transformer FC maps inputDim -> inputDim.
		attnFC, err := newMLPLayer(config.InputDim, config.InputDim)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d attentive: %w", i, err)
		}
		t.attentiveSteps[i] = attentiveTransformer{fc: attnFC}

		// Feature transformer: shared (inputDim -> ftDim*2) + step-specific (ftDim -> ftDim*2).
		shared, err := newMLPLayer(config.InputDim, ftDim*2)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d shared ft: %w", i, err)
		}
		stepFC, err := newMLPLayer(ftDim, ftDim*2)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d specific ft: %w", i, err)
		}
		t.featureTransformers[i] = featureTransformer{
			shared:       shared,
			stepSpecific: stepFC,
		}
	}

	// Output head: ftDim -> OutputDim.
	head, err := newMLPLayer(ftDim, config.OutputDim)
	if err != nil {
		return nil, fmt.Errorf("tabular: output head: %w", err)
	}
	t.outputHead = head

	return t, nil
}

// Forward runs the TabNet forward pass on input features [batch, inputDim].
// Returns logits [batch, outputDim].
func (t *TabNet) Forward(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	if len(shape) != 2 || shape[1] != t.config.InputDim {
		return nil, fmt.Errorf("tabular: expected input shape [batch, %d], got %v", t.config.InputDim, shape)
	}
	batch := shape[0]

	// Apply initial batch normalization (element-wise scale + bias).
	x, err := t.engine.Mul(ctx, input, t.initialBN.weights)
	if err != nil {
		return nil, fmt.Errorf("tabular: bn mul: %w", err)
	}
	x, err = t.engine.Add(ctx, x, t.initialBN.biases)
	if err != nil {
		return nil, fmt.Errorf("tabular: bn add: %w", err)
	}

	// Initialize prior scales to ones [batch, inputDim].
	priorData := make([]float32, batch*t.config.InputDim)
	for i := range priorData {
		priorData[i] = 1.0
	}
	priorScales, err := tensor.New[float32]([]int{batch, t.config.InputDim}, priorData)
	if err != nil {
		return nil, err
	}

	// Aggregate output across steps.
	aggData := make([]float32, batch*t.config.FeatureTransformerDim)
	aggregate, err := tensor.New[float32]([]int{batch, t.config.FeatureTransformerDim}, aggData)
	if err != nil {
		return nil, err
	}

	t.lastAttentionMasks = make([]*tensor.TensorNumeric[float32], t.config.NSteps)

	// In TabNet, the attentive transformer always operates on the BN'd features.
	// The step output (from the feature transformer) feeds into aggregation,
	// not back into the attentive transformer.

	for step := 0; step < t.config.NSteps; step++ {
		// Attentive transformer: h = FC(x), then prior * h, then sparsemax.
		h, err := t.linearForward(ctx, x, t.attentiveSteps[step].fc)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d attn fc: %w", step, err)
		}

		// Multiply by prior scales.
		masked, err := t.engine.Mul(ctx, priorScales, h)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d prior mul: %w", step, err)
		}

		// Apply sparsemax to get attention mask.
		attnMask, err := sparsemax(masked)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d sparsemax: %w", step, err)
		}
		t.lastAttentionMasks[step] = attnMask

		// Update prior scales: prior *= (relaxation - attnMask).
		relaxFactor := float32(t.config.RelaxationFactor)
		relaxScalar, err := t.engine.MulScalar(ctx, attnMask, -1.0)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d relax neg: %w", step, err)
		}
		relaxScalar, err = t.engine.AddScalar(ctx, relaxScalar, relaxFactor)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d relax add: %w", step, err)
		}
		priorScales, err = t.engine.Mul(ctx, priorScales, relaxScalar)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d prior update: %w", step, err)
		}

		// Apply attention mask to BN'd features.
		maskedFeatures, err := t.engine.Mul(ctx, attnMask, x)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d mask features: %w", step, err)
		}

		// Feature transformer: shared FC with GLU.
		sharedOut, err := t.linearForward(ctx, maskedFeatures, t.featureTransformers[step].shared)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d shared fc: %w", step, err)
		}
		gluOut, err := t.glu(ctx, sharedOut)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d shared glu: %w", step, err)
		}

		// Step-specific FC with GLU.
		stepOut, err := t.linearForward(ctx, gluOut, t.featureTransformers[step].stepSpecific)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d specific fc: %w", step, err)
		}
		stepResult, err := t.glu(ctx, stepOut)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d specific glu: %w", step, err)
		}

		// ReLU on step result.
		stepResult, err = t.engine.UnaryOp(ctx, stepResult, t.ops.ReLU)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d relu: %w", step, err)
		}

		// Aggregate += stepResult.
		aggregate, err = t.engine.Add(ctx, aggregate, stepResult)
		if err != nil {
			return nil, fmt.Errorf("tabular: step %d aggregate: %w", step, err)
		}
	}

	// Final output: linear head on aggregate.
	logits, err := t.linearForward(ctx, aggregate, t.outputHead)
	if err != nil {
		return nil, fmt.Errorf("tabular: output head: %w", err)
	}

	return logits, nil
}

// Predict runs inference and returns a Direction and confidence, matching the Model API.
func (t *TabNet) Predict(features []float64) (Direction, float64, error) {
	if len(features) != t.config.InputDim {
		return Flat, 0, fmt.Errorf("tabular: expected %d features, got %d", t.config.InputDim, len(features))
	}

	ctx := context.Background()

	f32 := make([]float32, len(features))
	for i, v := range features {
		f32[i] = float32(v)
	}
	input, err := tensor.New[float32]([]int{1, t.config.InputDim}, f32)
	if err != nil {
		return Flat, 0, err
	}

	logits, err := t.Forward(ctx, input)
	if err != nil {
		return Flat, 0, err
	}

	// Softmax to get probabilities (output dim must be >= 3 for direction).
	probs, err := t.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return Flat, 0, err
	}

	probData := probs.Data()
	if len(probData) < 3 {
		return Flat, 0, fmt.Errorf("tabular: output dim must be >= 3 for direction prediction, got %d", len(probData))
	}

	dir, conf := argmax(probData[:3])
	return dir, conf, nil
}

// AttentionMasks returns the attention masks from the last forward pass.
// Each mask has shape [batch, inputDim] and represents the feature importance at each step.
// Returns nil if no forward pass has been run.
func (t *TabNet) AttentionMasks() []*tensor.TensorNumeric[float32] {
	return t.lastAttentionMasks
}

// FeatureImportance returns the aggregate feature importance from the last forward pass.
// The result is the sum of attention masks across all steps, shape [batch, inputDim].
// Returns nil, error if no forward pass has been run.
func (t *TabNet) FeatureImportance(ctx context.Context) (*tensor.TensorNumeric[float32], error) {
	if t.lastAttentionMasks == nil || len(t.lastAttentionMasks) == 0 {
		return nil, fmt.Errorf("tabular: no attention masks available, run Forward first")
	}

	importance := t.lastAttentionMasks[0]
	var err error
	for i := 1; i < len(t.lastAttentionMasks); i++ {
		importance, err = t.engine.Add(ctx, importance, t.lastAttentionMasks[i])
		if err != nil {
			return nil, fmt.Errorf("tabular: aggregating attention masks: %w", err)
		}
	}
	return importance, nil
}

// linearForward computes a linear transformation via functional.Linear.
// mlpLayer stores weights as [in, out], so we transpose to [out, in] for
// functional.Linear which expects [out_features, in_features].
func (t *TabNet) linearForward(ctx context.Context, x *tensor.TensorNumeric[float32], l mlpLayer) (*tensor.TensorNumeric[float32], error) {
	wT, err := t.engine.Transpose(ctx, l.weights, []int{1, 0})
	if err != nil {
		return nil, err
	}
	return functional.Linear(ctx, t.engine, x, wT, l.biases)
}

// glu applies the Gated Linear Unit: given input [batch, 2*dim], splits into
// two halves and returns first_half * sigmoid(second_half).
func (t *TabNet) glu(ctx context.Context, x *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := x.Shape()
	if len(shape) != 2 || shape[1]%2 != 0 {
		return nil, fmt.Errorf("tabular: GLU input must be [batch, 2*dim], got %v", shape)
	}

	parts, err := t.engine.Split(ctx, x, 2, 1)
	if err != nil {
		return nil, fmt.Errorf("tabular: GLU split: %w", err)
	}

	// Sigmoid on second half.
	gate, err := t.engine.UnaryOp(ctx, parts[1], t.ops.Sigmoid)
	if err != nil {
		return nil, fmt.Errorf("tabular: GLU sigmoid: %w", err)
	}

	return t.engine.Mul(ctx, parts[0], gate)
}

// sparsemax computes the sparsemax activation along the last axis of a 2D tensor.
// Sparsemax projects each row onto the probability simplex, producing sparse outputs.
func sparsemax(input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("tabular: sparsemax expects 2D input, got %dD", len(shape))
	}

	batch := shape[0]
	dim := shape[1]
	data := input.Data()
	result := make([]float32, len(data))

	for b := 0; b < batch; b++ {
		offset := b * dim
		row := data[offset : offset+dim]

		// Sort in descending order.
		sorted := make([]float32, dim)
		copy(sorted, row)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] > sorted[j] })

		// Find the support: largest k such that 1 + k*z_k > cumsum(z_1..z_k).
		cumsum := float32(0)
		k := 0
		for i := 0; i < dim; i++ {
			cumsum += sorted[i]
			if 1+float32(i+1)*sorted[i] > cumsum {
				k = i + 1
			}
		}

		// Compute threshold tau.
		cumsum = 0
		for i := 0; i < k; i++ {
			cumsum += sorted[i]
		}
		tau := (cumsum - 1) / float32(k)

		// Apply sparsemax: max(z - tau, 0).
		for i := 0; i < dim; i++ {
			v := row[i] - tau
			if v > 0 {
				result[offset+i] = v
			}
		}
	}

	return tensor.New[float32](shape, result)
}

// SparsemaxDirect exposes sparsemax for direct testing.
func SparsemaxDirect(input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return sparsemax(input)
}

