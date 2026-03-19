package tabular

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// FTTransformerConfig holds the configuration for an FTTransformer.
type FTTransformerConfig struct {
	NumFeatures int     // number of numeric input features
	DToken      int     // embedding dimension per feature token
	NHeads      int     // number of attention heads
	NLayers     int     // number of transformer encoder layers
	DFFN        int     // hidden dimension of the feed-forward network
	DropoutRate float64 // dropout rate (unused at inference, reserved for training)
}

// ftTransformerLayer holds the weights for one transformer encoder layer.
type ftTransformerLayer struct {
	// Self-attention weights: Q, K, V projections and output projection.
	wQ mlpLayer // [DToken, DToken]
	wK mlpLayer // [DToken, DToken]
	wV mlpLayer // [DToken, DToken]
	wO mlpLayer // [DToken, DToken]

	// Layer norm 1 (pre-attention): gamma and beta, shape [1, DToken].
	ln1Gamma *tensor.TensorNumeric[float32]
	ln1Beta  *tensor.TensorNumeric[float32]

	// FFN: two linear layers.
	ffn1 mlpLayer // [DToken, DFFN]
	ffn2 mlpLayer // [DFFN, DToken]

	// Layer norm 2 (pre-FFN): gamma and beta, shape [1, DToken].
	ln2Gamma *tensor.TensorNumeric[float32]
	ln2Beta  *tensor.TensorNumeric[float32]
}

// FTTransformer implements the Feature Tokenizer + Transformer architecture
// for tabular data. Each numeric feature is tokenized via a learned embedding,
// a CLS token is prepended, and the sequence is processed by a stack of
// transformer encoder layers. The CLS token output feeds a linear head for
// 3-class classification (Long/Short/Flat).
type FTTransformer struct {
	config FTTransformerConfig
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]

	// Feature tokenizer: one embedding vector per feature, shape [NumFeatures, DToken].
	featureEmbeddings *tensor.TensorNumeric[float32]
	// Feature biases, shape [NumFeatures, DToken].
	featureBiases *tensor.TensorNumeric[float32]
	// CLS token, shape [1, DToken].
	clsToken *tensor.TensorNumeric[float32]

	// Transformer encoder layers.
	layers []ftTransformerLayer

	// Output head: CLS token -> 3 classes.
	head mlpLayer
}

// NewFTTransformer creates a new FTTransformer with the given configuration.
func NewFTTransformer(config FTTransformerConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*FTTransformer, error) {
	if config.NumFeatures <= 0 {
		return nil, fmt.Errorf("tabular: NumFeatures must be positive, got %d", config.NumFeatures)
	}
	if config.DToken <= 0 {
		return nil, fmt.Errorf("tabular: DToken must be positive, got %d", config.DToken)
	}
	if config.NHeads <= 0 {
		return nil, fmt.Errorf("tabular: NHeads must be positive, got %d", config.NHeads)
	}
	if config.DToken%config.NHeads != 0 {
		return nil, fmt.Errorf("tabular: DToken (%d) must be divisible by NHeads (%d)", config.DToken, config.NHeads)
	}
	if config.NLayers <= 0 {
		return nil, fmt.Errorf("tabular: NLayers must be positive, got %d", config.NLayers)
	}
	if config.DFFN <= 0 {
		return nil, fmt.Errorf("tabular: DFFN must be positive, got %d", config.DFFN)
	}
	if config.DropoutRate < 0 || config.DropoutRate >= 1 {
		return nil, fmt.Errorf("tabular: DropoutRate must be in [0, 1), got %f", config.DropoutRate)
	}

	ft := &FTTransformer{
		config: config,
		engine: engine,
		ops:    ops,
	}

	// Initialize feature tokenizer embeddings with Xavier/Glorot initialization.
	scale := float32(math.Sqrt(2.0 / float64(1+config.DToken)))
	embData := make([]float32, config.NumFeatures*config.DToken)
	for i := range embData {
		embData[i] = float32(rand.NormFloat64()) * scale
	}
	var err error
	ft.featureEmbeddings, err = tensor.New[float32]([]int{config.NumFeatures, config.DToken}, embData)
	if err != nil {
		return nil, fmt.Errorf("tabular: feature embeddings: %w", err)
	}

	biasData := make([]float32, config.NumFeatures*config.DToken)
	ft.featureBiases, err = tensor.New[float32]([]int{config.NumFeatures, config.DToken}, biasData)
	if err != nil {
		return nil, fmt.Errorf("tabular: feature biases: %w", err)
	}

	// CLS token initialized with Xavier.
	clsData := make([]float32, config.DToken)
	clsScale := float32(math.Sqrt(1.0 / float64(config.DToken)))
	for i := range clsData {
		clsData[i] = float32(rand.NormFloat64()) * clsScale
	}
	ft.clsToken, err = tensor.New[float32]([]int{1, config.DToken}, clsData)
	if err != nil {
		return nil, fmt.Errorf("tabular: cls token: %w", err)
	}

	// Build transformer layers.
	ft.layers = make([]ftTransformerLayer, config.NLayers)
	for i := 0; i < config.NLayers; i++ {
		layer, layerErr := newFTTransformerLayer(config.DToken, config.DFFN)
		if layerErr != nil {
			return nil, fmt.Errorf("tabular: transformer layer %d: %w", i, layerErr)
		}
		ft.layers[i] = layer
	}

	// Output head: DToken -> 3 classes.
	ft.head, err = newMLPLayer(config.DToken, 3)
	if err != nil {
		return nil, fmt.Errorf("tabular: output head: %w", err)
	}

	return ft, nil
}

// newFTTransformerLayer creates a single transformer encoder layer with
// initialized weights.
func newFTTransformerLayer(dToken, dFFN int) (ftTransformerLayer, error) {
	var layer ftTransformerLayer
	var err error

	// Attention projections.
	layer.wQ, err = newMLPLayer(dToken, dToken)
	if err != nil {
		return layer, fmt.Errorf("wQ: %w", err)
	}
	layer.wK, err = newMLPLayer(dToken, dToken)
	if err != nil {
		return layer, fmt.Errorf("wK: %w", err)
	}
	layer.wV, err = newMLPLayer(dToken, dToken)
	if err != nil {
		return layer, fmt.Errorf("wV: %w", err)
	}
	layer.wO, err = newMLPLayer(dToken, dToken)
	if err != nil {
		return layer, fmt.Errorf("wO: %w", err)
	}

	// Layer norm parameters (initialized to gamma=1, beta=0).
	gammaData := make([]float32, dToken)
	for i := range gammaData {
		gammaData[i] = 1.0
	}

	layer.ln1Gamma, err = tensor.New[float32]([]int{1, dToken}, gammaData)
	if err != nil {
		return layer, fmt.Errorf("ln1 gamma: %w", err)
	}
	betaData := make([]float32, dToken)
	layer.ln1Beta, err = tensor.New[float32]([]int{1, dToken}, betaData)
	if err != nil {
		return layer, fmt.Errorf("ln1 beta: %w", err)
	}

	gammaData2 := make([]float32, dToken)
	for i := range gammaData2 {
		gammaData2[i] = 1.0
	}
	layer.ln2Gamma, err = tensor.New[float32]([]int{1, dToken}, gammaData2)
	if err != nil {
		return layer, fmt.Errorf("ln2 gamma: %w", err)
	}
	betaData2 := make([]float32, dToken)
	layer.ln2Beta, err = tensor.New[float32]([]int{1, dToken}, betaData2)
	if err != nil {
		return layer, fmt.Errorf("ln2 beta: %w", err)
	}

	// FFN layers.
	layer.ffn1, err = newMLPLayer(dToken, dFFN)
	if err != nil {
		return layer, fmt.Errorf("ffn1: %w", err)
	}
	layer.ffn2, err = newMLPLayer(dFFN, dToken)
	if err != nil {
		return layer, fmt.Errorf("ffn2: %w", err)
	}

	return layer, nil
}

// Forward runs the FTTransformer forward pass on a batch of inputs.
// Input shape: [batch, NumFeatures]. Output shape: [batch, 3] (logits).
func (ft *FTTransformer) Forward(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	if len(shape) != 2 || shape[1] != ft.config.NumFeatures {
		return nil, fmt.Errorf("tabular: expected input shape [batch, %d], got %v", ft.config.NumFeatures, shape)
	}
	batch := shape[0]
	nf := ft.config.NumFeatures
	dt := ft.config.DToken
	seqLen := nf + 1

	inputData := input.Data()
	embData := ft.featureEmbeddings.Data()
	biasData := ft.featureBiases.Data()

	// Process each sample independently through the transformer.
	allLogits := make([]float32, batch*3)
	for b := 0; b < batch; b++ {
		// Tokenize features for this sample.
		tokenData := make([]float32, nf*dt)
		for i := 0; i < nf; i++ {
			fi := inputData[b*nf+i]
			for j := 0; j < dt; j++ {
				tokenData[i*dt+j] = fi*embData[i*dt+j] + biasData[i*dt+j]
			}
		}
		tokens, err := tensor.New[float32]([]int{nf, dt}, tokenData)
		if err != nil {
			return nil, fmt.Errorf("tabular: tokenize sample %d: %w", b, err)
		}

		// Prepend CLS token -> [seqLen, DToken].
		x, err := ft.engine.Concat(ctx, []*tensor.TensorNumeric[float32]{ft.clsToken, tokens}, 0)
		if err != nil {
			return nil, fmt.Errorf("tabular: concat cls sample %d: %w", b, err)
		}

		// Transformer encoder layers.
		for i, layer := range ft.layers {
			x, err = ft.transformerForward(ctx, x, layer, seqLen)
			if err != nil {
				return nil, fmt.Errorf("tabular: layer %d sample %d: %w", i, b, err)
			}
		}

		// Extract CLS token (first row) -> [1, DToken].
		clsData := x.Data()[:dt]
		cls, err := tensor.New[float32]([]int{1, dt}, clsData)
		if err != nil {
			return nil, fmt.Errorf("tabular: extract cls sample %d: %w", b, err)
		}

		// Linear head -> [1, 3].
		logits, err := ft.linearForward(ctx, cls, ft.head)
		if err != nil {
			return nil, fmt.Errorf("tabular: head sample %d: %w", b, err)
		}

		copy(allLogits[b*3:(b+1)*3], logits.Data())
	}

	return tensor.New[float32]([]int{batch, 3}, allLogits)
}

// Predict runs inference on the given features and returns a Direction and
// confidence score. The features slice must have length equal to NumFeatures.
func (ft *FTTransformer) Predict(features []float64) (Direction, float64, error) {
	if len(features) != ft.config.NumFeatures {
		return Flat, 0, fmt.Errorf("tabular: expected %d features, got %d", ft.config.NumFeatures, len(features))
	}

	ctx := context.Background()

	f32 := make([]float32, len(features))
	for i, v := range features {
		f32[i] = float32(v)
	}
	input, err := tensor.New[float32]([]int{1, ft.config.NumFeatures}, f32)
	if err != nil {
		return Flat, 0, err
	}

	logits, err := ft.Forward(ctx, input)
	if err != nil {
		return Flat, 0, err
	}

	probs, err := ft.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return Flat, 0, err
	}

	dir, conf := argmax(probs.Data())
	return dir, conf, nil
}

// transformerForward applies one transformer encoder layer.
// Input shape: [seqLen, DToken], output shape: [seqLen, DToken].
func (ft *FTTransformer) transformerForward(ctx context.Context, x *tensor.TensorNumeric[float32], layer ftTransformerLayer, seqLen int) (*tensor.TensorNumeric[float32], error) {
	dToken := ft.config.DToken
	nHeads := ft.config.NHeads
	dHead := dToken / nHeads

	// Pre-norm 1: layer norm before attention.
	normed, err := ft.layerNorm(ctx, x, layer.ln1Gamma, layer.ln1Beta, seqLen, dToken)
	if err != nil {
		return nil, fmt.Errorf("ln1: %w", err)
	}

	// Self-attention: Q, K, V projections.
	q, err := ft.linearForward(ctx, normed, layer.wQ) // [seqLen, DToken]
	if err != nil {
		return nil, fmt.Errorf("q proj: %w", err)
	}
	k, err := ft.linearForward(ctx, normed, layer.wK)
	if err != nil {
		return nil, fmt.Errorf("k proj: %w", err)
	}
	v, err := ft.linearForward(ctx, normed, layer.wV)
	if err != nil {
		return nil, fmt.Errorf("v proj: %w", err)
	}

	// Multi-head attention: process each head separately, then concatenate.
	// For single-sample inference, we work with [seqLen, DToken] tensors.
	// Split into heads: reshape to per-head views and compute attention.
	headOutputs := make([]*tensor.TensorNumeric[float32], nHeads)
	qData := q.Data()
	kData := k.Data()
	vData := v.Data()

	invSqrtDk := float32(1.0 / math.Sqrt(float64(dHead)))

	for h := 0; h < nHeads; h++ {
		// Extract head h: columns [h*dHead : (h+1)*dHead] from each row.
		qh := make([]float32, seqLen*dHead)
		kh := make([]float32, seqLen*dHead)
		vh := make([]float32, seqLen*dHead)
		for s := 0; s < seqLen; s++ {
			copy(qh[s*dHead:(s+1)*dHead], qData[s*dToken+h*dHead:s*dToken+(h+1)*dHead])
			copy(kh[s*dHead:(s+1)*dHead], kData[s*dToken+h*dHead:s*dToken+(h+1)*dHead])
			copy(vh[s*dHead:(s+1)*dHead], vData[s*dToken+h*dHead:s*dToken+(h+1)*dHead])
		}

		qhT, err := tensor.New[float32]([]int{seqLen, dHead}, qh)
		if err != nil {
			return nil, err
		}
		khT, err := tensor.New[float32]([]int{seqLen, dHead}, kh)
		if err != nil {
			return nil, err
		}
		vhT, err := tensor.New[float32]([]int{seqLen, dHead}, vh)
		if err != nil {
			return nil, err
		}

		// Attention scores: Q_h @ K_h^T / sqrt(d_head) -> [seqLen, seqLen]
		khTransposed, err := ft.engine.Transpose(ctx, khT, []int{1, 0})
		if err != nil {
			return nil, err
		}
		scores, err := ft.engine.MatMul(ctx, qhT, khTransposed)
		if err != nil {
			return nil, err
		}
		scores, err = ft.engine.MulScalar(ctx, scores, invSqrtDk)
		if err != nil {
			return nil, err
		}

		// Softmax over last dimension.
		attnWeights, err := ft.engine.Softmax(ctx, scores, -1)
		if err != nil {
			return nil, err
		}

		// Weighted sum: attn @ V_h -> [seqLen, dHead]
		headOut, err := ft.engine.MatMul(ctx, attnWeights, vhT)
		if err != nil {
			return nil, err
		}

		headOutputs[h] = headOut
	}

	// Concatenate heads: [seqLen, dHead] * nHeads -> [seqLen, DToken]
	concatData := make([]float32, seqLen*dToken)
	for h := 0; h < nHeads; h++ {
		hData := headOutputs[h].Data()
		for s := 0; s < seqLen; s++ {
			copy(concatData[s*dToken+h*dHead:s*dToken+(h+1)*dHead], hData[s*dHead:(s+1)*dHead])
		}
	}
	concat, err := tensor.New[float32]([]int{seqLen, dToken}, concatData)
	if err != nil {
		return nil, err
	}

	// Output projection.
	attnOut, err := ft.linearForward(ctx, concat, layer.wO)
	if err != nil {
		return nil, fmt.Errorf("o proj: %w", err)
	}

	// Residual connection.
	x, err = ft.engine.Add(ctx, x, attnOut)
	if err != nil {
		return nil, fmt.Errorf("residual 1: %w", err)
	}

	// Pre-norm 2: layer norm before FFN.
	normed2, err := ft.layerNorm(ctx, x, layer.ln2Gamma, layer.ln2Beta, seqLen, dToken)
	if err != nil {
		return nil, fmt.Errorf("ln2: %w", err)
	}

	// FFN: linear -> GELU -> linear.
	ffnOut, err := ft.linearForward(ctx, normed2, layer.ffn1)
	if err != nil {
		return nil, fmt.Errorf("ffn1: %w", err)
	}
	ffnOut, err = ft.engine.UnaryOp(ctx, ffnOut, geluScalar)
	if err != nil {
		return nil, fmt.Errorf("gelu: %w", err)
	}
	ffnOut, err = ft.linearForward(ctx, ffnOut, layer.ffn2)
	if err != nil {
		return nil, fmt.Errorf("ffn2: %w", err)
	}

	// Residual connection.
	x, err = ft.engine.Add(ctx, x, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("residual 2: %w", err)
	}

	return x, nil
}

// layerNorm applies layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.
// Input shape: [seqLen, dim]. Layer norm is applied per-row (over the dim axis).
func (ft *FTTransformer) layerNorm(ctx context.Context, x *tensor.TensorNumeric[float32], gamma, beta *tensor.TensorNumeric[float32], seqLen, dim int) (*tensor.TensorNumeric[float32], error) {
	const eps = 1e-5

	data := x.Data()
	out := make([]float32, seqLen*dim)
	gammaData := gamma.Data()
	betaData := beta.Data()

	for s := 0; s < seqLen; s++ {
		row := data[s*dim : (s+1)*dim]

		// Compute mean.
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(dim)

		// Compute variance.
		var variance float64
		for _, v := range row {
			diff := float64(v) - mean
			variance += diff * diff
		}
		variance /= float64(dim)

		invStd := float32(1.0 / math.Sqrt(variance+eps))

		// Normalize, scale, shift.
		for j := 0; j < dim; j++ {
			normalized := (row[j] - float32(mean)) * invStd
			out[s*dim+j] = normalized*gammaData[j] + betaData[j]
		}
	}

	return tensor.New[float32]([]int{seqLen, dim}, out)
}

// linearForward computes x @ W + b via the engine.
func (ft *FTTransformer) linearForward(ctx context.Context, x *tensor.TensorNumeric[float32], l mlpLayer) (*tensor.TensorNumeric[float32], error) {
	out, err := ft.engine.MatMul(ctx, x, l.weights)
	if err != nil {
		return nil, err
	}
	return ft.engine.Add(ctx, out, l.biases)
}
