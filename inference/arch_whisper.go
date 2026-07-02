package inference

import (
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/audio"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// WhisperConfig holds Whisper-specific model configuration.
type WhisperConfig struct {
	NumMels    int
	HiddenDim  int
	NumHeads   int
	NumLayers  int
	KernelSize int
}

// WhisperConfigFromGGUF extracts Whisper configuration from GGUF ModelConfig.
// Fields are mapped as: HiddenSize -> HiddenDim, NumHeads -> NumHeads,
// NumLayers -> NumLayers. NumMels defaults to 80, KernelSize defaults to 3.
func WhisperConfigFromGGUF(cfg *gguf.ModelConfig) WhisperConfig {
	numMels := 80
	kernelSize := 3
	numHeads := cfg.NumHeads
	if numHeads == 0 {
		numHeads = 6 // Whisper-base default
	}
	return WhisperConfig{
		NumMels:    numMels,
		HiddenDim:  cfg.HiddenSize,
		NumHeads:   numHeads,
		NumLayers:  cfg.NumLayers,
		KernelSize: kernelSize,
	}
}

// buildWhisperGraph constructs a computation graph for the Whisper audio encoder
// from pre-loaded GGUF tensors. It returns the graph and nil for the embedding
// weight (Whisper does not use text token embeddings in the encoder).
//
// The Whisper encoder architecture is:
//
//	MelSpectrogram -> Conv1(stride=2)+GELU -> Conv2(stride=2)+GELU -> PosEnc -> [LN -> SelfAttn -> Add -> LN -> FFN -> Add] x N -> LN
//
// Expected GGUF tensor names:
//
//	encoder.conv1.weight                    — [hidden_dim, num_mels, kernel_size]
//	encoder.conv1.bias                      — [hidden_dim]
//	encoder.conv2.weight                    — [hidden_dim, hidden_dim, kernel_size]
//	encoder.conv2.bias                      — [hidden_dim]
//	encoder.blocks.{i}.attn_ln.weight       — [hidden_dim]
//	encoder.blocks.{i}.attn_ln.bias         — [hidden_dim]
//	encoder.blocks.{i}.attn.query.weight    — [hidden_dim, hidden_dim]
//	encoder.blocks.{i}.attn.key.weight      — [hidden_dim, hidden_dim]
//	encoder.blocks.{i}.attn.value.weight    — [hidden_dim, hidden_dim]
//	encoder.blocks.{i}.attn.out.weight      — [hidden_dim, hidden_dim]
//	encoder.blocks.{i}.mlp_ln.weight        — [hidden_dim]
//	encoder.blocks.{i}.mlp_ln.bias          — [hidden_dim]
//	encoder.blocks.{i}.mlp.0.weight         — [4*hidden_dim, hidden_dim]
//	encoder.blocks.{i}.mlp.0.bias           — [4*hidden_dim]
//	encoder.blocks.{i}.mlp.2.weight         — [hidden_dim, 4*hidden_dim]
//	encoder.blocks.{i}.mlp.2.bias           — [hidden_dim]
//	encoder.ln_post.weight                  — [hidden_dim]
//	encoder.ln_post.bias                    — [hidden_dim]
func buildWhisperGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	wc := WhisperConfigFromGGUF(cfg)
	return BuildWhisperEncoder(wc, tensors, engine)
}

// BuildWhisperEncoder constructs a computation graph for Whisper encoder from a weight map.
// Exported for benchmark and integration tests that construct synthetic weight maps.
func BuildWhisperEncoder(
	wc WhisperConfig,
	tensors map[string]*tensor.TensorNumeric[float32],
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	enc, err := audio.NewWhisperEncoder[float32](
		"whisper_encoder", engine, ops,
		audio.WhisperEncoderConfig{
			NumMels:    wc.NumMels,
			HiddenDim:  wc.HiddenDim,
			NumHeads:   wc.NumHeads,
			NumLayers:  wc.NumLayers,
			KernelSize: wc.KernelSize,
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("create whisper encoder: %w", err)
	}

	// Load weights from GGUF tensors into the encoder's parameters.
	if err := loadWhisperWeights(enc, tensors, wc); err != nil {
		return nil, nil, fmt.Errorf("load whisper weights: %w", err)
	}

	// Build computation graph with encoder as the output node.
	builder := graph.NewBuilder[float32](engine)
	input := builder.Input([]int{1, wc.NumMels, -1}) // [batch, num_mels, T_frames]
	builder.AddNode(enc, input)
	g, err := builder.Build(enc)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	return g, nil, nil
}

// loadWhisperWeights maps GGUF tensor data onto WhisperEncoder parameters
// using positional indexing. Parameters() returns params in a fixed order:
//
//	[0] conv1_weight, [1] conv1_bias, [2] conv2_weight, [3] conv2_bias,
//	then per block (12 params each):
//	  [0] ln1.gamma, [1] ln1.beta, [2] q_weights, [3] k_weights,
//	  [4] v_weights, [5] o_weights, [6] ln2.gamma, [7] ln2.beta,
//	  [8] ffn1_linear_weights, [9] ffn1_bias_biases,
//	  [10] ffn2_linear_weights, [11] ffn2_bias_biases,
//	then: [0] lnPost.gamma, [1] lnPost.beta.
func loadWhisperWeights(
	enc *audio.WhisperEncoder[float32],
	tensors map[string]*tensor.TensorNumeric[float32],
	wc WhisperConfig,
) error {
	params := enc.Parameters()

	// Expected total: 4 (conv) + 12*numLayers (blocks) + 2 (ln_post).
	expectedCount := 4 + 12*wc.NumLayers + 2
	if len(params) != expectedCount {
		return fmt.Errorf("expected %d parameters, got %d", expectedCount, len(params))
	}

	// Build the GGUF-name-to-position mapping.
	ggufOrder := make([]string, 0, expectedCount)

	// Conv frontend.
	ggufOrder = append(ggufOrder,
		"encoder.conv1.weight",
		"encoder.conv1.bias",
		"encoder.conv2.weight",
		"encoder.conv2.bias",
	)

	// Per-block tensors.
	for i := 0; i < wc.NumLayers; i++ {
		prefix := fmt.Sprintf("encoder.blocks.%d.", i)
		ggufOrder = append(ggufOrder,
			prefix+"attn_ln.weight",
			prefix+"attn_ln.bias",
			prefix+"attn.query.weight",
			prefix+"attn.key.weight",
			prefix+"attn.value.weight",
			prefix+"attn.out.weight",
			prefix+"mlp_ln.weight",
			prefix+"mlp_ln.bias",
			prefix+"mlp.0.weight",
			prefix+"mlp.0.bias",
			prefix+"mlp.2.weight",
			prefix+"mlp.2.bias",
		)
	}

	// Post layer norm.
	ggufOrder = append(ggufOrder,
		"encoder.ln_post.weight",
		"encoder.ln_post.bias",
	)

	// Copy each GGUF tensor into the corresponding parameter by position.
	for i, ggufName := range ggufOrder {
		t, ok := tensors[ggufName]
		if !ok {
			return fmt.Errorf("missing GGUF tensor %q", ggufName)
		}
		copy(params[i].Value.Data(), t.Data())
	}

	return nil
}
