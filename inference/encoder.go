package inference

import (
	"context"
	"fmt"
	"io"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// encoderArchitectures lists architectures that are encoder-only (no KV cache,
// no autoregressive decoding).
var encoderArchitectures = map[string]bool{
	"bert":    true,
	"roberta": true,
}

// IsEncoderArchitecture reports whether the given architecture name is
// an encoder-only model (e.g., BERT, RoBERTa).
func IsEncoderArchitecture(arch string) bool {
	return encoderArchitectures[arch]
}

// EncoderModel represents a loaded encoder-only model (BERT, RoBERTa, etc.).
// Unlike the decoder Model type, EncoderModel has no KV cache, no generator,
// and no autoregressive decoding loop. It runs a single forward pass over the
// full input sequence and returns classification logits.
type EncoderModel struct {
	graph       *graph.Graph[float32]
	embedWeight *tensor.TensorNumeric[float32]
	config      *gguf.ModelConfig
	engine      compute.Engine[float32]
	closer      io.Closer // mmap reader, if applicable
}

// Forward runs the encoder on input token IDs and returns classification logits.
// The input is a slice of integer token IDs. The returned slice contains logits
// of shape [1, numClasses] flattened to []float32.
func (m *EncoderModel) Forward(ctx context.Context, inputIDs []int) ([]float32, error) {
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("encoder forward: empty input")
	}

	// Convert int token IDs to float32 for the graph input tensor.
	data := make([]float32, len(inputIDs))
	for i, id := range inputIDs {
		data[i] = float32(id)
	}

	input, err := tensor.New[float32]([]int{1, len(inputIDs)}, data)
	if err != nil {
		return nil, fmt.Errorf("encoder forward: create input tensor: %w", err)
	}

	output, err := m.graph.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("encoder forward: %w", err)
	}

	return output.Data(), nil
}

// OutputShape returns the expected output shape [batch, numClasses].
func (m *EncoderModel) OutputShape() []int {
	numLabels := m.config.NumLabels
	if numLabels <= 0 {
		numLabels = 2
	}
	return []int{1, numLabels}
}

// Config returns the underlying model configuration.
func (m *EncoderModel) Config() *gguf.ModelConfig {
	return m.config
}

// Graph returns the computation graph.
func (m *EncoderModel) Graph() *graph.Graph[float32] {
	return m.graph
}

// Engine returns the compute engine.
func (m *EncoderModel) Engine() compute.Engine[float32] {
	return m.engine
}

// Close releases resources held by the encoder model.
func (m *EncoderModel) Close() error {
	var firstErr error
	if c, ok := m.engine.(io.Closer); ok {
		firstErr = c.Close()
	}
	if m.closer != nil {
		if err := m.closer.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		m.closer = nil
	}
	return firstErr
}

// LoadEncoderFile loads an encoder-only model from a GGUF file.
// It verifies the architecture is encoder-only and returns an EncoderModel
// instead of a Generator-based Model. Returns an error if the architecture
// is not an encoder type.
func LoadEncoderFile(path string, opts ...Option) (*EncoderModel, error) {
	o := &loadOptions{device: "cpu"}
	for _, opt := range opts {
		opt(o)
	}

	gm, err := LoadGGUF(path)
	if err != nil {
		return nil, err
	}

	if !IsEncoderArchitecture(gm.Config.Architecture) {
		return nil, fmt.Errorf("LoadEncoderFile: architecture %q is not an encoder-only model", gm.Config.Architecture)
	}

	eng, err := createEngine(o.device)
	if err != nil {
		return nil, fmt.Errorf("create engine (%s): %w", o.device, err)
	}

	applyDType(eng, o.dtype)

	g, embWeight, err := buildArchGraph(gm.Config.Architecture, gm.Tensors, gm.Config, eng)
	if err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
	}

	// Upload model weights to GPU if the engine supports it.
	if uploader, ok := eng.(compute.WeightUploader); ok {
		tensors := g.ConstantTensors()
		if embWeight != nil {
			tensors = append(tensors, embWeight)
		}
		for _, p := range g.Parameters() {
			if p.Value != nil {
				tensors = append(tensors, p.Value)
			}
		}
		if err := uploader.UploadWeights(tensors); err != nil {
			return nil, fmt.Errorf("upload weights to GPU: %w", err)
		}
	}

	return &EncoderModel{
		graph:       g,
		embedWeight: embWeight,
		config:      gm.Config,
		engine:      eng,
	}, nil
}
