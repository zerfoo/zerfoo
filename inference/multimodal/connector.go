package multimodal

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// ConnectorConfig holds parameters for the vision-to-text projection.
type ConnectorConfig struct {
	VisionDim int
	TextDim   int
	WeightKey string // GGUF key for projection matrix; default "mm.projector.weight"
}

// ProjectionConnector projects vision encoder output into the text model's
// embedding space via a learned linear projection.
type ProjectionConnector[T tensor.Numeric] struct {
	cfg    ConnectorConfig
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T] // [VisionDim, TextDim]
}

// NewProjectionConnector creates a ProjectionConnector. The projection matrix
// is zero-initialized; call LoadWeights to populate it from model weights.
func NewProjectionConnector[T tensor.Numeric](cfg ConnectorConfig, e compute.Engine[T]) *ProjectionConnector[T] {
	if cfg.WeightKey == "" {
		cfg.WeightKey = "mm.projector.weight"
	}
	data := make([]T, cfg.VisionDim*cfg.TextDim)
	w, _ := tensor.New[T]([]int{cfg.VisionDim, cfg.TextDim}, data)
	return &ProjectionConnector[T]{
		cfg:    cfg,
		engine: e,
		weight: w,
	}
}

// Project applies linear projection: [numTokens, VisionDim] x [VisionDim, TextDim] -> [numTokens, TextDim].
func (p *ProjectionConnector[T]) Project(visionEmbeds []T, numTokens int) ([]T, error) {
	if numTokens <= 0 {
		return nil, fmt.Errorf("multimodal: numTokens must be positive, got %d", numTokens)
	}
	expected := numTokens * p.cfg.VisionDim
	if len(visionEmbeds) != expected {
		return nil, fmt.Errorf("multimodal: expected %d vision values (%d tokens x %d dim), got %d",
			expected, numTokens, p.cfg.VisionDim, len(visionEmbeds))
	}

	input, err := tensor.New[T]([]int{numTokens, p.cfg.VisionDim}, visionEmbeds)
	if err != nil {
		return nil, fmt.Errorf("multimodal: create input tensor: %w", err)
	}

	result, err := p.engine.MatMul(context.Background(), input, p.weight)
	if err != nil {
		return nil, fmt.Errorf("multimodal: projection matmul: %w", err)
	}

	return result.Data(), nil
}

// LoadWeights sets the projection matrix from a flat []float32 of shape [VisionDim, TextDim].
func (p *ProjectionConnector[T]) LoadWeights(weights []float32) error {
	expected := p.cfg.VisionDim * p.cfg.TextDim
	if len(weights) != expected {
		return fmt.Errorf("multimodal: expected %d weight values (%d x %d), got %d",
			expected, p.cfg.VisionDim, p.cfg.TextDim, len(weights))
	}

	data := make([]T, len(weights))
	for i, v := range weights {
		data[i] = T(v)
	}

	w, err := tensor.New[T]([]int{p.cfg.VisionDim, p.cfg.TextDim}, data)
	if err != nil {
		return fmt.Errorf("multimodal: create weight tensor: %w", err)
	}
	p.weight = w
	return nil
}

// VisionDim returns the input dimension (vision encoder hidden size).
func (p *ProjectionConnector[T]) VisionDim() int {
	return p.cfg.VisionDim
}

// TextDim returns the output dimension (text model embedding size).
func (p *ProjectionConnector[T]) TextDim() int {
	return p.cfg.TextDim
}
