package multimodal

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// VisionEncoder encodes image patches into hidden representations for
// vision-language model inference.
type VisionEncoder[T tensor.Numeric] interface {
	Encode(patches []float32, cfg PatchConfig) ([]T, error)
	HiddenSize() int
	NumLayers() int
}

// EncoderConfig holds the hyperparameters for a vision encoder.
type EncoderConfig struct {
	HiddenDim int
	NumHeads  int
	NumLayers int
	PatchCfg  PatchConfig
}

// SigLIPEncoder implements VisionEncoder using a SigLIP-style linear
// projection from patch embeddings into the hidden dimension.
type SigLIPEncoder[T tensor.Numeric] struct {
	cfg    EncoderConfig
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T] // [patch_dim, hidden_dim]
}

// NewSigLIPEncoder creates a SigLIPEncoder with randomly initialized weights.
func NewSigLIPEncoder[T tensor.Numeric](cfg EncoderConfig, e compute.Engine[T]) *SigLIPEncoder[T] {
	patchDim := PatchDim(cfg.PatchCfg)
	data := make([]T, patchDim*cfg.HiddenDim)
	for i := range data {
		data[i] = T(rand.Float64()*0.02 - 0.01)
	}
	w, _ := tensor.New[T]([]int{patchDim, cfg.HiddenDim}, data)
	return &SigLIPEncoder[T]{
		cfg:    cfg,
		engine: e,
		weight: w,
	}
}

// Encode projects patch embeddings through a linear layer.
// patches is a flat []float32 of shape [num_patches, patch_dim].
// Returns []T of length num_patches * HiddenDim.
func (s *SigLIPEncoder[T]) Encode(patches []float32, cfg PatchConfig) ([]T, error) {
	np := NumPatches(cfg)
	pd := PatchDim(cfg)
	if len(patches) != np*pd {
		return nil, fmt.Errorf("multimodal: expected %d patch values, got %d", np*pd, len(patches))
	}

	// Convert float32 patches to T for matmul.
	inputData := make([]T, len(patches))
	for i, v := range patches {
		inputData[i] = T(v)
	}

	input, err := tensor.New[T]([]int{np, pd}, inputData)
	if err != nil {
		return nil, fmt.Errorf("multimodal: create input tensor: %w", err)
	}

	// MatMul: [num_patches, patch_dim] x [patch_dim, hidden_dim] = [num_patches, hidden_dim]
	result, err := s.engine.MatMul(context.Background(), input, s.weight)
	if err != nil {
		return nil, fmt.Errorf("multimodal: linear projection: %w", err)
	}

	return result.Data(), nil
}

// HiddenSize returns the hidden dimension of the encoder output.
func (s *SigLIPEncoder[T]) HiddenSize() int {
	return s.cfg.HiddenDim
}

// NumLayers returns the number of encoder layers.
func (s *SigLIPEncoder[T]) NumLayers() int {
	return s.cfg.NumLayers
}
