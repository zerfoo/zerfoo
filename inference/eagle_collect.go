package inference

import (
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// TrainingPair holds a consecutive (input[t], target[t+1]) pair of hidden
// states from the penultimate transformer layer. These pairs are used to
// train the EAGLE head to predict the next hidden state.
type TrainingPair struct {
	Input  *tensor.TensorNumeric[float32] // [1, 1, hidden]
	Target *tensor.TensorNumeric[float32] // [1, 1, hidden]
}

// CollectPenultimateFeatures loads a GGUF model, tokenizes corpus text, runs
// a forward pass on token chunks, and extracts consecutive penultimate-layer
// hidden state pairs for EAGLE head training.
//
// The function returns (input[t], target[t+1]) pairs where input[t] is the
// penultimate hidden state at position t and target[t+1] is the penultimate
// hidden state at position t+1. maxSamples limits the number of pairs returned.
//
// This requires graph-level instrumentation to capture intermediate node outputs
// which is not yet implemented. Use GenerateSyntheticPairs for training loop
// validation and GGUF export testing.
func CollectPenultimateFeatures(
	modelPath string,
	corpusTokens []int,
	maxSamples int,
) ([]TrainingPair, error) {
	return nil, fmt.Errorf("CollectPenultimateFeatures: not yet implemented — graph-level intermediate output capture required; use GenerateSyntheticPairs for training loop validation")
}

// GenerateSyntheticPairs creates random training pairs for EAGLE head training
// loop validation. Each pair contains random [1, 1, hiddenDim] tensors.
func GenerateSyntheticPairs(hiddenDim, count int) ([]TrainingPair, error) {
	if hiddenDim <= 0 {
		return nil, fmt.Errorf("GenerateSyntheticPairs: hiddenDim must be positive, got %d", hiddenDim)
	}
	if count <= 0 {
		return nil, fmt.Errorf("GenerateSyntheticPairs: count must be positive, got %d", count)
	}

	pairs := make([]TrainingPair, count)
	for i := range count {
		// Use a simple deterministic pattern: input is a ramp scaled by position,
		// target is a slightly shifted version, so there is a learnable signal.
		inputData := make([]float32, hiddenDim)
		targetData := make([]float32, hiddenDim)
		for j := range hiddenDim {
			inputData[j] = float32(i*hiddenDim+j) * 0.001
			targetData[j] = float32(i*hiddenDim+j+1) * 0.001
		}

		input, err := tensor.New[float32]([]int{1, 1, hiddenDim}, inputData)
		if err != nil {
			return nil, fmt.Errorf("GenerateSyntheticPairs: create input tensor %d: %w", i, err)
		}
		target, err := tensor.New[float32]([]int{1, 1, hiddenDim}, targetData)
		if err != nil {
			return nil, fmt.Errorf("GenerateSyntheticPairs: create target tensor %d: %w", i, err)
		}

		pairs[i] = TrainingPair{Input: input, Target: target}
	}

	return pairs, nil
}
