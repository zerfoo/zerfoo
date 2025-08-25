package features

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestSpectralFingerprint_Forward(t *testing.T) {
	ctx := context.Background()
	outputDim := 8

	layer := NewSpectralFingerprint[float32](outputDim)

	inputData := make([]float32, 16)
	for i := range inputData {
		inputData[i] = float32(i)
	}
	input, _ := tensor.New[float32]([]int{1, 16}, inputData)

	output, err := layer.Forward(ctx, input)

	assert.NoError(t, err)
	assert.Equal(t, []int{1, outputDim}, output.Shape())
}

func TestSpectralFingerprint_Backward(t *testing.T) {
	ctx := context.Background()
	outputDim := 8

	layer := NewSpectralFingerprint[float32](outputDim)

	inputData := make([]float32, 16)
	input, _ := tensor.New[float32]([]int{1, 16}, inputData)

	outputGradient, _ := tensor.New[float32]([]int{1, outputDim}, make([]float32, outputDim))

	inputGradients, err := layer.Backward(ctx, types.FullBackprop, outputGradient, input)

	assert.NoError(t, err)
	assert.Len(t, inputGradients, 1)
	assert.Equal(t, input.Shape(), inputGradients[0].Shape())

	// Check if the gradient is all zeros
	for _, v := range inputGradients[0].Data() {
		assert.Equal(t, float32(0), v)
	}
}

func TestSpectralFingerprint_Parameters(t *testing.T) {
	layer := NewSpectralFingerprint[float32](8)
	assert.Empty(t, layer.Parameters())
}
