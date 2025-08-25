package core

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestFiLM_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	contextDim := 4
	featureDim := 8

	film, err := NewFiLM[float32]("test_film", engine, ops, contextDim, featureDim)
	assert.NoError(t, err)

	feature, _ := tensor.New[float32]([]int{1, featureDim}, make([]float32, featureDim))
	contextTensor, _ := tensor.New[float32]([]int{1, contextDim}, make([]float32, contextDim))

	output, err := film.Forward(ctx, feature, contextTensor)

	assert.NoError(t, err)
	assert.Equal(t, []int{1, featureDim}, output.Shape())
}

func TestFiLM_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	contextDim := 4
	featureDim := 8

	film, err := NewFiLM[float32]("test_film", engine, ops, contextDim, featureDim)
	assert.NoError(t, err)

	featureData := make([]float32, featureDim)
	for i := range featureData {
		featureData[i] = float32(i + 1)
	}
	feature, _ := tensor.New[float32]([]int{1, featureDim}, featureData)

	contextData := make([]float32, contextDim)
	for i := range contextData {
		contextData[i] = float32(i + 1)
	}
	contextTensor, _ := tensor.New[float32]([]int{1, contextDim}, contextData)

	// Forward pass to populate intermediate values
	_, err = film.Forward(ctx, feature, contextTensor)
	assert.NoError(t, err)

	// Dummy output gradient
	outputGradientData := make([]float32, featureDim)
	for i := range outputGradientData {
		outputGradientData[i] = 1.0
	}
	outputGradient, _ := tensor.New[float32]([]int{1, featureDim}, outputGradientData)

	inputGradients, err := film.Backward(ctx, types.FullBackprop, outputGradient, feature, contextTensor)

	assert.NoError(t, err)
	assert.Len(t, inputGradients, 2)

	dFeature := inputGradients[0]
	dContext := inputGradients[1]

	assert.Equal(t, feature.Shape(), dFeature.Shape())
	assert.Equal(t, contextTensor.Shape(), dContext.Shape())

	// Check that parameter gradients are populated
	params := film.Parameters()
	for _, p := range params {
		assert.NotNil(t, p.Gradient)
		assert.Equal(t, p.Value.Shape(), p.Gradient.Shape())
	}
}

func TestFiLM_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	contextDim := 4
	featureDim := 8

	film, err := NewFiLM[float32]("test_film", engine, ops, contextDim, featureDim)
	assert.NoError(t, err)

	params := film.Parameters()
	// scaleGen weights + biasGen weights
	assert.Len(t, params, 2)
}
