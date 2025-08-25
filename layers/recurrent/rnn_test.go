package recurrent

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestSimpleRNN_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	inputDim := 10
	hiddenDim := 20

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, inputDim, hiddenDim)
	assert.NoError(t, err)

	input, _ := tensor.New[float32]([]int{1, inputDim}, make([]float32, inputDim))

	output, err := rnn.Forward(ctx, input)

	assert.NoError(t, err)
	assert.Equal(t, []int{1, hiddenDim}, output.Shape())
}

func TestSimpleRNN_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	inputDim := 10
	hiddenDim := 20

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, inputDim, hiddenDim)
	assert.NoError(t, err)

	input, _ := tensor.New[float32]([]int{1, inputDim}, make([]float32, inputDim))

	// Forward pass to populate intermediate values
	_, err = rnn.Forward(ctx, input)
	assert.NoError(t, err)

	outputGradient, _ := tensor.New[float32]([]int{1, hiddenDim}, make([]float32, hiddenDim))

	inputGradients, err := rnn.Backward(ctx, types.OneStepApproximation, outputGradient, input)

	assert.NoError(t, err)
	assert.Len(t, inputGradients, 2)
	assert.Equal(t, []int{1, inputDim}, inputGradients[0].Shape())
	assert.Equal(t, []int{1, hiddenDim}, inputGradients[1].Shape())

	// Check that parameter gradients are populated
	params := rnn.Parameters()
	for _, p := range params {
		assert.NotNil(t, p.Gradient)
	}
}

func TestSimpleRNN_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	inputDim := 10
	hiddenDim := 20

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, inputDim, hiddenDim)
	assert.NoError(t, err)

	params := rnn.Parameters()
	// input weights + hidden weights + bias
	assert.Len(t, params, 3)
}
