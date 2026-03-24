package recurrent

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// computeMSE computes the mean squared error between output and target slices.
func computeMSE(output, target []float32) float32 {
	var sum float32
	for i := range output {
		d := output[i] - target[i]
		sum += d * d
	}
	return sum / float32(len(output))
}

// mseGradient computes d(MSE)/d(output) = 2*(output - target) / n.
func mseGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	n := float32(len(output))
	for i := range grad {
		grad[i] = 2 * (output[i] - target[i]) / n
	}
	return grad
}

// applyGradientDescent performs a single manual SGD step: param -= lr * grad.
func applyGradientDescent(params []*graph.Parameter[float32], lr float32) {
	for _, p := range params {
		if p.Gradient == nil {
			continue
		}
		pData := p.Value.Data()
		gData := p.Gradient.Data()
		for j := range pData {
			pData[j] -= lr * gData[j]
		}
	}
}

// hasNonZeroGradient returns true if at least one element in the gradient is non-zero.
func hasNonZeroGradient(p *graph.Parameter[float32]) bool {
	if p.Gradient == nil {
		return false
	}
	for _, v := range p.Gradient.Data() {
		if v != 0 {
			return true
		}
	}
	return false
}

// rnnForwardSequence runs the RNN over a 3D input [batch, seq, features] by
// iterating over the sequence dimension and calling Forward for each timestep.
// It returns the final hidden state (output of the last timestep).
func rnnForwardSequence(ctx context.Context, rnn *SimpleRNN[float32], input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	batch := shape[0]
	seqLen := shape[1]
	features := shape[2]
	data := input.Data()

	// Reset hidden state for a fresh sequence.
	rnn.hiddenState = nil

	var output *tensor.TensorNumeric[float32]
	for t := 0; t < seqLen; t++ {
		// Extract timestep slice: [batch, features]
		stepData := make([]float32, batch*features)
		for b := 0; b < batch; b++ {
			srcOff := b*seqLen*features + t*features
			dstOff := b * features
			copy(stepData[dstOff:dstOff+features], data[srcOff:srcOff+features])
		}
		stepTensor, err := tensor.New[float32]([]int{batch, features}, stepData)
		if err != nil {
			return nil, err
		}

		output, err = rnn.Forward(ctx, stepTensor)
		if err != nil {
			return nil, err
		}
	}
	return output, nil
}

// TestSimpleRNN_LossDecreases verifies that after Forward -> MSE -> Backward ->
// gradient descent, the loss decreases for a SimpleRNN processing a 3D input
// [batch=2, seq=4, features=8].
func TestSimpleRNN_LossDecreases(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	rng := rand.New(rand.NewPCG(42, 0))
	ctx := context.Background()

	const (
		batch     = 2
		seqLen    = 4
		features  = 8
		hiddenDim = 6
	)

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, features, hiddenDim)
	if err != nil {
		t.Fatalf("NewSimpleRNN: %v", err)
	}

	// Create 3D input [batch, seq, features].
	inputData := make([]float32, batch*seqLen*features)
	for i := range inputData {
		inputData[i] = float32(rng.NormFloat64()) * 0.5
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, features}, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	// Target for the final hidden state [batch, hiddenDim].
	target := make([]float32, batch*hiddenDim)
	for i := range target {
		target[i] = float32(rng.NormFloat64()) * 0.1
	}

	// Step 1: Forward over the full sequence.
	output, err := rnnForwardSequence(ctx, rnn, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outputData := output.Data()
	if len(outputData) != len(target) {
		t.Fatalf("output size %d != target size %d", len(outputData), len(target))
	}

	// Step 2: Compute initial MSE loss.
	loss1 := computeMSE(outputData, target)
	if math.IsNaN(float64(loss1)) || math.IsInf(float64(loss1), 0) {
		t.Fatalf("initial loss is NaN or Inf: %v", loss1)
	}
	t.Logf("initial loss: %f", loss1)

	// Step 3: Compute output gradient (dL/dOutput for MSE).
	gradData := mseGradient(outputData, target)
	outputGrad, err := tensor.New[float32](output.Shape(), gradData)
	if err != nil {
		t.Fatalf("tensor.New outputGrad: %v", err)
	}

	// Step 4: Backward pass (one-step approximation on the last timestep).
	_, err = rnn.Backward(ctx, types.OneStepApproximation, outputGrad, rnn.lastInput)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Step 5: Apply gradient descent.
	params := rnn.Parameters()
	lr := float32(0.01)
	applyGradientDescent(params, lr)

	// Step 6: Forward pass again with updated parameters.
	output2, err := rnnForwardSequence(ctx, rnn, input)
	if err != nil {
		t.Fatalf("Forward (after update): %v", err)
	}

	// Step 7: Compute new loss.
	loss2 := computeMSE(output2.Data(), target)
	if math.IsNaN(float64(loss2)) || math.IsInf(float64(loss2), 0) {
		t.Fatalf("updated loss is NaN or Inf: %v", loss2)
	}
	t.Logf("updated loss: %f", loss2)

	// Step 8: Verify loss decreased.
	if loss2 >= loss1 {
		t.Errorf("loss did not decrease: before=%f, after=%f", loss1, loss2)
	}
}

// TestSimpleRNN_ParameterGradientsNonZero verifies that after a backward pass,
// all SimpleRNN parameter gradients have at least one non-zero element.
func TestSimpleRNN_ParameterGradientsNonZero(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	rng := rand.New(rand.NewPCG(42, 0))
	ctx := context.Background()

	const (
		batch     = 2
		seqLen    = 4
		features  = 8
		hiddenDim = 6
	)

	rnn, err := NewSimpleRNN[float32]("test_rnn", engine, ops, features, hiddenDim)
	if err != nil {
		t.Fatalf("NewSimpleRNN: %v", err)
	}

	// Create 3D input [batch, seq, features].
	inputData := make([]float32, batch*seqLen*features)
	for i := range inputData {
		inputData[i] = float32(rng.NormFloat64()) * 0.5
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, features}, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	// Target for the final hidden state [batch, hiddenDim].
	target := make([]float32, batch*hiddenDim)
	for i := range target {
		target[i] = float32(rng.NormFloat64()) * 0.1
	}

	// Forward over the full sequence.
	output, err := rnnForwardSequence(ctx, rnn, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Compute MSE gradient as the output gradient.
	outputData := output.Data()
	gradData := mseGradient(outputData, target)
	outputGrad, err := tensor.New[float32](output.Shape(), gradData)
	if err != nil {
		t.Fatalf("tensor.New outputGrad: %v", err)
	}

	// Backward pass.
	_, err = rnn.Backward(ctx, types.OneStepApproximation, outputGrad, rnn.lastInput)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Verify each parameter has a non-zero gradient.
	for _, p := range rnn.Parameters() {
		if p.Gradient == nil {
			t.Errorf("parameter %q has nil gradient", p.Name)
			continue
		}
		if !hasNonZeroGradient(p) {
			t.Errorf("parameter %q has all-zero gradient", p.Name)
		}
	}
}
