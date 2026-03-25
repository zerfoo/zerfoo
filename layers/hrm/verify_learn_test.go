package hrm_test

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/hrm"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// vlComputeMSE computes the mean squared error between output and target slices.
func vlComputeMSE(output, target []float32) float32 {
	var sum float32
	for i := range output {
		d := output[i] - target[i]
		sum += d * d
	}
	return sum / float32(len(output))
}

// vlMSEGradient computes d(MSE)/d(output) = 2*(output - target) / n.
func vlMSEGradient(output, target []float32) []float32 {
	grad := make([]float32, len(output))
	n := float32(len(output))
	for i := range grad {
		grad[i] = 2 * (output[i] - target[i]) / n
	}
	return grad
}

// vlApplyGradientDescent performs a single manual SGD step: param -= lr * grad.
func vlApplyGradientDescent(params []*graph.Parameter[float32], lr float32) {
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

// vlHasNonZeroGradient returns true if at least one element in the gradient is non-zero.
func vlHasNonZeroGradient(p *graph.Parameter[float32]) bool {
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

// vlZeroGradients resets all parameter gradients to zero.
func vlZeroGradients(params []*graph.Parameter[float32]) {
	for _, p := range params {
		if p.Gradient == nil {
			continue
		}
		p.ClearGradient()
	}
}

const (
	vlModelDim = 16
	vlFFNDim   = 32
	vlNumHeads = 2
)

func newHModuleForLearn(t *testing.T) *hrm.HModule[float32] {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attn, err := attention.NewGlobalAttention[float32](engine, ops, vlModelDim, vlNumHeads, vlNumHeads)
	if err != nil {
		t.Fatalf("NewGlobalAttention: %v", err)
	}

	m, err := hrm.NewHModule[float32](engine, ops, vlModelDim, vlFFNDim, attn)
	if err != nil {
		t.Fatalf("NewHModule: %v", err)
	}

	// Scale parameters for numerical stability.
	for _, p := range m.Parameters() {
		d := p.Value.Data()
		for i := range d {
			d[i] *= 0.1
		}
	}

	return m
}

func newLModuleForLearn(t *testing.T) *hrm.LModule[float32] {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attn, err := attention.NewGlobalAttention[float32](engine, ops, vlModelDim, vlNumHeads, vlNumHeads)
	if err != nil {
		t.Fatalf("NewGlobalAttention: %v", err)
	}

	m, err := hrm.NewLModule[float32](engine, ops, vlModelDim, vlFFNDim, attn)
	if err != nil {
		t.Fatalf("NewLModule: %v", err)
	}

	for _, p := range m.Parameters() {
		d := p.Value.Data()
		for i := range d {
			d[i] *= 0.1
		}
	}

	return m
}

// ---------------------------------------------------------------------------
// HModule verify-learn tests
// ---------------------------------------------------------------------------

func TestHModule_LossDecreases(t *testing.T) {
	ctx := context.Background()
	m := newHModuleForLearn(t)

	const batch = 2

	// HModule.Forward takes lState [batch, modelDim].
	inputData := make([]float32, batch*vlModelDim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, vlModelDim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	target := make([]float32, batch*vlModelDim)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 30.0
	}

	params := m.Parameters()
	const lr = float32(0.01)
	const steps = 20

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		vlZeroGradients(params)

		output, fErr := m.Forward(ctx, input)
		if fErr != nil {
			t.Fatalf("step %d: Forward: %v", step, fErr)
		}

		loss := vlComputeMSE(output.Data(), target)
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d: loss is NaN or Inf", step)
		}

		if step == 0 {
			initialLoss = loss
			t.Logf("step 0: loss=%.6f", loss)
		}
		if step == steps-1 {
			finalLoss = loss
			t.Logf("step %d: loss=%.6f", step, loss)
		}

		gradData := vlMSEGradient(output.Data(), target)
		outputGrad, gErr := tensor.New[float32](output.Shape(), gradData)
		if gErr != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, gErr)
		}

		_, bErr := m.Backward(ctx, types.FullBackprop, outputGrad)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		vlApplyGradientDescent(params, lr)

		// Reset hidden state between steps to avoid accumulation effects.
		resetData := make([]float32, batch*vlModelDim)
		m.HiddenState, _ = tensor.New[float32]([]int{batch, vlModelDim}, resetData)
	}

	if finalLoss >= initialLoss {
		t.Errorf("HModule loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("HModule loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestHModule_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	m := newHModuleForLearn(t)

	const batch = 2

	inputData := make([]float32, batch*vlModelDim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, vlModelDim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	output, err := m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	onesData := make([]float32, len(output.Data()))
	for i := range onesData {
		onesData[i] = 1.0
	}
	dOut, err := tensor.New[float32](output.Shape(), onesData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	_, err = m.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// With seq_len=1, Q/K attention weights have zero gradients (trivial
	// softmax over a single position). Check that at least some parameters
	// received non-zero gradients.
	nonZeroCount := 0
	for _, p := range m.Parameters() {
		if vlHasNonZeroGradient(p) {
			nonZeroCount++
		}
	}
	if nonZeroCount == 0 {
		t.Error("HModule: no parameters have non-zero gradients after backward")
	} else {
		t.Logf("HModule: %d/%d parameters have non-zero gradients", nonZeroCount, len(m.Parameters()))
	}
}

// ---------------------------------------------------------------------------
// LModule verify-learn tests
// ---------------------------------------------------------------------------

func TestLModule_LossDecreases(t *testing.T) {
	ctx := context.Background()
	m := newLModuleForLearn(t)

	const batch = 2

	// LModule.Forward takes (hState, projectedInput) both [batch, modelDim].
	hStateData := make([]float32, batch*vlModelDim)
	for i := range hStateData {
		hStateData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	hState, err := tensor.New[float32]([]int{batch, vlModelDim}, hStateData)
	if err != nil {
		t.Fatalf("make hState: %v", err)
	}

	projInputData := make([]float32, batch*vlModelDim)
	for i := range projInputData {
		projInputData[i] = float32(((i*3+7)%13)-6) / 20.0
	}
	projInput, err := tensor.New[float32]([]int{batch, vlModelDim}, projInputData)
	if err != nil {
		t.Fatalf("make projInput: %v", err)
	}

	target := make([]float32, batch*vlModelDim)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 30.0
	}

	params := m.Parameters()
	const lr = float32(0.01)
	const steps = 20

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		vlZeroGradients(params)

		output, fErr := m.Forward(ctx, hState, projInput)
		if fErr != nil {
			t.Fatalf("step %d: Forward: %v", step, fErr)
		}

		loss := vlComputeMSE(output.Data(), target)
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d: loss is NaN or Inf", step)
		}

		if step == 0 {
			initialLoss = loss
			t.Logf("step 0: loss=%.6f", loss)
		}
		if step == steps-1 {
			finalLoss = loss
			t.Logf("step %d: loss=%.6f", step, loss)
		}

		gradData := vlMSEGradient(output.Data(), target)
		outputGrad, gErr := tensor.New[float32](output.Shape(), gradData)
		if gErr != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, gErr)
		}

		_, bErr := m.Backward(ctx, types.FullBackprop, outputGrad)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		vlApplyGradientDescent(params, lr)

		// Reset hidden state between steps to avoid accumulation effects.
		resetData := make([]float32, batch*vlModelDim)
		m.HiddenState, _ = tensor.New[float32]([]int{batch, vlModelDim}, resetData)
	}

	if finalLoss >= initialLoss {
		t.Errorf("LModule loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("LModule loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestLModule_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	m := newLModuleForLearn(t)

	const batch = 2

	hStateData := make([]float32, batch*vlModelDim)
	for i := range hStateData {
		hStateData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	hState, err := tensor.New[float32]([]int{batch, vlModelDim}, hStateData)
	if err != nil {
		t.Fatalf("make hState: %v", err)
	}

	projInputData := make([]float32, batch*vlModelDim)
	for i := range projInputData {
		projInputData[i] = float32(((i*3+7)%13)-6) / 20.0
	}
	projInput, err := tensor.New[float32]([]int{batch, vlModelDim}, projInputData)
	if err != nil {
		t.Fatalf("make projInput: %v", err)
	}

	output, err := m.Forward(ctx, hState, projInput)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	onesData := make([]float32, len(output.Data()))
	for i := range onesData {
		onesData[i] = 1.0
	}
	dOut, err := tensor.New[float32](output.Shape(), onesData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	_, err = m.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// With seq_len=1, Q/K attention weights have zero gradients (trivial
	// softmax over a single position). Check that at least some parameters
	// received non-zero gradients.
	nonZeroCount := 0
	for _, p := range m.Parameters() {
		if vlHasNonZeroGradient(p) {
			nonZeroCount++
		}
	}
	if nonZeroCount == 0 {
		t.Error("LModule: no parameters have non-zero gradients after backward")
	} else {
		t.Logf("LModule: %d/%d parameters have non-zero gradients", nonZeroCount, len(m.Parameters()))
	}
}
