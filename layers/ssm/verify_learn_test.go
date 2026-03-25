package ssm

import (
	"context"
	"math"
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

// zeroGradients resets all parameter gradients to zero.
func zeroGradients(params []*graph.Parameter[float32]) {
	for _, p := range params {
		if p.Gradient == nil {
			continue
		}
		p.ClearGradient()
	}
}

// sumElements computes the sum of all elements in a float32 slice.
func sumElements(data []float32) float32 {
	var s float32
	for _, v := range data {
		s += v
	}
	return s
}

// cloneParamData returns a copy of the parameter's value data.
func cloneParamData(p *graph.Parameter[float32]) []float32 {
	src := p.Value.Data()
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}

// newMambaForGradCheck creates a MambaBlock with scaled-down weights for
// numerically stable finite-difference gradient checking.
func newMambaForGradCheck(t *testing.T) (*MambaBlock[float32], compute.Engine[float32], numeric.Arithmetic[float32]) {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewMambaBlock[float32](
		"vl", engine, ops,
		8,  // dModel
		16, // dInner
		4,  // dState
		1,  // dtRank
		2,  // convKer
	)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	// Scale down parameters for stable gradient checking.
	for _, p := range block.Parameters() {
		d := p.Value.Data()
		for i := range d {
			d[i] *= 0.1
		}
	}

	return block, engine, ops
}

// newMambaForLearn creates a MambaBlock with default weights for training tests.
func newMambaForLearn(t *testing.T) (*MambaBlock[float32], compute.Engine[float32], numeric.Arithmetic[float32]) {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewMambaBlock[float32](
		"vl", engine, ops,
		8,  // dModel
		16, // dInner
		4,  // dState
		1,  // dtRank
		2,  // convKer
	)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	return block, engine, ops
}

// makeInput creates a deterministic [1, 4, 8] input tensor with moderate values.
func makeInput(t *testing.T) *tensor.TensorNumeric[float32] {
	t.Helper()
	const (
		batch  = 1
		seqLen = 4
		dModel = 8
	)
	n := batch * seqLen * dModel
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(((i*7+13)%19)-9) / 10.0
	}
	out, err := tensor.New[float32]([]int{batch, seqLen, dModel}, data)
	if err != nil {
		t.Fatalf("makeInput: %v", err)
	}
	return out
}

// forwardLoss runs the forward pass and returns sum-of-outputs as the scalar loss.
func forwardLoss(
	t *testing.T,
	ctx context.Context,
	block *MambaBlock[float32],
	input *tensor.TensorNumeric[float32],
) float32 {
	t.Helper()
	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	return sumElements(output.Data())
}

// TestMambaBlock_FiniteDiffParamGradients verifies analytical gradients of the A, D,
// and convWeight parameters against central finite differences.
func TestMambaBlock_FiniteDiffParamGradients(t *testing.T) {
	ctx := context.Background()
	block, _, _ := newMambaForGradCheck(t)
	input := makeInput(t)

	// --- Run forward + backward to get analytical gradients -----------------

	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Loss = sum(output). dLoss/dOutput = all-ones.
	onesData := make([]float32, len(output.Data()))
	for i := range onesData {
		onesData[i] = 1.0
	}
	dOut, err := tensor.New[float32](output.Shape(), onesData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	_, err = block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// --- Collect analytical gradients for the parameters we want to check ---

	type paramCheck struct {
		name     string
		param    *graph.Parameter[float32]
		analGrad []float32
	}

	checks := []paramCheck{
		{"A", block.A, nil},
		{"D", block.D, nil},
		{"convWeight", block.convWeight, nil},
	}
	for i := range checks {
		g := checks[i].param.Gradient
		if g == nil {
			t.Fatalf("parameter %s has nil gradient after backward", checks[i].name)
		}
		gd := g.Data()
		checks[i].analGrad = make([]float32, len(gd))
		copy(checks[i].analGrad, gd)
	}

	// --- Finite difference on each parameter element ------------------------

	const eps = float32(1e-4)
	const relTol = float32(1e-2)

	for _, pc := range checks {
		t.Run(pc.name, func(t *testing.T) {
			pData := pc.param.Value.Data()
			numFailed := 0
			numChecked := 0
			for i := range pData {
				orig := pData[i]

				// f(p + eps)
				pData[i] = orig + eps
				zeroGradients(block.Parameters())
				lPlus := forwardLoss(t, ctx, block, input)

				// f(p - eps)
				pData[i] = orig - eps
				zeroGradients(block.Parameters())
				lMinus := forwardLoss(t, ctx, block, input)

				pData[i] = orig

				numerical := (lPlus - lMinus) / (2 * float32(eps))
				analytical := pc.analGrad[i]

				diff := float32(math.Abs(float64(analytical - numerical)))
				denom := float32(math.Max(1.0, math.Max(math.Abs(float64(analytical)), math.Abs(float64(numerical)))))
				relErr := diff / denom

				numChecked++
				if relErr > relTol {
					numFailed++
					if numFailed <= 5 {
						t.Errorf("elem[%d]: analytical=%.6f numerical=%.6f relErr=%.4f",
							i, analytical, numerical, relErr)
					}
				}
			}
			if numFailed > 0 {
				t.Errorf("%d/%d elements exceeded relTol=%.4f", numFailed, numChecked, relTol)
			} else {
				t.Logf("%d elements passed finite-diff check (relTol=%.4f)", numChecked, relTol)
			}
		})
	}
}

// TestMambaBlock_LossDecreases verifies that running 20 steps of SGD with MSE loss
// against random targets reduces the loss.
func TestMambaBlock_LossDecreases(t *testing.T) {
	ctx := context.Background()
	block, _, _ := newMambaForLearn(t)
	input := makeInput(t)

	// Create a target: random but deterministic.
	// Output shape is [1, 4, 8] (batch=1, seq=4, dModel=8).
	const outputSize = 1 * 4 * 8
	target := make([]float32, outputSize)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 10.0
	}

	params := block.Parameters()
	const lr = float32(1.0)
	const steps = 20

	var initialLoss float32
	var finalLoss float32

	for step := 0; step < steps; step++ {
		// Zero gradients.
		zeroGradients(params)

		// Forward.
		output, err := block.Forward(ctx, input)
		if err != nil {
			t.Fatalf("step %d: Forward: %v", step, err)
		}

		outData := output.Data()
		if len(outData) != outputSize {
			t.Fatalf("step %d: output size %d != expected %d", step, len(outData), outputSize)
		}

		// Loss.
		loss := computeMSE(outData, target)
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d: loss is NaN or Inf: %v", step, loss)
		}

		if step == 0 {
			initialLoss = loss
			t.Logf("step 0: loss=%.6f", loss)
		}
		if step == steps-1 {
			finalLoss = loss
			t.Logf("step %d: loss=%.6f", step, loss)
		}

		// Backward.
		gradData := mseGradient(outData, target)
		outputGrad, err := tensor.New[float32](output.Shape(), gradData)
		if err != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, err)
		}

		_, err = block.Backward(ctx, types.FullBackprop, outputGrad, input)
		if err != nil {
			t.Fatalf("step %d: Backward: %v", step, err)
		}

		// SGD update.
		applyGradientDescent(params, lr)
	}

	if finalLoss >= initialLoss {
		t.Errorf("loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

// TestMambaBlock_GradientsNonZero verifies that after a backward pass all parameter
// gradients contain at least one non-zero element.
func TestMambaBlock_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	block, _, _ := newMambaForGradCheck(t)
	input := makeInput(t)

	output, err := block.Forward(ctx, input)
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

	_, err = block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for _, p := range block.Parameters() {
		if p.Gradient == nil {
			t.Errorf("parameter %q has nil gradient", p.Name)
			continue
		}
		if !hasNonZeroGradient(p) {
			t.Errorf("parameter %q has all-zero gradient", p.Name)
		} else {
			// Log a summary of the gradient magnitude.
			gData := p.Gradient.Data()
			var maxAbs float32
			for _, v := range gData {
				a := float32(math.Abs(float64(v)))
				if a > maxAbs {
					maxAbs = a
				}
			}
			t.Logf("parameter %q: %d elements, max|grad|=%.6f", p.Name, len(gData), maxAbs)
		}
	}
}

// TestMambaBlock_FiniteDiffInputGradient verifies the input gradient via finite
// differences (complementing the existing test but with the verify_learn pattern).
func TestMambaBlock_FiniteDiffInputGradient(t *testing.T) {
	ctx := context.Background()
	block, _, _ := newMambaForGradCheck(t)
	input := makeInput(t)

	// Forward + backward with ones output gradient (loss = sum(output)).
	output, err := block.Forward(ctx, input)
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

	grads, err := block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	analyticalGrad := make([]float32, len(grads[0].Data()))
	copy(analyticalGrad, grads[0].Data())

	const eps = float32(1e-4)
	const relTol = float32(1e-2)

	inputData := input.Data()
	numFailed := 0
	for i := range inputData {
		orig := inputData[i]

		inputData[i] = orig + eps
		lPlus := forwardLoss(t, ctx, block, input)

		inputData[i] = orig - eps
		lMinus := forwardLoss(t, ctx, block, input)

		inputData[i] = orig

		numerical := (lPlus - lMinus) / (2 * float32(eps))
		a := analyticalGrad[i]

		diff := float32(math.Abs(float64(a - numerical)))
		denom := float32(math.Max(1.0, math.Max(math.Abs(float64(a)), math.Abs(float64(numerical)))))
		relErr := diff / denom

		if relErr > relTol {
			numFailed++
			if numFailed <= 5 {
				t.Errorf("input grad[%d]: analytical=%.6f numerical=%.6f relErr=%.4f",
					i, a, numerical, relErr)
			}
		}
	}

	if numFailed > 0 {
		t.Errorf("input gradient: %d/%d exceeded relTol=%.4f", numFailed, len(inputData), relTol)
	} else {
		t.Logf("input gradient: %d elements passed finite-diff check (relTol=%.4f)", len(inputData), relTol)
	}
}

// ---------------------------------------------------------------------------
// S4 verify-learn tests
// ---------------------------------------------------------------------------

func newS4ForTest(t *testing.T) (*S4[float32], compute.Engine[float32], numeric.Arithmetic[float32]) {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	s4, err := NewS4[float32]("vl_s4", engine, ops, 4, 4)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Scale parameters to moderate range for numerical stability.
	for _, p := range s4.Parameters() {
		d := p.Value.Data()
		for i := range d {
			d[i] *= 0.3
		}
	}

	return s4, engine, ops
}

func TestS4_LossDecreases(t *testing.T) {
	ctx := context.Background()
	s4, _, _ := newS4ForTest(t)

	const (
		batch  = 1
		seqLen = 4
		dim    = 4
	)

	inputData := make([]float32, batch*seqLen*dim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	target := make([]float32, batch*seqLen*dim)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 30.0
	}

	params := s4.Parameters()
	const lr = float32(0.01)
	const steps = 20

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		zeroGradients(params)

		output, fErr := s4.Forward(ctx, input)
		if fErr != nil {
			t.Fatalf("step %d: Forward: %v", step, fErr)
		}

		loss := computeMSE(output.Data(), target)
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

		gradData := mseGradient(output.Data(), target)
		outputGrad, gErr := tensor.New[float32](output.Shape(), gradData)
		if gErr != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, gErr)
		}

		_, bErr := s4.Backward(ctx, types.FullBackprop, outputGrad, input)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		applyGradientDescent(params, lr)
	}

	if finalLoss >= initialLoss {
		t.Errorf("S4 loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("S4 loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestS4_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	s4, _, _ := newS4ForTest(t)

	const (
		batch  = 1
		seqLen = 4
		dim    = 4
	)

	inputData := make([]float32, batch*seqLen*dim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	output, err := s4.Forward(ctx, input)
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

	_, err = s4.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for _, p := range s4.Parameters() {
		if !hasNonZeroGradient(p) {
			t.Errorf("S4 parameter %q has all-zero gradient", p.Name)
		}
	}
}

// ---------------------------------------------------------------------------
// MIMO SSM verify-learn tests
// ---------------------------------------------------------------------------

func newMIMOForTest(t *testing.T) (*MIMOMambaBlock[float32], compute.Engine[float32], numeric.Arithmetic[float32]) {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewMIMOMambaBlock[float32](
		"vl_mimo", engine, ops,
		4, // dModel
		4, // dInner
		2, // dState
		1, // dtRank
		2, // convKer
		2, // numHeads
	)
	if err != nil {
		t.Fatalf("NewMIMOMambaBlock: %v", err)
	}

	// Keep parameters at original scale for MIMO (scaling too aggressively
	// can suppress gradient flow through the gating path).

	return block, engine, ops
}

func TestMIMOSSM_LossDecreases(t *testing.T) {
	ctx := context.Background()
	block, _, _ := newMIMOForTest(t)

	const (
		batch  = 1
		seqLen = 4
		dModel = 4
	)

	inputData := make([]float32, batch*seqLen*dModel)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dModel}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	target := make([]float32, batch*seqLen*dModel)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 10.0
	}

	params := block.Parameters()
	const lr = float32(0.5)
	const steps = 50

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		zeroGradients(params)

		output, fErr := block.Forward(ctx, input)
		if fErr != nil {
			t.Fatalf("step %d: Forward: %v", step, fErr)
		}

		loss := computeMSE(output.Data(), target)
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

		gradData := mseGradient(output.Data(), target)
		outputGrad, gErr := tensor.New[float32](output.Shape(), gradData)
		if gErr != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, gErr)
		}

		_, bErr := block.Backward(ctx, types.FullBackprop, outputGrad, input)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		applyGradientDescent(params, lr)
	}

	if finalLoss >= initialLoss {
		t.Errorf("MIMO SSM loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("MIMO SSM loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestMIMOSSM_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	block, _, _ := newMIMOForTest(t)

	const (
		batch  = 1
		seqLen = 4
		dModel = 4
	)

	inputData := make([]float32, batch*seqLen*dModel)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dModel}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	output, err := block.Forward(ctx, input)
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

	_, err = block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for _, p := range block.Parameters() {
		if !hasNonZeroGradient(p) {
			t.Errorf("MIMO SSM parameter %q has all-zero gradient", p.Name)
		}
	}
}

// ---------------------------------------------------------------------------
// ComplexSSMState verify-learn tests
// ---------------------------------------------------------------------------

func newComplexSSMForTest(t *testing.T) (*ComplexSSMState[float32], compute.Engine[float32], numeric.Arithmetic[float32]) {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	cs, err := NewComplexSSMState[float32](
		"vl_complex", engine, ops,
		4,  // dModel
		4,  // dInner
		4,  // dState (must be even)
		1,  // dtRank
		2,  // convKer
		16, // maxSeqLen
	)
	if err != nil {
		t.Fatalf("NewComplexSSMState: %v", err)
	}

	// Keep original parameter scale for ComplexSSMState to preserve
	// gradient flow through the gating path.

	return cs, engine, ops
}

func TestComplexSSMState_LossDecreases(t *testing.T) {
	ctx := context.Background()
	cs, _, _ := newComplexSSMForTest(t)

	const (
		batch  = 1
		seqLen = 4
		dModel = 4
	)

	inputData := make([]float32, batch*seqLen*dModel)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dModel}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	target := make([]float32, batch*seqLen*dModel)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 10.0
	}

	params := cs.Parameters()
	const lr = float32(0.5)
	const steps = 50

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		zeroGradients(params)

		output, fErr := cs.Forward(ctx, input)
		if fErr != nil {
			t.Fatalf("step %d: Forward: %v", step, fErr)
		}

		loss := computeMSE(output.Data(), target)
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

		gradData := mseGradient(output.Data(), target)
		outputGrad, gErr := tensor.New[float32](output.Shape(), gradData)
		if gErr != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, gErr)
		}

		_, bErr := cs.Backward(ctx, types.FullBackprop, outputGrad, input)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		applyGradientDescent(params, lr)
	}

	if finalLoss >= initialLoss {
		t.Errorf("ComplexSSMState loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("ComplexSSMState loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestComplexSSMState_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	cs, _, _ := newComplexSSMForTest(t)

	const (
		batch  = 1
		seqLen = 4
		dModel = 4
	)

	inputData := make([]float32, batch*seqLen*dModel)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dModel}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	output, err := cs.Forward(ctx, input)
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

	_, err = cs.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for _, p := range cs.Parameters() {
		if !hasNonZeroGradient(p) {
			t.Errorf("ComplexSSMState parameter %q has all-zero gradient", p.Name)
		}
	}
}

// ---------------------------------------------------------------------------
// BCNorm verify-learn tests
// ---------------------------------------------------------------------------

func TestBCNorm_LossDecreases(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	bn, err := NewBCNorm[float32]("vl_bcnorm", engine, ops, 4)
	if err != nil {
		t.Fatalf("NewBCNorm: %v", err)
	}

	const (
		batch  = 1
		seqLen = 3
		dim    = 4
	)

	inputData := make([]float32, batch*seqLen*dim)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dim}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	target := make([]float32, batch*seqLen*dim)
	for i := range target {
		target[i] = float32(((i*11+3)%17)-8) / 30.0
	}

	params := bn.Parameters()
	const lr = float32(0.01)
	const steps = 20

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		zeroGradients(params)

		output, fErr := bn.Forward(ctx, input)
		if fErr != nil {
			t.Fatalf("step %d: Forward: %v", step, fErr)
		}

		loss := computeMSE(output.Data(), target)
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

		gradData := mseGradient(output.Data(), target)
		outputGrad, gErr := tensor.New[float32](output.Shape(), gradData)
		if gErr != nil {
			t.Fatalf("step %d: tensor.New outputGrad: %v", step, gErr)
		}

		_, bErr := bn.Backward(ctx, outputGrad)
		if bErr != nil {
			t.Fatalf("step %d: Backward: %v", step, bErr)
		}

		applyGradientDescent(params, lr)
	}

	if finalLoss >= initialLoss {
		t.Errorf("BCNorm loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	} else {
		pctDrop := (initialLoss - finalLoss) / initialLoss * 100
		t.Logf("BCNorm loss decreased: initial=%.6f final=%.6f (%.1f%% drop)", initialLoss, finalLoss, pctDrop)
	}
}

func TestBCNorm_GradientsNonZero(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	bn, err := NewBCNorm[float32]("vl_bcnorm", engine, ops, 4)
	if err != nil {
		t.Fatalf("NewBCNorm: %v", err)
	}

	inputData := make([]float32, 12)
	for i := range inputData {
		inputData[i] = float32(((i*7+13)%19)-9) / 20.0
	}
	input, err := tensor.New[float32]([]int{1, 3, 4}, inputData)
	if err != nil {
		t.Fatalf("make input: %v", err)
	}

	output, err := bn.Forward(ctx, input)
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

	_, err = bn.Backward(ctx, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	for _, p := range bn.Parameters() {
		if !hasNonZeroGradient(p) {
			t.Errorf("BCNorm parameter %q has all-zero gradient", p.Name)
		}
	}
}

