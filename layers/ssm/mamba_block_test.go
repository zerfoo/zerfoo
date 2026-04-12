package ssm

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func makeTestTensor(t *testing.T, shape []int, seed int) *tensor.TensorNumeric[float32] {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(((i*7+seed)%19)-9) / 20.0
	}
	out, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("makeTestTensor: %v", err)
	}
	return out
}

func TestMambaBlock(t *testing.T) {
	tests := []struct {
		name    string
		dModel  int
		dInner  int
		dState  int
		dtRank  int
		convKer int
		batch   int
		seqLen  int
	}{
		{
			name:    "small",
			dModel:  4,
			dInner:  8,
			dState:  4,
			dtRank:  2,
			convKer: 4,
			batch:   1,
			seqLen:  6,
		},
		{
			name:    "batch2",
			dModel:  4,
			dInner:  8,
			dState:  4,
			dtRank:  2,
			convKer: 4,
			batch:   2,
			seqLen:  4,
		},
		{
			name:    "tiny",
			dModel:  2,
			dInner:  4,
			dState:  2,
			dtRank:  1,
			convKer: 2,
			batch:   1,
			seqLen:  3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine(ops)

			block, err := NewMambaBlock[float32](
				"test_mamba", engine, ops,
				tt.dModel, tt.dInner, tt.dState, tt.dtRank, tt.convKer,
			)
			if err != nil {
				t.Fatalf("NewMambaBlock: %v", err)
			}

			input := makeTestTensor(t, []int{tt.batch, tt.seqLen, tt.dModel}, 42)

			output, err := block.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			// Verify output shape
			expectedShape := []int{tt.batch, tt.seqLen, tt.dModel}
			gotShape := output.Shape()
			if len(gotShape) != len(expectedShape) {
				t.Fatalf("output shape length: got %d, want %d", len(gotShape), len(expectedShape))
			}
			for i := range expectedShape {
				if gotShape[i] != expectedShape[i] {
					t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], expectedShape[i])
				}
			}

			// Verify output is not all zeros
			outData := output.Data()
			allZero := true
			for _, v := range outData {
				if v != 0 {
					allZero = false
					break
				}
			}
			if allZero {
				t.Error("output is all zeros")
			}

			// Verify output contains finite values
			for i, v := range outData {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] is not finite: %v", i, v)
					break
				}
			}
		})
	}
}

func TestMambaBlock_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dModel := 4
	dInner := 8
	dState := 4
	dtRank := 2
	convKer := 4
	batch := 1
	seqLen := 4

	block, err := NewMambaBlock[float32](
		"test_mamba_bw", engine, ops,
		dModel, dInner, dState, dtRank, convKer,
	)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	input := makeTestTensor(t, []int{batch, seqLen, dModel}, 7)

	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create output gradient
	dOutData := make([]float32, len(output.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*11+5)%13)-6) / 10.0
	}
	dOut, err := tensor.New[float32](output.Shape(), dOutData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	grads, err := block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}

	dInput := grads[0]
	// Verify gradient shape matches input shape
	inputShape := input.Shape()
	gradShape := dInput.Shape()
	for i := range inputShape {
		if gradShape[i] != inputShape[i] {
			t.Errorf("grad shape[%d]: got %d, want %d", i, gradShape[i], inputShape[i])
		}
	}

	// Verify gradients are finite
	gradData := dInput.Data()
	for i, v := range gradData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("input grad[%d] is not finite: %v", i, v)
			break
		}
	}

	// Verify parameter gradients exist and are finite
	for _, p := range block.Parameters() {
		if p.Gradient == nil {
			t.Errorf("parameter %s has nil gradient", p.Name)
			continue
		}
		for i, v := range p.Gradient.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("parameter %s grad[%d] is not finite: %v", p.Name, i, v)
				break
			}
		}
	}
}

func TestMambaBlock_BackwardFiniteDiff(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	dModel := 2
	dInner := 4
	dState := 2
	dtRank := 1
	convKer := 2
	batch := 1
	seqLen := 3

	block, err := NewMambaBlock[float32](
		"test_mamba_fd", engine, ops,
		dModel, dInner, dState, dtRank, convKer,
	)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	input := makeTestTensor(t, []int{batch, seqLen, dModel}, 13)

	// Forward + backward
	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	dOutData := make([]float32, len(output.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*3+1)%7)-3) / 10.0
	}
	dOut, err := tensor.New[float32](output.Shape(), dOutData)
	if err != nil {
		t.Fatalf("creating dOut: %v", err)
	}

	grads, err := block.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	dInput := grads[0]

	// Finite difference check on input gradient
	eps := float32(1e-3)
	tol := float32(5e-3)

	analyticalGrad := make([]float32, len(dInput.Data()))
	copy(analyticalGrad, dInput.Data())

	inputData := input.Data()
	numFailed := 0
	for i := range inputData {
		orig := inputData[i]

		inputData[i] = orig + eps
		oPlus, err := block.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward(+eps): %v", err)
		}
		lPlus := dotProduct(oPlus.Data(), dOutData)

		inputData[i] = orig - eps
		oMinus, err := block.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward(-eps): %v", err)
		}
		lMinus := dotProduct(oMinus.Data(), dOutData)

		inputData[i] = orig

		numerical := (lPlus - lMinus) / (2 * eps)
		a := analyticalGrad[i]
		diff := float32(math.Abs(float64(a - numerical)))
		denom := float32(math.Max(1.0, math.Max(math.Abs(float64(a)), math.Abs(float64(numerical)))))

		if diff/denom > tol {
			numFailed++
			if numFailed <= 5 {
				t.Errorf("input grad[%d]: analytical=%.6f numerical=%.6f relErr=%.4f",
					i, a, numerical, diff/denom)
			}
		}
	}
	if numFailed > 0 {
		t.Errorf("input gradient: %d/%d exceeded tol=%.4f", numFailed, len(inputData), tol)
	} else {
		t.Logf("input gradient: %d elements passed finite-diff check", len(inputData))
	}
}

func TestMambaBlock_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewMambaBlock[float32]("test", engine, ops, 4, 8, 4, 2, 4)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	if got := block.OpType(); got != "MambaBlock" {
		t.Errorf("OpType: got %q, want %q", got, "MambaBlock")
	}
}

func TestMambaBlock_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	block, err := NewMambaBlock[float32]("test", engine, ops, 4, 8, 4, 2, 4)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	params := block.Parameters()
	if len(params) == 0 {
		t.Fatal("expected non-empty parameters")
	}

	// Should have: inProj weights, conv_weight, xProj weights, dtProj weights, A, D, outProj weights
	expectedNames := map[string]bool{
		"test_in_proj_weights":  true,
		"test_conv_weight":      true,
		"test_conv_bias":        true,
		"test_x_proj_weights":   true,
		"test_dt_proj_weights":  true,
		"test_A":                true,
		"test_D":                true,
		"test_out_proj_weights": true,
	}

	for _, p := range params {
		if !expectedNames[p.Name] {
			t.Errorf("unexpected parameter name: %q", p.Name)
		}
		delete(expectedNames, p.Name)
	}
	for name := range expectedNames {
		t.Errorf("missing expected parameter: %q", name)
	}
}

func TestMambaBlock_InvalidInputs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	ctx := context.Background()

	block, err := NewMambaBlock[float32]("test", engine, ops, 4, 8, 4, 2, 4)
	if err != nil {
		t.Fatalf("NewMambaBlock: %v", err)
	}

	// No inputs
	_, err = block.Forward(ctx)
	if err == nil {
		t.Error("expected error with no inputs")
	}

	// Wrong dimensionality
	bad2d, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))
	_, err = block.Forward(ctx, bad2d)
	if err == nil {
		t.Error("expected error with 2D input")
	}
}

func TestNewMambaBlock_Validation(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	_, err := NewMambaBlock[float32]("", engine, ops, 4, 8, 4, 2, 4)
	if err == nil {
		t.Error("expected error with empty name")
	}

	_, err = NewMambaBlock[float32]("test", engine, ops, 0, 8, 4, 2, 4)
	if err == nil {
		t.Error("expected error with zero dModel")
	}

	_, err = NewMambaBlock[float32]("test", engine, ops, 4, -1, 4, 2, 4)
	if err == nil {
		t.Error("expected error with negative dInner")
	}
}

func dotProduct(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// TestExpTrapDiscretization verifies the exponential-trapezoidal discretization
// (Mamba 3) produces different (and correct) B̄ values compared to ZOH, and
// that ZOH mode remains unchanged.
//
// Reference values are derived from the formula:
//
//	Ā = exp(Δ * A)
//	ZOH:     B̄ = Δ * B
//	ExpTrap: B̄ = Δ * (1 + Ā) / 2 * B
func TestExpTrapDiscretization(t *testing.T) {
	// We use a tiny SSM (dInner=1, dState=1) with controlled weights so we can
	// compute the expected hidden state exactly by hand.
	//
	// Setup:
	//   A (log-space): log(1) = 0  →  A_real = -exp(0) = -1.0
	//   dt = fixed positive value (via softplus-inverse so softplus gives ~0.5)
	//   B = 1.0, C = 1.0, D = 0.0 (no skip connection)
	//   x = 1.0 (single time step, single batch)
	//
	// Discretized values:
	//   Ā = exp(0.5 * (-1)) = exp(-0.5) ≈ 0.6065307
	//   ZOH:     B̄ = 0.5 * 1.0 = 0.5
	//   ExpTrap: B̄ = 0.5 * (1 + 0.6065307) / 2 * 1.0 ≈ 0.4016154
	//
	// With h[0] = 0 (initial state):
	//   h[1] = Ā * 0 + B̄ * x = B̄
	//   y[1] = C * h[1] + D * x = B̄
	//
	// So the scan output y equals B̄ directly.

	const (
		dt   = float64(0.5)
		aLog = float64(0) // A stored in log-space; A_real = -exp(aLog) = -1
	)

	aReal := -math.Exp(aLog)
	dA := math.Exp(dt * aReal) // ≈ 0.6065307

	refZOH := dt * 1.0                    // B̄_ZOH = Δ * B
	refExpTrap := dt * (1.0+dA) / 2.0 * 1.0 // B̄_ExpTrap = Δ * (1+Ā)/2 * B

	const tol = 1e-5

	tests := []struct {
		name     string
		mode     DiscretizationMode
		wantScan float64 // expected y output (= B̄ for this setup)
	}{
		{"ZOH", ZOH, refZOH},
		{"ExpTrap", ExpTrap, refExpTrap},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine(ops)

			// dModel=1 dInner=1 dState=1 dtRank=1 convKer=1
			block, err := NewMambaBlock[float32](
				"test_disc", engine, ops,
				1, 1, 1, 1, 1,
				WithDiscretizationMode[float32](tt.mode),
			)
			if err != nil {
				t.Fatalf("NewMambaBlock: %v", err)
			}

			// Override A to log(1)=0 (so A_real = -1) and D to 0 (no skip).
			aData := block.A.Value.Data()
			aData[0] = float32(aLog)
			dData := block.D.Value.Data()
			dData[0] = 0

			// Zero out all linear projection weights/biases to isolate the SSM
			// recurrence. We set them so the effective x=1, dt≈0.5, B=1, C=1
			// reach selectiveScan unchanged.
			//
			// Strategy: bypass projections by directly calling selectiveScan with
			// hand-crafted tensors.  We use softplus⁻¹(0.5) = log(exp(0.5)-1) ≈ 0.3133
			// for dt so that after softplus we get exactly 0.5.

			// softplus_inv(0.5) = log(exp(0.5) - 1)
			dtRaw := float32(math.Log(math.Exp(dt) - 1))

			batch, seqLen := 1, 1

			xTensor, _ := tensor.New[float32]([]int{batch, seqLen, 1}, []float32{1.0})
			dtTensor, _ := tensor.New[float32]([]int{batch, seqLen, 1}, []float32{dtRaw})
			bTensor, _ := tensor.New[float32]([]int{batch, seqLen, 1}, []float32{1.0})
			cTensor, _ := tensor.New[float32]([]int{batch, seqLen, 1}, []float32{1.0})

			// Apply softplus to dtTensor to get actual dt ≈ 0.5
			dtActual, err := block.applySoftplus(ctx, dtTensor)
			if err != nil {
				t.Fatalf("applySoftplus: %v", err)
			}

			y, _, err := block.selectiveScan(ctx, xTensor, dtActual, bTensor, cTensor, batch, seqLen)
			if err != nil {
				t.Fatalf("selectiveScan: %v", err)
			}

			got := float64(y.Data()[0])
			diff := math.Abs(got - tt.wantScan)
			if diff > tol {
				t.Errorf("mode=%s: scan output=%.8f, want=%.8f (diff=%.2e > tol=%.2e)",
					tt.name, got, tt.wantScan, diff, tol)
			} else {
				t.Logf("mode=%s: scan output=%.8f matches reference=%.8f (diff=%.2e)",
					tt.name, got, tt.wantScan, diff)
			}
		})
	}

	// Sanity check: ZOH and ExpTrap must produce different outputs.
	if math.Abs(refZOH-refExpTrap) < tol {
		t.Errorf("ZOH and ExpTrap reference values are indistinguishable: ZOH=%.8f ExpTrap=%.8f",
			refZOH, refExpTrap)
	}
}

// TestExpTrapDiscretization_ZOHUnchanged verifies that adding the new
// DiscretizationMode type does not change the behaviour of blocks created
// without the option (ZOH must remain the default).
func TestExpTrapDiscretization_ZOHUnchanged(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	// Block created with no option — must behave identically to explicit ZOH.
	blockDefault, err := NewMambaBlock[float32]("default", engine, ops, 4, 8, 4, 2, 4)
	if err != nil {
		t.Fatalf("NewMambaBlock (default): %v", err)
	}
	blockZOH, err := NewMambaBlock[float32]("explicit_zoh", engine, ops, 4, 8, 4, 2, 4,
		WithDiscretizationMode[float32](ZOH))
	if err != nil {
		t.Fatalf("NewMambaBlock (ZOH): %v", err)
	}

	if blockDefault.discMode != ZOH {
		t.Errorf("default discMode: got %v, want ZOH(%v)", blockDefault.discMode, ZOH)
	}
	if blockZOH.discMode != ZOH {
		t.Errorf("explicit ZOH discMode: got %v, want ZOH(%v)", blockZOH.discMode, ZOH)
	}

	// Both must produce identical outputs from the same input.
	input := makeTestTensor(t, []int{1, 3, 4}, 99)

	// Copy weights from blockDefault to blockZOH so both use the same parameters.
	for i, p := range blockDefault.Parameters() {
		pZOH := blockZOH.Parameters()[i]
		src := p.Value.Data()
		dst := pZOH.Value.Data()
		copy(dst, src)
	}

	outDefault, err := blockDefault.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward (default): %v", err)
	}
	outZOH, err := blockZOH.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward (ZOH): %v", err)
	}

	dDefault := outDefault.Data()
	dZOH := outZOH.Data()
	for i := range dDefault {
		diff := math.Abs(float64(dDefault[i] - dZOH[i]))
		if diff > 1e-6 {
			t.Errorf("output[%d]: default=%.8f, explicitZOH=%.8f (diff=%.2e)",
				i, dDefault[i], dZOH[i], diff)
		}
	}
}
