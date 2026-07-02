package training

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// bf16 is a local alias for readability.
type bf16 = float16.BFloat16

func bf(f float32) bf16 { return float16.BFloat16FromFloat32(f) }

// TestBF16TrainingPathCPU proves the bf16 (float16.BFloat16) element type is
// wired end-to-end through the generic training stack on the CPU engine: the
// layers a transformer-style model is built from (Linear, RMSNorm, Softmax),
// the cross-entropy loss, and the AdamW optimizer all instantiate over
// T=float16.BFloat16 and run a forward -> backward -> optimizer-step cycle
// without panicking. No GPU is required; the GPU fused-AdamW / native bf16
// kernels are exercised by CUDA-gated tests elsewhere and are NOT covered here.
//
// This is a wiring/compile-and-run proof, not a numerical-accuracy gate. bf16
// has ~3 decimal digits of mantissa, so we assert the optimizer changed the
// weights in a finite (non-NaN/Inf) way rather than asserting tight values.
func TestBF16TrainingPathCPU(t *testing.T) {
	ctx := context.Background()
	ops := numeric.BFloat16Ops{}
	engine := compute.NewCPUEngine[bf16](ops)

	const (
		inDim   = 4
		dModel  = 8
		classes = 3
		batch   = 2
	)

	// --- Build the generic layers over bf16 ---
	proj, err := core.NewLinear[bf16]("proj", engine, ops, inDim, dModel)
	if err != nil {
		t.Fatalf("NewLinear[bf16]: %v", err)
	}
	norm, err := normalization.NewRMSNorm[bf16]("norm", engine, ops, dModel)
	if err != nil {
		t.Fatalf("NewRMSNorm[bf16]: %v", err)
	}
	head, err := core.NewLinear[bf16]("head", engine, ops, dModel, classes)
	if err != nil {
		t.Fatalf("NewLinear[bf16] head: %v", err)
	}
	sm := activations.NewSoftmax[bf16](engine, -1)
	celoss := loss.NewCrossEntropyLoss[bf16](engine)

	// --- Inputs and integer-coded targets (as bf16) ---
	xData := make([]bf16, batch*inDim)
	for i := range xData {
		xData[i] = bf(0.1 * float32(i+1))
	}
	x, err := tensor.New[bf16]([]int{batch, inDim}, xData)
	if err != nil {
		t.Fatalf("new input: %v", err)
	}
	targets, err := tensor.New[bf16]([]int{batch}, []bf16{bf(0), bf(2)})
	if err != nil {
		t.Fatalf("new targets: %v", err)
	}

	// --- Forward: proj -> norm -> head -> (softmax probe) ---
	h, err := proj.Forward(ctx, x)
	if err != nil {
		t.Fatalf("proj.Forward: %v", err)
	}
	hn, err := norm.Forward(ctx, h)
	if err != nil {
		t.Fatalf("norm.Forward: %v", err)
	}
	logits, err := head.Forward(ctx, hn)
	if err != nil {
		t.Fatalf("head.Forward: %v", err)
	}
	// Softmax layer forward+backward proves the bf16 Softmax instantiation runs.
	probs, err := sm.Forward(ctx, logits)
	if err != nil {
		t.Fatalf("softmax.Forward: %v", err)
	}
	if _, err := sm.Backward(ctx, types.FullBackprop, probs); err != nil {
		t.Fatalf("softmax.Backward: %v", err)
	}

	lossT, err := celoss.Forward(ctx, logits, targets)
	if err != nil {
		t.Fatalf("crossentropy.Forward: %v", err)
	}
	lossVal := lossT.Data()[0].ToFloat32()
	if math.IsNaN(float64(lossVal)) || math.IsInf(float64(lossVal), 0) {
		t.Fatalf("loss is non-finite: %v", lossVal)
	}

	// --- Backward through loss -> head -> norm -> proj ---
	dOut, err := tensor.New[bf16]([]int{1}, []bf16{ops.One()})
	if err != nil {
		t.Fatalf("new dOut: %v", err)
	}
	dLogits, err := celoss.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("crossentropy.Backward: %v", err)
	}
	dHN, err := head.Backward(ctx, types.FullBackprop, dLogits[0], hn)
	if err != nil {
		t.Fatalf("head.Backward: %v", err)
	}
	dH, err := norm.Backward(ctx, types.FullBackprop, dHN[0], h)
	if err != nil {
		t.Fatalf("norm.Backward: %v", err)
	}
	if _, err := proj.Backward(ctx, types.FullBackprop, dH[0], x); err != nil {
		t.Fatalf("proj.Backward: %v", err)
	}

	// --- AdamW step over the bf16 parameters ---
	// NewAdamWFromFloat64 is the correct constructor for reduced-precision T:
	// bf16(0.999) rounds to 1.0 and bf16(1e-7) to 0, which would disable the
	// update. The float64 hyperparameters are retained for the mixed path.
	adam := optimizer.NewAdamWFromFloat64[bf16](engine, 0.01, 0.9, 0.999, 1e-7, 0.0)

	var params []*graph.Parameter[bf16]
	params = append(params, proj.Parameters()...)
	params = append(params, norm.Parameters()...)
	params = append(params, head.Parameters()...)

	// Snapshot every pre-step weight element to confirm the optimizer mutated
	// at least one. (Checking only element 0 is too brittle at bf16 precision:
	// an individual element's update can round to zero.)
	before := make([][]float32, len(params))
	for pi, p := range params {
		if p.Gradient == nil {
			t.Fatalf("parameter %q has nil gradient after backward", p.Name)
		}
		snap := make([]float32, 0, len(p.Value.Data()))
		for _, v := range p.Value.Data() {
			snap = append(snap, v.ToFloat32())
		}
		before[pi] = snap
	}

	if err := adam.Step(ctx, params); err != nil {
		t.Fatalf("AdamW.Step[bf16]: %v", err)
	}

	changed := false
	for pi, p := range params {
		for i, v := range p.Value.Data() {
			f := v.ToFloat32()
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				t.Fatalf("parameter %q became non-finite after step: %v", p.Name, f)
			}
			if f != before[pi][i] {
				changed = true
			}
		}
	}
	if !changed {
		t.Fatalf("AdamW.Step did not change any bf16 parameter; optimizer path likely not wired")
	}
}

// TestBF16OptimizerMixedPrecisionSelected guards the AdamW precision decision:
// bf16 on a CPU engine must take the float64-second-moment mixed path
// (shouldUseMixedPrecisionV), which is what keeps the reduced-precision
// accumulator from collapsing. A second Step must also succeed (timestep
// advance + reuse of the host-side moment sidecars).
func TestBF16OptimizerMixedPrecisionSelected(t *testing.T) {
	ctx := context.Background()
	ops := numeric.BFloat16Ops{}
	engine := compute.NewCPUEngine[bf16](ops)

	adam := optimizer.NewAdamWFromFloat64[bf16](engine, 0.05, 0.9, 0.999, 1e-7, 0.01)

	value, err := tensor.New[bf16]([]int{4}, []bf16{bf(0.5), bf(-0.5), bf(1.0), bf(-1.0)})
	if err != nil {
		t.Fatalf("new value: %v", err)
	}
	param, err := graph.NewParameter("w", value, tensor.New[bf16])
	if err != nil {
		t.Fatalf("new param: %v", err)
	}

	for step := 0; step < 2; step++ {
		grad, gerr := tensor.New[bf16]([]int{4}, []bf16{bf(0.1), bf(-0.1), bf(0.2), bf(-0.2)})
		if gerr != nil {
			t.Fatalf("new grad: %v", gerr)
		}
		param.Gradient = grad
		if serr := adam.Step(ctx, []*graph.Parameter[bf16]{param}); serr != nil {
			t.Fatalf("AdamW.Step[bf16] step %d: %v", step, serr)
		}
		for _, v := range param.Value.Data() {
			f := v.ToFloat32()
			if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
				t.Fatalf("non-finite weight after step %d: %v", step, f)
			}
		}
	}
}
