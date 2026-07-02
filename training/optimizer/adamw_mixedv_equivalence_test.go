package optimizer

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// referenceMixedV is an independent, self-contained reference implementation
// of the AdamW mixed-precision update that ADR 070 mandates: the second moment
// is accumulated in float64, the first moment and parameters stay in float32,
// and the m/(sqrt(v)+eps) division runs in float64. It deliberately does NOT
// share code with AdamW.stepMixedV so the equivalence test cross-checks the
// optimizer against the documented algorithm rather than against itself.
//
// step() mutates the param/m/v64 state in place so a caller can drive it
// step-by-step alongside the real optimizer.
type referenceMixedV struct {
	beta1, beta2, eps, lr, wd float64
	t                         int
	param                     []float32
	m                         []float32
	v64                       []float64
}

func newReferenceMixedV(initParam []float32, lr, beta1, beta2, eps, wd float64) *referenceMixedV {
	p := make([]float32, len(initParam))
	copy(p, initParam)
	return &referenceMixedV{
		beta1: beta1, beta2: beta2, eps: eps, lr: lr, wd: wd,
		param: p,
		m:     make([]float32, len(initParam)),
		v64:   make([]float64, len(initParam)),
	}
}

func (r *referenceMixedV) step(grad []float32) {
	r.t++
	numer := math.Sqrt(1.0 - math.Pow(r.beta2, float64(r.t)))
	denom := 1.0 - math.Pow(r.beta1, float64(r.t))
	alpha := r.lr * (numer / denom)
	lrWd := r.lr * r.wd

	for i := range r.param {
		g := float64(grad[i])
		mOld := float64(r.m[i])
		mNew := r.beta1*mOld + (1.0-r.beta1)*g
		r.m[i] = float32(mNew)

		r.v64[i] = r.beta2*r.v64[i] + (1.0-r.beta2)*g*g

		denomI := math.Sqrt(r.v64[i]) + r.eps
		update := alpha * mNew / denomI

		pv := float64(r.param[i])
		pv = pv - update - lrWd*pv
		r.param[i] = float32(pv)
	}
}

// TestAdamW_MixedV_MatchesReference asserts that the optimized host-round-trip-
// minimized stepMixedV path (host-only m/v64 sidecars + device-side gradient
// zeroing, ADR 070) produces a parameter trajectory bit-for-bit equal to the
// independent reference AdamW mixed update over several steps, across a table of
// shapes, hyperparameters, and gradient regimes (including the near-zero
// underflow regime that motivated the f64 second moment).
//
// The optimizer runs on a CPUEngine[float32], which takes the same stepMixedV
// path as the GB10 GPU engine (shouldUseMixedPrecisionV is true for both); the
// point of the test is numerical equivalence of the algorithm, independent of
// where the elementwise ops physically execute.
func TestAdamW_MixedV_MatchesReference(t *testing.T) {
	type gradFn func(step, i int) float32

	cases := []struct {
		name        string
		shape       []int
		initParam   []float32
		lr          float64
		beta1       float64
		beta2       float64
		eps         float64
		weightDecay float64
		steps       int
		grad        gradFn
	}{
		{
			name:      "typical_gradients_no_wd",
			shape:     []int{4},
			initParam: []float32{0.5, -0.25, 1.0, -1.0},
			lr:        1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.0,
			steps: 12,
			grad: func(step, i int) float32 {
				return float32(0.1*float64(i+1)) * float32(math.Cos(float64(step)))
			},
		},
		{
			name:      "with_weight_decay",
			shape:     []int{6},
			initParam: []float32{1, 2, 3, -1, -2, -3},
			lr:        5e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01,
			steps: 15,
			grad: func(step, i int) float32 {
				return float32(0.05 * float64((step%3)-1) * float64(i+1))
			},
		},
		{
			name:      "near_zero_underflow_regime",
			shape:     []int{4},
			initParam: []float32{0.5, 0.5, 0.5, 0.5},
			lr:        1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-5, weightDecay: 0.0,
			steps: 50,
			grad: func(step, i int) float32 {
				if i%2 == 0 {
					return 1e-10
				}
				return -1e-10
			},
		},
		{
			name:      "mixed_magnitude_gradients",
			shape:     []int{2, 3},
			initParam: []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6},
			lr:        2e-3, beta1: 0.95, beta2: 0.9995, eps: 1e-7, weightDecay: 0.005,
			steps: 20,
			grad: func(step, i int) float32 {
				scale := math.Pow(10, float64(i-3)) // spans 1e-3 .. 1e2
				return float32(scale * math.Sin(float64(step+i)))
			},
		},
	}

	ops := numeric.Float32Ops{}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine := compute.NewCPUEngine[float32](ops)

			paramTensor, err := tensor.New[float32](tc.shape, append([]float32(nil), tc.initParam...))
			if err != nil {
				t.Fatalf("new param tensor: %v", err)
			}
			param, err := graph.NewParameter[float32]("w", paramTensor, tensor.New[float32])
			if err != nil {
				t.Fatalf("new parameter: %v", err)
			}

			adam := NewAdamW[float32](engine, float32(tc.lr), float32(tc.beta1),
				float32(tc.beta2), float32(tc.eps), float32(tc.weightDecay))
			if !adam.useMixedV {
				t.Fatalf("expected mixed-precision path active for float32 CPU engine")
			}

			ref := newReferenceMixedV(tc.initParam, tc.lr, tc.beta1, tc.beta2, tc.eps, tc.weightDecay)

			n := len(tc.initParam)
			for step := 0; step < tc.steps; step++ {
				gradVals := make([]float32, n)
				for i := range gradVals {
					gradVals[i] = tc.grad(step, i)
				}

				gradTensor, err := tensor.New[float32](tc.shape, append([]float32(nil), gradVals...))
				if err != nil {
					t.Fatalf("step %d: new grad tensor: %v", step, err)
				}
				if err := param.AddGradient(gradTensor); err != nil {
					t.Fatalf("step %d: add grad: %v", step, err)
				}

				if err := adam.Step(context.Background(), []*graph.Parameter[float32]{param}); err != nil {
					t.Fatalf("step %d: optimizer step: %v", step, err)
				}
				ref.step(gradVals)

				got := param.Value.Data()
				for i := range got {
					// Both implementations perform the identical float64
					// arithmetic and round to float32 the same way, so the
					// results must be bit-identical. A tiny tolerance guards
					// against incidental reassociation only.
					if diff := math.Abs(float64(got[i]) - float64(ref.param[i])); diff > 1e-6 {
						t.Fatalf("case %q step %d param[%d]: optimizer=%v reference=%v (diff=%g)",
							tc.name, step, i, got[i], ref.param[i], diff)
					}
				}

				// Gradient must be zeroed after the step (device-side Fill on
				// GPU, host loop on CPU); a stale gradient would corrupt the
				// next step.
				for i, gv := range param.Gradient.Data() {
					if gv != 0 {
						t.Fatalf("case %q step %d: gradient[%d] not zeroed: %v", tc.name, step, i, gv)
					}
				}
			}
		})
	}
}

// referenceFullF64 is a straightforward, fully-float64 AdamW reference: the
// parameters, both moments, and every intermediate live in float64 with no
// rounding to float32 anywhere. It is the "ground truth" trajectory that the
// mixed-precision path (f32 param/m, f64 v) is supposed to track closely;
// ADR 070's premise is that promoting only v captures essentially all of the
// f64 stability.
type referenceFullF64 struct {
	beta1, beta2, eps, lr, wd float64
	t                         int
	param                     []float64
	m                         []float64
	v                         []float64
}

func newReferenceFullF64(initParam []float32, lr, beta1, beta2, eps, wd float64) *referenceFullF64 {
	p := make([]float64, len(initParam))
	for i, v := range initParam {
		p[i] = float64(v)
	}
	return &referenceFullF64{
		beta1: beta1, beta2: beta2, eps: eps, lr: lr, wd: wd,
		param: p,
		m:     make([]float64, len(initParam)),
		v:     make([]float64, len(initParam)),
	}
}

func (r *referenceFullF64) step(grad []float32) {
	r.t++
	numer := math.Sqrt(1.0 - math.Pow(r.beta2, float64(r.t)))
	denom := 1.0 - math.Pow(r.beta1, float64(r.t))
	alpha := r.lr * (numer / denom)
	lrWd := r.lr * r.wd

	for i := range r.param {
		g := float64(grad[i])
		r.m[i] = r.beta1*r.m[i] + (1.0-r.beta1)*g
		r.v[i] = r.beta2*r.v[i] + (1.0-r.beta2)*g*g

		update := alpha * r.m[i] / (math.Sqrt(r.v[i]) + r.eps)
		r.param[i] = r.param[i] - update - lrWd*r.param[i]
	}
}

// TestAdamW_MixedV_TracksFullF64Reference asserts that the mixed-precision
// stepMixedV path stays within a small tolerance of a straightforward
// all-float64 AdamW over many steps, including gradient regimes where an
// all-f32 second moment would lose precision (near-zero gradients whose
// squares underflow f32, and gradients spanning several orders of magnitude).
// Unlike TestAdamW_MixedV_MatchesReference -- which checks bit-equivalence
// against an independent implementation of the same mixed algorithm -- this
// test bounds the mixed path's drift from true f64 AdamW, i.e. it verifies
// that promoting only v (per ADR 070) is sufficient to track the full-f64
// trajectory.
func TestAdamW_MixedV_TracksFullF64Reference(t *testing.T) {
	type gradFn func(step, i int) float32

	cases := []struct {
		name        string
		shape       []int
		initParam   []float32
		lr          float64
		beta1       float64
		beta2       float64
		eps         float64
		weightDecay float64
		steps       int
		relTol      float64 // tolerance relative to max(1, |ref param|)
		grad        gradFn
	}{
		{
			name:      "typical_gradients",
			shape:     []int{4},
			initParam: []float32{0.5, -0.25, 1.0, -1.0},
			lr:        1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.0,
			steps:  25,
			relTol: 1e-5,
			grad: func(step, i int) float32 {
				return float32(0.1*float64(i+1)) * float32(math.Cos(float64(step)))
			},
		},
		{
			name:      "weight_decay_and_large_gradients",
			shape:     []int{5},
			initParam: []float32{1, -2, 3, -4, 5},
			lr:        1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01,
			steps:  20,
			relTol: 1e-5,
			grad: func(step, i int) float32 {
				// Large-magnitude gradients: up to 1e3.
				return float32(math.Pow(10, float64(i%4)) * math.Sin(float64(step+i)))
			},
		},
		{
			name:      "tiny_gradients_f32_v_would_underflow",
			shape:     []int{4},
			initParam: []float32{0.5, 0.5, 0.5, 0.5},
			lr:        1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-5, weightDecay: 0.0,
			steps: 50,
			// Updates are ~lr in magnitude but param stays ~0.5; the only
			// f32 rounding is on param/m, so drift stays tiny.
			relTol: 1e-5,
			grad: func(step, i int) float32 {
				// g^2 = 1e-20 underflows toward the f32 denormal floor; an
				// f32 v would make sqrt(v)+eps collapse to eps.
				if i%2 == 0 {
					return 1e-10
				}
				return -1e-10
			},
		},
		{
			name:      "wide_dynamic_range",
			shape:     []int{8},
			initParam: []float32{0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4},
			lr:        2e-3, beta1: 0.95, beta2: 0.9995, eps: 1e-7, weightDecay: 0.005,
			steps:  30,
			relTol: 1e-5,
			grad: func(step, i int) float32 {
				// Spans 1e-6 .. 1e1 across elements in the same tensor.
				scale := math.Pow(10, float64(i-6))
				return float32(scale * math.Sin(float64(step+i)))
			},
		},
	}

	ops := numeric.Float32Ops{}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine := compute.NewCPUEngine[float32](ops)

			paramTensor, err := tensor.New[float32](tc.shape, append([]float32(nil), tc.initParam...))
			if err != nil {
				t.Fatalf("new param tensor: %v", err)
			}
			param, err := graph.NewParameter[float32]("w", paramTensor, tensor.New[float32])
			if err != nil {
				t.Fatalf("new parameter: %v", err)
			}

			adam := NewAdamW[float32](engine, float32(tc.lr), float32(tc.beta1),
				float32(tc.beta2), float32(tc.eps), float32(tc.weightDecay))
			if !adam.useMixedV {
				t.Fatalf("expected mixed-precision path active for float32 CPU engine")
			}

			ref := newReferenceFullF64(tc.initParam, tc.lr, tc.beta1, tc.beta2, tc.eps, tc.weightDecay)

			n := len(tc.initParam)
			for step := 0; step < tc.steps; step++ {
				gradVals := make([]float32, n)
				for i := range gradVals {
					gradVals[i] = tc.grad(step, i)
				}

				gradTensor, err := tensor.New[float32](tc.shape, append([]float32(nil), gradVals...))
				if err != nil {
					t.Fatalf("step %d: new grad tensor: %v", step, err)
				}
				if err := param.AddGradient(gradTensor); err != nil {
					t.Fatalf("step %d: add grad: %v", step, err)
				}

				if err := adam.Step(context.Background(), []*graph.Parameter[float32]{param}); err != nil {
					t.Fatalf("step %d: optimizer step: %v", step, err)
				}
				ref.step(gradVals)

				got := param.Value.Data()
				for i := range got {
					scale := math.Max(1.0, math.Abs(ref.param[i]))
					if diff := math.Abs(float64(got[i]) - ref.param[i]); diff > tc.relTol*scale {
						t.Fatalf("case %q step %d param[%d]: mixed=%v fullF64=%v (diff=%g, tol=%g)",
							tc.name, step, i, got[i], ref.param[i], diff, tc.relTol*scale)
					}
				}
			}
		})
	}
}
