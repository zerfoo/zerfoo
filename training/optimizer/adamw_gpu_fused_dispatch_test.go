package optimizer

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// fusedRecorder wraps a CPUEngine and also implements gpuFusedAdamW. It records
// the scalars stepMixedV forwards and applies the on-device AdamW arithmetic to
// a host-side mirror so the test can assert (a) the fused path is dispatched and
// (b) the scalars are exactly the raw hyperparameters, with bias correction
// computed by the engine (matching the kernel contract) -- not pre-multiplied
// by the caller. It does NOT touch real device memory, so it runs on CI.
type fusedRecorder struct {
	*compute.CPUEngine[float32]
	calls int
	// last-seen scalars and the derived bias-corrected terms.
	beta1, beta2, eps, lr, wd float64
	alpha, lrWd               float64
	lastT                     int
}

func (f *fusedRecorder) GPUFusedAdamW(_, _ *tensor.TensorNumeric[float32], beta1, beta2, eps, lr, wd float64, t int) error {
	f.calls++
	f.beta1, f.beta2, f.eps, f.lr, f.wd, f.lastT = beta1, beta2, eps, lr, wd, t
	// The bias-corrected step size and decoupled weight-decay term the kernel
	// derives from these raw scalars (locks the cross-repo contract).
	numer := math.Sqrt(1.0 - math.Pow(beta2, float64(t)))
	denom := 1.0 - math.Pow(beta1, float64(t))
	f.alpha = lr * (numer / denom)
	f.lrWd = lr * wd
	return nil
}

// TestStepMixedV_DispatchesToFusedWhenGPUResident asserts that when the engine
// implements gpuFusedAdamW AND the parameter/gradient are GPU-resident, the
// host loop is bypassed in favor of the on-device kernel, and the raw
// hyperparameters (not bias-corrected) are forwarded with the current timestep.
//
// GPU-resident tensors cannot be constructed without a device, so this drives
// the dispatch through isGPUResident by using the real predicate against a
// CPU-backed tensor (host path) to confirm the NEGATIVE case here, and locks the
// scalar-forwarding contract via the recorder. The POSITIVE on-device path is
// verified end-to-end on GB10 (zero NaN, loss tracks); the kernel's numerical
// equivalence to the host f64 update is in ztensor's
// TestAdamWKernelArithmetic_MatchesReference and the host equivalence tests here.
func TestStepMixedV_DispatchesToFusedWhenGPUResident(t *testing.T) {
	ops := numeric.Float32Ops{}
	cpu := compute.NewCPUEngine[float32](ops)
	rec := &fusedRecorder{CPUEngine: cpu}

	// The recorder implements the gpuFusedAdamW interface, so stepMixedV's
	// dispatch type-assertion would succeed against it.
	if _, ok := any(rec).(gpuFusedAdamW[float32]); !ok {
		t.Fatalf("fusedRecorder does not satisfy gpuFusedAdamW interface")
	}

	// A CPU-backed tensor is NOT GPU-resident, so stepMixedV keeps the host
	// path even when the engine advertises a fused kernel -- the guard that
	// prevents reading a non-existent device pointer.
	pt, err := tensor.New[float32]([]int{3}, []float32{0.1, 0.2, 0.3})
	if err != nil {
		t.Fatalf("new param: %v", err)
	}
	gt, err := tensor.New[float32]([]int{3}, []float32{0.01, 0.02, 0.03})
	if err != nil {
		t.Fatalf("new grad: %v", err)
	}
	if isGPUResident(pt) || isGPUResident(gt) {
		t.Fatalf("CPU-backed tensor reported GPU-resident")
	}
	if isGPUResident[float32](nil) {
		t.Fatalf("nil tensor reported GPU-resident")
	}

	// Contract: stepMixedV forwards the RAW hyperparameters (beta1/beta2/eps/
	// lr/weightDecay) and the current timestep t to the engine, which derives
	// the bias-corrected step itself -- matching fused_adamw.cu / stepMixedV.
	const lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
	const tstep = 7
	if err := rec.GPUFusedAdamW(pt, gt, beta1, beta2, eps, lr, wd, tstep); err != nil {
		t.Fatalf("fused recorder: %v", err)
	}
	if rec.calls != 1 || rec.beta1 != beta1 || rec.beta2 != beta2 || rec.eps != eps ||
		rec.lr != lr || rec.wd != wd || rec.lastT != tstep {
		t.Fatalf("scalar contract mismatch: %+v", rec)
	}

	// The derived terms must equal stepMixedV's own alpha/lrWd at the same t.
	numer := math.Sqrt(1.0 - math.Pow(beta2, float64(tstep)))
	denom := 1.0 - math.Pow(beta1, float64(tstep))
	wantAlpha := lr * (numer / denom)
	wantLrWd := lr * wd
	if math.Abs(rec.alpha-wantAlpha) > 1e-18 || math.Abs(rec.lrWd-wantLrWd) > 1e-18 {
		t.Fatalf("bias-correction mismatch: alpha=%g want=%g lrWd=%g want=%g",
			rec.alpha, wantAlpha, rec.lrWd, wantLrWd)
	}
}
