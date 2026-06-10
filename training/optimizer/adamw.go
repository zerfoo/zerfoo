// Package optimizer provides various optimization algorithms for neural networks.
package optimizer

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// AdamW implements the AdamW optimizer.
//
// When T is float32 or a sub-float32 precision (float16, float8) AND the
// engine is a CPU engine, the second-moment accumulator (v) is held as a
// float64 sidecar instead of T and the per-element "sqrt(v) + epsilon"
// and "m / (sqrt(v) + epsilon)" computations run in float64. This removes
// the underflow cliff where v drifts into denormals and sqrt(v) + eps
// collapses to eps, producing runaway update magnitudes. Storage for
// param.Value and param.Gradient is unchanged. On GPU engines the
// original T-only path is preserved; mixed-precision GPU kernels are a
// follow-up.
type AdamW[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	learningRate T
	beta1        T
	beta2        T
	epsilon      T
	weightDecay  T
	maxGradNorm  float64 // If > 0, clip global gradient norm to this value.

	// State variables for each parameter.
	m map[*graph.Parameter[T]]*tensor.TensorNumeric[T] // First moment estimates (T-precision)
	v map[*graph.Parameter[T]]*tensor.TensorNumeric[T] // Second moment estimates (T-precision, GPU path)

	// Mixed-precision sidecars. When useMixedV is true the second moment is
	// held host-side in v64[p] (float64) and the first moment is held host-side
	// in mMixed[p] (T-precision); the device tensors v[p] and m[p] are unused.
	// Keeping both moments host-only is the ADR-070 optimization: on a GPU
	// engine GPUStorage.Data() is a D2H copy and Set() an H2D copy, so holding
	// the optimizer state that only the host update touches in plain Go slices
	// removes those per-step round-trips entirely. Allocated lazily per
	// parameter on the first Step call.
	v64       map[*graph.Parameter[T]][]float64
	mMixed    map[*graph.Parameter[T]][]T
	useMixedV bool

	t int // Timestep
}

// NewAdamW creates a new AdamW optimizer.
func NewAdamW[T tensor.Numeric](engine compute.Engine[T], learningRate, beta1, beta2, epsilon, weightDecay T) *AdamW[T] {
	return &AdamW[T]{
		engine:       engine,
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		weightDecay:  weightDecay,
		m:            make(map[*graph.Parameter[T]]*tensor.TensorNumeric[T]),
		v:            make(map[*graph.Parameter[T]]*tensor.TensorNumeric[T]),
		v64:          make(map[*graph.Parameter[T]][]float64),
		mMixed:       make(map[*graph.Parameter[T]][]T),
		useMixedV:    shouldUseMixedPrecisionV[T](engine),
		t:            0,
	}
}

// shouldUseMixedPrecisionV returns true when AdamW should keep the
// second-moment accumulator in float64 instead of T. True when T is float32
// or below (float64 T doesn't need promotion) on a CPU or CUDA GPU engine.
//
// An all-T (f32) second moment is numerically unstable for GPU f32 training:
// sqrt(v)+eps in the denominator loses precision when gradients span a wide
// dynamic range, so m/sqrt(v) drifts the weights until they overflow to NaN a
// few optimizer steps in -- the "CrossAsset cliff" GPU-only blow-up. The CPU
// path has always used an f64 v and trains the same model cleanly, so GPU must
// match. stepMixedV does the f64 update on host then writes param/m/grad back
// to device storage (a no-op for host storage), so it is correct on the GB10
// unified-memory engine and on discrete CUDA GPUs alike.
func shouldUseMixedPrecisionV[T tensor.Numeric](engine compute.Engine[T]) bool {
	switch engine.(type) {
	case *compute.CPUEngine[T], *compute.GPUEngine[T]:
	default:
		return false
	}
	var zero T
	switch any(zero).(type) {
	case float64:
		return false // Already at max useful precision.
	case float32, float16.Float16, float16.BFloat16, float8.Float8:
		return true
	default:
		return false // Integer Ts don't use AdamW in practice.
	}
}

// SetMaxGradNorm sets the maximum gradient norm for gradient clipping.
// If maxGradNorm <= 0, gradient clipping is disabled.
func (a *AdamW[T]) SetMaxGradNorm(maxGradNorm float64) {
	a.maxGradNorm = maxGradNorm
}

// Step updates the parameters based on their gradients.
func (a *AdamW[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	// NaN/Inf guard and optional gradient clipping.
	if err := a.guardAndClipGradients(ctx, params); err != nil {
		return err
	}

	a.t++ // Increment timestep

	if a.useMixedV {
		return a.stepMixedV(ctx, params)
	}
	return a.stepEngine(ctx, params)
}

// stepEngine is the original all-T path: every Adam arithmetic step goes
// through the engine. Preserved for GPU engines and for float64 T where
// further promotion has no benefit.
func (a *AdamW[T]) stepEngine(ctx context.Context, params []*graph.Parameter[T]) error {
	// Bias correction terms.
	ops := a.engine.Ops()
	one := ops.FromFloat64(1.0)
	tAsT := ops.FromFloat64(float64(a.t))
	numer := ops.Sqrt(ops.Sub(one, ops.Pow(a.beta2, tAsT)))
	denom := ops.Sub(one, ops.Pow(a.beta1, tAsT))
	biasCorr := ops.Div(numer, denom)
	alpha := ops.Mul(a.learningRate, biasCorr)

	for _, param := range params {
		grad := param.Gradient

		if grad == nil {
			continue
		}

		if _, ok := a.m[param]; !ok {
			mTensor, err := tensor.New[T](param.Value.Shape(), nil)
			if err != nil {
				return err
			}

			if err := a.engine.Zeros(ctx, mTensor, param.Value.Shape()); err != nil {
				return err
			}

			a.m[param] = mTensor

			vTensor, err := tensor.New[T](param.Value.Shape(), nil)
			if err != nil {
				return err
			}

			if err := a.engine.Zeros(ctx, vTensor, param.Value.Shape()); err != nil {
				return err
			}

			a.v[param] = vTensor
		}

		m := a.m[param]
		v := a.v[param]
		paramValue := param.Value

		mNew, err := a.engine.MulScalar(ctx, m, a.beta1, nil)
		if err != nil {
			return err
		}

		gradScaled, err := a.engine.MulScalar(ctx, grad, ops.Sub(one, a.beta1), nil)
		if err != nil {
			return err
		}

		m, err = a.engine.Add(ctx, mNew, gradScaled, m)
		if err != nil {
			return err
		}

		vNew, err := a.engine.MulScalar(ctx, v, a.beta2, nil)
		if err != nil {
			return err
		}

		gradSquared, err := a.engine.Mul(ctx, grad, grad, nil)
		if err != nil {
			return err
		}

		gradSquaredScaled, err := a.engine.MulScalar(ctx, gradSquared, ops.Sub(one, a.beta2), nil)
		if err != nil {
			return err
		}

		v, err = a.engine.Add(ctx, vNew, gradSquaredScaled, v)
		if err != nil {
			return err
		}

		sqrtV, err := a.engine.Sqrt(ctx, v, nil)
		if err != nil {
			return err
		}

		sqrtVPlusEpsilon, err := a.engine.AddScalar(ctx, sqrtV, a.epsilon, nil)
		if err != nil {
			return err
		}

		updateTerm, err := a.engine.Div(ctx, m, sqrtVPlusEpsilon, nil)
		if err != nil {
			return err
		}

		updateTermScaled, err := a.engine.MulScalar(ctx, updateTerm, alpha, nil)
		if err != nil {
			return err
		}

		lrWd := ops.Mul(a.learningRate, a.weightDecay)
		weightDecayTerm, err := a.engine.MulScalar(ctx, paramValue, lrWd, nil)
		if err != nil {
			return err
		}

		paramNew, err := a.engine.Sub(ctx, paramValue, updateTermScaled, nil)
		if err != nil {
			return err
		}

		param.Value, err = a.engine.Sub(ctx, paramNew, weightDecayTerm, paramValue)
		if err != nil {
			return err
		}

		var zero T
		if err := a.engine.Fill(ctx, param.Gradient, zero); err != nil {
			param.ClearGradient()
		}
	}

	return nil
}

// stepMixedV implements the Adam update with the second-moment accumulator
// held in float64. The parameter values stay in T; the first moment is kept
// host-side in T. Both CPU and GPU (GB10 CUDA) engines reach this path for
// float32-and-below T (see shouldUseMixedPrecisionV).
//
// Numerics: sqrt(v) + epsilon never collapses to just epsilon due to float32
// underflow on near-zero gradients, which in the all-T path can yield update
// magnitudes of m/epsilon = m * 1e5 and cause weights to drift rapidly toward
// the float32 overflow edge. This is the optimizer fix for the GPU "CrossAsset
// cliff" (ADR 070).
//
// Host round-trip minimization (ADR 070, decision 2): GPUStorage.Data() returns
// a host *copy* (D2H) and Set() performs an H2D copy, so naively reading and
// writing every piece of optimizer state through the device storage every step
// is a per-step host<->device sync. We keep the second moment (v64) and the
// first moment (mMixed) as host-only Go slices that the engine never touches,
// so they incur no transfer at all. Only the param value is genuinely shared
// with the device: we read it (D2H), update it on host, and write it back
// (H2D). The gradient is read once (D2H) and then zeroed *on device* via
// engine.Fill, avoiding an H2D write-back of a zero buffer. Per parameter per
// step this is 2 reads + 1 write + 1 device fill, down from the previous 3
// reads + 3 writes.
//
// A true on-device f64 accumulator (ADR 070's preferred end state) is NOT
// implemented here because the GPU engine's elementwise/scalar ops are gated on
// isFloat32[T] and fall back to the CPU engine for any non-f32 T -- an f64
// device tensor would execute every op on host anyway. Promoting the update
// fully on-device requires native f64 CUDA kernels (Sqrt/Div/Add/Mul/...),
// which is out of scope here and tracked as the ADR-070 follow-up.
func (a *AdamW[T]) stepMixedV(ctx context.Context, params []*graph.Parameter[T]) error {
	ops := a.engine.Ops()
	one64 := 1.0
	t64 := float64(a.t)
	beta1F := numericToFloat64(a.beta1)
	beta2F := numericToFloat64(a.beta2)
	epsF := numericToFloat64(a.epsilon)
	lrF := numericToFloat64(a.learningRate)
	wdF := numericToFloat64(a.weightDecay)

	numer := math.Sqrt(one64 - math.Pow(beta2F, t64))
	denom := one64 - math.Pow(beta1F, t64)
	alpha := lrF * (numer / denom)
	lrWd := lrF * wdF

	for _, param := range params {
		grad := param.Gradient
		if grad == nil {
			continue
		}

		if _, ok := a.mMixed[param]; !ok {
			total := 1
			for _, d := range param.Value.Shape() {
				total *= d
			}
			// Host-only first and second moment. These slices are never read
			// or written by the engine, so on a GPU engine they cost zero
			// host<->device transfers (cf. the device m tensor in stepEngine).
			a.mMixed[param] = make([]T, total)
			a.v64[param] = make([]float64, total)
		}

		mData := a.mMixed[param]
		v64 := a.v64[param]
		paramData := param.Value.Data() // D2H copy on GPU storage.
		gradData := grad.Data()         // D2H copy on GPU storage.

		if len(v64) != len(paramData) {
			return fmt.Errorf("adamw: v64 size mismatch for parameter %q: %d vs %d",
				param.Name, len(v64), len(paramData))
		}
		if len(mData) != len(paramData) {
			return fmt.Errorf("adamw: mMixed size mismatch for parameter %q: %d vs %d",
				param.Name, len(mData), len(paramData))
		}

		for i := range paramData {
			g := numericToFloat64(gradData[i])
			mOld := numericToFloat64(mData[i])
			mNew := beta1F*mOld + (one64-beta1F)*g
			mData[i] = ops.FromFloat64(mNew)

			v64[i] = beta2F*v64[i] + (one64-beta2F)*g*g

			denomI := math.Sqrt(v64[i]) + epsF
			update := alpha * mNew / denomI

			pv := numericToFloat64(paramData[i])
			pv = pv - update - lrWd*pv
			paramData[i] = ops.FromFloat64(pv)
		}

		// Persist the updated weights to the parameter's storage. For CPU
		// storage Data() returned the backing slice and this is a no-op
		// reassignment; for GPU storage Data() returned a host copy, so the new
		// weights must be copied back to the device (H2D) or the step would be
		// silently discarded. The first/second moments live host-only in
		// mMixed/v64 and need no write-back.
		param.Value.GetStorage().Set(paramData)

		// Zero the gradient IN PLACE, preserving its storage buffer. Do NOT use
		// engine.Fill here: on the GPU engine Fill reallocates the tensor's
		// storage from the arena pool (gpuFill -> pool.Alloc + SetStorage), which
		// moves param.Gradient INTO the arena. Callers that accumulate gradients
		// across a batch and reset the arena per sample (e.g. Wolf crossasset)
		// rely on the gradient living in a persistent, NON-arena buffer; an arena
		// reset would then reclaim that buffer mid-accumulation and corrupt the
		// next step's gradient -- a GPU-only, timing-sensitive NaN. Writing zeros
		// back through the existing storage keeps the same buffer (a same-size
		// Set is an in-place memcpy / H2D, not a realloc). gradData is the host
		// copy already read above, so this reuses it.
		var zero T
		for i := range gradData {
			gradData[i] = zero
		}
		grad.GetStorage().Set(gradData)
	}

	return nil
}

// numericToFloat64 converts a tensor.Numeric value to float64.
func numericToFloat64[T tensor.Numeric](v T) float64 {
	switch val := any(v).(type) {
	case float32:
		return float64(val)
	case float64:
		return val
	case int:
		return float64(val)
	case int8:
		return float64(val)
	case int16:
		return float64(val)
	case int32:
		return float64(val)
	case int64:
		return float64(val)
	case uint:
		return float64(val)
	case uint8:
		return float64(val)
	case uint32:
		return float64(val)
	case uint64:
		return float64(val)
	case float16.Float16:
		return float64(val.ToFloat32())
	case float16.BFloat16:
		return float64(val.ToFloat32())
	case float8.Float8:
		return val.ToFloat64()
	default:
		return 0
	}
}

// guardAndClipGradients checks all gradient values for NaN/Inf and optionally
// clips the global gradient norm to MaxGradNorm.
//
// Detection uses Engine ReduceSum to collapse each gradient to a single scalar
// (1 D2H copy per parameter) instead of iterating every element via .Data().
func (a *AdamW[T]) guardAndClipGradients(ctx context.Context, params []*graph.Parameter[T]) error {
	var globalNormSq float64

	for _, param := range params {
		grad := param.Gradient
		if grad == nil {
			continue
		}

		sumTensor, err := a.engine.ReduceSum(ctx, grad, -1, false)
		if err != nil {
			return fmt.Errorf("adamw: ReduceSum failed for parameter %q: %w", param.Name, err)
		}

		sumVal := numericToFloat64(sumTensor.Data()[0])

		if math.IsNaN(sumVal) {
			return fmt.Errorf("adamw: NaN detected in gradient of parameter %q", param.Name)
		}

		if math.IsInf(sumVal, 0) {
			return fmt.Errorf("adamw: Inf detected in gradient of parameter %q", param.Name)
		}

		gradSquared, err := a.engine.Mul(ctx, grad, grad, nil)
		if err != nil {
			return fmt.Errorf("adamw: Mul failed for parameter %q: %w", param.Name, err)
		}

		sqSumTensor, err := a.engine.ReduceSum(ctx, gradSquared, -1, false)
		if err != nil {
			return fmt.Errorf("adamw: ReduceSum failed for parameter %q: %w", param.Name, err)
		}

		globalNormSq += numericToFloat64(sqSumTensor.Data()[0])
	}

	if a.maxGradNorm > 0 {
		globalNorm := math.Sqrt(globalNormSq)
		if globalNorm > a.maxGradNorm {
			scaleF64 := a.maxGradNorm / globalNorm
			scaleT := a.engine.Ops().FromFloat64(scaleF64)

			for _, param := range params {
				grad := param.Gradient
				if grad == nil {
					continue
				}

				clipped, err := a.engine.MulScalar(ctx, grad, scaleT, grad)
				if err != nil {
					return fmt.Errorf("adamw: MulScalar failed for parameter %q: %w", param.Name, err)
				}

				param.Gradient = clipped
			}
		}
	}

	return nil
}

// SetLR sets the learning rate. This is typically called by a scheduler.
func (a *AdamW[T]) SetLR(lr T) {
	a.learningRate = lr
}

// Statically assert that the type implements the Optimizer interface.
var _ Optimizer[float32] = (*AdamW[float32])(nil)
