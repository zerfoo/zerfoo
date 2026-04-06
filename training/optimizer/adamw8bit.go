package optimizer

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// blockSize is the number of elements per quantization block.
const blockSize = 256

// Int8State holds a block-wise INT8-quantized representation of a float32 slice.
// Each block of blockSize elements shares a single scale factor, reducing memory
// from 4 bytes/element to ~1 byte/element (+ negligible scale overhead).
type Int8State struct {
	data   []int8
	scales []float32
}

// quantizeToInt8 quantizes src into block-wise INT8 representation.
// Each block of blockSize elements is independently scaled to fit in [-127, 127].
func quantizeToInt8(src []float32) Int8State {
	n := len(src)
	nBlocks := (n + blockSize - 1) / blockSize
	state := Int8State{
		data:   make([]int8, n),
		scales: make([]float32, nBlocks),
	}

	for b := range nBlocks {
		start := b * blockSize
		end := start + blockSize
		if end > n {
			end = n
		}

		// Find absmax in block.
		var absMax float32
		for _, v := range src[start:end] {
			av := float32(math.Abs(float64(v)))
			if av > absMax {
				absMax = av
			}
		}

		if absMax == 0 {
			state.scales[b] = 0
			// data already zeroed by make
			continue
		}

		scale := absMax / 127.0
		state.scales[b] = scale
		invScale := 127.0 / absMax

		for i := start; i < end; i++ {
			// Round to nearest, clamp to [-127, 127].
			q := int(math.Round(float64(src[i] * invScale)))
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			state.data[i] = int8(q)
		}
	}

	return state
}

// dequantizeFromInt8 reconstructs a float32 slice from its INT8 representation.
func dequantizeFromInt8(s Int8State) []float32 {
	n := len(s.data)
	out := make([]float32, n)

	for b := range len(s.scales) {
		start := b * blockSize
		end := start + blockSize
		if end > n {
			end = n
		}

		scale := s.scales[b]
		for i := start; i < end; i++ {
			out[i] = float32(s.data[i]) * scale
		}
	}

	return out
}

// memoryBytes returns the approximate memory used by the Int8State.
func (s Int8State) memoryBytes() int {
	return len(s.data) + len(s.scales)*4
}

// AdamW8bit implements the AdamW optimizer with block-wise INT8 quantization
// for first and second moment estimates. Parameters remain in full precision.
// This reduces optimizer state memory by ~4x compared to FP32 AdamW.
type AdamW8bit[T tensor.Numeric] struct {
	engine                    compute.Engine[T]
	lr, beta1, beta2, eps, wd float32
	step                      int
	m, v                      map[*graph.Parameter[T]]*Int8State
}

// NewAdamW8bit creates a new 8-bit AdamW optimizer.
func NewAdamW8bit[T tensor.Numeric](engine compute.Engine[T], lr, beta1, beta2, eps, wd float32) *AdamW8bit[T] {
	return &AdamW8bit[T]{
		engine: engine,
		lr:     lr,
		beta1:  beta1,
		beta2:  beta2,
		eps:    eps,
		wd:     wd,
		m:      make(map[*graph.Parameter[T]]*Int8State),
		v:      make(map[*graph.Parameter[T]]*Int8State),
	}
}

// Step updates parameters based on their gradients. Moment estimates are stored
// in INT8 and dequantized for computation, then re-quantized after update.
func (a *AdamW8bit[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	a.step++

	// Bias correction factors.
	ops := a.engine.Ops()
	one := ops.FromFloat64(1.0)
	bc1 := 1.0 - math.Pow(float64(a.beta1), float64(a.step))
	bc2 := 1.0 - math.Pow(float64(a.beta2), float64(a.step))
	alphaScalar := ops.FromFloat64(float64(a.lr) * math.Sqrt(bc2) / bc1)
	b1T := ops.FromFloat64(float64(a.beta1))
	oneMinusB1 := ops.Sub(one, b1T)
	b2T := ops.FromFloat64(float64(a.beta2))
	oneMinusB2 := ops.Sub(one, b2T)
	epsT := ops.FromFloat64(float64(a.eps))
	lrWd := ops.FromFloat64(float64(a.lr) * float64(a.wd))

	for _, param := range params {
		grad := param.Gradient
		if grad == nil {
			continue
		}

		shape := param.Value.Shape()
		n := len(param.Value.Data())

		// Dequantize or initialize moment estimates as float32 slices,
		// then wrap as tensors for vectorized engine ops.
		var mf, vf []float32
		if ms, ok := a.m[param]; ok {
			mf = dequantizeFromInt8(*ms)
			vf = dequantizeFromInt8(*a.v[param])
		} else {
			mf = make([]float32, n)
			vf = make([]float32, n)
		}

		// Create moment tensors with the same shape as the parameter.
		mTensor, err := tensor.New[T](shape, float32SliceToT[T](mf, ops))
		if err != nil {
			return err
		}
		vTensor, err := tensor.New[T](shape, float32SliceToT[T](vf, ops))
		if err != nil {
			return err
		}

		// m = beta1 * m + (1 - beta1) * grad
		mScaled, err := a.engine.MulScalar(ctx, mTensor, b1T)
		if err != nil {
			return err
		}
		gScaled, err := a.engine.MulScalar(ctx, grad, oneMinusB1)
		if err != nil {
			return err
		}
		mTensor, err = a.engine.Add(ctx, mScaled, gScaled)
		if err != nil {
			return err
		}

		// v = beta2 * v + (1 - beta2) * grad^2
		vScaled, err := a.engine.MulScalar(ctx, vTensor, b2T)
		if err != nil {
			return err
		}
		gradSq, err := a.engine.Mul(ctx, grad, grad)
		if err != nil {
			return err
		}
		gSqScaled, err := a.engine.MulScalar(ctx, gradSq, oneMinusB2)
		if err != nil {
			return err
		}
		vTensor, err = a.engine.Add(ctx, vScaled, gSqScaled)
		if err != nil {
			return err
		}

		// update = alpha * m / (sqrt(v) + eps)
		sqrtV, err := a.engine.Sqrt(ctx, vTensor)
		if err != nil {
			return err
		}
		sqrtVEps, err := a.engine.AddScalar(ctx, sqrtV, epsT)
		if err != nil {
			return err
		}
		update, err := a.engine.Div(ctx, mTensor, sqrtVEps)
		if err != nil {
			return err
		}
		update, err = a.engine.MulScalar(ctx, update, alphaScalar)
		if err != nil {
			return err
		}

		// decay = lr * wd * param
		decay, err := a.engine.MulScalar(ctx, param.Value, lrWd)
		if err != nil {
			return err
		}

		// param = param - update - decay
		paramNew, err := a.engine.Sub(ctx, param.Value, update)
		if err != nil {
			return err
		}
		param.Value, err = a.engine.Sub(ctx, paramNew, decay)
		if err != nil {
			return err
		}

		// Re-quantize moments to INT8.
		mq := quantizeToInt8(tSliceToFloat32(mTensor.Data()))
		vq := quantizeToInt8(tSliceToFloat32(vTensor.Data()))
		a.m[param] = &mq
		a.v[param] = &vq

		// Clear gradient.
		var zero T
		if err := a.engine.Fill(ctx, param.Gradient, zero); err != nil {
			param.ClearGradient()
		}
	}

	return nil
}

// float32SliceToT converts a []float32 to []T using the arithmetic ops.
func float32SliceToT[T tensor.Numeric](src []float32, ops numeric.Arithmetic[T]) []T {
	dst := make([]T, len(src))
	for i, v := range src {
		dst[i] = ops.FromFloat64(float64(v))
	}
	return dst
}

// tSliceToFloat32 converts a []T to []float32.
func tSliceToFloat32[T tensor.Numeric](src []T) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = float32(numericToFloat64(v))
	}
	return dst
}

// Statically assert that AdamW8bit implements the Optimizer interface.
var _ Optimizer[float32] = (*AdamW8bit[float32])(nil)
