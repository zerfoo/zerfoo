package optimizer

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/graph"
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
	lr, beta1, beta2, eps, wd float32
	step                      int
	m, v                      map[*graph.Parameter[T]]*Int8State
}

// NewAdamW8bit creates a new 8-bit AdamW optimizer.
func NewAdamW8bit[T tensor.Numeric](lr, beta1, beta2, eps, wd float32) *AdamW8bit[T] {
	return &AdamW8bit[T]{
		lr:    lr,
		beta1: beta1,
		beta2: beta2,
		eps:   eps,
		wd:    wd,
		m:     make(map[*graph.Parameter[T]]*Int8State),
		v:     make(map[*graph.Parameter[T]]*Int8State),
	}
}

// Step updates parameters based on their gradients. Moment estimates are stored
// in INT8 and dequantized for computation, then re-quantized after update.
func (a *AdamW8bit[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	a.step++

	// Bias correction factors.
	bc1 := 1.0 - math.Pow(float64(a.beta1), float64(a.step))
	bc2 := 1.0 - math.Pow(float64(a.beta2), float64(a.step))
	alpha := float64(a.lr) * math.Sqrt(bc2) / bc1

	b1 := float64(a.beta1)
	b2 := float64(a.beta2)

	for _, param := range params {
		grad := param.Gradient
		if grad == nil {
			continue
		}

		paramData := param.Value.Data()
		gradData := grad.Data()
		n := len(paramData)

		// Dequantize or initialize moment estimates.
		var mf, vf []float32
		if ms, ok := a.m[param]; ok {
			mf = dequantizeFromInt8(*ms)
			vf = dequantizeFromInt8(*a.v[param])
		} else {
			mf = make([]float32, n)
			vf = make([]float32, n)
		}

		// Update moments and parameters in a single pass.
		for i := range n {
			g := float64(gradData[i])

			// m = beta1 * m + (1 - beta1) * g
			mf[i] = float32(b1*float64(mf[i]) + (1-b1)*g)

			// v = beta2 * v + (1 - beta2) * g^2
			vf[i] = float32(b2*float64(vf[i]) + (1-b2)*g*g)

			// param = param - alpha * m / (sqrt(v) + eps) - lr * wd * param
			update := alpha * float64(mf[i]) / (math.Sqrt(float64(vf[i])) + float64(a.eps))
			decay := float64(a.lr) * float64(a.wd) * float64(paramData[i])
			paramData[i] = T(float64(paramData[i]) - update - decay)
		}

		// Re-quantize moments to INT8.
		mq := quantizeToInt8(mf)
		vq := quantizeToInt8(vf)
		a.m[param] = &mq
		a.v[param] = &vq

		// Clear gradient.
		for i := range gradData {
			gradData[i] = 0
		}
	}

	return nil
}

// Statically assert that AdamW8bit implements the Optimizer interface.
var _ Optimizer[float32] = (*AdamW8bit[float32])(nil)
