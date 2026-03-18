package ssm

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BCNorm implements L2 normalization with a learnable gain for the B and C
// matrices of a state space model. It stabilizes the SSM recurrence by
// preventing the B/C values from growing unbounded, which is especially
// important when complex-valued RoPE rotations are applied.
//
// For an input x of shape [..., dim]:
//
//	norm = sqrt(sum(x^2, dim=-1) / dim + eps)
//	out = gain * x / norm
//
// This is similar to RMSNorm but applied specifically to the SSM projection
// outputs before they enter the selective scan.
type BCNorm[T tensor.Numeric] struct {
	name    string
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	dim     int
	epsilon T
	gain    *graph.Parameter[T]

	// Cached for backward
	cachedInput *tensor.TensorNumeric[T]
	cachedNorm  []T // per-vector norms
}

// NewBCNorm creates a new BCNorm layer.
func NewBCNorm[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], dim int) (*BCNorm[T], error) {
	if name == "" {
		return nil, fmt.Errorf("BCNorm name cannot be empty")
	}
	if dim <= 0 {
		return nil, fmt.Errorf("BCNorm dim must be positive, got %d", dim)
	}

	gainData := make([]T, dim)
	for i := range gainData {
		gainData[i] = ops.One()
	}
	gainTensor, err := tensor.New[T]([]int{dim}, gainData)
	if err != nil {
		return nil, err
	}
	gainParam, err := graph.NewParameter[T](name+"_gain", gainTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &BCNorm[T]{
		name:    name,
		engine:  engine,
		ops:     ops,
		dim:     dim,
		epsilon: ops.FromFloat64(1e-6),
		gain:    gainParam,
	}, nil
}

// Forward applies BCNorm: gain * x / rms(x).
func (bn *BCNorm[T]) Forward(_ context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	if len(shape) == 0 {
		return nil, fmt.Errorf("BCNorm: empty input shape")
	}
	lastDim := shape[len(shape)-1]
	if lastDim != bn.dim {
		return nil, fmt.Errorf("BCNorm: last dim %d != expected %d", lastDim, bn.dim)
	}

	bn.cachedInput = input

	data := input.Data()
	numVectors := len(data) / bn.dim
	outData := make([]T, len(data))
	bn.cachedNorm = make([]T, numVectors)
	gainData := bn.gain.Value.Data()

	for v := 0; v < numVectors; v++ {
		off := v * bn.dim
		// Compute RMS
		var sumSq float64
		for i := 0; i < bn.dim; i++ {
			val := float64(data[off+i])
			sumSq += val * val
		}
		rms := math.Sqrt(sumSq/float64(bn.dim) + float64(bn.epsilon))
		bn.cachedNorm[v] = T(rms)

		// Normalize and scale by gain
		for i := 0; i < bn.dim; i++ {
			outData[off+i] = bn.ops.Mul(gainData[i], T(float64(data[off+i])/rms))
		}
	}

	return tensor.New[T](shape, outData)
}

// Backward computes gradients for BCNorm.
func (bn *BCNorm[T]) Backward(_ context.Context, dOut *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	data := bn.cachedInput.Data()
	dOutData := dOut.Data()
	gainData := bn.gain.Value.Data()
	numVectors := len(data) / bn.dim

	dInData := make([]T, len(data))
	dGainData := make([]T, bn.dim)

	for v := 0; v < numVectors; v++ {
		off := v * bn.dim
		rms := float64(bn.cachedNorm[v])
		invRms := 1.0 / rms
		rms2 := rms * rms

		// sum_i(dOut_i * gain_i * x_i) for the cross-term
		var dotProd float64
		for i := 0; i < bn.dim; i++ {
			dotProd += float64(dOutData[off+i]) * float64(gainData[i]) * float64(data[off+i])
		}

		for i := 0; i < bn.dim; i++ {
			xi := float64(data[off+i])
			xNorm := xi * invRms
			// dGain accumulation: dL/dg_i = dL/dy_i * x_i / rms
			dGainData[i] = bn.ops.Add(dGainData[i], T(float64(dOutData[off+i])*xNorm))
			// dInput: dL/dx_j = dOut_j * g_j / rms - x_j * dotProd / (dim * rms^3)
			dInData[off+i] = T(float64(dOutData[off+i])*float64(gainData[i])*invRms - xi*dotProd/(float64(bn.dim)*rms2*rms))
		}
	}

	// Accumulate gain gradient
	dGain, _ := tensor.New[T](bn.gain.Value.Shape(), dGainData)
	if bn.gain.Gradient != nil {
		bn.gain.Gradient, _ = bn.engine.Add(context.Background(), bn.gain.Gradient, dGain)
	} else {
		bn.gain.Gradient = dGain
	}

	return tensor.New[T](bn.cachedInput.Shape(), dInData)
}

// Parameters returns the trainable parameters (gain).
func (bn *BCNorm[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{bn.gain}
}

// Name returns the layer name.
func (bn *BCNorm[T]) Name() string { return bn.name }
