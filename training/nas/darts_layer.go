package nas

import (
	"context"
	"errors"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// DARTSLayer implements a DARTS (Differentiable Architecture Search) mixed-operation
// layer. It computes a softmax-weighted mixture of candidate operations, where the
// architecture parameters (alpha) are learnable and the forward pass is differentiable
// through the softmax weights.
type DARTSLayer[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	candidates []graph.Node[T]
	alpha      *graph.Parameter[T]

	// Cached forward pass state for backward.
	weights   []T // softmax(alpha)
	lastInput *tensor.TensorNumeric[T]
	opOutputs []*tensor.TensorNumeric[T]
}

// NewDARTSLayer creates a new DARTS mixed-operation layer with the given candidate
// operations. The alpha architecture parameters are initialized to zero, giving
// uniform softmax weights. At least 2 candidates are required.
func NewDARTSLayer[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], candidates []graph.Node[T]) (*DARTSLayer[T], error) {
	if len(candidates) < 2 {
		return nil, errors.New("nas: DARTSLayer requires at least 2 candidate operations")
	}

	numOps := len(candidates)
	alphaData := make([]T, numOps)
	alphaVal, err := tensor.New[T]([]int{numOps}, alphaData)
	if err != nil {
		return nil, err
	}

	alphaParam, err := graph.NewParameter[T]("alpha", alphaVal, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &DARTSLayer[T]{
		engine:     engine,
		ops:        ops,
		candidates: candidates,
		alpha:      alphaParam,
	}, nil
}

// softmax computes softmax over alpha values using the Arithmetic ops for type safety.
func (d *DARTSLayer[T]) softmax() []T {
	alpha := d.alpha.Value.Data()
	n := len(alpha)
	weights := make([]T, n)

	// Find max for numerical stability.
	maxVal := alpha[0]
	for i := 1; i < n; i++ {
		if d.ops.GreaterThan(alpha[i], maxVal) {
			maxVal = alpha[i]
		}
	}

	// Compute exp(alpha_i - max) and sum.
	var sum T
	for i, a := range alpha {
		diff := d.ops.Sub(a, maxVal)
		expVal := d.ops.Exp(diff)
		weights[i] = expVal
		sum = d.ops.Add(sum, expVal)
	}

	// Normalize.
	for i := range weights {
		weights[i] = d.ops.Div(weights[i], sum)
	}
	return weights
}

// Forward computes the softmax-weighted mixture of all candidate operations.
// output = sum_i softmax(alpha)_i * op_i(input)
func (d *DARTSLayer[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) == 0 {
		return nil, errors.New("nas: DARTSLayer.Forward requires at least 1 input")
	}
	input := inputs[0]
	d.lastInput = input

	d.weights = d.softmax()
	d.opOutputs = make([]*tensor.TensorNumeric[T], len(d.candidates))

	var result *tensor.TensorNumeric[T]
	for i, candidate := range d.candidates {
		out, err := candidate.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
		d.opOutputs[i] = out

		scaled, err := d.engine.MulScalar(ctx, out, d.weights[i])
		if err != nil {
			return nil, err
		}

		if result == nil {
			result = scaled
		} else {
			result, err = d.engine.Add(ctx, result, scaled)
			if err != nil {
				return nil, err
			}
		}
	}

	return result, nil
}

// Backward computes gradients for both the input and the alpha architecture parameters.
//
// Given output = sum_i w_i * op_i(x) where w = softmax(alpha):
//   - dInput = sum_i w_i * op_i.Backward(dOut)
//   - dAlpha_k = sum_j dOut_j * (sum_i op_i(x)_j * (delta_{ik} * w_i - w_i * w_k))
//     which simplifies to: dAlpha_k = w_k * dot(dOut, op_k(x) - output)
func (d *DARTSLayer[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	input := d.lastInput
	if len(inputs) > 0 {
		input = inputs[0]
	}

	// Compute weighted output for alpha gradient: output = sum_i w_i * op_i(x).
	// We already have opOutputs and weights from forward, but recompute if needed.
	if d.opOutputs == nil {
		_, err := d.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
	}

	// Compute the mixed output for the Jacobian of softmax.
	var mixedOutput *tensor.TensorNumeric[T]
	for i, out := range d.opOutputs {
		scaled, err := d.engine.MulScalar(ctx, out, d.weights[i])
		if err != nil {
			return nil, err
		}
		if mixedOutput == nil {
			mixedOutput = scaled
		} else {
			mixedOutput, err = d.engine.Add(ctx, mixedOutput, scaled)
			if err != nil {
				return nil, err
			}
		}
	}

	// Gradient w.r.t. alpha: dAlpha_k = w_k * dot(dOut, op_k(x) - mixedOutput)
	alphaGrad := make([]T, len(d.candidates))
	dOutData := dOut.Data()
	mixedData := mixedOutput.Data()

	for k, out := range d.opOutputs {
		outData := out.Data()
		var dotProd T
		for j := range dOutData {
			diff := d.ops.Sub(outData[j], mixedData[j])
			dotProd = d.ops.Add(dotProd, d.ops.Mul(dOutData[j], diff))
		}
		alphaGrad[k] = d.ops.Mul(d.weights[k], dotProd)
	}

	alphaGradTensor, err := tensor.New[T](d.alpha.Value.Shape(), alphaGrad)
	if err != nil {
		return nil, err
	}
	if err := d.alpha.AddGradient(alphaGradTensor); err != nil {
		return nil, err
	}

	// Gradient w.r.t. input: dInput = sum_i w_i * op_i.Backward(dOut)
	var inputGrad *tensor.TensorNumeric[T]
	for i, candidate := range d.candidates {
		opGrads, err := candidate.Backward(ctx, mode, dOut, input)
		if err != nil {
			return nil, err
		}
		if len(opGrads) == 0 {
			continue
		}

		scaled, err := d.engine.MulScalar(ctx, opGrads[0], d.weights[i])
		if err != nil {
			return nil, err
		}

		if inputGrad == nil {
			inputGrad = scaled
		} else {
			inputGrad, err = d.engine.Add(ctx, inputGrad, scaled)
			if err != nil {
				return nil, err
			}
		}
	}

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// Parameters returns the learnable architecture parameters (alpha).
func (d *DARTSLayer[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{d.alpha}
}

// OpType returns the operation type identifier.
func (d *DARTSLayer[T]) OpType() string {
	return "DARTSMixedOp"
}

// Attributes returns the layer's attributes.
func (d *DARTSLayer[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_ops": len(d.candidates),
	}
}

// OutputShape returns the output shape, which matches the first candidate's output shape.
func (d *DARTSLayer[T]) OutputShape() []int {
	if len(d.candidates) > 0 {
		return d.candidates[0].OutputShape()
	}
	return nil
}

// Weights returns the current softmax weights over candidate operations.
func (d *DARTSLayer[T]) Weights() []T {
	return d.softmax()
}

// Statically assert that DARTSLayer implements graph.Node.
var _ graph.Node[float32] = (*DARTSLayer[float32])(nil)
