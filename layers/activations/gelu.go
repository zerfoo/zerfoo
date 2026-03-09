package activations

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Gelu represents a standard Gelu activation layer.
// Implements: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// All operations are composed from Engine primitives so they appear in the
// ExecutionPlan instruction tape.
type Gelu[T tensor.Float] struct {
	graph.NoParameters[T]
	engine    compute.Engine[T]
	ops       numeric.Arithmetic[T]
	lastInput *tensor.TensorNumeric[T]
}

// NewGelu creates a new standard Gelu activation layer.
func NewGelu[T tensor.Float](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Gelu[T] {
	return &Gelu[T]{engine: engine, ops: ops}
}

// OpType returns the operation type of the activation.
func (g *Gelu[T]) OpType() string { return "Gelu" }

// Attributes returns the attributes of the activation.
func (g *Gelu[T]) Attributes() map[string]interface{} { return nil }

// OutputShape returns the output shape of the activation.
func (g *Gelu[T]) OutputShape() []int {
	if g.lastInput != nil {
		return g.lastInput.Shape()
	}
	return nil
}

// Forward performs the forward pass using only engine primitives.
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func (g *Gelu[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Gelu: expected 1 input, got %d", len(inputs))
	}

	x := inputs[0]
	g.lastInput = x
	ops := g.ops

	// x^2
	x2, err := g.engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}

	// x^3
	x3, err := g.engine.Mul(ctx, x2, x)
	if err != nil {
		return nil, err
	}

	// 0.044715 * x^3
	term1, err := g.engine.MulScalar(ctx, x3, ops.FromFloat64(0.044715))
	if err != nil {
		return nil, err
	}

	// x + 0.044715 * x^3
	term2, err := g.engine.Add(ctx, x, term1)
	if err != nil {
		return nil, err
	}

	// sqrt(2/pi) * (x + 0.044715 * x^3)
	term3, err := g.engine.MulScalar(ctx, term2, ops.FromFloat64(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}

	// tanh(...) -- uses engine.Tanh, visible in instruction tape
	tanhResult, err := g.engine.Tanh(ctx, term3)
	if err != nil {
		return nil, err
	}

	// 1 + tanh(...)
	term4, err := g.engine.AddScalar(ctx, tanhResult, ops.One())
	if err != nil {
		return nil, err
	}

	// x * (1 + tanh(...))
	term5, err := g.engine.Mul(ctx, x, term4)
	if err != nil {
		return nil, err
	}

	// 0.5 * x * (1 + tanh(...))
	output, err := g.engine.MulScalar(ctx, term5, ops.FromFloat64(0.5))
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward performs the backward pass using only engine primitives.
// d/dx[0.5 * x * (1 + tanh(u))] where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
// du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
func (g *Gelu[T]) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	x := g.lastInput
	ops := g.ops

	// x^2
	x2, err := g.engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}

	// x^3
	x3, err := g.engine.Mul(ctx, x2, x)
	if err != nil {
		return nil, err
	}

	// 0.044715 * x^3
	term1, err := g.engine.MulScalar(ctx, x3, ops.FromFloat64(0.044715))
	if err != nil {
		return nil, err
	}

	// x + 0.044715 * x^3
	term2, err := g.engine.Add(ctx, x, term1)
	if err != nil {
		return nil, err
	}

	// u = sqrt(2/pi) * (x + 0.044715 * x^3)
	u, err := g.engine.MulScalar(ctx, term2, ops.FromFloat64(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}

	// tanh(u)
	tanhU, err := g.engine.Tanh(ctx, u)
	if err != nil {
		return nil, err
	}

	// sech^2(u) = 1 - tanh^2(u)
	tanhU2, err := g.engine.Mul(ctx, tanhU, tanhU)
	if err != nil {
		return nil, err
	}
	// sechSq = 1 - tanh^2(u): negate tanh^2 then add 1
	negTanhU2, err := g.engine.MulScalar(ctx, tanhU2, ops.FromFloat64(-1))
	if err != nil {
		return nil, err
	}
	sechSq, err := g.engine.AddScalar(ctx, negTanhU2, ops.One())
	if err != nil {
		return nil, err
	}

	// du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
	// 3 * 0.044715 * x^2
	dterm1, err := g.engine.MulScalar(ctx, x2, ops.FromFloat64(3*0.044715))
	if err != nil {
		return nil, err
	}
	// 1 + 3 * 0.044715 * x^2
	dterm2, err := g.engine.AddScalar(ctx, dterm1, ops.One())
	if err != nil {
		return nil, err
	}
	// sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
	dudx, err := g.engine.MulScalar(ctx, dterm2, ops.FromFloat64(math.Sqrt(2/math.Pi)))
	if err != nil {
		return nil, err
	}

	// 0.5 * (1 + tanh(u))
	onePlusTanh, err := g.engine.AddScalar(ctx, tanhU, ops.One())
	if err != nil {
		return nil, err
	}
	halfOnePlusTanh, err := g.engine.MulScalar(ctx, onePlusTanh, ops.FromFloat64(0.5))
	if err != nil {
		return nil, err
	}

	// 0.5 * x * sech^2(u) * du/dx
	xSechSq, err := g.engine.Mul(ctx, x, sechSq)
	if err != nil {
		return nil, err
	}
	xSechSqDudx, err := g.engine.Mul(ctx, xSechSq, dudx)
	if err != nil {
		return nil, err
	}
	halfXSechSqDudx, err := g.engine.MulScalar(ctx, xSechSqDudx, ops.FromFloat64(0.5))
	if err != nil {
		return nil, err
	}

	// derivative = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
	derivative, err := g.engine.Add(ctx, halfOnePlusTanh, halfXSechSqDudx)
	if err != nil {
		return nil, err
	}

	// inputGrad = derivative * outputGradient
	inputGrad, err := g.engine.Mul(ctx, outputGradient, derivative)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// BuildGelu constructs a standard Gelu layer for the registry.
func BuildGelu[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewGelu(engine, ops), nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Gelu[float32])(nil)
