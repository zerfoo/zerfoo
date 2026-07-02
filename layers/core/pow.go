package core //nolint:dupl // Pow follows the same binary-op pattern as Add/Sub/Mul/Div

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Pow represents an element-wise power node.
type Pow[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

// NewPow creates a new Pow node.
func NewPow[T tensor.Numeric](engine compute.Engine[T]) *Pow[T] {
	return &Pow[T]{engine: engine}
}

func (p *Pow[T]) OpType() string                  { return "Pow" }
func (p *Pow[T]) Attributes() map[string]any       { return nil }
func (p *Pow[T]) OutputShape() []int               { return nil }
func (p *Pow[T]) Parameters() []*graph.Parameter[T] { return nil }

func (p *Pow[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Pow requires 2 inputs, got %d", len(inputs))
	}
	return p.engine.Pow(ctx, inputs[0], inputs[1])
}

func (p *Pow[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Pow backward requires 2 inputs, got %d", len(inputs))
	}

	base := inputs[0]
	exp := inputs[1]
	ops := p.engine.Ops()

	// d/da(a^n) = n * a^(n-1)
	oneData := make([]T, len(exp.Data()))
	one := ops.One()
	for i := range oneData {
		oneData[i] = one
	}
	onesTensor, err := tensor.New[T](exp.Shape(), oneData)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: creating ones tensor: %w", err)
	}

	expM1, err := p.engine.Sub(ctx, exp, onesTensor)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: exp-1: %w", err)
	}

	basePowExpM1, err := p.engine.Pow(ctx, base, expM1)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: base^(exp-1): %w", err)
	}

	nTimesBasePow, err := p.engine.Mul(ctx, exp, basePowExpM1)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: n*base^(n-1): %w", err)
	}

	gradBase, err := p.engine.Mul(ctx, dOut, nTimesBasePow)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: dOut*gradBase: %w", err)
	}

	// d/dn(a^n) = a^n * ln(a)
	output, err := p.engine.Pow(ctx, base, exp)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: base^exp: %w", err)
	}

	logBase, err := p.engine.Log(ctx, base)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: log(base): %w", err)
	}

	outputTimesLog, err := p.engine.Mul(ctx, output, logBase)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: output*log(base): %w", err)
	}

	gradExp, err := p.engine.Mul(ctx, dOut, outputTimesLog)
	if err != nil {
		return nil, fmt.Errorf("Pow backward: dOut*gradExp: %w", err)
	}

	return []*tensor.TensorNumeric[T]{gradBase, gradExp}, nil
}

// BuildPow constructs a Pow node from attributes.
func BuildPow[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return NewPow[T](engine), nil
}

var _ graph.Node[float32] = (*Pow[float32])(nil)
