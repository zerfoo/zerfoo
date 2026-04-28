// Package lora provides Low-Rank Adaptation layers for parameter-efficient fine-tuning.
package lora

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// LoraLinear wraps an existing linear layer with low-rank adaptation matrices.
// During forward: y = base(x) + (alpha/rank) * B @ A @ x
// Only A and B are trainable; the base layer weights are frozen.
type LoraLinear[T tensor.Numeric] struct {
	name   string
	base   graph.Node[T]
	A      *graph.Parameter[T] // r x d_in, initialized N(0,1)
	B      *graph.Parameter[T] // d_out x r, initialized zero
	rank   int
	alpha  float32
	scale  T
	engine compute.Engine[T]

	// cached for backward
	lastInput   *tensor.TensorNumeric[T]
	lastAx      *tensor.TensorNumeric[T]
	lastBaseOut *tensor.TensorNumeric[T]
}

// NewLoraLinear creates a LoRA adapter around an existing base layer.
// rank is the low-rank dimension; alpha is the scaling factor.
// A is initialized with N(0,1) random values; B is initialized to zero.
func NewLoraLinear[T tensor.Numeric](
	name string,
	base graph.Node[T],
	rank int,
	alpha float32,
	engine compute.Engine[T],
	dIn, dOut int,
) (*LoraLinear[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if rank <= 0 {
		return nil, fmt.Errorf("rank must be positive, got %d", rank)
	}
	if dIn <= 0 || dOut <= 0 {
		return nil, fmt.Errorf("input and output dimensions must be positive")
	}

	// A: r x d_in, initialized with N(0,1)
	aData := make([]T, rank*dIn)
	for i := range aData {
		aData[i] = T(rand.NormFloat64())
	}
	aTensor, err := tensor.New[T]([]int{rank, dIn}, aData)
	if err != nil {
		return nil, fmt.Errorf("failed to create A tensor: %w", err)
	}
	aParam, err := graph.NewParameter[T](name+"_lora_a", aTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("failed to create A parameter: %w", err)
	}

	// B: d_out x r, initialized to zero
	bData := make([]T, dOut*rank)
	bTensor, err := tensor.New[T]([]int{dOut, rank}, bData)
	if err != nil {
		return nil, fmt.Errorf("failed to create B tensor: %w", err)
	}
	bParam, err := graph.NewParameter[T](name+"_lora_b", bTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("failed to create B parameter: %w", err)
	}

	return &LoraLinear[T]{
		name:   name,
		base:   base,
		A:      aParam,
		B:      bParam,
		rank:   rank,
		alpha:  alpha,
		scale:  T(alpha / float32(rank)),
		engine: engine,
	}, nil
}

// OpType returns the operation type.
func (l *LoraLinear[T]) OpType() string {
	return "LoraLinear"
}

// Attributes returns layer attributes.
func (l *LoraLinear[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"rank":  l.rank,
		"alpha": l.alpha,
	}
}

// OutputShape returns the output shape of the base layer.
func (l *LoraLinear[T]) OutputShape() []int {
	return l.base.OutputShape()
}

// Forward computes y = base(x) + (alpha/rank) * B @ A @ x.
func (l *LoraLinear[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LoraLinear requires exactly one input, got %d", len(inputs))
	}
	x := inputs[0]

	// Base forward pass (frozen)
	baseOut, err := l.base.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("base forward failed: %w", err)
	}

	// LoRA path: (alpha/rank) * B @ A @ x
	// x is (batch, d_in), A is (r, d_in), so A^T is (d_in, r)
	// x @ A^T => (batch, r)
	aT, err := l.engine.Transpose(ctx, l.A.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	ax, err := l.engine.MatMul(ctx, x, aT)
	if err != nil {
		return nil, err
	}

	// ax @ B^T => (batch, d_out), where B is (d_out, r), B^T is (r, d_out)
	bT, err := l.engine.Transpose(ctx, l.B.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	bax, err := l.engine.MatMul(ctx, ax, bT)
	if err != nil {
		return nil, err
	}

	// Scale by alpha/rank
	scale := l.scale
	scaledBAx, err := l.engine.UnaryOp(ctx, bax, func(v T) T { return v * scale })
	if err != nil {
		return nil, err
	}

	// Cache for backward
	l.lastInput = x
	l.lastAx = ax
	l.lastBaseOut = baseOut

	// y = base_out + scaled_BAx
	return l.engine.Add(ctx, baseOut, scaledBAx)
}

// Backward computes gradients for A and B (not the base layer weights).
func (l *LoraLinear[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LoraLinear requires exactly one input for backward, got %d", len(inputs))
	}
	x := inputs[0]
	scale := l.scale

	// Scale the output gradient by alpha/rank for LoRA path
	scaledGrad, err := l.engine.UnaryOp(ctx, outputGradient, func(v T) T { return v * scale })
	if err != nil {
		return nil, err
	}

	// dB: gradient w.r.t. B
	// scaledGrad is (batch, d_out), ax is (batch, r)
	// dB = scaledGrad^T @ ax => (d_out, r)
	scaledGradT, err := l.engine.Transpose(ctx, scaledGrad, []int{1, 0})
	if err != nil {
		return nil, err
	}

	// Recompute ax if not cached
	ax := l.lastAx
	if ax == nil {
		aT, err := l.engine.Transpose(ctx, l.A.Value, []int{1, 0})
		if err != nil {
			return nil, err
		}
		ax, err = l.engine.MatMul(ctx, x, aT)
		if err != nil {
			return nil, err
		}
	}

	dB, err := l.engine.MatMul(ctx, scaledGradT, ax)
	if err != nil {
		return nil, err
	}
	if l.B.Gradient == nil {
		l.B.Gradient = dB
	} else {
		l.B.Gradient, err = l.engine.Add(ctx, l.B.Gradient, dB)
		if err != nil {
			return nil, err
		}
	}

	// dA: gradient w.r.t. A
	// scaledGrad @ B => (batch, r), x is (batch, d_in)
	// dA = (scaledGrad @ B)^T @ x => (r, d_in)
	gradB, err := l.engine.MatMul(ctx, scaledGrad, l.B.Value)
	if err != nil {
		return nil, err
	}
	gradBT, err := l.engine.Transpose(ctx, gradB, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dA, err := l.engine.MatMul(ctx, gradBT, x)
	if err != nil {
		return nil, err
	}
	if l.A.Gradient == nil {
		l.A.Gradient = dA
	} else {
		l.A.Gradient, err = l.engine.Add(ctx, l.A.Gradient, dA)
		if err != nil {
			return nil, err
		}
	}

	// Propagate gradient to input through base layer
	dxBase, err := l.base.Backward(ctx, mode, outputGradient, inputs...)
	if err != nil {
		return nil, err
	}

	// LoRA contribution to input gradient: scale * B^T @ A @ dx_lora
	// dx_lora = scaledGrad @ B @ A => (batch, d_in)... but actually:
	// dy/dx through lora path = scale * (B @ A), so dx_lora = scaledGrad @ B @ A
	// gradB is already scaledGrad @ B => (batch, r)
	aValue := l.A.Value
	dxLora, err := l.engine.MatMul(ctx, gradB, aValue)
	if err != nil {
		return nil, err
	}

	// Total input gradient = base gradient + lora gradient
	if len(dxBase) == 0 {
		return []*tensor.TensorNumeric[T]{dxLora}, nil
	}
	dxTotal, err := l.engine.Add(ctx, dxBase[0], dxLora)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{dxTotal}, nil
}

// Parameters returns only the LoRA parameters A and B (not the base layer weights).
func (l *LoraLinear[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{l.A, l.B}
}

// Name returns the name of the layer.
func (l *LoraLinear[T]) Name() string {
	return l.name
}

// SetName sets the name of the layer.
func (l *LoraLinear[T]) SetName(name string) {
	l.name = name
	l.A.Name = name + "_lora_a"
	l.B.Name = name + "_lora_b"
}

// Rank returns the LoRA rank.
func (l *LoraLinear[T]) Rank() int {
	return l.rank
}

// Alpha returns the LoRA alpha scaling factor.
func (l *LoraLinear[T]) Alpha() float32 {
	return l.alpha
}
