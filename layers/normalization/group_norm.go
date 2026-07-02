package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// GroupNormalization implements Group Normalization (Wu & He, 2018) -- the
// canonical normalization of convolutional VAE/UNet/diffusion decoders (E127
// video VAE, T127.4.1). For input X [N, C, *spatial] it splits the C channels
// into `numGroups` groups and normalizes each group's (C/groups channels x all
// spatial positions) jointly, per sample, then applies a learnable per-channel
// affine (scale, bias of shape [C]):
//
//	y = scale[c] * (x - mean_g) / sqrt(var_g + epsilon) + bias[c]
//
// where mean_g/var_g are computed over the C/groups channels and every spatial
// position within group g. All math flows through Engine primitives so it
// appears in the ExecutionPlan tape and runs on CPU/GPU. groups must divide C.
//
// When scale/bias are graph.Parameters (NewGroupNormalizationWithParams),
// Backward accumulates their gradients and returns dX. The group statistics
// fall out of the same reduce/elementwise ops as LayerNorm, applied to the
// grouped [N*groups, (C/groups)*S] view.
type GroupNormalization[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	ops       numeric.Arithmetic[T]
	numGroups int
	epsilon   T

	scale *graph.Parameter[T] // [C], optional
	bias  *graph.Parameter[T] // [C], optional

	// Cache for backward (registered with the save-for-backward contract).
	normGrouped *tensor.TensorNumeric[T] // normalized, grouped view [N*groups, M]
	invStd      *tensor.TensorNumeric[T] // 1/sqrt(var+eps) per group [N*groups, 1]
	scaleBC     *tensor.TensorNumeric[T] // scale reshaped to [1,C,1,...] (nil if no scale)
	inputShape  []int
	groupedRows int // N*groups
	groupM      int // (C/groups)*S
	saver       graph.Saver[T]
}

// SetSaver implements graph.SaverAware.
func (g *GroupNormalization[T]) SetSaver(sv graph.Saver[T]) { g.saver = sv }

// NewGroupNormalization creates an inference layer with no affine (scale/bias nil).
func NewGroupNormalization[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], numGroups int, epsilon T) *GroupNormalization[T] {
	return &GroupNormalization[T]{engine: engine, ops: ops, numGroups: numGroups, epsilon: epsilon}
}

// NewGroupNormalizationWithParams creates a layer with learnable per-channel
// scale and bias (each shape [C]).
func NewGroupNormalizationWithParams[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], numGroups int, epsilon T, scale, bias *graph.Parameter[T]) *GroupNormalization[T] {
	return &GroupNormalization[T]{engine: engine, ops: ops, numGroups: numGroups, epsilon: epsilon, scale: scale, bias: bias}
}

// Forward computes group normalization. inputs: [X].
func (g *GroupNormalization[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GroupNormalization requires 1 input (X), got %d", len(inputs))
	}
	x := inputs[0]
	xShape := x.Shape()
	if len(xShape) < 2 {
		return nil, fmt.Errorf("GroupNormalization: X must have at least rank 2 [N,C,...], got %v", xShape)
	}
	n, c := xShape[0], xShape[1]
	if g.numGroups <= 0 || c%g.numGroups != 0 {
		return nil, fmt.Errorf("GroupNormalization: C %d not divisible by numGroups %d", c, g.numGroups)
	}
	s := 1
	for i := 2; i < len(xShape); i++ {
		s *= xShape[i]
	}
	rows := n * g.numGroups
	m := (c / g.numGroups) * s

	e := g.engine
	grouped, err := e.Reshape(ctx, x, []int{rows, m})
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: reshape grouped: %w", err)
	}
	mean, err := e.ReduceMean(ctx, grouped, 1, true)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: mean: %w", err)
	}
	centered, err := e.Sub(ctx, grouped, mean)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: center: %w", err)
	}
	sq, err := e.Mul(ctx, centered, centered)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: square: %w", err)
	}
	variance, err := e.ReduceMean(ctx, sq, 1, true)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: var: %w", err)
	}
	veps, err := e.AddScalar(ctx, variance, g.epsilon)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: var+eps: %w", err)
	}
	invStd, err := e.Rsqrt(ctx, veps)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: rsqrt: %w", err)
	}
	normGrouped, err := e.Mul(ctx, centered, invStd)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: normalize: %w", err)
	}

	out, err := e.Reshape(ctx, normGrouped, xShape)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization: reshape back: %w", err)
	}

	// Per-channel affine via [1, C, 1, ...] broadcast.
	bShape := make([]int, len(xShape))
	bShape[0], bShape[1] = 1, c
	for i := 2; i < len(xShape); i++ {
		bShape[i] = 1
	}
	var scaleBC *tensor.TensorNumeric[T]
	if g.scale != nil {
		scaleBC, err = e.Reshape(ctx, g.scale.Value, bShape)
		if err != nil {
			return nil, fmt.Errorf("GroupNormalization: reshape scale: %w", err)
		}
		out, err = e.Mul(ctx, out, scaleBC)
		if err != nil {
			return nil, fmt.Errorf("GroupNormalization: mul scale: %w", err)
		}
	}
	if g.bias != nil {
		biasBC, err := e.Reshape(ctx, g.bias.Value, bShape)
		if err != nil {
			return nil, fmt.Errorf("GroupNormalization: reshape bias: %w", err)
		}
		out, err = e.Add(ctx, out, biasBC)
		if err != nil {
			return nil, fmt.Errorf("GroupNormalization: add bias: %w", err)
		}
	}

	g.normGrouped = normGrouped
	g.invStd = invStd
	g.scaleBC = scaleBC
	g.inputShape = xShape
	g.groupedRows = rows
	g.groupM = m
	if g.saver != nil {
		if scaleBC != nil {
			g.saver.SaveForBackward(normGrouped, invStd, scaleBC)
		} else {
			g.saver.SaveForBackward(normGrouped, invStd)
		}
	}
	return out, nil
}

// Backward returns dX and accumulates scale/bias gradients. inputs: [X].
func (g *GroupNormalization[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GroupNormalization: %w, expected 1 input for backward, got %d", graph.ErrInvalidInputCount, len(inputs))
	}
	if g.normGrouped == nil || g.invStd == nil {
		return nil, fmt.Errorf("GroupNormalization: backward called before forward")
	}
	e := g.engine

	// normalized in original shape, for per-channel parameter gradients.
	normalized, err := e.Reshape(ctx, g.normGrouped, g.inputShape)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: reshape normalized: %w", err)
	}

	// Parameter gradients: reduce over all dims except channel (dim 1).
	if g.scale != nil {
		dScaleFull, err := e.Mul(ctx, dOut, normalized)
		if err != nil {
			return nil, fmt.Errorf("GroupNormalization backward: dScale mul: %w", err)
		}
		if err := g.reduceToChannelAndAccum(ctx, dScaleFull, g.scale); err != nil {
			return nil, err
		}
	}
	if g.bias != nil {
		if err := g.reduceToChannelAndAccum(ctx, dOut, g.bias); err != nil {
			return nil, err
		}
	}

	// dNormalized = dOut * scale (broadcast); then group-norm input gradient.
	dNorm := dOut
	if g.scaleBC != nil {
		dNorm, err = e.Mul(ctx, dOut, g.scaleBC)
		if err != nil {
			return nil, fmt.Errorf("GroupNormalization backward: dNorm mul scale: %w", err)
		}
	}
	dNormG, err := e.Reshape(ctx, dNorm, []int{g.groupedRows, g.groupM})
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: reshape dNorm: %w", err)
	}
	// dX_g = invStd * (dNorm - mean_M(dNorm) - normGrouped * mean_M(dNorm*normGrouped))
	m1, err := e.ReduceMean(ctx, dNormG, 1, true)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: mean(dNorm): %w", err)
	}
	dn, err := e.Mul(ctx, dNormG, g.normGrouped)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: dNorm*norm: %w", err)
	}
	m2, err := e.ReduceMean(ctx, dn, 1, true)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: mean(dNorm*norm): %w", err)
	}
	t1, err := e.Sub(ctx, dNormG, m1)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: sub m1: %w", err)
	}
	nm2, err := e.Mul(ctx, g.normGrouped, m2)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: norm*m2: %w", err)
	}
	t2, err := e.Sub(ctx, t1, nm2)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: sub nm2: %w", err)
	}
	dXg, err := e.Mul(ctx, t2, g.invStd)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: mul invStd: %w", err)
	}
	dX, err := e.Reshape(ctx, dXg, g.inputShape)
	if err != nil {
		return nil, fmt.Errorf("GroupNormalization backward: reshape dX: %w", err)
	}
	return []*tensor.TensorNumeric[T]{dX}, nil
}

// reduceToChannelAndAccum reduces `full` ([N,C,*]) over every dim except the
// channel dim, reshapes to the parameter's [C] shape, and accumulates.
func (g *GroupNormalization[T]) reduceToChannelAndAccum(ctx context.Context, full *tensor.TensorNumeric[T], param *graph.Parameter[T]) error {
	red := full
	var err error
	for dim := len(red.Shape()) - 1; dim >= 0; dim-- {
		if dim == 1 {
			continue
		}
		red, err = g.engine.ReduceSum(ctx, red, dim, true)
		if err != nil {
			return fmt.Errorf("GroupNormalization backward: reduce dim %d: %w", dim, err)
		}
	}
	red, err = red.Reshape(param.Value.Shape())
	if err != nil {
		return fmt.Errorf("GroupNormalization backward: reshape param grad: %w", err)
	}
	if param.Gradient == nil {
		param.Gradient = red
	} else {
		param.Gradient, err = g.engine.Add(ctx, param.Gradient, red, param.Gradient)
		if err != nil {
			return fmt.Errorf("GroupNormalization backward: accumulate param grad: %w", err)
		}
	}
	return nil
}

// OpType returns "GroupNormalization".
func (g *GroupNormalization[T]) OpType() string { return "GroupNormalization" }

// Attributes returns the layer configuration.
func (g *GroupNormalization[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"num_groups": g.numGroups, "epsilon": g.epsilon}
}

// OutputShape returns the output shape (same as input; populated after Forward).
func (g *GroupNormalization[T]) OutputShape() []int { return g.inputShape }

// Parameters returns the learnable affine parameters when set.
func (g *GroupNormalization[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	if g.scale != nil {
		params = append(params, g.scale)
	}
	if g.bias != nil {
		params = append(params, g.bias)
	}
	return params
}

// BuildGroupNormalization constructs a GroupNormalization layer for the registry.
// Reads "num_groups" (int/int64, default 32) and "epsilon" (default 1e-5).
func BuildGroupNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	numGroups := 32
	if v, ok := attributes["num_groups"]; ok {
		switch g := v.(type) {
		case int64:
			numGroups = int(g)
		case int:
			numGroups = g
		}
	}
	eps := 1e-5
	if v, ok := attributes["epsilon"]; ok {
		switch e := v.(type) {
		case float64:
			eps = e
		case float32:
			eps = float64(e)
		}
	}
	scale, bias := params["scale"], params["bias"]
	if scale != nil || bias != nil {
		return NewGroupNormalizationWithParams(engine, ops, numGroups, ops.FromFloat64(eps), scale, bias), nil
	}
	return NewGroupNormalization(engine, ops, numGroups, ops.FromFloat64(eps)), nil
}

// Statically assert that GroupNormalization implements graph.Node + SaverAware.
var (
	_ graph.Node[float32]       = (*GroupNormalization[float32])(nil)
	_ graph.SaverAware[float32] = (*GroupNormalization[float32])(nil)
)
