package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Conv2d performs 2D convolution (inference-only).
// Forward expects inputs (X, W [,B]) where:
//   - X:  [N, C_in, H, W]
//   - W:  [C_out, C_in/groups, kH, kW]
//   - B:  [C_out] (optional)
type Conv2d[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	strides     [2]int
	pads        [4]int // top, left, bottom, right
	dilations   [2]int
	groups      int
	outputShape []int
}

// NewConv2d creates a Conv2d layer.
// strides: [strideH, strideW]; pads: [top, left, bottom, right]; dilations: [dH, dW].
func NewConv2d[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	strides []int,
	pads []int,
	dilations []int,
	groups int,
) *Conv2d[T] {
	c := &Conv2d[T]{engine: engine, ops: ops, groups: groups}
	if groups <= 0 {
		c.groups = 1
	}
	if len(strides) >= 2 {
		c.strides = [2]int{strides[0], strides[1]}
	} else {
		c.strides = [2]int{1, 1}
	}
	if len(pads) >= 4 {
		c.pads = [4]int{pads[0], pads[1], pads[2], pads[3]}
	}
	if len(dilations) >= 2 {
		c.dilations = [2]int{dilations[0], dilations[1]}
	} else {
		c.dilations = [2]int{1, 1}
	}
	return c
}

// im2col extracts input patches into a column matrix for one batch sample and group.
// Returns a tensor of shape [cInG*kH*kW, outH*outW].
func im2col[T tensor.Numeric](
	xData []T,
	inH, inW int,
	cInG, kH, kW int,
	sH, sW, padT, padL, dH, dW int,
	outH, outW int,
	icOffset int,
) []T {
	colRows := cInG * kH * kW
	colCols := outH * outW
	col := make([]T, colRows*colCols)
	for ic := range cInG {
		for kh := range kH {
			for kw := range kW {
				row := ic*kH*kW + kh*kW + kw
				for oh := range outH {
					for ow := range outW {
						ih := oh*sH - padT + kh*dH
						iw := ow*sW - padL + kw*dW
						if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
							col[row*colCols+oh*outW+ow] = xData[(icOffset+ic)*inH*inW+ih*inW+iw]
						}
					}
				}
			}
		}
	}
	return col
}

// Forward computes 2D convolution via im2col + engine.MatMul.
func (c *Conv2d[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 2 || len(inputs) > 3 {
		return nil, fmt.Errorf("Conv2d requires 2 or 3 inputs (X, W [,B]), got %d", len(inputs))
	}

	x, w := inputs[0], inputs[1]
	xShape := x.Shape()
	wShape := w.Shape()

	if len(xShape) != 4 {
		return nil, fmt.Errorf("Conv2d: X must be 4D [N,C,H,W], got shape %v", xShape)
	}
	if len(wShape) != 4 {
		return nil, fmt.Errorf("Conv2d: W must be 4D [C_out,C_in/groups,kH,kW], got shape %v", wShape)
	}

	n, cIn, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3]
	cOut, cInG, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	sH, sW := c.strides[0], c.strides[1]
	padT, padL, padB, padR := c.pads[0], c.pads[1], c.pads[2], c.pads[3]
	dH, dW := c.dilations[0], c.dilations[1]

	outH := (inH+padT+padB-dH*(kH-1)-1)/sH + 1
	outW := (inW+padL+padR-dW*(kW-1)-1)/sW + 1

	groups := c.groups
	cOutPerGroup := cOut / groups
	colK := cInG * kH * kW // shared inner dimension
	spatialSize := outH * outW

	// Reshape weight per group: [cOutPerGroup, cInG*kH*kW].
	wData := w.Data()

	outData := make([]T, n*cOut*spatialSize)
	xData := x.Data()

	for ni := range n {
		batchOffset := ni * cIn * inH * inW
		for g := range groups {
			// Build im2col matrix for this batch/group.
			colData := im2col[T](xData[batchOffset:], inH, inW,
				cInG, kH, kW, sH, sW, padT, padL, dH, dW,
				outH, outW, g*cInG)

			colTensor, err := tensor.New[T]([]int{colK, spatialSize}, colData)
			if err != nil {
				return nil, fmt.Errorf("Conv2d: im2col tensor: %w", err)
			}

			// Extract weight slice for this group: [cOutPerGroup, colK].
			wGroupStart := g * cOutPerGroup * colK
			wGroupEnd := wGroupStart + cOutPerGroup*colK
			wGroup, err := tensor.New[T]([]int{cOutPerGroup, colK}, wData[wGroupStart:wGroupEnd])
			if err != nil {
				return nil, fmt.Errorf("Conv2d: weight reshape: %w", err)
			}

			// MatMul: [cOutPerGroup, colK] x [colK, spatialSize] = [cOutPerGroup, spatialSize]
			result, err := c.engine.MatMul(ctx, wGroup, colTensor)
			if err != nil {
				return nil, fmt.Errorf("Conv2d: MatMul: %w", err)
			}

			// Copy result into output buffer at the correct position.
			rData := result.Data()
			ocOffset := g * cOutPerGroup
			for oc := range cOutPerGroup {
				dstStart := ni*cOut*spatialSize + (ocOffset+oc)*spatialSize
				copy(outData[dstStart:dstStart+spatialSize], rData[oc*spatialSize:(oc+1)*spatialSize])
			}
		}
	}

	out, err := tensor.New[T]([]int{n, cOut, outH, outW}, outData)
	if err != nil {
		return nil, fmt.Errorf("Conv2d: failed to create output tensor: %w", err)
	}

	// Add bias if provided (B shape [C_out]) via engine.Add with broadcasting.
	if len(inputs) == 3 {
		// Reshape bias from [cOut] to [1, cOut, 1, 1] for broadcasting.
		biasReshaped, err := c.engine.Reshape(ctx, inputs[2], []int{1, cOut, 1, 1})
		if err != nil {
			return nil, fmt.Errorf("Conv2d: bias reshape: %w", err)
		}
		out, err = c.engine.Add(ctx, out, biasReshaped)
		if err != nil {
			return nil, fmt.Errorf("Conv2d: bias add: %w", err)
		}
	}

	c.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only).
func (c *Conv2d[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "Conv".
func (c *Conv2d[T]) OpType() string { return "Conv" }

// Attributes returns the layer configuration.
func (c *Conv2d[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"strides":   c.strides,
		"pads":      c.pads,
		"dilations": c.dilations,
		"group":     c.groups,
	}
}

// OutputShape returns the output shape (populated after Forward).
func (c *Conv2d[T]) OutputShape() []int { return c.outputShape }

// Parameters returns nil (no embedded trainable parameters).
func (c *Conv2d[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildConv2d constructs a Conv2d layer from registry attributes.
// Expects optional int64 or []int64 attributes: "strides", "pads", "dilations", "group".
func BuildConv2d[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	strides := extractInts(attributes, "strides", []int{1, 1})
	pads := extractInts(attributes, "pads", []int{0, 0, 0, 0})
	dilations := extractInts(attributes, "dilations", []int{1, 1})
	group := 1
	if v, ok := attributes["group"]; ok {
		switch g := v.(type) {
		case int64:
			group = int(g)
		case int:
			group = g
		}
	}
	return NewConv2d(engine, ops, strides, pads, dilations, group), nil
}

// extractInts reads a []int64 or []int attribute, falling back to def.
func extractInts(attrs map[string]interface{}, key string, def []int) []int {
	v, ok := attrs[key]
	if !ok {
		return def
	}
	switch val := v.(type) {
	case []int64:
		out := make([]int, len(val))
		for i, x := range val {
			out[i] = int(x)
		}
		return out
	case []int:
		return val
	}
	return def
}

// Statically assert that Conv2d implements graph.Node.
var _ graph.Node[float32] = (*Conv2d[float32])(nil)
