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

// Forward computes 2D convolution.
func (c *Conv2d[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
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

	outData := make([]T, n*cOut*outH*outW)
	xData := x.Data()
	wData := w.Data()

	groups := c.groups
	cOutPerGroup := cOut / groups

	_ = cIn // cIn is checked implicitly via cInG * groups

	for ni := range n {
		for g := range groups {
			icOffset := g * cInG
			ocOffset := g * cOutPerGroup
			for oc := range cOutPerGroup {
				absOC := ocOffset + oc
				for oh := range outH {
					for ow := range outW {
						val := c.ops.FromFloat64(0)
						for ic := range cInG {
							for kh := range kH {
								for kw := range kW {
									ih := oh*sH - padT + kh*dH
									iw := ow*sW - padL + kw*dW
									if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
										xIdx := ni*cIn*inH*inW + (icOffset+ic)*inH*inW + ih*inW + iw
										wIdx := absOC*cInG*kH*kW + ic*kH*kW + kh*kW + kw
										val = c.ops.Add(val, c.ops.Mul(xData[xIdx], wData[wIdx]))
									}
								}
							}
						}
						outIdx := ni*cOut*outH*outW + absOC*outH*outW + oh*outW + ow
						outData[outIdx] = val
					}
				}
			}
		}
	}

	// Add bias if provided (B shape [C_out]).
	if len(inputs) == 3 {
		bData := inputs[2].Data()
		for ni := range n {
			for oc := range cOut {
				for oh := range outH {
					for ow := range outW {
						idx := ni*cOut*outH*outW + oc*outH*outW + oh*outW + ow
						outData[idx] = c.ops.Add(outData[idx], bData[oc])
					}
				}
			}
		}
	}

	out, err := tensor.New[T]([]int{n, cOut, outH, outW}, outData)
	if err != nil {
		return nil, fmt.Errorf("Conv2d: failed to create output tensor: %w", err)
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
