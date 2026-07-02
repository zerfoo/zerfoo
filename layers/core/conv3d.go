package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Conv3d performs 3D convolution (inference-only) -- the canonical video/
// volumetric convolution used by causal 3D VAEs/UNets (E127 video VAE decode,
// T127.4.2). Forward expects inputs (X, W [,B]) where:
//   - X: [N, C_in, D, H, W]
//   - W: [C_out, C_in/groups, kD, kH, kW]
//   - B: [C_out] (optional)
//
// It composes im2col + engine.MatMul exactly like Conv2d, so the heavy GEMM
// runs on the engine (CPU/GPU). Inference-only: Backward returns nil; conv
// backward (for future VAE training) is tracked as a separate deferred issue
// (ADR-092). Unlocks the 3D-conv primitive for the whole diffusion/vision class.
type Conv3d[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	strides     [3]int
	pads        [6]int // dBegin, hBegin, wBegin, dEnd, hEnd, wEnd (ONNX N-D order)
	dilations   [3]int
	groups      int
	outputShape []int
}

// NewConv3d creates a Conv3d layer.
// strides: [sD,sH,sW]; pads: [dBegin,hBegin,wBegin,dEnd,hEnd,wEnd]; dilations: [dD,dH,dW].
func NewConv3d[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	strides []int,
	pads []int,
	dilations []int,
	groups int,
) *Conv3d[T] {
	c := &Conv3d[T]{engine: engine, ops: ops, groups: groups}
	if groups <= 0 {
		c.groups = 1
	}
	if len(strides) >= 3 {
		c.strides = [3]int{strides[0], strides[1], strides[2]}
	} else {
		c.strides = [3]int{1, 1, 1}
	}
	if len(pads) >= 6 {
		c.pads = [6]int{pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]}
	}
	if len(dilations) >= 3 {
		c.dilations = [3]int{dilations[0], dilations[1], dilations[2]}
	} else {
		c.dilations = [3]int{1, 1, 1}
	}
	return c
}

// im2col3d extracts input patches into a column matrix for one sample and
// group. Returns [cInG*kD*kH*kW, outD*outH*outW]. Mirrors im2col (2D).
func im2col3d[T tensor.Numeric](
	xData []T,
	inD, inH, inW int,
	cInG, kD, kH, kW int,
	sD, sH, sW, padD, padH, padW, dD, dH, dW int,
	outD, outH, outW int,
	icOffset int,
) []T {
	colRows := cInG * kD * kH * kW
	colCols := outD * outH * outW
	col := make([]T, colRows*colCols)
	for ic := range cInG {
		for kd := range kD {
			for kh := range kH {
				for kw := range kW {
					row := ((ic*kD+kd)*kH+kh)*kW + kw
					for od := range outD {
						id := od*sD - padD + kd*dD
						if id < 0 || id >= inD {
							continue
						}
						for oh := range outH {
							ih := oh*sH - padH + kh*dH
							if ih < 0 || ih >= inH {
								continue
							}
							for ow := range outW {
								iw := ow*sW - padW + kw*dW
								if iw >= 0 && iw < inW {
									col[row*colCols+(od*outH+oh)*outW+ow] =
										xData[(((icOffset+ic)*inD+id)*inH+ih)*inW+iw]
								}
							}
						}
					}
				}
			}
		}
	}
	return col
}

// Forward computes 3D convolution via im2col + engine.MatMul.
func (c *Conv3d[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 2 || len(inputs) > 3 {
		return nil, fmt.Errorf("Conv3d requires 2 or 3 inputs (X, W [,B]), got %d", len(inputs))
	}
	x, w := inputs[0], inputs[1]
	xShape := x.Shape()
	wShape := w.Shape()
	if len(xShape) != 5 {
		return nil, fmt.Errorf("Conv3d: X must be 5D [N,C,D,H,W], got shape %v", xShape)
	}
	if len(wShape) != 5 {
		return nil, fmt.Errorf("Conv3d: W must be 5D [C_out,C_in/groups,kD,kH,kW], got shape %v", wShape)
	}

	n, cIn, inD, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3], xShape[4]
	cOut, cInG, kD, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3], wShape[4]
	sD, sH, sW := c.strides[0], c.strides[1], c.strides[2]
	padDb, padHb, padWb := c.pads[0], c.pads[1], c.pads[2]
	padDe, padHe, padWe := c.pads[3], c.pads[4], c.pads[5]
	dD, dH, dW := c.dilations[0], c.dilations[1], c.dilations[2]

	if cInG*c.groups != cIn {
		return nil, fmt.Errorf("Conv3d: C_in %d != C_in/groups %d * groups %d", cIn, cInG, c.groups)
	}

	outD := (inD+padDb+padDe-dD*(kD-1)-1)/sD + 1
	outH := (inH+padHb+padHe-dH*(kH-1)-1)/sH + 1
	outW := (inW+padWb+padWe-dW*(kW-1)-1)/sW + 1
	if outD <= 0 || outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("Conv3d: non-positive output spatial size [%d,%d,%d]", outD, outH, outW)
	}

	groups := c.groups
	cOutPerGroup := cOut / groups
	colK := cInG * kD * kH * kW
	spatialSize := outD * outH * outW

	wData := w.Data()
	xData := x.Data()
	outData := make([]T, n*cOut*spatialSize)

	for ni := range n {
		batchOffset := ni * cIn * inD * inH * inW
		for g := range groups {
			colData := im2col3d[T](xData[batchOffset:], inD, inH, inW,
				cInG, kD, kH, kW, sD, sH, sW, padDb, padHb, padWb, dD, dH, dW,
				outD, outH, outW, g*cInG)

			colTensor, err := tensor.New[T]([]int{colK, spatialSize}, colData)
			if err != nil {
				return nil, fmt.Errorf("Conv3d: im2col tensor: %w", err)
			}

			wGroupStart := g * cOutPerGroup * colK
			wGroup, err := tensor.New[T]([]int{cOutPerGroup, colK}, wData[wGroupStart:wGroupStart+cOutPerGroup*colK])
			if err != nil {
				return nil, fmt.Errorf("Conv3d: weight reshape: %w", err)
			}

			result, err := c.engine.MatMul(ctx, wGroup, colTensor)
			if err != nil {
				return nil, fmt.Errorf("Conv3d: MatMul: %w", err)
			}

			rData := result.Data()
			ocOffset := g * cOutPerGroup
			for oc := range cOutPerGroup {
				dstStart := ni*cOut*spatialSize + (ocOffset+oc)*spatialSize
				copy(outData[dstStart:dstStart+spatialSize], rData[oc*spatialSize:(oc+1)*spatialSize])
			}
		}
	}

	out, err := tensor.New[T]([]int{n, cOut, outD, outH, outW}, outData)
	if err != nil {
		return nil, fmt.Errorf("Conv3d: failed to create output tensor: %w", err)
	}

	if len(inputs) == 3 {
		biasReshaped, err := c.engine.Reshape(ctx, inputs[2], []int{1, cOut, 1, 1, 1})
		if err != nil {
			return nil, fmt.Errorf("Conv3d: bias reshape: %w", err)
		}
		out, err = c.engine.Add(ctx, out, biasReshaped)
		if err != nil {
			return nil, fmt.Errorf("Conv3d: bias add: %w", err)
		}
	}

	c.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only; see ADR-092).
func (c *Conv3d[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "Conv".
func (c *Conv3d[T]) OpType() string { return "Conv" }

// Attributes returns the layer configuration.
func (c *Conv3d[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"strides":   c.strides,
		"pads":      c.pads,
		"dilations": c.dilations,
		"group":     c.groups,
	}
}

// OutputShape returns the output shape (populated after Forward).
func (c *Conv3d[T]) OutputShape() []int { return c.outputShape }

// Parameters returns nil (weight/bias are passed as forward inputs).
func (c *Conv3d[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildConv3d constructs a Conv3d layer from registry attributes.
func BuildConv3d[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	strides := extractInts(attributes, "strides", []int{1, 1, 1})
	pads := extractInts(attributes, "pads", []int{0, 0, 0, 0, 0, 0})
	dilations := extractInts(attributes, "dilations", []int{1, 1, 1})
	group := 1
	if v, ok := attributes["group"]; ok {
		switch g := v.(type) {
		case int64:
			group = int(g)
		case int:
			group = g
		}
	}
	return NewConv3d(engine, ops, strides, pads, dilations, group), nil
}

// Statically assert that Conv3d implements graph.Node.
var _ graph.Node[float32] = (*Conv3d[float32])(nil)
