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

// ConvTranspose3d performs 3D transposed (fractionally-strided) convolution --
// the upsampling convolution used by convolutional VAE/UNet decoders (E127
// video VAE decode, T127.4.3). Forward expects inputs (X, W [,B]) where:
//   - X: [N, C_in, D, H, W]
//   - W: [C_in, C_out, kD, kH, kW]   (torch conv_transpose weight layout)
//   - B: [C_out] (optional)
//
// It computes cols = Wᵀ @ X via engine.MatMul (heavy GEMM on the engine) then
// scatters them into the (larger) output with col2im -- the exact transpose of
// Conv3d's im2col path. Inference-only: Backward returns nil (ADR-092). groups
// are currently limited to 1. Unlocks the transposed-conv primitive for the
// whole convolutional-decoder class.
type ConvTranspose3d[T tensor.Numeric] struct {
	engine        compute.Engine[T]
	ops           numeric.Arithmetic[T]
	strides       [3]int
	pads          [6]int // dBegin, hBegin, wBegin, dEnd, hEnd, wEnd
	dilations     [3]int
	outputPadding [3]int
	groups        int
	outputShape   []int
}

// NewConvTranspose3d creates a ConvTranspose3d layer (groups=1 supported).
func NewConvTranspose3d[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	strides []int,
	pads []int,
	dilations []int,
	outputPadding []int,
	groups int,
) *ConvTranspose3d[T] {
	c := &ConvTranspose3d[T]{engine: engine, ops: ops, groups: groups}
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
	if len(outputPadding) >= 3 {
		c.outputPadding = [3]int{outputPadding[0], outputPadding[1], outputPadding[2]}
	}
	return c
}

// Forward computes 3D transposed convolution via engine.MatMul (Wᵀ@X) + col2im.
func (c *ConvTranspose3d[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 2 || len(inputs) > 3 {
		return nil, fmt.Errorf("ConvTranspose3d requires 2 or 3 inputs (X, W [,B]), got %d", len(inputs))
	}
	if c.groups != 1 {
		return nil, fmt.Errorf("ConvTranspose3d: only groups=1 is supported, got %d", c.groups)
	}
	x, w := inputs[0], inputs[1]
	xShape := x.Shape()
	wShape := w.Shape()
	if len(xShape) != 5 {
		return nil, fmt.Errorf("ConvTranspose3d: X must be 5D [N,C,D,H,W], got %v", xShape)
	}
	if len(wShape) != 5 {
		return nil, fmt.Errorf("ConvTranspose3d: W must be 5D [C_in,C_out,kD,kH,kW], got %v", wShape)
	}

	n, cIn, inD, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3], xShape[4]
	wCin, cOut, kD, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3], wShape[4]
	if wCin != cIn {
		return nil, fmt.Errorf("ConvTranspose3d: W C_in %d != X C_in %d", wCin, cIn)
	}
	sD, sH, sW := c.strides[0], c.strides[1], c.strides[2]
	padDb, padHb, padWb := c.pads[0], c.pads[1], c.pads[2]
	padDe, padHe, padWe := c.pads[3], c.pads[4], c.pads[5]
	dD, dH, dW := c.dilations[0], c.dilations[1], c.dilations[2]
	opD, opH, opW := c.outputPadding[0], c.outputPadding[1], c.outputPadding[2]

	outD := (inD-1)*sD - (padDb + padDe) + dD*(kD-1) + opD + 1
	outH := (inH-1)*sH - (padHb + padHe) + dH*(kH-1) + opH + 1
	outW := (inW-1)*sW - (padWb + padWe) + dW*(kW-1) + opW + 1
	if outD <= 0 || outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("ConvTranspose3d: non-positive output spatial size [%d,%d,%d]", outD, outH, outW)
	}

	kSize := kD * kH * kW
	sIn := inD * inH * inW
	outS := outD * outH * outW

	// Wmat = reshape(W, [C_in, C_out*K]); WmatT = [C_out*K, C_in].
	wMat, err := tensor.New[T]([]int{cIn, cOut * kSize}, w.Data())
	if err != nil {
		return nil, fmt.Errorf("ConvTranspose3d: weight reshape: %w", err)
	}
	wMatT, err := c.engine.Transpose(ctx, wMat, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("ConvTranspose3d: weight transpose: %w", err)
	}

	xData := x.Data()
	outData := make([]T, n*cOut*outS)

	for ni := range n {
		// Xmat for this sample: [C_in, sIn] (already contiguous in xData).
		xMat, err := tensor.New[T]([]int{cIn, sIn}, xData[ni*cIn*sIn:(ni+1)*cIn*sIn])
		if err != nil {
			return nil, fmt.Errorf("ConvTranspose3d: input reshape: %w", err)
		}
		// cols[C_out*K, sIn] = WmatT @ Xmat (sum over C_in).
		cols, err := c.engine.MatMul(ctx, wMatT, xMat)
		if err != nil {
			return nil, fmt.Errorf("ConvTranspose3d: MatMul: %w", err)
		}
		colData := cols.Data()
		base := ni * cOut * outS
		// col2im: scatter each (cout, k, input-pos) into the strided output pos.
		for oc := range cOut {
			for kd := range kD {
				for kh := range kH {
					for kw := range kW {
						kidx := (kd*kH+kh)*kW + kw
						rowBase := (oc*kSize + kidx) * sIn
						for id := range inD {
							od := id*sD - padDb + kd*dD
							if od < 0 || od >= outD {
								continue
							}
							for ih := range inH {
								oh := ih*sH - padHb + kh*dH
								if oh < 0 || oh >= outH {
									continue
								}
								for iw := range inW {
									ow := iw*sW - padWb + kw*dW
									if ow < 0 || ow >= outW {
										continue
									}
									src := colData[rowBase+(id*inH+ih)*inW+iw]
									dst := base + ((oc*outD+od)*outH+oh)*outW + ow
									outData[dst] += src
								}
							}
						}
					}
				}
			}
		}
	}

	out, err := tensor.New[T]([]int{n, cOut, outD, outH, outW}, outData)
	if err != nil {
		return nil, fmt.Errorf("ConvTranspose3d: failed to create output tensor: %w", err)
	}

	if len(inputs) == 3 {
		biasReshaped, err := c.engine.Reshape(ctx, inputs[2], []int{1, cOut, 1, 1, 1})
		if err != nil {
			return nil, fmt.Errorf("ConvTranspose3d: bias reshape: %w", err)
		}
		out, err = c.engine.Add(ctx, out, biasReshaped)
		if err != nil {
			return nil, fmt.Errorf("ConvTranspose3d: bias add: %w", err)
		}
	}

	c.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only; see ADR-092).
func (c *ConvTranspose3d[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "ConvTranspose".
func (c *ConvTranspose3d[T]) OpType() string { return "ConvTranspose" }

// Attributes returns the layer configuration.
func (c *ConvTranspose3d[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"strides":        c.strides,
		"pads":           c.pads,
		"dilations":      c.dilations,
		"output_padding": c.outputPadding,
		"group":          c.groups,
	}
}

// OutputShape returns the output shape (populated after Forward).
func (c *ConvTranspose3d[T]) OutputShape() []int { return c.outputShape }

// Parameters returns nil (weight/bias are passed as forward inputs).
func (c *ConvTranspose3d[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildConvTranspose3d constructs a ConvTranspose3d layer from registry attributes.
func BuildConvTranspose3d[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	strides := extractInts(attributes, "strides", []int{1, 1, 1})
	pads := extractInts(attributes, "pads", []int{0, 0, 0, 0, 0, 0})
	dilations := extractInts(attributes, "dilations", []int{1, 1, 1})
	outputPadding := extractInts(attributes, "output_padding", []int{0, 0, 0})
	group := 1
	if v, ok := attributes["group"]; ok {
		switch g := v.(type) {
		case int64:
			group = int(g)
		case int:
			group = g
		}
	}
	return NewConvTranspose3d(engine, ops, strides, pads, dilations, outputPadding, group), nil
}

// Statically assert that ConvTranspose3d implements graph.Node.
var _ graph.Node[float32] = (*ConvTranspose3d[float32])(nil)
