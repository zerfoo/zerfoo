package core

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Resize resamples a 4D input tensor to a new spatial size.
// Only nearest-neighbor interpolation is implemented.
// Either scales or sizes (but not both) must be provided at build time.
// Forward expects exactly one input: X [N, C, H, W].
type Resize[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	mode        string    // "nearest" (default)
	scales      []float64 // [scaleN, scaleC, scaleH, scaleW]; nil if sizes used
	sizes       []int64   // absolute [N, C, outH, outW]; nil if scales used
	outputShape []int
}

// NewResize creates a Resize layer.
// Provide either non-nil scales or non-nil sizes (not both).
func NewResize[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	mode string,
	scales []float64,
	sizes []int64,
) *Resize[T] {
	if mode == "" {
		mode = "nearest"
	}
	return &Resize[T]{engine: engine, ops: ops, mode: mode, scales: scales, sizes: sizes}
}

// Forward resamples X to the target spatial size.
func (r *Resize[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Resize requires exactly 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	inShape := x.Shape()
	if len(inShape) != 4 {
		return nil, fmt.Errorf("Resize requires 4D input [N,C,H,W], got shape %v", inShape)
	}

	n, c, inH, inW := inShape[0], inShape[1], inShape[2], inShape[3]

	var outH, outW int
	switch {
	case len(r.sizes) == 4:
		outH = int(r.sizes[2])
		outW = int(r.sizes[3])
	case len(r.scales) == 4:
		outH = max(1, int(float64(inH)*r.scales[2]))
		outW = max(1, int(float64(inW)*r.scales[3]))
	default:
		return nil, errors.New("Resize: either scales or sizes must be set")
	}

	xData := x.Data()
	outData := make([]T, n*c*outH*outW)

	scaleH := float64(inH) / float64(outH)
	scaleW := float64(inW) / float64(outW)

	for ni := range n {
		for ci := range c {
			for oh := range outH {
				for ow := range outW {
					ih := int(float64(oh) * scaleH)
					iw := int(float64(ow) * scaleW)
					if ih >= inH {
						ih = inH - 1
					}
					if iw >= inW {
						iw = inW - 1
					}
					srcIdx := ni*c*inH*inW + ci*inH*inW + ih*inW + iw
					dstIdx := ni*c*outH*outW + ci*outH*outW + oh*outW + ow
					outData[dstIdx] = xData[srcIdx]
				}
			}
		}
	}

	out, err := tensor.New[T]([]int{n, c, outH, outW}, outData)
	if err != nil {
		return nil, fmt.Errorf("Resize: failed to create output tensor: %w", err)
	}
	r.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only).
func (r *Resize[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "Resize".
func (r *Resize[T]) OpType() string { return "Resize" }

// Attributes returns the resize configuration.
func (r *Resize[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"mode":   r.mode,
		"scales": r.scales,
		"sizes":  r.sizes,
	}
}

// OutputShape returns the output shape (populated after Forward).
func (r *Resize[T]) OutputShape() []int { return r.outputShape }

// Parameters returns nil.
func (r *Resize[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildResize constructs a Resize layer from registry attributes.
// Reads "mode" (string), "scales" ([]float64), and/or "sizes" ([]int64).
// Returns an error if neither scales nor sizes are provided.
func BuildResize[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	mode := "nearest"
	if v, ok := attributes["mode"]; ok {
		if s, ok2 := v.(string); ok2 {
			mode = s
		}
	}

	var scales []float64
	if v, ok := attributes["scales"]; ok {
		switch s := v.(type) {
		case []float64:
			scales = s
		case []float32:
			scales = make([]float64, len(s))
			for i, f := range s {
				scales[i] = float64(f)
			}
		}
	}

	var sizes []int64
	if v, ok := attributes["sizes"]; ok {
		if s, ok2 := v.([]int64); ok2 {
			sizes = s
		}
	}

	if scales == nil && sizes == nil {
		return nil, errors.New("BuildResize: attributes must contain 'scales' or 'sizes'")
	}

	return NewResize(engine, ops, mode, scales, sizes), nil
}

// Statically assert that Resize implements graph.Node.
var _ graph.Node[float32] = (*Resize[float32])(nil)
