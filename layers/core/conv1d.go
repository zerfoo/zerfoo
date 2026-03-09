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

// Conv1D implements a 1D convolution layer.
// Input shape: [batch, channels_in, length]
// Weight shape: [channels_out, channels_in, kernel_size]
// Output shape: [batch, channels_out, output_length]
// where output_length = (length + 2*padding - kernel_size) / stride + 1
type Conv1D[T tensor.Numeric] struct {
	name        string
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	weight      *graph.Parameter[T]
	bias        *graph.Parameter[T]
	inChannels  int
	outChannels int
	kernelSize  int
	stride      int
	padding     int
	useBias     bool
}

// NewConv1D creates a new Conv1D layer.
func NewConv1D[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inChannels, outChannels, kernelSize int,
	opts ...Conv1DOption,
) (*Conv1D[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inChannels <= 0 || outChannels <= 0 || kernelSize <= 0 {
		return nil, fmt.Errorf("inChannels, outChannels, and kernelSize must be positive")
	}

	c := &Conv1D[T]{
		name:        name,
		engine:      engine,
		ops:         ops,
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      1,
		padding:     0,
		useBias:     true,
	}
	for _, opt := range opts {
		opt(c)
	}

	// Weight: [outChannels, inChannels, kernelSize]
	wData := randomData[T](outChannels * inChannels * kernelSize)
	wTensor, err := tensor.New[T]([]int{outChannels, inChannels, kernelSize}, wData)
	if err != nil {
		return nil, err
	}
	c.weight, err = graph.NewParameter[T](name+"_weight", wTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	if c.useBias {
		bData := make([]T, outChannels) // zero-initialized
		bTensor, err := tensor.New[T]([]int{outChannels}, bData)
		if err != nil {
			return nil, err
		}
		c.bias, err = graph.NewParameter[T](name+"_bias", bTensor, tensor.New[T])
		if err != nil {
			return nil, err
		}
	}

	return c, nil
}

// Conv1DOption configures Conv1D construction.
type Conv1DOption func(c interface{})

// Conv1DStride sets the stride.
func Conv1DStride(s int) Conv1DOption {
	return func(c interface{}) {
		if cc, ok := c.(interface{ setStride(int) }); ok {
			cc.setStride(s)
		}
	}
}

// Conv1DPadding sets the padding.
func Conv1DPadding(p int) Conv1DOption {
	return func(c interface{}) {
		if cc, ok := c.(interface{ setPadding(int) }); ok {
			cc.setPadding(p)
		}
	}
}

// Conv1DWithoutBias disables bias.
func Conv1DWithoutBias() Conv1DOption {
	return func(c interface{}) {
		if cc, ok := c.(interface{ setUseBias(bool) }); ok {
			cc.setUseBias(false)
		}
	}
}

func (c *Conv1D[T]) setStride(s int)   { c.stride = s }
func (c *Conv1D[T]) setPadding(p int)  { c.padding = p }
func (c *Conv1D[T]) setUseBias(b bool) { c.useBias = b }

func (c *Conv1D[T]) OpType() string { return "Conv1D" }

func (c *Conv1D[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"in_channels":  c.inChannels,
		"out_channels": c.outChannels,
		"kernel_size":  c.kernelSize,
		"stride":       c.stride,
		"padding":      c.padding,
		"use_bias":     c.useBias,
	}
}

func (c *Conv1D[T]) outputLength(inputLength int) int {
	return (inputLength + 2*c.padding - c.kernelSize) / c.stride + 1
}

func (c *Conv1D[T]) OutputShape() []int {
	return []int{-1, c.outChannels, -1}
}

// Forward computes 1D convolution.
// Input: [batch, inChannels, length]
// Output: [batch, outChannels, outputLength]
func (c *Conv1D[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Conv1D requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("Conv1D input must be 3D [batch, channels, length], got %v", shape)
	}

	batch := shape[0]
	inCh := shape[1]
	inLen := shape[2]
	outLen := c.outputLength(inLen)

	if inCh != c.inChannels {
		return nil, fmt.Errorf("Conv1D: input channels %d != expected %d", inCh, c.inChannels)
	}

	inputData := input.Data()
	weightData := c.weight.Value.Data()
	outData := make([]T, batch*c.outChannels*outLen)

	for b := 0; b < batch; b++ {
		for oc := 0; oc < c.outChannels; oc++ {
			for ol := 0; ol < outLen; ol++ {
				var sum T
				for ic := 0; ic < c.inChannels; ic++ {
					for k := 0; k < c.kernelSize; k++ {
						inIdx := ol*c.stride - c.padding + k
						if inIdx >= 0 && inIdx < inLen {
							inputVal := inputData[b*inCh*inLen+ic*inLen+inIdx]
							weightVal := weightData[oc*c.inChannels*c.kernelSize+ic*c.kernelSize+k]
							sum = c.ops.Add(sum, c.ops.Mul(inputVal, weightVal))
						}
					}
				}
				if c.useBias {
					sum = c.ops.Add(sum, c.bias.Value.Data()[oc])
				}
				outData[b*c.outChannels*outLen+oc*outLen+ol] = sum
			}
		}
	}

	return tensor.New[T]([]int{batch, c.outChannels, outLen}, outData)
}

// Backward computes gradients for Conv1D.
func (c *Conv1D[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Conv1D requires exactly 1 input for backward, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	batch := shape[0]
	inLen := shape[2]
	outLen := c.outputLength(inLen)

	inputData := input.Data()
	weightData := c.weight.Value.Data()
	gradData := outputGradient.Data()

	// Gradient w.r.t. weight
	dwData := make([]T, c.outChannels*c.inChannels*c.kernelSize)
	for b := 0; b < batch; b++ {
		for oc := 0; oc < c.outChannels; oc++ {
			for ic := 0; ic < c.inChannels; ic++ {
				for k := 0; k < c.kernelSize; k++ {
					var sum T
					for ol := 0; ol < outLen; ol++ {
						inIdx := ol*c.stride - c.padding + k
						if inIdx >= 0 && inIdx < inLen {
							g := gradData[b*c.outChannels*outLen+oc*outLen+ol]
							x := inputData[b*c.inChannels*inLen+ic*inLen+inIdx]
							sum = c.ops.Add(sum, c.ops.Mul(g, x))
						}
					}
					dwData[oc*c.inChannels*c.kernelSize+ic*c.kernelSize+k] = c.ops.Add(
						dwData[oc*c.inChannels*c.kernelSize+ic*c.kernelSize+k], sum)
				}
			}
		}
	}
	dwTensor, err := tensor.New[T](c.weight.Value.Shape(), dwData)
	if err != nil {
		return nil, err
	}
	c.weight.Gradient, err = c.engine.Add(ctx, c.weight.Gradient, dwTensor)
	if err != nil {
		return nil, err
	}

	// Gradient w.r.t. bias
	if c.useBias {
		dbData := make([]T, c.outChannels)
		for b := 0; b < batch; b++ {
			for oc := 0; oc < c.outChannels; oc++ {
				for ol := 0; ol < outLen; ol++ {
					dbData[oc] = c.ops.Add(dbData[oc], gradData[b*c.outChannels*outLen+oc*outLen+ol])
				}
			}
		}
		dbTensor, err := tensor.New[T](c.bias.Value.Shape(), dbData)
		if err != nil {
			return nil, err
		}
		c.bias.Gradient, err = c.engine.Add(ctx, c.bias.Gradient, dbTensor)
		if err != nil {
			return nil, err
		}
	}

	// Gradient w.r.t. input (transposed convolution / full correlation)
	dxData := make([]T, batch*c.inChannels*inLen)
	for b := 0; b < batch; b++ {
		for oc := 0; oc < c.outChannels; oc++ {
			for ol := 0; ol < outLen; ol++ {
				g := gradData[b*c.outChannels*outLen+oc*outLen+ol]
				for ic := 0; ic < c.inChannels; ic++ {
					for k := 0; k < c.kernelSize; k++ {
						inIdx := ol*c.stride - c.padding + k
						if inIdx >= 0 && inIdx < inLen {
							w := weightData[oc*c.inChannels*c.kernelSize+ic*c.kernelSize+k]
							idx := b*c.inChannels*inLen + ic*inLen + inIdx
							dxData[idx] = c.ops.Add(dxData[idx], c.ops.Mul(g, w))
						}
					}
				}
			}
		}
	}
	dxTensor, err := tensor.New[T](shape, dxData)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dxTensor}, nil
}

func (c *Conv1D[T]) Parameters() []*graph.Parameter[T] {
	params := []*graph.Parameter[T]{c.weight}
	if c.useBias {
		params = append(params, c.bias)
	}
	return params
}
