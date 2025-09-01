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

// MatMulNBits represents a matrix multiplication layer with N-bit quantized weights.
// This is commonly used for 4-bit quantized models like Gemma 3.
// 
// The layer performs: output = input @ dequantize(quantized_weights)
// Where dequantize unpacks N-bit weights and applies scale/zero-point transformation.
type MatMulNBits[T tensor.Numeric] struct {
	name    string
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	
	// Quantized weights stored as packed N-bit values in uint8 array
	quantizedWeights *tensor.TensorNumeric[uint8]
	
	// Quantization parameters
	scale     *tensor.TensorNumeric[T]
	zeroPoint *tensor.TensorNumeric[uint8]
	
	// Configuration
	nbits     int  // Number of bits per weight (typically 4)
	symmetric bool // Whether to use symmetric quantization
	
	// Output shape
	outputShape []int
	
	// Cache for dequantized weights to avoid recomputation
	dequantizedWeights *tensor.TensorNumeric[T]
	cacheValid         bool
}

// NewMatMulNBits creates a new N-bit quantized matrix multiplication layer.
func NewMatMulNBits[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	quantizedWeights *tensor.TensorNumeric[uint8],
	scale *tensor.TensorNumeric[T],
	zeroPoint *tensor.TensorNumeric[uint8],
	nbits int,
	symmetric bool,
) (*MatMulNBits[T], error) {
	if nbits != 4 {
		return nil, fmt.Errorf("only 4-bit quantization is currently supported, got %d bits", nbits)
	}
	
	if quantizedWeights == nil {
		return nil, fmt.Errorf("quantized weights cannot be nil")
	}
	
	if scale == nil {
		return nil, fmt.Errorf("scale tensor cannot be nil")
	}
	
	// Validate dimensions
	weightsShape := quantizedWeights.Shape()
	if len(weightsShape) != 2 {
		return nil, fmt.Errorf("quantized weights must be 2D, got shape %v", weightsShape)
	}
	
	// For 4-bit weights, each uint8 stores 2 weight values
	// So actual weight matrix is [weightsShape[0], weightsShape[1]*2]
	actualRows := weightsShape[0]
	actualCols := weightsShape[1] * 2
	
	scaleShape := scale.Shape()
	if len(scaleShape) != 1 {
		return nil, fmt.Errorf("scale must be 1D, got shape %v", scaleShape)
	}
	
	// Scale can be per-row or per-tensor
	if scaleShape[0] != actualRows && scaleShape[0] != 1 {
		return nil, fmt.Errorf("scale shape [%d] incompatible with weight matrix rows %d", scaleShape[0], actualRows)
	}
	
	// Zero point validation (optional for symmetric quantization)
	if !symmetric && zeroPoint != nil {
		zpShape := zeroPoint.Shape()
		if len(zpShape) != 1 {
			return nil, fmt.Errorf("zero point must be 1D, got shape %v", zpShape)
		}
		if zpShape[0] != scaleShape[0] {
			return nil, fmt.Errorf("zero point shape [%d] must match scale shape [%d]", zpShape[0], scaleShape[0])
		}
	}
	
	// Compute output shape for the layer
	outputShape := []int{actualRows, actualCols}
	
	return &MatMulNBits[T]{
		name:               name,
		engine:             engine,
		ops:                ops,
		quantizedWeights:   quantizedWeights,
		scale:              scale,
		zeroPoint:          zeroPoint,
		nbits:              nbits,
		symmetric:          symmetric,
		outputShape:        outputShape,
		dequantizedWeights: nil,
		cacheValid:         false,
	}, nil
}

// dequantizeWeights unpacks the N-bit weights and applies dequantization.
// This is cached to avoid recomputation on multiple forward passes.
func (m *MatMulNBits[T]) dequantizeWeights() (*tensor.TensorNumeric[T], error) {
	if m.cacheValid && m.dequantizedWeights != nil {
		return m.dequantizedWeights, nil
	}
	
	quantData := m.quantizedWeights.Data()
	scaleData := m.scale.Data()
	
	var zeroPointData []uint8
	if !m.symmetric && m.zeroPoint != nil {
		zeroPointData = m.zeroPoint.Data()
	}
	
	// Unpack 4-bit weights
	unpackedWeights := numeric.Unpack4BitSlice(quantData)
	
	// Convert to float and apply dequantization
	weightsShape := m.quantizedWeights.Shape()
	actualRows := weightsShape[0]
	actualCols := weightsShape[1] * 2
	
	dequantizedData := make([]T, len(unpackedWeights))
	
	perRowScale := len(scaleData) == actualRows
	perRowZeroPoint := len(zeroPointData) == actualRows
	
	for i := 0; i < actualRows; i++ {
		// Get scale and zero point for this row
		var scale T
		var zeroPoint uint8
		
		if perRowScale {
			scale = scaleData[i]
		} else {
			scale = scaleData[0] // Global scale
		}
		
		if !m.symmetric && zeroPointData != nil {
			if perRowZeroPoint {
				zeroPoint = zeroPointData[i]
			} else {
				zeroPoint = zeroPointData[0] // Global zero point
			}
		} else if m.symmetric {
			zeroPoint = 128 // Middle of uint8 range for symmetric quantization
		}
		
		// Dequantize this row: dequantized = scale * (quantized - zero_point)
		for j := 0; j < actualCols; j++ {
			idx := i*actualCols + j
			quantizedVal := unpackedWeights[idx]
			dequantizedData[idx] = scale * (T(quantizedVal) - T(zeroPoint))
		}
	}
	
	// Create dequantized weight tensor
	var err error
	m.dequantizedWeights, err = tensor.New[T]([]int{actualRows, actualCols}, dequantizedData)
	if err != nil {
		return nil, fmt.Errorf("failed to create dequantized weights tensor: %w", err)
	}
	
	m.cacheValid = true
	return m.dequantizedWeights, nil
}

// OpType returns the operation type of the node.
func (m *MatMulNBits[T]) OpType() string {
	return "MatMulNBits"
}

// Attributes returns the node's non-tensor attributes.
func (m *MatMulNBits[T]) Attributes() map[string]interface{} {
	attrs := map[string]interface{}{
		"nbits":     m.nbits,
		"symmetric": m.symmetric,
	}
	return attrs
}

// Parameters returns the trainable parameters (empty for quantized layers).
func (m *MatMulNBits[T]) Parameters() []*graph.Parameter[T] {
	// Quantized weights are typically frozen, so no trainable parameters
	return []*graph.Parameter[T]{}
}

// OutputShape returns the shape of the output tensor.
func (m *MatMulNBits[T]) OutputShape() []int {
	return m.outputShape
}

// Forward performs the forward pass: output = input @ dequantized_weights
func (m *MatMulNBits[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MatMulNBits expects exactly 1 input, got %d", len(inputs))
	}
	
	input := inputs[0]
	// Dequantize weights if needed
	weights, err := m.dequantizeWeights()
	if err != nil {
		return nil, fmt.Errorf("failed to dequantize weights: %w", err)
	}
	
	// Validate input dimensions
	inputShape := input.Shape()
	weightsShape := weights.Shape()
	
	if len(inputShape) < 2 {
		return nil, fmt.Errorf("input must be at least 2D, got shape %v", inputShape)
	}
	
	lastDim := inputShape[len(inputShape)-1]
	if lastDim != weightsShape[0] {
		return nil, fmt.Errorf("input last dimension %d must match weights first dimension %d", lastDim, weightsShape[0])
	}
	
	// Compute output shape
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	outputShape[len(outputShape)-1] = weightsShape[1] // Replace last dim with weights output dim
	
	// Perform matrix multiplication using the engine
	result, err := m.engine.MatMul(ctx, input, weights)
	if err != nil {
		return nil, fmt.Errorf("matrix multiplication failed: %w", err)
	}
	
	return result, nil
}

// Backward performs the backward pass for MatMulNBits.
// Computes gradients with respect to input (weights are quantized and typically frozen).
func (m *MatMulNBits[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MatMulNBits expects exactly 1 input for backward pass, got %d", len(inputs))
	}
	
	// Get dequantized weights for gradient computation
	weights, err := m.dequantizeWeights()
	if err != nil {
		return nil, fmt.Errorf("failed to dequantize weights for backward pass: %w", err)
	}
	
	// Gradient w.r.t. input: grad_input = grad_output @ weights.T
	weightsT, err := m.engine.Transpose(ctx, weights, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("failed to transpose weights: %w", err)
	}
	
	gradInput, err := m.engine.MatMul(ctx, outputGradient, weightsT)
	if err != nil {
		return nil, fmt.Errorf("failed to compute input gradient: %w", err)
	}
	
	// Note: We don't compute gradients for quantized weights since they're typically frozen
	// In a full training setup, you might want to compute gradients w.r.t. scale/zero_point
	
	return []*tensor.TensorNumeric[T]{gradInput}, nil
}

// InvalidateCache marks the dequantized weights cache as invalid.
// Call this if quantized weights, scale, or zero point are modified.
func (m *MatMulNBits[T]) InvalidateCache() {
	m.cacheValid = false
	m.dequantizedWeights = nil
}

// GetDequantizedWeights returns the cached dequantized weights for inspection.
// Useful for debugging and testing.
func (m *MatMulNBits[T]) GetDequantizedWeights() (*tensor.TensorNumeric[T], error) {
	return m.dequantizeWeights()
}

// QuantizationInfo returns information about the quantization configuration.
func (m *MatMulNBits[T]) QuantizationInfo() map[string]interface{} {
	info := map[string]interface{}{
		"nbits":      m.nbits,
		"symmetric":  m.symmetric,
		"has_scale":  m.scale != nil,
		"has_zero_point": m.zeroPoint != nil,
	}
	
	if m.scale != nil {
		info["scale_shape"] = m.scale.Shape()
	}
	
	if m.zeroPoint != nil {
		info["zero_point_shape"] = m.zeroPoint.Shape()
	}
	
	if m.quantizedWeights != nil {
		qShape := m.quantizedWeights.Shape()
		info["quantized_shape"] = qShape
		info["actual_weight_shape"] = []int{qShape[0], qShape[1] * 2} // 4-bit unpacking
	}
	
	return info
}

// String returns a string representation of the layer.
func (m *MatMulNBits[T]) String() string {
	if m.quantizedWeights == nil {
		return "MatMulNBits(uninitialized)"
	}
	
	qShape := m.quantizedWeights.Shape()
	actualShape := []int{qShape[0], qShape[1] * 2}
	
	return fmt.Sprintf("MatMulNBits(%dx%d, %d-bit, symmetric=%t)", 
		actualShape[0], actualShape[1], m.nbits, m.symmetric)
}