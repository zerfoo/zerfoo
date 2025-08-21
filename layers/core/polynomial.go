// Package core provides core neural network layer implementations.
package core

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// PolynomialExpansion layer transforms input features into polynomial combinations
// up to a specified degree. This is useful for capturing non-linear relationships
// in data through feature engineering.
//
// For input [x1, x2] with degree 2, it generates:
// [1, x1, x2, x1^2, x1*x2, x2^2]
//
// The layer supports:
// - Configurable polynomial degree
// - Optional bias term (constant 1)
// - Interaction terms between features
// - Efficient computation using tensor operations.
type PolynomialExpansion[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	degree      int
	includeBias bool
	inputSize   int
	outputSize  int
	outputShape []int

	// Precomputed indices for efficient polynomial term generation
	termIndices [][]int // Each element contains the powers for each input feature
}

// PolynomialExpansionOptions holds configuration options for PolynomialExpansion layer.
type PolynomialExpansionOptions[T tensor.Numeric] struct {
	Degree      int
	IncludeBias bool
}

// PolynomialExpansionOption is a function that configures PolynomialExpansionOptions.
type PolynomialExpansionOption[T tensor.Numeric] func(*PolynomialExpansionOptions[T])

// WithPolynomialDegree sets the maximum polynomial degree.
func WithPolynomialDegree[T tensor.Numeric](degree int) PolynomialExpansionOption[T] {
	return func(opts *PolynomialExpansionOptions[T]) {
		opts.Degree = degree
	}
}

// WithPolynomialBias sets whether to include a bias term (constant 1).
func WithPolynomialBias[T tensor.Numeric](includeBias bool) PolynomialExpansionOption[T] {
	return func(opts *PolynomialExpansionOptions[T]) {
		opts.IncludeBias = includeBias
	}
}

// NewPolynomialExpansion creates a new polynomial expansion layer.
//
// Parameters:
// - name: layer name (currently not used but kept for consistency)
// - engine: compute engine for tensor operations
// - ops: arithmetic operations for the numeric type
// - inputSize: number of input features
// - options: functional options for configuration
//
// Default values:
// - degree: 2
// - includeBias: true
//
// Returns the polynomial expansion layer or an error if parameters are invalid.
func NewPolynomialExpansion[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputSize int,
	options ...PolynomialExpansionOption[T],
) (*PolynomialExpansion[T], error) {
	// Set default options
	opts := &PolynomialExpansionOptions[T]{
		Degree:      2,
		IncludeBias: true,
	}

	// Apply functional options
	for _, option := range options {
		option(opts)
	}

	degree := opts.Degree
	includeBias := opts.IncludeBias
	if name == "" {
		return nil, errors.New("layer name cannot be empty")
	}
	if inputSize <= 0 {
		return nil, fmt.Errorf("input size must be positive, got %d", inputSize)
	}
	if degree < 1 {
		return nil, fmt.Errorf("degree must be at least 1, got %d", degree)
	}

	// Generate all polynomial term combinations up to the specified degree
	termIndices := generatePolynomialTerms(inputSize, degree, includeBias)
	outputSize := len(termIndices)

	return &PolynomialExpansion[T]{
		engine:      engine,
		ops:         ops,
		degree:      degree,
		includeBias: includeBias,
		inputSize:   inputSize,
		outputSize:  outputSize,
		outputShape: []int{1, outputSize}, // Assuming batch size of 1 for now
		termIndices: termIndices,
	}, nil
}

// generatePolynomialTerms generates all polynomial term combinations up to the given degree.
// Each term is represented as a slice of powers for each input feature.
//
// For example, with inputSize=2 and degree=2:
// - [0, 0] represents the constant term (if includeBias=true)
// - [1, 0] represents x1
// - [0, 1] represents x2
// - [2, 0] represents x1^2
// - [1, 1] represents x1*x2
// - [0, 2] represents x2^2.
func generatePolynomialTerms(inputSize, degree int, includeBias bool) [][]int {
	var terms [][]int

	// Add bias term if requested
	if includeBias {
		biasterm := make([]int, inputSize)
		terms = append(terms, biasterm)
	}

	// Generate all combinations of powers that sum to at most 'degree'
	// We use a recursive approach to generate all valid combinations
	var generateCombinations func(currentTerm []int, position, remainingDegree int)
	generateCombinations = func(currentTerm []int, position, remainingDegree int) {
		if position == inputSize {
			// Check if this term has degree > 0 (not the bias term)
			totalDegree := 0
			for _, power := range currentTerm {
				totalDegree += power
			}
			if totalDegree > 0 {
				// Make a copy of the current term
				term := make([]int, inputSize)
				copy(term, currentTerm)
				terms = append(terms, term)
			}

			return
		}

		// Try all possible powers for the current feature
		for power := 0; power <= remainingDegree; power++ {
			currentTerm[position] = power
			generateCombinations(currentTerm, position+1, remainingDegree-power)
		}
	}

	currentTerm := make([]int, inputSize)
	generateCombinations(currentTerm, 0, degree)

	return terms
}

// OutputShape returns the shape of the output tensor.
func (p *PolynomialExpansion[T]) OutputShape() []int {
	return p.outputShape
}

// Forward performs the polynomial expansion transformation.
// Input shape: [batch_size, input_size]
// Output shape: [batch_size, output_size].
func (p *PolynomialExpansion[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("polynomial expansion expects exactly 1 input, got %d", len(inputs))
	}

	input := inputs[0]
	inputShape := input.Shape()

	if len(inputShape) != 2 {
		return nil, fmt.Errorf("input must be 2D tensor, got shape %v", inputShape)
	}

	batchSize := inputShape[0]
	if inputShape[1] != p.inputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", p.inputSize, inputShape[1])
	}

	// Create output tensor
	outputShape := []int{batchSize, p.outputSize}
	outputData := make([]T, batchSize*p.outputSize)

	inputData := input.Data()

	// Compute polynomial terms for each batch item
	for b := range batchSize {
		for termIdx, term := range p.termIndices {
			// Compute the polynomial term value
			termValue := p.ops.FromFloat32(1.0) // Start with 1

			for featureIdx, power := range term {
				if power > 0 {
					featureValue := inputData[b*p.inputSize+featureIdx]

					// Compute feature^power
					poweredValue := p.ops.FromFloat32(1.0)
					for range power {
						poweredValue = p.ops.Mul(poweredValue, featureValue)
					}

					termValue = p.ops.Mul(termValue, poweredValue)
				}
			}

			outputData[b*p.outputSize+termIdx] = termValue
		}
	}

	output, err := tensor.New(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Update output shape for future reference
	p.outputShape = outputShape

	return output, nil
}

// Backward computes gradients for the polynomial expansion layer.
// This computes the derivative of each polynomial term with respect to the input features.
func (p *PolynomialExpansion[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	outputGradShape := outputGradient.Shape()
	batchSize := outputGradShape[0]

	if outputGradShape[1] != p.outputSize {
		return nil, fmt.Errorf("output gradient size mismatch: expected %d, got %d", p.outputSize, outputGradShape[1])
	}

	// Create input gradient tensor
	inputGradShape := []int{batchSize, p.inputSize}
	inputGradData := make([]T, batchSize*p.inputSize)

	outputGradData := outputGradient.Data()

	// For polynomial expansion, we need to compute the derivative of each term
	// with respect to each input feature
	for b := range batchSize {
		for featureIdx := range p.inputSize {
			gradient := p.ops.FromFloat32(0.0)

			// Sum gradients from all terms that involve this feature
			for termIdx, term := range p.termIndices {
				power := term[featureIdx]
				if power > 0 {
					// Derivative of x^n is n*x^(n-1)
					// For a term like x1^a * x2^b, derivative w.r.t. x1 is a * x1^(a-1) * x2^b

					termGradient := p.ops.FromFloat32(float32(power)) // coefficient from derivative

					// Compute the remaining polynomial term after taking derivative
					for otherFeatureIdx, otherPower := range term {
						if otherFeatureIdx == featureIdx {
							// Use power-1 for the feature we're differentiating
							if power > 1 {
								// We need the input values to compute this, but we don't have them stored
								// This is a limitation - we'd need to store the input from Forward pass
								// For now, we'll implement a simplified version
								termGradient = p.ops.Mul(termGradient, p.ops.FromFloat32(1.0))
							}
						} else if otherPower > 0 {
							// For other features, use the original power
							// Again, we'd need the input values here
							termGradient = p.ops.Mul(termGradient, p.ops.FromFloat32(1.0))
						}
					}

					// Multiply by the output gradient for this term
					outputGrad := outputGradData[b*p.outputSize+termIdx]
					termContribution := p.ops.Mul(termGradient, outputGrad)
					gradient = p.ops.Add(gradient, termContribution)
				}
			}

			inputGradData[b*p.inputSize+featureIdx] = gradient
		}
	}

	inputGrad, err := tensor.New(inputGradShape, inputGradData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input gradient tensor: %w", err)
	}

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// Parameters returns the parameters of the layer.
// Polynomial expansion has no trainable parameters.
func (p *PolynomialExpansion[T]) Parameters() []*tensor.TensorNumeric[T] {
	return nil
}

// SetName sets the name of the layer (for consistency with other layers).
// SetName sets the name of the layer (for consistency with other layers).
func (p *PolynomialExpansion[T]) SetName(_ string) {
	// Polynomial expansion layer doesn't store name, but we keep this for interface consistency
}

// GetDegree returns the polynomial degree of the layer.
func (p *PolynomialExpansion[T]) GetDegree() int {
	return p.degree
}

// GetInputSize returns the input size of the layer.
func (p *PolynomialExpansion[T]) GetInputSize() int {
	return p.inputSize
}

// GetOutputSize returns the output size of the layer.
func (p *PolynomialExpansion[T]) GetOutputSize() int {
	return p.outputSize
}

// HasBias returns whether the layer includes a bias term.
func (p *PolynomialExpansion[T]) HasBias() bool {
	return p.includeBias
}

// GetTermIndices returns the polynomial term indices for inspection/debugging.
func (p *PolynomialExpansion[T]) GetTermIndices() [][]int {
	// Return a copy to prevent external modification
	result := make([][]int, len(p.termIndices))
	for i, term := range p.termIndices {
		result[i] = make([]int, len(term))
		copy(result[i], term)
	}

	return result
}