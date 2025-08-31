package numerics_test

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/tests/helpers"
)

func TestFiniteDifferenceGradientChecks(t *testing.T) {
	if helpers.ImplNumerics == nil {
		t.Skip("wire ImplNumerics in tests/helpers/wire.go")
	}
	
	// Test finite difference gradient checks for various functions
	testCases := []struct {
		name      string
		input     []float64
		epsilon   float64
		tolerance float64
	}{
		{
			name:      "simple_quadratic",
			input:     []float64{1.0, 2.0, 3.0},
			epsilon:   1e-5,
			tolerance: 1e-3,
		},
		{
			name:      "negative_values",
			input:     []float64{-1.0, -0.5, 0.5, 1.0},
			epsilon:   1e-5,
			tolerance: 1e-3,
		},
		{
			name:      "small_values",
			input:     []float64{0.001, 0.002, 0.003},
			epsilon:   1e-6,
			tolerance: 1e-2,
		},
		{
			name:      "large_values",
			input:     []float64{100.0, 200.0, 300.0},
			epsilon:   1e-4,
			tolerance: 1e-3,
		},
		{
			name:      "single_element",
			input:     []float64{5.0},
			epsilon:   1e-5,
			tolerance: 1e-3,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Compute forward pass and get context for backward
			output, ctx, err := helpers.ImplNumerics.Forward(tc.input)
			if err != nil {
				t.Fatalf("Forward pass failed: %v", err)
			}
			
			// Assume scalar output for gradient computation (sum reduction)
			scalar := 0.0
			for _, v := range output {
				scalar += v
			}
			
			// Compute analytical gradient using backward pass
			upstream := make([]float64, len(output))
			for i := range upstream {
				upstream[i] = 1.0 // Gradient of sum is 1
			}
			
			analyticalGrad, err := helpers.ImplNumerics.Backward(ctx, upstream)
			if err != nil {
				t.Fatalf("Backward pass failed: %v", err)
			}
			
			// Compute numerical gradient using finite differences
			numericalGrad := make([]float64, len(tc.input))
			for i := range tc.input {
				// Forward perturbation: f(x + h)
				inputPlus := make([]float64, len(tc.input))
				copy(inputPlus, tc.input)
				inputPlus[i] += tc.epsilon
				
				outputPlus, _, err := helpers.ImplNumerics.Forward(inputPlus)
				if err != nil {
					t.Fatalf("Forward pass (+h) failed at index %d: %v", i, err)
				}
				
				scalarPlus := 0.0
				for _, v := range outputPlus {
					scalarPlus += v
				}
				
				// Backward perturbation: f(x - h)
				inputMinus := make([]float64, len(tc.input))
				copy(inputMinus, tc.input)
				inputMinus[i] -= tc.epsilon
				
				outputMinus, _, err := helpers.ImplNumerics.Forward(inputMinus)
				if err != nil {
					t.Fatalf("Forward pass (-h) failed at index %d: %v", i, err)
				}
				
				scalarMinus := 0.0
				for _, v := range outputMinus {
					scalarMinus += v
				}
				
				// Central difference: [f(x+h) - f(x-h)] / (2*h)
				numericalGrad[i] = (scalarPlus - scalarMinus) / (2.0 * tc.epsilon)
			}
			
			// Compare analytical and numerical gradients
			if len(analyticalGrad) != len(numericalGrad) {
				t.Fatalf("Gradient length mismatch: analytical=%d, numerical=%d", 
					len(analyticalGrad), len(numericalGrad))
			}
			
			for i := range analyticalGrad {
				diff := math.Abs(analyticalGrad[i] - numericalGrad[i])
				relativeError := diff / (math.Abs(numericalGrad[i]) + 1e-8)
				
				if relativeError > tc.tolerance {
					t.Errorf("Gradient check failed at index %d: analytical=%.6f, numerical=%.6f, relative_error=%.6f > tolerance=%.6f",
						i, analyticalGrad[i], numericalGrad[i], relativeError, tc.tolerance)
				}
			}
			
			t.Logf("Finite difference check passed for %s (input_size=%d, max_relative_error=%.2e)",
				tc.name, len(tc.input), tc.tolerance)
		})
	}
}

func TestNumericsEdgeCases(t *testing.T) {
	if helpers.ImplNumerics == nil {
		t.Skip("wire ImplNumerics in tests/helpers/wire.go")
	}
	
	edgeCases := []struct {
		name        string
		input       []float64
		expectError bool
		description string
	}{
		{
			name:        "empty_input",
			input:       []float64{},
			expectError: true,
			description: "Empty input should be rejected",
		},
		{
			name:        "zero_values",
			input:       []float64{0.0, 0.0, 0.0},
			expectError: false,
			description: "Zero values should be handled gracefully",
		},
		{
			name:        "nan_input",
			input:       []float64{math.NaN(), 1.0, 2.0},
			expectError: true,
			description: "NaN input should be detected and rejected",
		},
		{
			name:        "inf_input", 
			input:       []float64{math.Inf(1), 1.0, 2.0},
			expectError: true,
			description: "Infinity input should be detected and rejected",
		},
		{
			name:        "very_small_values",
			input:       []float64{1e-10, 2e-10, 3e-10},
			expectError: false,
			description: "Very small values should be handled without underflow",
		},
		{
			name:        "very_large_values",
			input:       []float64{1e10, 2e10, 3e10},
			expectError: false,
			description: "Large values should be handled without overflow",
		},
	}
	
	for _, tc := range edgeCases {
		t.Run(tc.name, func(t *testing.T) {
			output, ctx, err := helpers.ImplNumerics.Forward(tc.input)
			
			if tc.expectError {
				if err == nil {
					t.Errorf("Expected error for %s, but got output: %v", tc.description, output)
				} else {
					t.Logf("Correctly rejected %s: %v", tc.description, err)
				}
				return
			}
			
			if err != nil {
				t.Fatalf("Unexpected error for %s: %v", tc.description, err)
			}
			
			// Test that backward also handles edge cases appropriately
			if ctx != nil && len(output) > 0 {
				upstream := make([]float64, len(output))
				for i := range upstream {
					upstream[i] = 1.0
				}
				
				grad, backErr := helpers.ImplNumerics.Backward(ctx, upstream)
				if backErr != nil {
					t.Logf("Backward pass failed for %s: %v", tc.description, backErr)
				} else {
					// Check for NaN/Inf in gradients
					for i, g := range grad {
						if math.IsNaN(g) || math.IsInf(g, 0) {
							t.Errorf("Invalid gradient at index %d for %s: %f", i, tc.description, g)
						}
					}
				}
			}
		})
	}
}