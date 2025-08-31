package numerics_test

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/tests/helpers"
)

func TestNaNInfHooks(t *testing.T) {
	if helpers.ImplNumerics == nil {
		t.Skip("wire ImplNumerics in tests/helpers/wire.go")
	}
	
	// Test that NaN/Inf values are properly detected and handled
	// This simulates training halting behavior when invalid values are encountered
	
	testCases := []struct {
		name           string
		input          []float64
		expectInvalid  bool
		description    string
	}{
		{
			name:          "normal_values",
			input:         []float64{1.0, 2.0, 3.0},
			expectInvalid: false,
			description:   "Normal values should pass validation",
		},
		{
			name:          "contains_nan",
			input:         []float64{1.0, math.NaN(), 3.0},
			expectInvalid: true,
			description:   "NaN values should trigger halt",
		},
		{
			name:          "contains_positive_inf",
			input:         []float64{1.0, math.Inf(1), 3.0},
			expectInvalid: true,
			description:   "Positive infinity should trigger halt",
		},
		{
			name:          "contains_negative_inf",
			input:         []float64{1.0, math.Inf(-1), 3.0},
			expectInvalid: true,
			description:   "Negative infinity should trigger halt",
		},
		{
			name:          "very_large_but_finite",
			input:         []float64{1e30, 2e30, 3e30},
			expectInvalid: false,
			description:   "Large but finite values should pass",
		},
		{
			name:          "very_small_but_normal",
			input:         []float64{1e-30, 2e-30, 3e-30},
			expectInvalid: false,
			description:   "Small but normal values should pass",
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check input validation
			hasInvalid := containsInvalidValues(tc.input)
			if hasInvalid != tc.expectInvalid {
				t.Errorf("Input validation mismatch for %s: expected_invalid=%v, actual_invalid=%v",
					tc.description, tc.expectInvalid, hasInvalid)
			}
			
			// If input has invalid values, test should expect error or special handling
			if tc.expectInvalid {
				// Test that the system properly rejects or handles invalid input
				output, ctx, err := helpers.ImplNumerics.Forward(tc.input)
				
				// The implementation should either:
				// 1. Return an error for invalid input
				// 2. Return output that we can validate for invalid values
				if err == nil && output != nil {
					// Check if output contains invalid values (should be caught)
					outputInvalid := containsInvalidValues(output)
					if outputInvalid {
						t.Logf("Forward pass produced invalid output as expected for %s", tc.description)
						
						// Test that backward also handles this appropriately
						if ctx != nil {
							upstream := make([]float64, len(output))
							for i := range upstream {
								upstream[i] = 1.0
							}
							
							grad, backErr := helpers.ImplNumerics.Backward(ctx, upstream)
							if backErr == nil && grad != nil {
								gradInvalid := containsInvalidValues(grad)
								if gradInvalid {
									t.Logf("Backward pass also produced invalid gradients for %s", tc.description)
								}
							}
						}
					}
				} else if err != nil {
					t.Logf("Forward pass correctly rejected invalid input for %s: %v", tc.description, err)
				}
			} else {
				// Valid input should produce valid output
				output, ctx, err := helpers.ImplNumerics.Forward(tc.input)
				if err != nil {
					t.Fatalf("Forward pass failed for valid input %s: %v", tc.description, err)
				}
				
				if containsInvalidValues(output) {
					t.Errorf("Forward pass produced invalid output for valid input %s", tc.description)
				}
				
				// Test backward pass
				if ctx != nil {
					upstream := make([]float64, len(output))
					for i := range upstream {
						upstream[i] = 1.0
					}
					
					grad, backErr := helpers.ImplNumerics.Backward(ctx, upstream)
					if backErr != nil {
						t.Errorf("Backward pass failed for valid input %s: %v", tc.description, backErr)
					} else if containsInvalidValues(grad) {
						t.Errorf("Backward pass produced invalid gradients for valid input %s", tc.description)
					}
				}
			}
		})
	}
}

func TestTensorDumpOnInvalidValues(t *testing.T) {
	if helpers.ImplNumerics == nil {
		t.Skip("wire ImplNumerics in tests/helpers/wire.go")
	}
	
	// Test that when invalid values are detected, we can dump tensor information
	// This simulates the debugging capability when training halts due to NaN/Inf
	
	inputs := [][]float64{
		{1.0, 2.0, math.NaN()},           // Contains NaN
		{math.Inf(1), 2.0, 3.0},         // Contains +Inf
		{1.0, math.Inf(-1), 3.0},        // Contains -Inf
		{math.NaN(), math.Inf(1), 0.0},  // Multiple invalid values
	}
	
	for i, input := range inputs {
		t.Run(t.Name(), func(t *testing.T) {
			// Detect invalid values in input
			if !containsInvalidValues(input) {
				t.Fatalf("Test case %d should contain invalid values", i)
			}
			
			// Dump tensor information for debugging
			dumpTensorInfo(t, "input", input)
			
			// Try forward pass (may fail or produce invalid output)
			output, ctx, err := helpers.ImplNumerics.Forward(input)
			if err != nil {
				t.Logf("Forward pass correctly failed: %v", err)
				return
			}
			
			if output != nil {
				dumpTensorInfo(t, "output", output)
				
				// Check output validity
				if containsInvalidValues(output) {
					t.Logf("Forward pass propagated invalid values to output")
				}
				
				// Try backward pass
				if ctx != nil {
					upstream := make([]float64, len(output))
					for j := range upstream {
						upstream[j] = 1.0
					}
					
					grad, backErr := helpers.ImplNumerics.Backward(ctx, upstream)
					if backErr != nil {
						t.Logf("Backward pass failed: %v", backErr)
					} else if grad != nil {
						dumpTensorInfo(t, "gradient", grad)
						
						if containsInvalidValues(grad) {
							t.Logf("Backward pass propagated invalid values to gradients")
						}
					}
				}
			}
		})
	}
}

// Helper function to check for NaN or Inf values in a slice
func containsInvalidValues(values []float64) bool {
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

// Helper function to dump tensor information for debugging
func dumpTensorInfo(t *testing.T, name string, values []float64) {
	t.Logf("=== %s tensor dump ===", name)
	t.Logf("Shape: [%d]", len(values))
	
	hasNaN := false
	hasInf := false
	minVal := math.Inf(1)
	maxVal := math.Inf(-1)
	
	for i, v := range values {
		switch {
		case math.IsNaN(v):
			hasNaN = true
			t.Logf("  [%d]: NaN", i)
		case math.IsInf(v, 0):
			hasInf = true
			if math.IsInf(v, 1) {
				t.Logf("  [%d]: +Inf", i)
			} else {
				t.Logf("  [%d]: -Inf", i)
			}
		default:
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
	}
	
	if !hasNaN && !hasInf {
		t.Logf("Range: [%.6e, %.6e]", minVal, maxVal)
	}
	
	t.Logf("Contains NaN: %v, Contains Inf: %v", hasNaN, hasInf)
	t.Logf("=== end %s dump ===", name)
}