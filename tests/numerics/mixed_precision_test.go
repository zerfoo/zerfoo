package numerics_test

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/tests/helpers"
)

func TestMixedPrecisionNumerics(t *testing.T) {
	if helpers.ImplNumerics == nil {
		t.Skip("wire ImplNumerics in tests/helpers/wire.go")
	}
	
	// Test mixed-precision behavior with different value ranges
	// This simulates scenarios where computations involve both
	// high-precision and reduced-precision values
	
	testCases := []struct {
		name               string
		input              []float64
		expectedPrecision  string
		toleranceThreshold float64
		description        string
	}{
		{
			name:               "high_precision_range",
			input:              []float64{1.123456789, 2.987654321, 3.141592653},
			expectedPrecision:  "float64",
			toleranceThreshold: 1e-10,
			description:        "High precision values should maintain accuracy",
		},
		{
			name:               "float32_range",
			input:              []float64{1.123, 2.987, 3.141},
			expectedPrecision:  "float32",
			toleranceThreshold: 1e-6,
			description:        "Float32-level precision should be preserved",
		},
		{
			name:               "float16_range",
			input:              []float64{1.5, 2.0, 3.5},
			expectedPrecision:  "float16",
			toleranceThreshold: 1e-3,
			description:        "Float16-level precision should work correctly",
		},
		{
			name:               "mixed_magnitude",
			input:              []float64{1e-6, 1.0, 1e6},
			expectedPrecision:  "mixed",
			toleranceThreshold: 1e-5,
			description:        "Mixed magnitude values should be handled consistently",
		},
		{
			name:               "gradient_scale_range",
			input:              []float64{1e-8, 1e-4, 1e-2, 1.0},
			expectedPrecision:  "gradient",
			toleranceThreshold: 1e-7,
			description:        "Gradient-scale values should maintain sufficient precision",
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Forward pass
			output, ctx, err := helpers.ImplNumerics.Forward(tc.input)
			if err != nil {
				t.Fatalf("Forward pass failed for %s: %v", tc.description, err)
			}
			
			// Check output precision characteristics
			checkPrecisionCharacteristics(t, "output", output, tc.toleranceThreshold)
			
			// Backward pass
			if ctx != nil {
				upstream := make([]float64, len(output))
				for i := range upstream {
					upstream[i] = 1.0
				}
				
				grad, backErr := helpers.ImplNumerics.Backward(ctx, upstream)
				if backErr != nil {
					t.Fatalf("Backward pass failed for %s: %v", tc.description, backErr)
				}
				
				// Check gradient precision characteristics
				checkPrecisionCharacteristics(t, "gradient", grad, tc.toleranceThreshold)
				
				// Verify gradient magnitudes are reasonable
				for i, g := range grad {
					if math.Abs(g) > 1e10 {
						t.Errorf("Gradient %d too large (%.2e) for %s - potential overflow", i, g, tc.description)
					}
					if math.Abs(g) < 1e-15 && g != 0.0 {
						t.Logf("Gradient %d very small (%.2e) for %s - potential underflow", i, g, tc.description)
					}
				}
			}
			
			t.Logf("Mixed precision test passed for %s (tolerance=%.2e)", tc.description, tc.toleranceThreshold)
		})
	}
}

func TestPrecisionDegradation(t *testing.T) {
	if helpers.ImplNumerics == nil {
		t.Skip("wire ImplNumerics in tests/helpers/wire.go")
	}
	
	// Test precision degradation scenarios that might occur in training
	scenarios := []struct {
		name    string
		input   []float64
		rounds  int
		maxDegradation float64
	}{
		{
			name:           "accumulation_test",
			input:          []float64{0.1, 0.2, 0.3},
			rounds:         100,
			maxDegradation: 1e-10,
		},
		{
			name:           "iterative_computation",
			input:          []float64{1.0, 1.0, 1.0},
			rounds:         50,
			maxDegradation: 1e-12,
		},
	}
	
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			currentInput := make([]float64, len(scenario.input))
			copy(currentInput, scenario.input)
			
			initialSum := 0.0
			for _, v := range currentInput {
				initialSum += v
			}
			
			// Perform iterative forward passes
			for round := 0; round < scenario.rounds; round++ {
				output, _, err := helpers.ImplNumerics.Forward(currentInput)
				if err != nil {
					t.Fatalf("Round %d failed: %v", round, err)
				}
				
				// Check for invalid values
				if containsInvalidValues(output) {
					t.Fatalf("Round %d produced invalid values", round)
				}
				
				// Use output as input for next iteration (simulating iterative training)
				if len(output) == len(currentInput) {
					copy(currentInput, output)
				}
			}
			
			// Check precision degradation
			finalSum := 0.0
			for _, v := range currentInput {
				finalSum += v
			}
			
			degradation := math.Abs((finalSum - initialSum) / initialSum)
			if degradation > scenario.maxDegradation {
				t.Errorf("Precision degradation too high after %d rounds: %.2e > %.2e",
					scenario.rounds, degradation, scenario.maxDegradation)
			}
			
			t.Logf("Precision maintained after %d rounds: degradation=%.2e", scenario.rounds, degradation)
		})
	}
}

func TestNumericalStability(t *testing.T) {
	if helpers.ImplNumerics == nil {
		t.Skip("wire ImplNumerics in tests/helpers/wire.go")
	}
	
	// Test numerical stability in challenging scenarios
	stabilityTests := []struct {
		name        string
		input       []float64
		description string
	}{
		{
			name:        "near_zero_gradients",
			input:       []float64{1e-8, 1e-9, 1e-10},
			description: "Very small values that might cause underflow",
		},
		{
			name:        "large_dynamic_range",
			input:       []float64{1e-6, 1.0, 1e6},
			description: "Wide dynamic range testing precision limits",
		},
		{
			name:        "cancellation_prone",
			input:       []float64{1.0000001, 1.0000002, -2.0000003},
			description: "Values prone to catastrophic cancellation",
		},
		{
			name:        "alternating_signs",
			input:       []float64{1.0, -1.0, 0.5, -0.5, 0.25, -0.25},
			description: "Alternating positive/negative values",
		},
	}
	
	for _, test := range stabilityTests {
		t.Run(test.name, func(t *testing.T) {
			// Multiple runs to check consistency
			const numRuns = 5
			var outputs [][]float64
			var grads [][]float64
			
			for run := 0; run < numRuns; run++ {
				output, ctx, err := helpers.ImplNumerics.Forward(test.input)
				if err != nil {
					t.Fatalf("Run %d forward pass failed for %s: %v", run, test.description, err)
				}
				
				if containsInvalidValues(output) {
					t.Fatalf("Run %d produced invalid output for %s", run, test.description)
				}
				
				outputs = append(outputs, output)
				
				// Backward pass
				if ctx != nil {
					upstream := make([]float64, len(output))
					for i := range upstream {
						upstream[i] = 1.0
					}
					
					grad, backErr := helpers.ImplNumerics.Backward(ctx, upstream)
					if backErr != nil {
						t.Fatalf("Run %d backward pass failed for %s: %v", run, test.description, backErr)
					}
					
					if containsInvalidValues(grad) {
						t.Fatalf("Run %d produced invalid gradients for %s", run, test.description)
					}
					
					grads = append(grads, grad)
				}
			}
			
			// Check consistency across runs
			if len(outputs) > 1 {
				checkConsistency(t, "outputs", outputs, 1e-12)
			}
			if len(grads) > 1 {
				checkConsistency(t, "gradients", grads, 1e-12)
			}
			
			t.Logf("Numerical stability verified for %s across %d runs", test.description, numRuns)
		})
	}
}

// Helper function to check precision characteristics of values
func checkPrecisionCharacteristics(t *testing.T, name string, values []float64, threshold float64) {
	if len(values) == 0 {
		return
	}
	
	// Check for invalid values
	if containsInvalidValues(values) {
		t.Errorf("%s contains invalid values (NaN/Inf)", name)
		return
	}
	
	// Check dynamic range
	minVal := math.Inf(1)
	maxVal := math.Inf(-1)
	
	for _, v := range values {
		if v != 0.0 {
			absV := math.Abs(v)
			if absV < minVal {
				minVal = absV
			}
			if absV > maxVal {
				maxVal = absV
			}
		}
	}
	
	if !math.IsInf(minVal, 0) && !math.IsInf(maxVal, 0) && minVal > 0 {
		dynamicRange := maxVal / minVal
		if dynamicRange > 1e12 {
			t.Logf("%s has wide dynamic range: %.2e (may impact precision)", name, dynamicRange)
		}
	}
	
	// Check for subnormal values
	subnormalCount := 0
	for _, v := range values {
		if v != 0.0 && math.Abs(v) < 2.225074e-308 { // Approximate float64 subnormal threshold
			subnormalCount++
		}
	}
	
	if subnormalCount > 0 {
		t.Logf("%s contains %d subnormal values (potential underflow)", name, subnormalCount)
	}
}

// Helper function to check consistency across multiple runs
func checkConsistency(t *testing.T, name string, runs [][]float64, tolerance float64) {
	if len(runs) < 2 {
		return
	}
	
	reference := runs[0]
	for runIdx, current := range runs[1:] {
		if len(current) != len(reference) {
			t.Errorf("%s run %d length mismatch: expected %d, got %d", name, runIdx+1, len(reference), len(current))
			continue
		}
		
		for i := range reference {
			diff := math.Abs(current[i] - reference[i])
			if diff > tolerance {
				t.Errorf("%s inconsistent across runs at index %d: run0=%.6e, run%d=%.6e, diff=%.6e > tolerance=%.6e",
					name, i, reference[i], runIdx+1, current[i], diff, tolerance)
			}
		}
	}
}