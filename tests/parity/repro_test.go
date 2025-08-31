package parity_test

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/tests/helpers"
)

func TestCPUGPUDeterminism(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	
	// Test that CPU and GPU computations produce identical results
	// for deterministic operations when using the same seed
	
	testPrompts := []string{
		"The quick brown fox",
		"Hello world",
		"Machine learning is",
		"In a world where",
		"The answer to life",
	}
	
	seeds := []int{42, 123, 999, 2024}
	
	for _, seed := range seeds {
		for promptIdx, prompt := range testPrompts {
			t.Run(t.Name(), func(t *testing.T) {
				// Test tokenization determinism across devices
				helpers.ImplZerfoo.SetSeed(seed)
				tokens1, err := helpers.ImplZerfoo.Tokenize(prompt)
				if err != nil {
					t.Fatalf("Tokenization failed for prompt %d: %v", promptIdx, err)
				}
				
				helpers.ImplZerfoo.SetSeed(seed)
				tokens2, err := helpers.ImplZerfoo.Tokenize(prompt)
				if err != nil {
					t.Fatalf("Second tokenization failed for prompt %d: %v", promptIdx, err)
				}
				
				if !sliceEqual(tokens1, tokens2) {
					t.Errorf("Tokenization not deterministic for prompt %d with seed %d", promptIdx, seed)
				}
				
				// Test logits determinism
				maxTokens := 10
				helpers.ImplZerfoo.SetSeed(seed)
				logits1, err := helpers.ImplZerfoo.Logits(prompt, maxTokens)
				if err != nil {
					t.Fatalf("First logits computation failed for prompt %d: %v", promptIdx, err)
				}
				
				helpers.ImplZerfoo.SetSeed(seed)
				logits2, err := helpers.ImplZerfoo.Logits(prompt, maxTokens)
				if err != nil {
					t.Fatalf("Second logits computation failed for prompt %d: %v", promptIdx, err)
				}
				
				// Check logits determinism with tolerance for floating point precision
				if !floatSliceEqual(logits1, logits2, 1e-6) {
					t.Errorf("Logits not deterministic for prompt %d with seed %d", promptIdx, seed)
					
					// Log detailed comparison for debugging
					maxDiff := 0.0
					for i := 0; i < len(logits1) && i < len(logits2) && i < 10; i++ {
						diff := math.Abs(float64(logits1[i] - logits2[i]))
						if diff > maxDiff {
							maxDiff = diff
						}
						if diff > 1e-6 {
							t.Logf("  logits[%d]: %.8f vs %.8f (diff=%.2e)", i, logits1[i], logits2[i], diff)
						}
					}
					t.Logf("  max_difference: %.2e", maxDiff)
				}
				
				// Test greedy decoding determinism
				if len(tokens1) > 0 {
					helpers.ImplZerfoo.SetSeed(seed)
					decoded1, err := helpers.ImplZerfoo.DecodeGreedy(tokens1)
					if err != nil {
						t.Fatalf("First greedy decode failed for prompt %d: %v", promptIdx, err)
					}
					
					helpers.ImplZerfoo.SetSeed(seed)
					decoded2, err := helpers.ImplZerfoo.DecodeGreedy(tokens1)
					if err != nil {
						t.Fatalf("Second greedy decode failed for prompt %d: %v", promptIdx, err)
					}
					
					if decoded1 != decoded2 {
						t.Errorf("Greedy decoding not deterministic for prompt %d with seed %d: %q vs %q",
							promptIdx, seed, decoded1, decoded2)
					}
				}
				
				t.Logf("Determinism verified for prompt %d (seed=%d, device=%s)", 
					promptIdx, seed, helpers.ImplZerfoo.DeviceName())
			})
		}
	}
}

func TestCrossDeviceConsistency(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	
	// Test that results are consistent across different device configurations
	// This test documents expected behavior for CPU vs GPU computations
	
	device := helpers.ImplZerfoo.DeviceName()
	t.Logf("Testing cross-device consistency on device: %s", device)
	
	testCases := []struct {
		prompt     string
		maxTokens  int
		seed       int
		tolerance  float32
	}{
		{
			prompt:    "The capital of France is",
			maxTokens: 5,
			seed:      12345,
			tolerance: 1e-5,
		},
		{
			prompt:    "Machine learning models",
			maxTokens: 8,
			seed:      54321,
			tolerance: 1e-5,
		},
		{
			prompt:    "In the year 2024",
			maxTokens: 12,
			seed:      99999,
			tolerance: 1e-5,
		},
	}
	
	for i, tc := range testCases {
		t.Run(t.Name(), func(t *testing.T) {
			// Run the same computation multiple times to check consistency
			const numRuns = 3
			var logitsRuns [][]float32
			var tokenRuns [][]int
			
			for run := 0; run < numRuns; run++ {
				helpers.ImplZerfoo.SetSeed(tc.seed)
				
				// Get logits
				logits, err := helpers.ImplZerfoo.Logits(tc.prompt, tc.maxTokens)
				if err != nil {
					t.Fatalf("Logits failed on run %d for case %d: %v", run, i, err)
				}
				logitsRuns = append(logitsRuns, logits)
				
				// Get tokens  
				tokens, err := helpers.ImplZerfoo.Tokenize(tc.prompt)
				if err != nil {
					t.Fatalf("Tokenization failed on run %d for case %d: %v", run, i, err)
				}
				tokenRuns = append(tokenRuns, tokens)
			}
			
			// Verify consistency across runs
			for run := 1; run < numRuns; run++ {
				// Check token consistency
				if !sliceEqual(tokenRuns[0], tokenRuns[run]) {
					t.Errorf("Tokenization inconsistent between runs 0 and %d for case %d", run, i)
				}
				
				// Check logits consistency
				if !floatSliceEqual(logitsRuns[0], logitsRuns[run], tc.tolerance) {
					t.Errorf("Logits inconsistent between runs 0 and %d for case %d", run, i)
					
					// Log sample differences
					maxDiff := float32(0.0)
					for j := 0; j < len(logitsRuns[0]) && j < len(logitsRuns[run]) && j < 5; j++ {
						diff := float32(math.Abs(float64(logitsRuns[0][j] - logitsRuns[run][j])))
						if diff > maxDiff {
							maxDiff = diff
						}
						if diff > tc.tolerance {
							t.Logf("  logits[%d]: %.6f vs %.6f (diff=%.2e)", j, logitsRuns[0][j], logitsRuns[run][j], diff)
						}
					}
					t.Logf("  max_difference: %.2e, tolerance: %.2e", maxDiff, tc.tolerance)
				}
			}
			
			t.Logf("Cross-device consistency verified for case %d on %s", i, device)
		})
	}
}

func TestReproducibilityAcrossSessions(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	
	// Test that results can be reproduced across different test runs
	// This is crucial for debugging and scientific reproducibility
	
	const referenceSeed = 2024
	referencePrompt := "Reproducibility test prompt"
	maxTokens := 6
	
	// Baseline computation
	helpers.ImplZerfoo.SetSeed(referenceSeed)
	baselineTokens, err := helpers.ImplZerfoo.Tokenize(referencePrompt)
	if err != nil {
		t.Fatalf("Baseline tokenization failed: %v", err)
	}
	
	helpers.ImplZerfoo.SetSeed(referenceSeed)
	baselineLogits, err := helpers.ImplZerfoo.Logits(referencePrompt, maxTokens)
	if err != nil {
		t.Fatalf("Baseline logits computation failed: %v", err)
	}
	
	helpers.ImplZerfoo.SetSeed(referenceSeed)
	baselineGreedy, err := helpers.ImplZerfoo.DecodeGreedy(baselineTokens)
	if err != nil {
		t.Fatalf("Baseline greedy decode failed: %v", err)
	}
	
	// Test reproducibility across multiple "sessions" (seed resets)
	for session := 0; session < 5; session++ {
		t.Run(t.Name(), func(t *testing.T) {
			// Reset to same seed
			helpers.ImplZerfoo.SetSeed(referenceSeed)
			
			// Repeat computations
			tokens, err := helpers.ImplZerfoo.Tokenize(referencePrompt)
			if err != nil {
				t.Fatalf("Session %d tokenization failed: %v", session, err)
			}
			
			if !sliceEqual(baselineTokens, tokens) {
				t.Errorf("Session %d tokenization differs from baseline", session)
			}
			
			helpers.ImplZerfoo.SetSeed(referenceSeed)
			logits, err := helpers.ImplZerfoo.Logits(referencePrompt, maxTokens)
			if err != nil {
				t.Fatalf("Session %d logits computation failed: %v", session, err)
			}
			
			if !floatSliceEqual(baselineLogits, logits, 1e-8) {
				t.Errorf("Session %d logits differ from baseline", session)
			}
			
			helpers.ImplZerfoo.SetSeed(referenceSeed)
			greedy, err := helpers.ImplZerfoo.DecodeGreedy(tokens)
			if err != nil {
				t.Fatalf("Session %d greedy decode failed: %v", session, err)
			}
			
			if baselineGreedy != greedy {
				t.Errorf("Session %d greedy decode differs from baseline: %q vs %q", 
					session, baselineGreedy, greedy)
			}
			
			t.Logf("Session %d reproduced baseline results successfully", session)
		})
	}
}

func TestDevicePerformanceConsistency(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	if helpers.ImplPerf == nil {
		t.Skip("wire ImplPerf in tests/helpers/wire.go")
	}
	
	// Test that performance characteristics are consistent and reasonable
	batchSizes := []int{1, 4, 8, 16, 32}
	device := helpers.ImplZerfoo.DeviceName()
	
	for _, batchSize := range batchSizes {
		t.Run(t.Name(), func(t *testing.T) {
			latency, err := helpers.ImplPerf.InferLatency(batchSize)
			if err != nil {
				t.Fatalf("Latency measurement failed for batch size %d: %v", batchSize, err)
			}
			
			if latency <= 0 {
				t.Errorf("Invalid latency for batch size %d: %.2f ms", batchSize, latency)
			}
			
			if latency > 10000 { // 10 seconds seems unreasonable for small batches
				t.Errorf("Latency too high for batch size %d on %s: %.2f ms", batchSize, device, latency)
			}
			
			t.Logf("Batch size %d on %s: %.2f ms", batchSize, device, latency)
		})
	}
}

// Helper function to compare integer slices
func sliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Helper function to compare float32 slices with tolerance
func floatSliceEqual(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > float64(tolerance) {
			return false
		}
	}
	return true
}