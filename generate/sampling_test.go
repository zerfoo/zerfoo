package generate

import (
	"math"
	"testing"
)

func TestArgmax(t *testing.T) {
	tests := []struct {
		name   string
		logits []float64
		want   int
	}{
		{"single", []float64{5.0}, 0},
		{"first highest", []float64{10.0, 5.0, 3.0}, 0},
		{"middle highest", []float64{1.0, 9.0, 2.0}, 1},
		{"last highest", []float64{1.0, 2.0, 3.0}, 2},
		{"negative values", []float64{-3.0, -1.0, -2.0}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := argmax(tc.logits)
			if got != tc.want {
				t.Errorf("argmax(%v) = %d, want %d", tc.logits, got, tc.want)
			}
		})
	}
}

func TestApplyTemperature(t *testing.T) {
	tests := []struct {
		name   string
		logits []float64
		temp   float64
		want   []float64
	}{
		{"divide by 2", []float64{4.0, 6.0, 8.0}, 2.0, []float64{2.0, 3.0, 4.0}},
		{"divide by 0.5", []float64{2.0, 4.0}, 0.5, []float64{4.0, 8.0}},
		{"temp 1.0 no change", []float64{1.0, 2.0}, 1.0, []float64{1.0, 2.0}},
		{"zero temp is no-op", []float64{1.0, 2.0}, 0, []float64{1.0, 2.0}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logits := make([]float64, len(tc.logits))
			copy(logits, tc.logits)
			applyTemperature(logits, tc.temp)
			for i, v := range logits {
				if math.Abs(v-tc.want[i]) > 1e-9 {
					t.Errorf("logits[%d] = %f, want %f", i, v, tc.want[i])
				}
			}
		})
	}
}

func TestApplyTopK(t *testing.T) {
	t.Run("keeps top 2", func(t *testing.T) {
		logits := []float64{1.0, 5.0, 3.0, 7.0, 2.0}
		applyTopK(logits, 2)
		// Top-2: indices 3 (7.0) and 1 (5.0).
		if logits[3] != 7.0 {
			t.Errorf("logits[3] = %f, want 7.0", logits[3])
		}
		if logits[1] != 5.0 {
			t.Errorf("logits[1] = %f, want 5.0", logits[1])
		}
		// Others should be -Inf.
		for _, i := range []int{0, 2, 4} {
			if !math.IsInf(logits[i], -1) {
				t.Errorf("logits[%d] = %f, want -Inf", i, logits[i])
			}
		}
	})

	t.Run("k >= len is no-op", func(t *testing.T) {
		logits := []float64{1.0, 2.0, 3.0}
		applyTopK(logits, 5)
		if logits[0] != 1.0 || logits[1] != 2.0 || logits[2] != 3.0 {
			t.Errorf("logits should be unchanged: %v", logits)
		}
	})

	t.Run("k=0 is no-op", func(t *testing.T) {
		logits := []float64{1.0, 2.0}
		applyTopK(logits, 0)
		if logits[0] != 1.0 || logits[1] != 2.0 {
			t.Errorf("logits should be unchanged: %v", logits)
		}
	})
}

func TestApplyTopP(t *testing.T) {
	t.Run("keeps high probability tokens", func(t *testing.T) {
		// Logits that produce a clear distribution after softmax.
		logits := []float64{10.0, 1.0, 0.0, -1.0}
		applyTopP(logits, 0.9)
		// Token 0 has very high probability (softmax of 10 vs 1,0,-1).
		// It should be kept; at least some of the low-probability tokens zeroed.
		if math.IsInf(logits[0], -1) {
			t.Error("highest probability token should not be masked")
		}
	})

	t.Run("p=1.0 is no-op", func(t *testing.T) {
		logits := []float64{1.0, 2.0, 3.0}
		orig := make([]float64, len(logits))
		copy(orig, logits)
		applyTopP(logits, 1.0)
		for i, v := range logits {
			if v != orig[i] {
				t.Errorf("logits[%d] = %f, want %f", i, v, orig[i])
			}
		}
	})

	t.Run("p=0 is no-op", func(t *testing.T) {
		logits := []float64{1.0, 2.0}
		orig := make([]float64, len(logits))
		copy(orig, logits)
		applyTopP(logits, 0)
		for i, v := range logits {
			if v != orig[i] {
				t.Errorf("logits[%d] = %f, want %f", i, v, orig[i])
			}
		}
	})
}

func TestApplyRepetitionPenalty(t *testing.T) {
	t.Run("positive logit divided by penalty", func(t *testing.T) {
		logits := []float64{4.0, 2.0, 6.0}
		applyRepetitionPenalty(logits, []int{0, 2}, 2.0)
		if logits[0] != 2.0 {
			t.Errorf("logits[0] = %f, want 2.0", logits[0])
		}
		if logits[1] != 2.0 {
			t.Errorf("logits[1] = %f, want 2.0 (unpenalized)", logits[1])
		}
		if logits[2] != 3.0 {
			t.Errorf("logits[2] = %f, want 3.0", logits[2])
		}
	})

	t.Run("negative logit multiplied by penalty", func(t *testing.T) {
		logits := []float64{-4.0, 2.0}
		applyRepetitionPenalty(logits, []int{0}, 2.0)
		if logits[0] != -8.0 {
			t.Errorf("logits[0] = %f, want -8.0", logits[0])
		}
	})

	t.Run("penalty 1.0 is no-op", func(t *testing.T) {
		logits := []float64{1.0, 2.0}
		applyRepetitionPenalty(logits, []int{0}, 1.0)
		if logits[0] != 1.0 {
			t.Errorf("logits[0] = %f, want 1.0", logits[0])
		}
	})

	t.Run("out of range token ignored", func(t *testing.T) {
		logits := []float64{1.0, 2.0}
		applyRepetitionPenalty(logits, []int{-1, 5}, 2.0)
		if logits[0] != 1.0 || logits[1] != 2.0 {
			t.Errorf("logits should be unchanged: %v", logits)
		}
	})
}

func TestSoftmax(t *testing.T) {
	t.Run("uniform", func(t *testing.T) {
		probs := softmax([]float64{0.0, 0.0, 0.0})
		for i, p := range probs {
			if math.Abs(p-1.0/3.0) > 1e-6 {
				t.Errorf("probs[%d] = %f, want ~0.333", i, p)
			}
		}
	})

	t.Run("sums to 1", func(t *testing.T) {
		probs := softmax([]float64{1.0, 2.0, 3.0})
		sum := 0.0
		for _, p := range probs {
			sum += p
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("sum = %f, want 1.0", sum)
		}
	})

	t.Run("handles -Inf", func(t *testing.T) {
		probs := softmax([]float64{5.0, math.Inf(-1), math.Inf(-1)})
		if math.Abs(probs[0]-1.0) > 1e-6 {
			t.Errorf("probs[0] = %f, want ~1.0", probs[0])
		}
	})
}

func TestSampleFromDistribution(t *testing.T) {
	t.Run("deterministic when one token dominates", func(t *testing.T) {
		// Token 1 has extremely high logit; all sampling should return 1.
		logits := []float64{-100.0, 100.0, -100.0}
		for range 20 {
			got := sampleFromDistribution(logits)
			if got != 1 {
				t.Errorf("sampleFromDistribution = %d, want 1", got)
			}
		}
	})

	t.Run("returns valid index", func(t *testing.T) {
		logits := []float64{1.0, 1.0, 1.0, 1.0}
		for range 50 {
			got := sampleFromDistribution(logits)
			if got < 0 || got >= 4 {
				t.Errorf("sampleFromDistribution = %d, want [0,4)", got)
			}
		}
	})

	t.Run("all negative inf returns valid index", func(t *testing.T) {
		logits := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)}
		got := sampleFromDistribution(logits)
		if got < 0 || got >= 3 {
			t.Errorf("sampleFromDistribution = %d, want [0,3)", got)
		}
	})
}

func TestSoftmax_AllNegInf(t *testing.T) {
	probs := softmax([]float64{math.Inf(-1), math.Inf(-1)})
	// When all inputs are -Inf, exp returns 0, sum is 0.
	// softmax guard should prevent division by zero.
	for i, p := range probs {
		if math.IsNaN(p) {
			t.Errorf("probs[%d] is NaN", i)
		}
	}
}
