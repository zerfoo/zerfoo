package timeseries

import (
	"math"
	"testing"
)

func TestTimeMixerDecompositionRoundtrip(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    32,
		OutputLen:   8,
		NumFeatures: 3,
		NumScales:   4,
	}
	m := NewTimeMixer(cfg)

	// Create deterministic input.
	input := make([][]float64, cfg.NumFeatures)
	for f := 0; f < cfg.NumFeatures; f++ {
		input[f] = make([]float64, cfg.InputLen)
		for i := 0; i < cfg.InputLen; i++ {
			input[f][i] = math.Sin(float64(i)*0.3+float64(f)) + float64(f)*0.5
		}
	}

	out, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// For each scale, trend + seasonal must equal the original input.
	for s, sc := range out.Scales {
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.InputLen; i++ {
				reconstructed := sc.trend[f][i] + sc.seasonal[f][i]
				diff := math.Abs(reconstructed - input[f][i])
				if diff > 1e-6 {
					t.Errorf("scale %d feature %d index %d: trend+seasonal=%.10f, want %.10f (diff=%.2e)",
						s, f, i, reconstructed, input[f][i], diff)
				}
			}
		}
	}
}

func TestTimeMixerShapeCorrectness(t *testing.T) {
	tests := []struct {
		name      string
		numScales int
		inputLen  int
		features  int
	}{
		{"2scales_16len_1feat", 2, 16, 1},
		{"4scales_64len_3feat", 4, 64, 3},
		{"6scales_128len_5feat", 6, 128, 5},
		{"1scale_8len_2feat", 1, 8, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := TimeMixerConfig{
				InputLen:    tt.inputLen,
				OutputLen:   4,
				NumFeatures: tt.features,
				NumScales:   tt.numScales,
			}
			m := NewTimeMixer(cfg)

			input := make([][]float64, tt.features)
			for f := range input {
				input[f] = make([]float64, tt.inputLen)
				for i := range input[f] {
					input[f][i] = float64(i) * 0.1
				}
			}

			out, err := m.Forward(input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			if len(out.Scales) != tt.numScales {
				t.Fatalf("expected %d scales, got %d", tt.numScales, len(out.Scales))
			}

			for s, sc := range out.Scales {
				if len(sc.trend) != tt.features {
					t.Errorf("scale %d: trend has %d features, want %d", s, len(sc.trend), tt.features)
				}
				if len(sc.seasonal) != tt.features {
					t.Errorf("scale %d: seasonal has %d features, want %d", s, len(sc.seasonal), tt.features)
				}
				for f := 0; f < tt.features; f++ {
					if len(sc.trend[f]) != tt.inputLen {
						t.Errorf("scale %d feature %d: trend length %d, want %d", s, f, len(sc.trend[f]), tt.inputLen)
					}
					if len(sc.seasonal[f]) != tt.inputLen {
						t.Errorf("scale %d feature %d: seasonal length %d, want %d", s, f, len(sc.seasonal[f]), tt.inputLen)
					}
				}
			}

			// Verify MA kernel sizes: 2^(s+1).
			for s := 0; s < tt.numScales; s++ {
				w := m.MAWeights(s)
				expectedKernel := 1 << (s + 1)
				if len(w) != expectedKernel {
					t.Errorf("scale %d: kernel size %d, want %d", s, len(w), expectedKernel)
				}
			}
		})
	}
}

func TestTimeMixerDifferentNumFeatures(t *testing.T) {
	for _, nf := range []int{1, 2, 5, 10} {
		t.Run("features_"+itoa(nf), func(t *testing.T) {
			cfg := TimeMixerConfig{
				InputLen:    24,
				OutputLen:   6,
				NumFeatures: nf,
				NumScales:   3,
			}
			m := NewTimeMixer(cfg)

			input := make([][]float64, nf)
			for f := range input {
				input[f] = make([]float64, cfg.InputLen)
				for i := range input[f] {
					input[f][i] = math.Cos(float64(i)*0.2) * float64(f+1)
				}
			}

			out, err := m.Forward(input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Roundtrip check for all features and scales.
			for s, sc := range out.Scales {
				if len(sc.trend) != nf {
					t.Errorf("scale %d: got %d features, want %d", s, len(sc.trend), nf)
				}
				for f := 0; f < nf; f++ {
					for i := 0; i < cfg.InputLen; i++ {
						reconstructed := sc.trend[f][i] + sc.seasonal[f][i]
						diff := math.Abs(reconstructed - input[f][i])
						if diff > 1e-6 {
							t.Errorf("scale %d feature %d index %d: roundtrip diff %.2e", s, f, i, diff)
						}
					}
				}
			}
		})
	}
}

func TestTimeMixerForwardValidation(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    16,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   2,
	}
	m := NewTimeMixer(cfg)

	// Empty input.
	_, err := m.Forward(nil)
	if err == nil {
		t.Error("expected error for nil input")
	}

	// Wrong number of features.
	_, err = m.Forward(make([][]float64, 3))
	if err == nil {
		t.Error("expected error for wrong feature count")
	}

	// Wrong input length.
	bad := make([][]float64, 2)
	bad[0] = make([]float64, 16)
	bad[1] = make([]float64, 10)
	_, err = m.Forward(bad)
	if err == nil {
		t.Error("expected error for wrong input length")
	}
}

func TestTimeMixerMAWeightsSumToOne(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    32,
		OutputLen:   8,
		NumFeatures: 1,
		NumScales:   5,
	}
	m := NewTimeMixer(cfg)

	for s := 0; s < cfg.NumScales; s++ {
		w := m.MAWeights(s)
		sum := 0.0
		for _, v := range w {
			if v < 0 {
				t.Errorf("scale %d: negative weight %.6f", s, v)
			}
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("scale %d: weights sum to %.15f, want 1.0", s, sum)
		}
	}
}

// itoa is a simple int-to-string for test names without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}
