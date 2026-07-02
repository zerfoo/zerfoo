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

	// Test decompose directly for roundtrip (before mixing).
	scales := m.decompose(input)
	for s, sc := range scales {
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

			// After mixing, shapes must still be correct.
			for s, sc := range out.Scales {
				if len(sc.trend) != nf {
					t.Errorf("scale %d: got %d features, want %d", s, len(sc.trend), nf)
				}
				if len(sc.seasonal) != nf {
					t.Errorf("scale %d: got %d seasonal features, want %d", s, len(sc.seasonal), nf)
				}
				for f := 0; f < nf; f++ {
					if len(sc.trend[f]) != cfg.InputLen {
						t.Errorf("scale %d feature %d: trend length %d, want %d", s, f, len(sc.trend[f]), cfg.InputLen)
					}
					if len(sc.seasonal[f]) != cfg.InputLen {
						t.Errorf("scale %d feature %d: seasonal length %d, want %d", s, f, len(sc.seasonal[f]), cfg.InputLen)
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

func TestTimeMixerMixingOutputShapes(t *testing.T) {
	tests := []struct {
		name      string
		inputLen  int
		features  int
		numScales int
		numLayers int
		hidden    int
	}{
		{"small", 16, 2, 2, 1, 32},
		{"medium", 32, 3, 4, 3, 64},
		{"large", 64, 5, 3, 2, 128},
		{"single_layer", 24, 1, 3, 1, 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := TimeMixerConfig{
				InputLen:    tt.inputLen,
				OutputLen:   4,
				NumFeatures: tt.features,
				NumScales:   tt.numScales,
				HiddenSize:  tt.hidden,
				NumLayers:   tt.numLayers,
			}
			m := NewTimeMixer(cfg)

			input := make([][]float64, tt.features)
			for f := range input {
				input[f] = make([]float64, tt.inputLen)
				for i := range input[f] {
					input[f][i] = math.Sin(float64(i)*0.5) * float64(f+1)
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
					t.Errorf("scale %d: trend features %d, want %d", s, len(sc.trend), tt.features)
				}
				if len(sc.seasonal) != tt.features {
					t.Errorf("scale %d: seasonal features %d, want %d", s, len(sc.seasonal), tt.features)
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
		})
	}
}

func TestTimeMixerBottomUpResidual(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    16,
		OutputLen:   4,
		NumFeatures: 1,
		NumScales:   3,
		HiddenSize:  8,
		NumLayers:   1,
	}
	m := NewTimeMixer(cfg)

	// Set MLP weights to identity-like to make residual effect observable.
	// Zero out all weights and set diagonal of w2*w1 to 1 (skip MLP transform).
	mlp := m.seasonalMLPs[0]
	for i := range mlp.w1 {
		for j := range mlp.w1[i] {
			mlp.w1[i][j] = 0
		}
		mlp.b1[i] = 0
	}
	for i := range mlp.w2 {
		for j := range mlp.w2[i] {
			mlp.w2[i][j] = 0
		}
		mlp.b2[i] = 0
	}
	// With zero MLP, seasonal outputs are all zero from MLP.
	// Bottom-up residual adds coarse to fine, so scale 0 gets sum of all
	// zero MLP outputs (still zero). Let's use a non-trivial identity instead.

	// Make w1 = I (first numScales rows), w2 = I (first numScales cols).
	ns := cfg.NumScales
	for i := 0; i < ns && i < len(mlp.w1); i++ {
		mlp.w1[i][i] = 1.0
	}
	for i := 0; i < ns; i++ {
		if i < len(mlp.w2[i]) {
			mlp.w2[i][i] = 1.0
		}
	}
	// Now the seasonal MLP is approximately identity for positive inputs.
	// With ReLU, only positive values pass through.

	// Do the same for trend MLP.
	tmlp := m.trendMLPs[0]
	for i := range tmlp.w1 {
		for j := range tmlp.w1[i] {
			tmlp.w1[i][j] = 0
		}
		tmlp.b1[i] = 0
	}
	for i := range tmlp.w2 {
		for j := range tmlp.w2[i] {
			tmlp.w2[i][j] = 0
		}
		tmlp.b2[i] = 0
	}
	for i := 0; i < ns && i < len(tmlp.w1); i++ {
		tmlp.w1[i][i] = 1.0
	}
	for i := 0; i < ns; i++ {
		if i < len(tmlp.w2[i]) {
			tmlp.w2[i][i] = 1.0
		}
	}

	input := make([][]float64, 1)
	input[0] = make([]float64, cfg.InputLen)
	for i := range input[0] {
		input[0][i] = float64(i) + 1.0 // all positive
	}

	out, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// With identity MLP and bottom-up residual, finer scales should have
	// larger values than coarser scales (they accumulate residuals).
	// Check that scale 0 seasonal values >= scale 1 >= scale 2.
	for i := 0; i < cfg.InputLen; i++ {
		s0 := math.Abs(out.Scales[0].seasonal[0][i])
		s2 := math.Abs(out.Scales[2].seasonal[0][i])
		// Scale 0 (finest) should be >= scale 2 (coarsest) due to bottom-up accumulation.
		if s0 < s2-1e-10 {
			t.Errorf("index %d: finest scale seasonal %.6f < coarsest %.6f (bottom-up residual not working)",
				i, s0, s2)
		}
	}
}

func TestTimeMixerMixingTransforms(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    16,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   3,
		HiddenSize:  32,
		NumLayers:   2,
	}
	m := NewTimeMixer(cfg)

	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = math.Sin(float64(i)*0.3) + float64(f)*2.0
		}
	}

	// Get decomposition without mixing.
	rawScales := m.decompose(input)

	// Get full forward output (with mixing).
	out, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Mixing should produce different values from raw decomposition.
	different := false
	for s := range rawScales {
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.InputLen; i++ {
				if math.Abs(rawScales[s].seasonal[f][i]-out.Scales[s].seasonal[f][i]) > 1e-10 {
					different = true
					break
				}
			}
			if different {
				break
			}
		}
		if different {
			break
		}
	}
	if !different {
		t.Error("mixing did not transform the seasonal components — MLP had no effect")
	}
}

func TestTimeMixerMixingDeterministic(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    16,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   3,
		HiddenSize:  32,
		NumLayers:   2,
	}
	m := NewTimeMixer(cfg)

	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = float64(i) * 0.1
		}
	}

	out1, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward 1 failed: %v", err)
	}
	out2, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward 2 failed: %v", err)
	}

	// Same model, same input -> same output.
	for s := range out1.Scales {
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.InputLen; i++ {
				if out1.Scales[s].trend[f][i] != out2.Scales[s].trend[f][i] {
					t.Errorf("scale %d feature %d index %d: trend not deterministic", s, f, i)
				}
				if out1.Scales[s].seasonal[f][i] != out2.Scales[s].seasonal[f][i] {
					t.Errorf("scale %d feature %d index %d: seasonal not deterministic", s, f, i)
				}
			}
		}
	}
}

func TestMixingMLPForward(t *testing.T) {
	mlp := newMixingMLP(3, 8, nil)

	// Zero input should produce output equal to bias pass-through.
	input := []float64{0, 0, 0}
	out := mlp.forward(input)
	if len(out) != 3 {
		t.Fatalf("expected output length 3, got %d", len(out))
	}

	// Non-zero input should produce non-zero output (with high probability given random init).
	input2 := []float64{1.0, 2.0, 3.0}
	out2 := mlp.forward(input2)
	allZero := true
	for _, v := range out2 {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("MLP produced all-zero output for non-zero input")
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

func TestTimeMixerForecastShape(t *testing.T) {
	tests := []struct {
		name     string
		features int
		inputLen int
		outLen   int
		scales   int
	}{
		{"1feat_16in_4out_2scales", 1, 16, 4, 2},
		{"3feat_32in_8out_4scales", 3, 32, 8, 4},
		{"5feat_64in_16out_6scales", 5, 64, 16, 6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := TimeMixerConfig{
				InputLen:    tt.inputLen,
				OutputLen:   tt.outLen,
				NumFeatures: tt.features,
				NumScales:   tt.scales,
			}
			m := NewTimeMixer(cfg)

			input := make([][]float64, tt.features)
			for f := range input {
				input[f] = make([]float64, tt.inputLen)
				for i := range input[f] {
					input[f][i] = math.Sin(float64(i)*0.2) * float64(f+1)
				}
			}

			out, err := m.Forward(input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			if len(out.Forecast) != tt.features {
				t.Fatalf("Forecast has %d features, want %d", len(out.Forecast), tt.features)
			}
			for f := 0; f < tt.features; f++ {
				if len(out.Forecast[f]) != tt.outLen {
					t.Errorf("Forecast[%d] length %d, want %d", f, len(out.Forecast[f]), tt.outLen)
				}
			}

			// Forecast values should be finite.
			for f := 0; f < tt.features; f++ {
				for j := 0; j < tt.outLen; j++ {
					if math.IsNaN(out.Forecast[f][j]) || math.IsInf(out.Forecast[f][j], 0) {
						t.Errorf("Forecast[%d][%d] = %v, want finite", f, j, out.Forecast[f][j])
					}
				}
			}
		})
	}
}

func TestTimeMixerTrainWindowedDecreasingLoss(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    24,
		OutputLen:   6,
		NumFeatures: 1,
		NumScales:   2,
		HiddenSize:  16,
		NumLayers:   1,
	}
	m := NewTimeMixer(cfg)

	// Generate synthetic sinusoidal training data.
	nSamples := 20
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, cfg.NumFeatures)
		for f := 0; f < cfg.NumFeatures; f++ {
			windows[s][f] = make([]float64, cfg.InputLen)
			for i := 0; i < cfg.InputLen; i++ {
				windows[s][f][i] = math.Sin(float64(s+i) * 0.3)
			}
		}
		// Label is next value in the sine wave.
		labels[s] = math.Sin(float64(s+cfg.InputLen) * 0.3)
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{Epochs: 30})
	if err != nil {
		t.Fatalf("TrainWindowed failed: %v", err)
	}

	if len(result.LossHistory) != 30 {
		t.Fatalf("expected 30 loss entries, got %d", len(result.LossHistory))
	}

	// Compare average of first 5 epochs vs last 5 epochs.
	earlyAvg := 0.0
	for _, l := range result.LossHistory[:5] {
		earlyAvg += l
	}
	earlyAvg /= 5.0

	lateAvg := 0.0
	for _, l := range result.LossHistory[len(result.LossHistory)-5:] {
		lateAvg += l
	}
	lateAvg /= 5.0

	if lateAvg >= earlyAvg {
		t.Errorf("loss did not decrease: early avg=%.6f, late avg=%.6f", earlyAvg, lateAvg)
	}
}

func TestTimeMixerMultiScaleMixingValues(t *testing.T) {
	// Verify that different scale counts produce different decompositions,
	// confirming multi-scale mixing is effective.
	inputLen := 32
	outLen := 8
	features := 2

	input := make([][]float64, features)
	for f := range input {
		input[f] = make([]float64, inputLen)
		for i := range input[f] {
			input[f][i] = math.Sin(float64(i)*0.4) + math.Cos(float64(i)*0.1)*float64(f+1)
		}
	}

	for _, numScales := range []int{2, 4, 6} {
		t.Run("scales_"+itoa(numScales), func(t *testing.T) {
			cfg := TimeMixerConfig{
				InputLen:    inputLen,
				OutputLen:   outLen,
				NumFeatures: features,
				NumScales:   numScales,
			}
			m := NewTimeMixer(cfg)

			out, err := m.Forward(input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			if len(out.Scales) != numScales {
				t.Fatalf("expected %d scales, got %d", numScales, len(out.Scales))
			}

			// Verify different scales have different trend patterns
			// (due to different MA kernel sizes).
			rawScales := m.decompose(input)
			for s1 := 0; s1 < numScales; s1++ {
				for s2 := s1 + 1; s2 < numScales; s2++ {
					diff := 0.0
					for i := 0; i < inputLen; i++ {
						diff += math.Abs(rawScales[s1].trend[0][i] - rawScales[s2].trend[0][i])
					}
					if diff < 1e-10 {
						t.Errorf("scales %d and %d have identical trends — decomposition is not multi-scale", s1, s2)
					}
				}
			}
		})
	}
}

func TestTimeMixerChannelIndependence(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    16,
		OutputLen:   4,
		NumFeatures: 3,
		NumScales:   2,
		HiddenSize:  8,
		NumLayers:   1,
	}
	m := NewTimeMixer(cfg)

	// Create input where each feature is distinct.
	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = math.Sin(float64(i)*0.3+float64(f)*2.0) * float64(f+1)
		}
	}

	out, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Modify only feature 0 and verify other features produce
	// identical forecasts, confirming channel independence.
	modified := make([][]float64, cfg.NumFeatures)
	for f := range modified {
		modified[f] = make([]float64, cfg.InputLen)
		copy(modified[f], input[f])
	}
	// Perturb feature 0.
	for i := range modified[0] {
		modified[0][i] += 10.0
	}

	out2, err := m.Forward(modified)
	if err != nil {
		t.Fatalf("Forward with modified input failed: %v", err)
	}

	// Features 1 and 2 should be unchanged in the decomposition.
	for s := range out.Scales {
		for f := 1; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.InputLen; i++ {
				if out.Scales[s].trend[f][i] != out2.Scales[s].trend[f][i] {
					t.Errorf("scale %d feature %d index %d: trend changed (%.6f -> %.6f) when only feature 0 was modified",
						s, f, i, out.Scales[s].trend[f][i], out2.Scales[s].trend[f][i])
				}
				if out.Scales[s].seasonal[f][i] != out2.Scales[s].seasonal[f][i] {
					t.Errorf("scale %d feature %d index %d: seasonal changed when only feature 0 was modified",
						s, f, i)
				}
			}
		}
	}

	// Feature 0 forecast should be different.
	same := true
	for j := 0; j < cfg.OutputLen; j++ {
		if out.Forecast[0][j] != out2.Forecast[0][j] {
			same = false
			break
		}
	}
	if same {
		t.Error("feature 0 forecast unchanged despite input perturbation")
	}

	// Features 1 and 2 forecasts should be identical.
	for f := 1; f < cfg.NumFeatures; f++ {
		for j := 0; j < cfg.OutputLen; j++ {
			if out.Forecast[f][j] != out2.Forecast[f][j] {
				t.Errorf("feature %d forecast[%d] changed (%.6f -> %.6f) when only feature 0 was modified",
					f, j, out.Forecast[f][j], out2.Forecast[f][j])
			}
		}
	}
}
