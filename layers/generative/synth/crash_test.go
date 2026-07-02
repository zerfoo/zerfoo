package synth

import (
	"math"
	"testing"
)

func trainedVAE(t *testing.T) *MarketVAE {
	t.Helper()
	inputDim := 4
	data := generateCorrelatedData(500, inputDim, 42)
	vae := NewMarketVAE(VAEConfig{
		InputDim:     inputDim,
		LatentDim:    2,
		HiddenDims:   []int{16, 8},
		LearningRate: 0.001,
		NEpochs:      80,
		Seed:         42,
	})
	if err := vae.Train(data); err != nil {
		t.Fatalf("Train error: %v", err)
	}
	return vae
}

func TestCrashGenerator_ExtremeEvents(t *testing.T) {
	vae := trainedVAE(t)
	inputDim := vae.config.InputDim

	tests := []struct {
		name     string
		severity float64
		duration int
		n        int
	}{
		{name: "low severity", severity: 1.0, duration: 5, n: 20},
		{name: "medium severity", severity: 5.0, duration: 10, n: 20},
		{name: "high severity", severity: 10.0, duration: 10, n: 20},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cg := NewCrashGenerator(vae, CrashConfig{
				Severity:         tt.severity,
				Duration:         tt.duration,
				CorrelationSpike: 0.5,
				Seed:             99,
			})

			samples := cg.Generate(tt.n)
			if len(samples) != tt.n {
				t.Fatalf("expected %d samples, got %d", tt.n, len(samples))
			}

			expectedLen := tt.duration * inputDim
			for i, s := range samples {
				if len(s) != expectedLen {
					t.Fatalf("sample %d: expected length %d, got %d", i, expectedLen, len(s))
				}
				for j, v := range s {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						t.Fatalf("sample[%d][%d] is not finite: %v", i, j, v)
					}
				}
			}
		})
	}

	// Higher severity should produce more extreme values on average.
	lowCG := NewCrashGenerator(vae, CrashConfig{
		Severity:         1.0,
		Duration:         5,
		CorrelationSpike: 0.3,
		Seed:             123,
	})
	highCG := NewCrashGenerator(vae, CrashConfig{
		Severity:         10.0,
		Duration:         5,
		CorrelationSpike: 0.3,
		Seed:             123,
	})

	lowSamples := lowCG.Generate(100)
	highSamples := highCG.Generate(100)

	lowAbsMean := absMean(lowSamples)
	highAbsMean := absMean(highSamples)

	if highAbsMean <= lowAbsMean {
		t.Errorf("high severity abs mean (%.4f) should exceed low severity abs mean (%.4f)",
			highAbsMean, lowAbsMean)
	}
}

func TestCrashGenerator_GenerateWithSeverity(t *testing.T) {
	vae := trainedVAE(t)

	cg := NewCrashGenerator(vae, CrashConfig{
		Severity:         5.0,
		Duration:         3,
		CorrelationSpike: 0.5,
		Seed:             42,
	})

	tests := []struct {
		name     string
		severity float64
	}{
		{name: "below minimum clamps to 1", severity: 0.5},
		{name: "minimum severity", severity: 1.0},
		{name: "mid severity", severity: 5.0},
		{name: "maximum severity", severity: 10.0},
		{name: "above maximum clamps to 10", severity: 15.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			samples := cg.GenerateWithSeverity(10, tt.severity)
			if len(samples) != 10 {
				t.Fatalf("expected 10 samples, got %d", len(samples))
			}
			expectedLen := 3 * vae.config.InputDim
			for i, s := range samples {
				if len(s) != expectedLen {
					t.Errorf("sample %d: expected length %d, got %d", i, expectedLen, len(s))
				}
			}
		})
	}
}

func TestCrashGenerator_CorrelationSpike(t *testing.T) {
	vae := trainedVAE(t)
	inputDim := vae.config.InputDim

	// Generate scenarios with no correlation spike.
	noCorrCG := NewCrashGenerator(vae, CrashConfig{
		Severity:         5.0,
		Duration:         1,
		CorrelationSpike: 0.0,
		Seed:             42,
	})
	// Generate scenarios with maximum correlation spike.
	highCorrCG := NewCrashGenerator(vae, CrashConfig{
		Severity:         5.0,
		Duration:         1,
		CorrelationSpike: 1.0,
		Seed:             42,
	})

	n := 200
	noCorrSamples := noCorrCG.Generate(n)
	highCorrSamples := highCorrCG.Generate(n)

	// With high correlation spike, assets should be more correlated.
	// Measure average pairwise correlation across features.
	noCorrAvg := avgPairwiseCorrelation(noCorrSamples, inputDim)
	highCorrAvg := avgPairwiseCorrelation(highCorrSamples, inputDim)

	if highCorrAvg < noCorrAvg {
		t.Errorf("high correlation spike (avg corr=%.4f) should produce higher correlations than none (avg corr=%.4f)",
			highCorrAvg, noCorrAvg)
	}
}

func TestCrashGenerator_ConfigDefaults(t *testing.T) {
	vae := trainedVAE(t)

	tests := []struct {
		name              string
		severity          float64
		duration          int
		wantSeverityCap   float64
		wantDurationFloor int
	}{
		{name: "below min severity", severity: 0.0, duration: 5, wantSeverityCap: 1.0, wantDurationFloor: 5},
		{name: "above max severity", severity: 20.0, duration: 5, wantSeverityCap: 10.0, wantDurationFloor: 5},
		{name: "zero duration", severity: 5.0, duration: 0, wantSeverityCap: 5.0, wantDurationFloor: 1},
		{name: "negative duration", severity: 5.0, duration: -1, wantSeverityCap: 5.0, wantDurationFloor: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cg := NewCrashGenerator(vae, CrashConfig{
				Severity: tt.severity,
				Duration: tt.duration,
				Seed:     42,
			})
			if cg.config.Severity != tt.wantSeverityCap {
				t.Errorf("severity: got %.1f, want %.1f", cg.config.Severity, tt.wantSeverityCap)
			}
			if cg.config.Duration != tt.wantDurationFloor {
				t.Errorf("duration: got %d, want %d", cg.config.Duration, tt.wantDurationFloor)
			}
		})
	}
}

// absMean computes the mean of absolute values across all samples.
func absMean(samples [][]float64) float64 {
	var sum float64
	var count int
	for _, s := range samples {
		for _, v := range s {
			sum += math.Abs(v)
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

// avgPairwiseCorrelation computes the average absolute Pearson correlation
// across all pairs of features (columns) in the samples. Each sample is
// a flat slice of length inputDim (Duration=1).
func avgPairwiseCorrelation(samples [][]float64, inputDim int) float64 {
	n := len(samples)
	if n < 2 || inputDim < 2 {
		return 0
	}

	// Extract columns.
	cols := make([][]float64, inputDim)
	for j := 0; j < inputDim; j++ {
		cols[j] = make([]float64, n)
		for i := 0; i < n; i++ {
			cols[j][i] = samples[i][j]
		}
	}

	var totalCorr float64
	var pairs int
	for a := 0; a < inputDim; a++ {
		for b := a + 1; b < inputDim; b++ {
			r := pearson(cols[a], cols[b])
			totalCorr += math.Abs(r)
			pairs++
		}
	}
	if pairs == 0 {
		return 0
	}
	return totalCorr / float64(pairs)
}

// pearson computes the Pearson correlation coefficient between x and y.
func pearson(x, y []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	var sx, sy, sxx, syy, sxy float64
	for i := 0; i < n; i++ {
		sx += x[i]
		sy += y[i]
		sxx += x[i] * x[i]
		syy += y[i] * y[i]
		sxy += x[i] * y[i]
	}
	fn := float64(n)
	num := fn*sxy - sx*sy
	den := math.Sqrt((fn*sxx - sx*sx) * (fn*syy - sy*sy))
	if den == 0 {
		return 0
	}
	return num / den
}
