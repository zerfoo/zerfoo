package synth

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/layers/functional"
)

// generateCorrelatedData creates n samples of d-dimensional data where features
// are correlated: x[i] = base + noise. This gives the VAE structure to learn.
func generateCorrelatedData(n, d int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		base := rng.NormFloat64()
		row := make([]float64, d)
		for j := 0; j < d; j++ {
			row[j] = base + 0.3*rng.NormFloat64()
		}
		data[i] = row
	}
	return data
}

func meanStd(data [][]float64, dim int) (float64, float64) {
	n := len(data)
	var sum float64
	for _, row := range data {
		sum += row[dim]
	}
	mean := sum / float64(n)
	var ss float64
	for _, row := range data {
		d := row[dim] - mean
		ss += d * d
	}
	std := math.Sqrt(ss / float64(n))
	return mean, std
}

func TestMarketVAE_Generation(t *testing.T) {
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

	synthetic := vae.Generate(500)
	if len(synthetic) != 500 {
		t.Fatalf("expected 500 samples, got %d", len(synthetic))
	}
	if len(synthetic[0]) != inputDim {
		t.Fatalf("expected dimension %d, got %d", inputDim, len(synthetic[0]))
	}

	// Check that generated data has similar statistics to the original.
	for d := 0; d < inputDim; d++ {
		origMean, origStd := meanStd(data, d)
		synMean, synStd := meanStd(synthetic, d)

		// Allow generous tolerance — the VAE won't perfectly match statistics,
		// but should be in the same ballpark.
		if math.Abs(synMean-origMean) > 2.0 {
			t.Errorf("dim %d: mean mismatch: original=%.3f synthetic=%.3f", d, origMean, synMean)
		}
		if synStd < 0.1 {
			t.Errorf("dim %d: synthetic std too low (%.3f), VAE may have collapsed", d, synStd)
		}
		if math.Abs(synStd-origStd) > origStd*3 {
			t.Errorf("dim %d: std mismatch: original=%.3f synthetic=%.3f", d, origStd, synStd)
		}
	}
}

func TestMarketVAE_LatentSpace(t *testing.T) {
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

	latent := vae.Encode(data)
	if len(latent) != len(data) {
		t.Fatalf("expected %d latent vectors, got %d", len(data), len(latent))
	}
	if len(latent[0]) != 2 {
		t.Fatalf("expected latent dim 2, got %d", len(latent[0]))
	}

	// Verify latent space is smooth and continuous:
	// 1. Check that latent representations have finite values.
	for i, z := range latent {
		for j, v := range z {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("latent[%d][%d] is not finite: %v", i, j, v)
			}
		}
	}

	// 2. Check that latent space has non-trivial variance (not collapsed).
	for d := 0; d < 2; d++ {
		_, std := meanStd(latent, d)
		if std < 0.01 {
			t.Errorf("latent dim %d: std=%.6f is too small, posterior may have collapsed", d, std)
		}
	}

	// 3. Check continuity: nearby latent points should decode to similar outputs.
	// Take two close latent points and verify decoded outputs are close.
	z1 := latent[0]
	z2 := make([]float64, len(z1))
	for i := range z1 {
		z2[i] = z1[i] + 0.01
	}

	// Decode both points.
	ctx := context.Background()
	z1T, _ := tensor.New[float64]([]int{1, len(z1)}, z1)
	z2T, _ := tensor.New[float64]([]int{1, len(z2)}, z2)
	acts1 := vae.decoderForward(ctx, z1T)
	out1 := acts1[len(acts1)-1].Data()
	acts2 := vae.decoderForward(ctx, z2T)
	out2 := acts2[len(acts2)-1].Data()

	var dist float64
	for i := range out1 {
		d := out1[i] - out2[i]
		dist += d * d
	}
	dist = math.Sqrt(dist)

	// Nearby latent points should produce nearby outputs.
	if dist > 5.0 {
		t.Errorf("decoded outputs for nearby latent points are too far apart: distance=%.4f", dist)
	}
}

func TestMarketVAE_TrainErrors(t *testing.T) {
	tests := []struct {
		name    string
		data    [][]float64
		config  VAEConfig
		wantErr string
	}{
		{
			name:    "empty data",
			data:    nil,
			config:  VAEConfig{InputDim: 4, LatentDim: 2, HiddenDims: []int{8}},
			wantErr: "at least one sample",
		},
		{
			name:    "dimension mismatch",
			data:    [][]float64{{1, 2, 3}},
			config:  VAEConfig{InputDim: 4, LatentDim: 2, HiddenDims: []int{8}},
			wantErr: "does not match",
		},
		{
			name:    "jagged rows",
			data:    [][]float64{{1, 2, 3, 4}, {1, 2}},
			config:  VAEConfig{InputDim: 4, LatentDim: 2, HiddenDims: []int{8}},
			wantErr: "columns",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vae := NewMarketVAE(tt.config)
			err := vae.Train(tt.data)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !containsSubstr(err.Error(), tt.wantErr) {
				t.Errorf("error %q does not contain %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestMarketVAE_Encode(t *testing.T) {
	inputDim := 3
	vae := NewMarketVAE(VAEConfig{
		InputDim:   inputDim,
		LatentDim:  2,
		HiddenDims: []int{8},
		Seed:       42,
	})

	data := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	latent := vae.Encode(data)
	if len(latent) != 2 {
		t.Fatalf("expected 2 latent vectors, got %d", len(latent))
	}
	if len(latent[0]) != 2 {
		t.Fatalf("expected latent dim 2, got %d", len(latent[0]))
	}

	// Same input should produce same encoding (deterministic).
	latent2 := vae.Encode(data)
	for i := range latent {
		for j := range latent[i] {
			if latent[i][j] != latent2[i][j] {
				t.Errorf("Encode not deterministic: latent[%d][%d] = %v vs %v", i, j, latent[i][j], latent2[i][j])
			}
		}
	}
}

func TestMarketVAE_Generate(t *testing.T) {
	vae := NewMarketVAE(VAEConfig{
		InputDim:   3,
		LatentDim:  2,
		HiddenDims: []int{8},
		Seed:       42,
	})

	samples := vae.Generate(10)
	if len(samples) != 10 {
		t.Fatalf("expected 10 samples, got %d", len(samples))
	}
	for i, s := range samples {
		if len(s) != 3 {
			t.Errorf("sample %d has dimension %d, expected 3", i, len(s))
		}
		for j, v := range s {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("sample[%d][%d] is not finite: %v", i, j, v)
			}
		}
	}
}

func TestLinearForward(t *testing.T) {
	// Simple 2->3 linear layer via functional.Linear through the VAE's engine path.
	rng := rand.New(rand.NewSource(42))
	wData := make([]float64, 6) // [3, 2] row-major (outDim=3, inDim=2)
	for i := range wData {
		wData[i] = rng.NormFloat64() * 0.1
	}

	vae := NewMarketVAE(VAEConfig{InputDim: 2, LatentDim: 1, HiddenDims: []int{3}, Seed: 42})
	ctx := context.Background()

	xData := []float64{1.0, 2.0}
	xT, _ := tensor.New[float64]([]int{1, 2}, xData)
	wT, _ := tensor.New[float64]([]int{3, 2}, wData)
	bT, _ := tensor.New[float64]([]int{3}, make([]float64, 3))

	y := vae.linearFwd(ctx, xT, wT, bT)
	yData := y.Data()
	if len(yData) != 3 {
		t.Fatalf("expected output dim 3, got %d", len(yData))
	}

	// Check: y[j] = sum_i(x[i]*w[j*2+i]) for weight [outDim, inDim].
	for j := 0; j < 3; j++ {
		expected := xData[0]*wData[j*2] + xData[1]*wData[j*2+1]
		if math.Abs(yData[j]-expected) > 1e-10 {
			t.Errorf("y[%d] = %f, expected %f", j, yData[j], expected)
		}
	}
}

func TestReluForwardBackward(t *testing.T) {
	vae := NewMarketVAE(VAEConfig{InputDim: 5, LatentDim: 1, HiddenDims: []int{5}, Seed: 42})
	ctx := context.Background()

	xData := []float64{-2, -1, 0, 1, 2}
	xT, _ := tensor.New[float64]([]int{1, 5}, xData)
	y, _ := functional.ReLU(ctx, vae.engine, vae.ops, xT)
	yData := y.Data()
	expected := []float64{0, 0, 0, 1, 2}
	for i := range yData {
		if yData[i] != expected[i] {
			t.Errorf("relu[%d] = %f, expected %f", i, yData[i], expected[i])
		}
	}

	dOutData := []float64{1, 1, 1, 1, 1}
	dOutT, _ := tensor.New[float64]([]int{1, 5}, dOutData)
	dIn := vae.reluBwd(ctx, y, dOutT)
	dInData := dIn.Data()
	expectedGrad := []float64{0, 0, 0, 1, 1}
	for i := range dInData {
		if dInData[i] != expectedGrad[i] {
			t.Errorf("relu_backward[%d] = %f, expected %f", i, dInData[i], expectedGrad[i])
		}
	}
}

func containsSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
