package multimodal

import (
	"math"
	"testing"
)

func TestExtractMelSpectrogram(t *testing.T) {
	cfg := DefaultAudioConfig()

	// Generate 1 second of 440Hz sine wave at 16kHz.
	numSamples := cfg.SampleRate
	samples := make([]float32, numSamples)
	for i := range samples {
		samples[i] = float32(math.Sin(2.0 * math.Pi * 440.0 * float64(i) / float64(cfg.SampleRate)))
	}

	mel, err := ExtractMelSpectrogram(samples, cfg)
	if err != nil {
		t.Fatalf("ExtractMelSpectrogram: %v", err)
	}

	expectedFrames := 1 + (numSamples-cfg.FFTSize)/cfg.HopLength
	if mel.NumFrames != expectedFrames {
		t.Errorf("NumFrames = %d, want %d", mel.NumFrames, expectedFrames)
	}
	if mel.NumMels != cfg.NumMels {
		t.Errorf("NumMels = %d, want %d", mel.NumMels, cfg.NumMels)
	}
	if len(mel.Data) != mel.NumFrames*mel.NumMels {
		t.Errorf("len(Data) = %d, want %d", len(mel.Data), mel.NumFrames*mel.NumMels)
	}

	// Verify all values are finite (no NaN or Inf).
	for i, v := range mel.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("Data[%d] = %f, want finite", i, v)
			break
		}
	}
}

func TestExtractMelSpectrogramErrors(t *testing.T) {
	cfg := DefaultAudioConfig()

	_, err := ExtractMelSpectrogram(nil, cfg)
	if err == nil {
		t.Error("expected error for nil samples")
	}

	_, err = ExtractMelSpectrogram(make([]float32, 10), cfg)
	if err == nil {
		t.Error("expected error for audio shorter than FFTSize")
	}
}

func TestNormalizeAudio(t *testing.T) {
	samples := []float32{-3.0, 0.5, 2.0, -1.0, 4.0}
	norm := NormalizeAudio(samples)

	if len(norm) != len(samples) {
		t.Fatalf("len(norm) = %d, want %d", len(norm), len(samples))
	}

	for i, v := range norm {
		a := v
		if a < 0 {
			a = -a
		}
		if a > 1.0+1e-6 {
			t.Errorf("norm[%d] = %f, want abs <= 1.0", i, v)
		}
	}

	// The max absolute value sample should become exactly 1.0 or -1.0.
	var maxAbs float32
	for _, v := range norm {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	if math.Abs(float64(maxAbs)-1.0) > 1e-6 {
		t.Errorf("max abs = %f, want 1.0", maxAbs)
	}

	// Original should not be modified.
	if samples[4] != 4.0 {
		t.Errorf("original modified: samples[4] = %f, want 4.0", samples[4])
	}
}

func TestNormalizeAudioZero(t *testing.T) {
	samples := []float32{0, 0, 0}
	norm := NormalizeAudio(samples)
	for i, v := range norm {
		if v != 0 {
			t.Errorf("norm[%d] = %f, want 0", i, v)
		}
	}
}

func TestNormalizeAudioNil(t *testing.T) {
	norm := NormalizeAudio(nil)
	if norm != nil {
		t.Errorf("expected nil for nil input, got %v", norm)
	}
}

func TestHannWindow(t *testing.T) {
	size := 400
	w := HannWindow(size)

	if len(w) != size {
		t.Fatalf("len(window) = %d, want %d", len(w), size)
	}

	// First sample should be near 0.
	if w[0] > 1e-6 {
		t.Errorf("w[0] = %f, want near 0", w[0])
	}

	// Last sample should be near 0.
	if w[size-1] > 1e-6 {
		t.Errorf("w[%d] = %f, want near 0", size-1, w[size-1])
	}

	// Middle sample should be near 1.
	mid := size / 2
	if math.Abs(float64(w[mid])-1.0) > 0.01 {
		t.Errorf("w[%d] = %f, want near 1.0", mid, w[mid])
	}

	// All values should be in [0, 1].
	for i, v := range w {
		if v < 0 || v > 1.0+1e-6 {
			t.Errorf("w[%d] = %f, want in [0, 1]", i, v)
			break
		}
	}
}

func TestMelFilterbank(t *testing.T) {
	numMels := 80
	fftSize := 400
	sampleRate := 16000
	var fMin, fMax float32 = 0, 8000

	fb := MelFilterbank(numMels, fftSize, sampleRate, fMin, fMax)

	expectedFreqBins := fftSize/2 + 1
	if len(fb) != numMels {
		t.Fatalf("len(filterbank) = %d, want %d", len(fb), numMels)
	}
	for m, row := range fb {
		if len(row) != expectedFreqBins {
			t.Errorf("filterbank[%d] length = %d, want %d", m, len(row), expectedFreqBins)
		}
	}

	// Each filter should have non-negative values.
	for m, row := range fb {
		for k, v := range row {
			if v < 0 {
				t.Errorf("filterbank[%d][%d] = %f, want >= 0", m, k, v)
			}
		}
	}

	// Each filter should have at least one non-zero value.
	for m, row := range fb {
		var sum float32
		for _, v := range row {
			sum += v
		}
		if sum <= 0 {
			t.Errorf("filterbank[%d] has zero sum, want > 0", m)
		}
	}
}
