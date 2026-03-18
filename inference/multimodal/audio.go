// Package multimodal provides audio preprocessing for audio-language model inference.
package multimodal

import (
	"fmt"
	"math"
)

// AudioConfig specifies parameters for mel-spectrogram extraction.
type AudioConfig struct {
	SampleRate int     // Audio sample rate in Hz (default 16000).
	NumMels    int     // Number of mel filterbank channels (default 80).
	FFTSize    int     // FFT window size in samples (default 400).
	HopLength  int     // Hop length between frames in samples (default 160).
	FMin       float32 // Minimum frequency for mel filterbank (default 0).
	FMax       float32 // Maximum frequency for mel filterbank (default 8000).
}

// DefaultAudioConfig returns an AudioConfig with standard Whisper-style defaults.
func DefaultAudioConfig() AudioConfig {
	return AudioConfig{
		SampleRate: 16000,
		NumMels:    80,
		FFTSize:    400,
		HopLength:  160,
		FMin:       0,
		FMax:       8000,
	}
}

// MelSpectrogram holds a log-mel spectrogram with shape [NumFrames, NumMels].
// Data is stored in row-major order: Data[frame*NumMels + mel].
type MelSpectrogram struct {
	Data      []float32
	NumFrames int
	NumMels   int
}

// ExtractMelSpectrogram computes a log-mel spectrogram from raw audio samples.
// It applies a Hann window to each frame, computes the magnitude spectrum via
// DFT, applies a mel filterbank, and returns log-scaled mel energies.
// Output shape: [NumFrames, NumMels].
func ExtractMelSpectrogram(samples []float32, cfg AudioConfig) (*MelSpectrogram, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("multimodal: empty audio samples")
	}
	if cfg.FFTSize <= 0 {
		return nil, fmt.Errorf("multimodal: FFTSize must be positive, got %d", cfg.FFTSize)
	}
	if cfg.HopLength <= 0 {
		return nil, fmt.Errorf("multimodal: HopLength must be positive, got %d", cfg.HopLength)
	}
	if cfg.NumMels <= 0 {
		return nil, fmt.Errorf("multimodal: NumMels must be positive, got %d", cfg.NumMels)
	}

	numFrames := 1 + (len(samples)-cfg.FFTSize)/cfg.HopLength
	if numFrames <= 0 {
		return nil, fmt.Errorf("multimodal: audio too short (%d samples) for FFTSize %d", len(samples), cfg.FFTSize)
	}

	window := HannWindow(cfg.FFTSize)
	filterbank := MelFilterbank(cfg.NumMels, cfg.FFTSize, cfg.SampleRate, cfg.FMin, cfg.FMax)
	freqBins := cfg.FFTSize/2 + 1

	data := make([]float32, numFrames*cfg.NumMels)
	frame := make([]float64, cfg.FFTSize)

	for f := 0; f < numFrames; f++ {
		offset := f * cfg.HopLength

		// Apply Hann window to frame.
		for i := 0; i < cfg.FFTSize; i++ {
			frame[i] = float64(samples[offset+i]) * float64(window[i])
		}

		// Compute magnitude spectrum via DFT.
		mag := dftMagnitude(frame, freqBins)

		// Apply mel filterbank and log scaling.
		for m := 0; m < cfg.NumMels; m++ {
			var energy float64
			for k := 0; k < freqBins; k++ {
				energy += mag[k] * float64(filterbank[m][k])
			}
			data[f*cfg.NumMels+m] = float32(math.Log(energy + 1e-6))
		}
	}

	return &MelSpectrogram{
		Data:      data,
		NumFrames: numFrames,
		NumMels:   cfg.NumMels,
	}, nil
}

// NormalizeAudio normalizes audio samples to the range [-1, 1] by dividing
// by the maximum absolute value. Returns a new slice without modifying the input.
func NormalizeAudio(samples []float32) []float32 {
	if len(samples) == 0 {
		return nil
	}

	var maxAbs float32
	for _, s := range samples {
		a := s
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}

	out := make([]float32, len(samples))
	if maxAbs == 0 {
		copy(out, samples)
		return out
	}

	for i, s := range samples {
		out[i] = s / maxAbs
	}
	return out
}

// HannWindow returns Hann window coefficients of the given size.
// w[n] = 0.5 * (1 - cos(2*pi*n / (size-1)))
func HannWindow(size int) []float32 {
	w := make([]float32, size)
	if size <= 1 {
		if size == 1 {
			w[0] = 1
		}
		return w
	}
	factor := 2 * math.Pi / float64(size-1)
	for i := 0; i < size; i++ {
		w[i] = float32(0.5 * (1.0 - math.Cos(factor*float64(i))))
	}
	return w
}

// MelFilterbank computes a bank of triangular mel-scale filters.
// Returns a matrix of shape [numMels, fftSize/2+1] where each row is a
// triangular filter spanning from its lower to upper mel-band edge.
func MelFilterbank(numMels, fftSize, sampleRate int, fMin, fMax float32) [][]float32 {
	freqBins := fftSize/2 + 1
	melMin := hzToMel(float64(fMin))
	melMax := hzToMel(float64(fMax))

	// Create numMels+2 equally spaced points in mel space.
	melPoints := make([]float64, numMels+2)
	step := (melMax - melMin) / float64(numMels+1)
	for i := range melPoints {
		melPoints[i] = melMin + float64(i)*step
	}

	// Convert mel points to FFT bin indices.
	binIndices := make([]float64, numMels+2)
	for i, m := range melPoints {
		freq := melToHz(m)
		binIndices[i] = freq * float64(fftSize) / float64(sampleRate)
	}

	fb := make([][]float32, numMels)
	for m := 0; m < numMels; m++ {
		fb[m] = make([]float32, freqBins)
		lower := binIndices[m]
		center := binIndices[m+1]
		upper := binIndices[m+2]

		for k := 0; k < freqBins; k++ {
			fk := float64(k)
			if fk >= lower && fk <= center && center > lower {
				fb[m][k] = float32((fk - lower) / (center - lower))
			} else if fk > center && fk <= upper && upper > center {
				fb[m][k] = float32((upper - fk) / (upper - center))
			}
		}
	}
	return fb
}

// dftMagnitude computes the magnitude of the first freqBins frequency bins
// using a direct DFT computation. This avoids external FFT dependencies.
func dftMagnitude(frame []float64, freqBins int) []float64 {
	n := len(frame)
	mag := make([]float64, freqBins)
	for k := 0; k < freqBins; k++ {
		var real, imag float64
		w := -2.0 * math.Pi * float64(k) / float64(n)
		for t := 0; t < n; t++ {
			angle := w * float64(t)
			real += frame[t] * math.Cos(angle)
			imag += frame[t] * math.Sin(angle)
		}
		mag[k] = math.Sqrt(real*real + imag*imag)
	}
	return mag
}

// hzToMel converts a frequency in Hz to the mel scale.
func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// melToHz converts a mel-scale value back to Hz.
func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}
