package audio

import (
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// MelConfig configures mel spectrogram extraction.
type MelConfig struct {
	SampleRate int // audio sample rate in Hz (default 16000)
	FFTSize    int // FFT window size (default 400)
	HopLength  int // hop between windows (default 160)
	NumMels    int // number of mel filter banks (default 128)
}

// DefaultMelConfig returns defaults matching Whisper/Voxtral.
func DefaultMelConfig() MelConfig {
	return MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    128,
	}
}

// MelExtractor extracts log mel spectrograms from raw audio samples.
type MelExtractor struct {
	cfg        MelConfig
	window     []float64
	filterbank [][]float64 // [numMels][fftSize/2+1]
}

// NewMelExtractor creates a mel spectrogram extractor.
func NewMelExtractor(cfg MelConfig) *MelExtractor {
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 16000
	}
	if cfg.FFTSize == 0 {
		cfg.FFTSize = 400
	}
	if cfg.HopLength == 0 {
		cfg.HopLength = 160
	}
	if cfg.NumMels == 0 {
		cfg.NumMels = 128
	}

	// Hann window.
	window := make([]float64, cfg.FFTSize)
	for i := range window {
		window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(cfg.FFTSize)))
	}

	// Mel filterbank.
	numFreqs := cfg.FFTSize/2 + 1
	filterbank := buildMelFilterbank(cfg.NumMels, numFreqs, cfg.SampleRate)

	return &MelExtractor{
		cfg:        cfg,
		window:     window,
		filterbank: filterbank,
	}
}

// Extract computes a log mel spectrogram from raw float32 audio samples.
// samples should be in [-1, 1] range at the configured sample rate.
// Returns a tensor of shape [numMels, numFrames].
func (m *MelExtractor) Extract(samples []float32) (*tensor.TensorNumeric[float32], error) {
	n := len(samples)
	numFrames := 1 + (n-m.cfg.FFTSize)/m.cfg.HopLength
	if numFrames < 1 {
		numFrames = 1
	}

	numFreqs := m.cfg.FFTSize/2 + 1
	numMels := m.cfg.NumMels

	// Compute STFT magnitude spectrum per frame.
	magnitudes := make([]float64, numFrames*numFreqs)
	for f := range numFrames {
		offset := f * m.cfg.HopLength
		for k := range numFreqs {
			var re, im float64
			for j := range m.cfg.FFTSize {
				idx := offset + j
				var sample float64
				if idx < n {
					sample = float64(samples[idx])
				}
				windowed := sample * m.window[j]
				angle := -2.0 * math.Pi * float64(k) * float64(j) / float64(m.cfg.FFTSize)
				re += windowed * math.Cos(angle)
				im += windowed * math.Sin(angle)
			}
			magnitudes[f*numFreqs+k] = re*re + im*im // power spectrum
		}
	}

	// Apply mel filterbank and log.
	melData := make([]float32, numMels*numFrames)
	for mel := range numMels {
		for f := range numFrames {
			var sum float64
			for k := range numFreqs {
				sum += m.filterbank[mel][k] * magnitudes[f*numFreqs+k]
			}
			// Log mel with floor to avoid log(0).
			if sum < 1e-10 {
				sum = 1e-10
			}
			melData[mel*numFrames+f] = float32(math.Log10(sum))
		}
	}

	return tensor.New[float32]([]int{numMels, numFrames}, melData)
}

// Config returns the extractor's configuration.
func (m *MelExtractor) Config() MelConfig { return m.cfg }

// hzToMel converts frequency in Hz to mel scale.
func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// melToHz converts mel scale back to Hz.
func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// buildMelFilterbank creates triangular mel filter banks.
func buildMelFilterbank(numMels, numFreqs, sampleRate int) [][]float64 {
	fMin := 0.0
	fMax := float64(sampleRate) / 2.0

	melMin := hzToMel(fMin)
	melMax := hzToMel(fMax)

	// numMels+2 points evenly spaced in mel scale.
	melPoints := make([]float64, numMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(numMels+1)
	}

	// Convert back to Hz, then to FFT bin indices.
	binIndices := make([]float64, numMels+2)
	for i, m := range melPoints {
		hz := melToHz(m)
		binIndices[i] = hz * float64(numFreqs-1) * 2.0 / float64(sampleRate)
	}

	// Build triangular filters.
	filterbank := make([][]float64, numMels)
	for i := range numMels {
		filterbank[i] = make([]float64, numFreqs)
		left := binIndices[i]
		center := binIndices[i+1]
		right := binIndices[i+2]

		for k := range numFreqs {
			fk := float64(k)
			if fk >= left && fk < center && center > left {
				filterbank[i][k] = (fk - left) / (center - left)
			} else if fk >= center && fk <= right && right > center {
				filterbank[i][k] = (right - fk) / (right - center)
			}
		}
	}

	return filterbank
}

// ChunkAudio splits audio samples into fixed-length chunks.
// The last chunk is padded with silence (zeros) to maxSeconds.
func ChunkAudio(samples []float32, sampleRate int, maxSeconds float64) [][]float32 {
	chunkSamples := int(float64(sampleRate) * maxSeconds)
	if chunkSamples <= 0 {
		return [][]float32{samples}
	}

	var chunks [][]float32
	for i := 0; i < len(samples); i += chunkSamples {
		end := i + chunkSamples
		if end > len(samples) {
			// Pad with silence.
			chunk := make([]float32, chunkSamples)
			copy(chunk, samples[i:])
			chunks = append(chunks, chunk)
		} else {
			chunk := make([]float32, chunkSamples)
			copy(chunk, samples[i:end])
			chunks = append(chunks, chunk)
		}
	}

	if len(chunks) == 0 {
		chunks = append(chunks, make([]float32, chunkSamples))
	}

	return chunks
}

// ParseWAV reads a 16-bit PCM WAV file and returns float32 samples in [-1, 1].
// Only supports mono 16-bit PCM. Returns samples and sample rate.
func ParseWAV(data []byte) (samples []float32, sampleRate int, err error) {
	if len(data) < 44 {
		return nil, 0, errShortWAV
	}

	// RIFF header.
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, 0, errNotWAV
	}

	// Find fmt and data chunks.
	pos := 12
	var numChannels, bitsPerSample int
	var dataStart, dataLen int

	for pos+8 <= len(data) {
		chunkID := string(data[pos : pos+4])
		chunkSize := int(data[pos+4]) | int(data[pos+5])<<8 | int(data[pos+6])<<16 | int(data[pos+7])<<24

		if chunkID == "fmt " && pos+24 <= len(data) {
			// audioFormat := int(data[pos+8]) | int(data[pos+9])<<8
			numChannels = int(data[pos+10]) | int(data[pos+11])<<8
			sampleRate = int(data[pos+12]) | int(data[pos+13])<<8 | int(data[pos+14])<<16 | int(data[pos+15])<<24
			bitsPerSample = int(data[pos+22]) | int(data[pos+23])<<8
		} else if chunkID == "data" {
			dataStart = pos + 8
			dataLen = chunkSize
		}

		pos += 8 + chunkSize
		if chunkSize%2 != 0 {
			pos++ // padding byte
		}
	}

	if dataStart == 0 || sampleRate == 0 {
		return nil, 0, errNoData
	}

	if bitsPerSample != 16 {
		return nil, 0, errNot16Bit
	}

	// Convert int16 samples to float32.
	bytesPerSample := bitsPerSample / 8
	numSamples := dataLen / bytesPerSample
	if dataStart+dataLen > len(data) {
		numSamples = (len(data) - dataStart) / bytesPerSample
	}

	samples = make([]float32, numSamples/numChannels)
	for i := range samples {
		idx := dataStart + i*numChannels*bytesPerSample
		if idx+1 >= len(data) {
			break
		}
		raw := int16(data[idx]) | int16(data[idx+1])<<8
		samples[i] = float32(raw) / 32768.0
	}

	return samples, sampleRate, nil
}

type melError string

func (e melError) Error() string { return string(e) }

const (
	errShortWAV melError = "WAV data too short"
	errNotWAV   melError = "not a WAV file"
	errNoData   melError = "no data chunk found"
	errNot16Bit melError = "only 16-bit PCM WAV supported"
)
