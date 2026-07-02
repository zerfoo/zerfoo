package audio

import (
	"math"
	"testing"
)

func TestMelExtractor_FilterbankShape(t *testing.T) {
	cfg := DefaultMelConfig()
	ext := NewMelExtractor(cfg)
	if len(ext.filterbank) != cfg.NumMels {
		t.Errorf("filterbank has %d mels, want %d", len(ext.filterbank), cfg.NumMels)
	}
	numFreqs := cfg.FFTSize/2 + 1
	for i, f := range ext.filterbank {
		if len(f) != numFreqs {
			t.Errorf("filterbank[%d] has %d bins, want %d", i, len(f), numFreqs)
		}
	}
}

func TestMelExtractor_ExtractShape(t *testing.T) {
	cfg := DefaultMelConfig()
	ext := NewMelExtractor(cfg)

	// 1 second of silence at 16kHz.
	samples := make([]float32, 16000)
	mel, err := ext.Extract(samples)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	shape := mel.Shape()
	if len(shape) != 2 {
		t.Fatalf("shape rank = %d, want 2", len(shape))
	}
	if shape[0] != cfg.NumMels {
		t.Errorf("shape[0] = %d, want %d mels", shape[0], cfg.NumMels)
	}
	expectedFrames := 1 + (16000-cfg.FFTSize)/cfg.HopLength
	if shape[1] != expectedFrames {
		t.Errorf("shape[1] = %d, want %d frames", shape[1], expectedFrames)
	}
}

func TestMelExtractor_SineWave(t *testing.T) {
	cfg := DefaultMelConfig()
	ext := NewMelExtractor(cfg)

	// 440 Hz sine wave, 0.1 seconds.
	n := 1600 // 0.1s at 16kHz
	samples := make([]float32, n)
	for i := range samples {
		samples[i] = float32(math.Sin(2.0 * math.Pi * 440.0 * float64(i) / 16000.0))
	}

	mel, err := ext.Extract(samples)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	// Output should be finite and non-zero somewhere.
	data := mel.Data()
	allZero := true
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("mel contains NaN or Inf")
		}
		if v != 0 {
			allZero = false
		}
	}
	if allZero {
		t.Error("mel is all zeros for a sine wave input")
	}
}

func TestMelExtractor_80Mels(t *testing.T) {
	cfg := MelConfig{NumMels: 80}
	ext := NewMelExtractor(cfg)
	if len(ext.filterbank) != 80 {
		t.Errorf("filterbank has %d mels, want 80", len(ext.filterbank))
	}
}

func TestChunkAudio(t *testing.T) {
	tests := []struct {
		name       string
		numSamples int
		maxSec     float64
		wantChunks int
		wantLen    int
	}{
		{"exact", 48000, 1.0, 3, 16000},
		{"short", 8000, 1.0, 1, 16000},
		{"partial", 24000, 1.0, 2, 16000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			samples := make([]float32, tt.numSamples)
			chunks := ChunkAudio(samples, 16000, tt.maxSec)
			if len(chunks) != tt.wantChunks {
				t.Errorf("got %d chunks, want %d", len(chunks), tt.wantChunks)
			}
			for i, c := range chunks {
				if len(c) != tt.wantLen {
					t.Errorf("chunk[%d] len = %d, want %d", i, len(c), tt.wantLen)
				}
			}
		})
	}
}

func TestParseWAV_InvalidHeader(t *testing.T) {
	_, _, err := ParseWAV([]byte("not a wav"))
	if err == nil {
		t.Error("expected error for invalid WAV")
	}
}

func TestParseWAV_SyntheticMono16(t *testing.T) {
	// Build a minimal valid WAV: 4 samples of mono 16-bit PCM at 16000 Hz.
	wav := buildTestWAV(16000, 1, 16, []int16{0, 16384, -16384, 0})

	samples, sr, err := ParseWAV(wav)
	if err != nil {
		t.Fatalf("ParseWAV: %v", err)
	}
	if sr != 16000 {
		t.Errorf("sample rate = %d, want 16000", sr)
	}
	if len(samples) != 4 {
		t.Fatalf("got %d samples, want 4", len(samples))
	}
	// 16384 / 32768 = 0.5
	if samples[1] < 0.49 || samples[1] > 0.51 {
		t.Errorf("samples[1] = %f, want ~0.5", samples[1])
	}
	if samples[2] < -0.51 || samples[2] > -0.49 {
		t.Errorf("samples[2] = %f, want ~-0.5", samples[2])
	}
}

func buildTestWAV(sampleRate, numChannels, bitsPerSample int, data []int16) []byte {
	dataSize := len(data) * 2
	fileSize := 36 + dataSize

	b := make([]byte, 44+len(data)*2)
	copy(b[0:4], "RIFF")
	b[4] = byte(fileSize)
	b[5] = byte(fileSize >> 8)
	b[6] = byte(fileSize >> 16)
	b[7] = byte(fileSize >> 24)
	copy(b[8:12], "WAVE")

	// fmt chunk.
	copy(b[12:16], "fmt ")
	b[16] = 16 // chunk size
	b[20] = 1  // PCM
	b[22] = byte(numChannels)
	b[24] = byte(sampleRate)
	b[25] = byte(sampleRate >> 8)
	b[26] = byte(sampleRate >> 16)
	b[27] = byte(sampleRate >> 24)
	byteRate := sampleRate * numChannels * bitsPerSample / 8
	b[28] = byte(byteRate)
	b[29] = byte(byteRate >> 8)
	b[30] = byte(byteRate >> 16)
	b[31] = byte(byteRate >> 24)
	blockAlign := numChannels * bitsPerSample / 8
	b[32] = byte(blockAlign)
	b[33] = byte(blockAlign >> 8)
	b[34] = byte(bitsPerSample)
	b[35] = byte(bitsPerSample >> 8)

	// data chunk.
	copy(b[36:40], "data")
	b[40] = byte(dataSize)
	b[41] = byte(dataSize >> 8)
	b[42] = byte(dataSize >> 16)
	b[43] = byte(dataSize >> 24)

	for i, s := range data {
		b[44+i*2] = byte(s)
		b[44+i*2+1] = byte(s >> 8)
	}

	return b
}

func TestHzToMelRoundtrip(t *testing.T) {
	for _, hz := range []float64{0, 440, 1000, 4000, 8000} {
		mel := hzToMel(hz)
		got := melToHz(mel)
		if math.Abs(got-hz) > 0.01 {
			t.Errorf("roundtrip(%f): got %f", hz, got)
		}
	}
}
