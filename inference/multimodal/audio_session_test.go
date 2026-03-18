package multimodal

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/zerfoo/layers/audio"
)

// newTestAudioSession creates a small Whisper encoder and audio session for testing.
// hiddenDim is the encoder hidden size, embedDim is the language model embedding size.
func newTestAudioSession(t *testing.T, hiddenDim, embedDim int) (*AudioTextSession[float32], audio.WhisperEncoderConfig) {
	t.Helper()

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	encCfg := audio.WhisperEncoderConfig{
		NumMels:    80,
		HiddenDim:  hiddenDim,
		NumHeads:   2,
		NumLayers:  1,
		KernelSize: 3,
	}

	enc, err := audio.NewWhisperEncoder[float32]("test_whisper", engine, ops, encCfg)
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}

	connector := NewProjectionConnector[float32](ConnectorConfig{
		VisionDim: hiddenDim,
		TextDim:   embedDim,
	}, engine)

	// Set projection weights to scaled identity so signal passes through.
	projWeights := make([]float32, hiddenDim*embedDim)
	for i := 0; i < hiddenDim && i < embedDim; i++ {
		projWeights[i*embedDim+i] = 1.0
	}
	if err := connector.LoadWeights(projWeights); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}

	sessCfg := AudioSessionConfig{
		AudioCfg:       DefaultAudioConfig(),
		AudioTokenID:   88888,
		MaxAudioTokens: 0,
		EmbedDim:       embedDim,
	}

	session, err := NewAudioTextSession[float32](sessCfg, engine, ops, enc, connector)
	if err != nil {
		t.Fatalf("NewAudioTextSession: %v", err)
	}

	return session, encCfg
}

// generateSineWave produces a mono sine wave of the given frequency and duration.
func generateSineWave(freq float64, sampleRate, numSamples int) []float32 {
	samples := make([]float32, numSamples)
	for i := range samples {
		samples[i] = float32(math.Sin(2.0 * math.Pi * freq * float64(i) / float64(sampleRate)))
	}
	return samples
}

// computeAudioFrames calculates the number of downsampled frames the Whisper
// encoder produces for a given number of audio samples.
func computeAudioFrames(numSamples int, cfg AudioConfig, kernelSize int) int {
	// Mel spectrogram frames.
	numFrames := 1 + (numSamples-cfg.FFTSize)/cfg.HopLength

	// Conv1 with stride=2, padding=(kernelSize-1)/2.
	padding := (kernelSize - 1) / 2
	after1 := (numFrames+2*padding-kernelSize)/2 + 1

	// Conv2 with stride=2, same padding.
	after2 := (after1+2*padding-kernelSize)/2 + 1

	return after2
}

func TestAudioTextSession(t *testing.T) {
	const (
		hiddenDim  = 16
		embedDim   = 16
		sampleRate = 16000
	)

	session, encCfg := newTestAudioSession(t, hiddenDim, embedDim)

	// Generate 0.5 seconds of 440Hz sine wave.
	numSamples := sampleRate / 2
	pcm := generateSineWave(440.0, sampleRate, numSamples)

	// Compute expected audio frames after encoder downsampling.
	audioFrames := computeAudioFrames(numSamples, session.cfg.AudioCfg, encCfg.KernelSize)
	if audioFrames <= 0 {
		t.Fatalf("computed audioFrames = %d, need positive", audioFrames)
	}

	// Build token sequence: [BOS, <audio>..., EOS].
	tokenIDs := make([]int, audioFrames+2)
	tokenIDs[0] = 1 // BOS
	for i := 1; i <= audioFrames; i++ {
		tokenIDs[i] = 88888
	}
	tokenIDs[audioFrames+1] = 2 // EOS

	seqLen := len(tokenIDs)
	textEmbeds := make([]float32, seqLen*embedDim)

	input := AudioTextInput{
		PCMAudio:       pcm,
		TokenIDs:       tokenIDs,
		TextEmbeddings: textEmbeds,
	}

	output, err := session.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if output.SeqLen != seqLen {
		t.Errorf("SeqLen = %d, want %d", output.SeqLen, seqLen)
	}
	if output.EmbedDim != embedDim {
		t.Errorf("EmbedDim = %d, want %d", output.EmbedDim, embedDim)
	}
	if output.AudioFrames != audioFrames {
		t.Errorf("AudioFrames = %d, want %d", output.AudioFrames, audioFrames)
	}
	if len(output.MergedEmbeddings) != seqLen*embedDim {
		t.Errorf("MergedEmbeddings length = %d, want %d", len(output.MergedEmbeddings), seqLen*embedDim)
	}

	// Verify merged embeddings contain non-zero values at audio positions.
	hasNonZero := false
	for i := 1; i <= audioFrames; i++ {
		for d := 0; d < embedDim; d++ {
			if output.MergedEmbeddings[i*embedDim+d] != 0 {
				hasNonZero = true
				break
			}
		}
		if hasNonZero {
			break
		}
	}
	if !hasNonZero {
		t.Error("audio positions in merged embeddings are all zeros — audio signal lost")
	}

	// Verify all values are finite.
	for i, v := range output.MergedEmbeddings {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("MergedEmbeddings[%d] = %f, want finite", i, v)
		}
	}
}

func TestAudioTextSessionErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	enc, err := audio.NewWhisperEncoder[float32]("test", engine, ops, audio.WhisperEncoderConfig{
		NumMels:    80,
		HiddenDim:  16,
		NumHeads:   2,
		NumLayers:  1,
		KernelSize: 3,
	})
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}

	connector := NewProjectionConnector[float32](ConnectorConfig{
		VisionDim: 16,
		TextDim:   16,
	}, engine)

	t.Run("nil_encoder", func(t *testing.T) {
		_, err := NewAudioTextSession[float32](AudioSessionConfig{EmbedDim: 16}, engine, ops, nil, connector)
		if err == nil {
			t.Error("expected error for nil encoder")
		}
	})

	t.Run("nil_connector", func(t *testing.T) {
		_, err := NewAudioTextSession[float32](AudioSessionConfig{EmbedDim: 16}, engine, ops, enc, nil)
		if err == nil {
			t.Error("expected error for nil connector")
		}
	})

	t.Run("zero_embed_dim", func(t *testing.T) {
		_, err := NewAudioTextSession[float32](AudioSessionConfig{EmbedDim: 0}, engine, ops, enc, connector)
		if err == nil {
			t.Error("expected error for zero EmbedDim")
		}
	})

	session, err := NewAudioTextSession[float32](AudioSessionConfig{
		AudioCfg:     DefaultAudioConfig(),
		AudioTokenID: 88888,
		EmbedDim:     16,
	}, engine, ops, enc, connector)
	if err != nil {
		t.Fatalf("NewAudioTextSession: %v", err)
	}

	t.Run("empty_pcm", func(t *testing.T) {
		_, err := session.Run(context.Background(), AudioTextInput{
			PCMAudio: nil,
			TokenIDs: []int{1, 2},
		})
		if err == nil {
			t.Error("expected error for empty PCM")
		}
	})

	t.Run("empty_tokens", func(t *testing.T) {
		_, err := session.Run(context.Background(), AudioTextInput{
			PCMAudio: []float32{0.1, 0.2, 0.3},
			TokenIDs: nil,
		})
		if err == nil {
			t.Error("expected error for empty token IDs")
		}
	})

	t.Run("no_audio_placeholders", func(t *testing.T) {
		// Enough audio for mel spectrogram extraction but no audio token placeholders.
		pcm := generateSineWave(440.0, 16000, 16000)
		_, err := session.Run(context.Background(), AudioTextInput{
			PCMAudio:       pcm,
			TokenIDs:       []int{1, 2, 3},
			TextEmbeddings: make([]float32, 3*16),
		})
		if err == nil {
			t.Error("expected error when no audio token placeholders present")
		}
	})
}

func TestNumAudioTokens(t *testing.T) {
	tokens := []int{1, 88888, 88888, 2, 88888, 3}
	got := NumAudioTokens(tokens, 88888)
	if got != 3 {
		t.Errorf("NumAudioTokens = %d, want 3", got)
	}

	got = NumAudioTokens(tokens, 99999)
	if got != 0 {
		t.Errorf("NumAudioTokens for missing token = %d, want 0", got)
	}

	got = NumAudioTokens(nil, 88888)
	if got != 0 {
		t.Errorf("NumAudioTokens for nil = %d, want 0", got)
	}
}
