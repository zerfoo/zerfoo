package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/zerfoo/generate"
	tokenizer "github.com/zerfoo/ztoken"
)

// buildTranscribeWAV builds a minimal mono 16-bit PCM WAV file with the given
// number of samples at 16000 Hz. The samples are a simple sine wave.
func buildTranscribeWAV(numSamples int) []byte {
	sampleRate := 16000
	numChannels := 1
	bitsPerSample := 16
	dataSize := numSamples * 2
	fileSize := 36 + dataSize

	b := make([]byte, 44+dataSize)
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

	for i := 0; i < numSamples; i++ {
		s := int16(8000 * math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)))
		b[44+i*2] = byte(s)
		b[44+i*2+1] = byte(s >> 8)
	}
	return b
}

// buildVoxtralTestModel constructs a minimal Voxtral Model with synthetic weights
// for testing Transcribe.
func buildVoxtralTestModel(t *testing.T) *Model {
	t.Helper()
	cfg, vc := testVoxtralConfig()
	tensors := makeVoxtralTestTensors(cfg, vc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg.AudioHiddenSize = vc.AudioHiddenDim
	cfg.AudioNumLayers = vc.AudioNumLayers
	cfg.AudioNumHeads = vc.AudioNumHeads
	cfg.AudioNumMels = vc.AudioNumMels
	cfg.AudioIntermediateSize = vc.AudioIntermediateSize
	cfg.AudioProjectorStackFactor = vc.StackFactor

	g, _, err := BuildVoxtralModel(vc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildVoxtralModel: %v", err)
	}

	tok := tokenizer.NewWhitespaceTokenizer()

	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  cfg.VocabSize,
		MaxSeqLen:  64,
		EOSTokenID: 0,
		NumLayers:  cfg.NumLayers,
	})

	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    engine,
		config: ModelMetadata{
			Architecture: "voxtral",
			VocabSize:    cfg.VocabSize,
			EOSTokenID:   0,
			AudioNumMels: vc.AudioNumMels,
		},
	}
}

func TestModelTranscribe_ProducesOutput(t *testing.T) {
	model := buildVoxtralTestModel(t)

	// 1 second of audio at 16kHz — enough for several mel frames.
	wav := buildTranscribeWAV(16000)

	text, err := model.Transcribe(context.Background(), wav)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	// The model has random weights so the transcript is nonsense, but it must not error.
	_ = text
}

func TestModelTranscribe_InvalidWAV(t *testing.T) {
	model := buildVoxtralTestModel(t)

	_, err := model.Transcribe(context.Background(), []byte("not a wav"))
	if err == nil {
		t.Fatal("expected error for invalid WAV")
	}
}

func TestModelTranscribe_EmptyAudio(t *testing.T) {
	model := buildVoxtralTestModel(t)

	// WAV with zero data samples.
	wav := buildTranscribeWAV(0)

	_, err := model.Transcribe(context.Background(), wav)
	// Should return an error (no audio data) or empty string.
	if err != nil {
		// Acceptable: no audio data -> no chunks.
		return
	}
}
