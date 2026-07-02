package multimodal

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/audio"
)

// AudioSessionConfig holds configuration for an audio+text inference session.
type AudioSessionConfig struct {
	// AudioCfg controls mel-spectrogram extraction parameters.
	AudioCfg AudioConfig

	// AudioTokenID is the token ID used as a placeholder for audio frames
	// in the text token sequence.
	AudioTokenID int

	// MaxAudioTokens is the maximum number of audio tokens allowed in a
	// single sequence. Zero means no limit.
	MaxAudioTokens int

	// EmbedDim is the embedding dimension of the language model.
	EmbedDim int
}

// AudioTextSession orchestrates audio+text inference: it accepts raw PCM audio
// and a text prompt, runs mel-spectrogram extraction, encodes audio through a
// Whisper-style encoder, projects audio embeddings into text space, merges them
// with text embeddings, and produces a unified embedding sequence ready for
// language model decoding.
type AudioTextSession[T tensor.Numeric] struct {
	cfg       AudioSessionConfig
	engine    compute.Engine[T]
	ops       numeric.Arithmetic[T]
	encoder   *audio.WhisperEncoder[T]
	connector *ProjectionConnector[T]
}

// NewAudioTextSession creates a new audio+text inference session.
// The encoder produces embeddings of dimension audioDim, which the connector
// projects into the language model's embedding space (cfg.EmbedDim).
func NewAudioTextSession[T tensor.Numeric](
	cfg AudioSessionConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	encoder *audio.WhisperEncoder[T],
	connector *ProjectionConnector[T],
) (*AudioTextSession[T], error) {
	if cfg.EmbedDim <= 0 {
		return nil, fmt.Errorf("multimodal: EmbedDim must be positive, got %d", cfg.EmbedDim)
	}
	if encoder == nil {
		return nil, fmt.Errorf("multimodal: encoder must not be nil")
	}
	if connector == nil {
		return nil, fmt.Errorf("multimodal: connector must not be nil")
	}
	return &AudioTextSession[T]{
		cfg:       cfg,
		engine:    engine,
		ops:       ops,
		encoder:   encoder,
		connector: connector,
	}, nil
}

// AudioTextInput holds the inputs for a single audio+text inference call.
type AudioTextInput struct {
	// PCMAudio is raw PCM audio samples (mono, float32, at AudioCfg.SampleRate).
	PCMAudio []float32

	// TokenIDs is the tokenized text prompt, with AudioTokenID placeholders
	// where audio embeddings should be inserted.
	TokenIDs []int

	// TextEmbeddings is the text token embedding matrix of shape
	// [len(TokenIDs), EmbedDim]. Positions with AudioTokenID will be
	// replaced by projected audio embeddings.
	TextEmbeddings []float32
}

// AudioTextOutput holds the result of an audio+text inference session.
type AudioTextOutput struct {
	// MergedEmbeddings is the merged [SeqLen, EmbedDim] embedding sequence
	// with audio embeddings replacing AudioTokenID positions.
	MergedEmbeddings []float32

	// SeqLen is the sequence length of the merged output.
	SeqLen int

	// EmbedDim is the embedding dimension of the merged output.
	EmbedDim int

	// AudioFrames is the number of audio frames (downsampled) produced by
	// the encoder.
	AudioFrames int
}

// Run executes the full audio+text inference pipeline:
//  1. Normalize raw PCM audio
//  2. Extract mel-spectrogram
//  3. Run Whisper encoder to produce audio embeddings
//  4. Project audio embeddings into text embedding space
//  5. Merge audio embeddings with text embeddings at AudioTokenID positions
func (s *AudioTextSession[T]) Run(ctx context.Context, input AudioTextInput) (*AudioTextOutput, error) {
	if len(input.PCMAudio) == 0 {
		return nil, fmt.Errorf("multimodal: PCMAudio must not be empty")
	}
	if len(input.TokenIDs) == 0 {
		return nil, fmt.Errorf("multimodal: TokenIDs must not be empty")
	}

	// Step 1: Normalize audio.
	normalized := NormalizeAudio(input.PCMAudio)

	// Step 2: Extract mel-spectrogram.
	mel, err := ExtractMelSpectrogram(normalized, s.cfg.AudioCfg)
	if err != nil {
		return nil, fmt.Errorf("multimodal: mel spectrogram: %w", err)
	}

	// Step 3: Run Whisper encoder.
	// Encoder expects input shape [batch, num_mels, T_frames].
	// Convert mel data from [NumFrames, NumMels] (row-major) to [1, NumMels, NumFrames].
	encoderInput, err := s.prepareMelTensor(mel)
	if err != nil {
		return nil, fmt.Errorf("multimodal: prepare mel tensor: %w", err)
	}

	encoderOutput, err := s.encoder.Forward(ctx, encoderInput)
	if err != nil {
		return nil, fmt.Errorf("multimodal: whisper encoder: %w", err)
	}

	// Encoder output: [T_downsampled, hidden_dim]
	encShape := encoderOutput.Shape()
	audioFrames := encShape[0]
	audioDim := encShape[1]

	// Step 4: Project audio embeddings to text embedding space.
	audioEmbeds := encoderOutput.Data()
	projected, err := s.connector.Project(audioEmbeds, audioFrames)
	if err != nil {
		return nil, fmt.Errorf("multimodal: audio projection: %w", err)
	}

	// Step 5: Count audio token placeholders and validate.
	numAudioTokens := NumAudioTokens(input.TokenIDs, s.cfg.AudioTokenID)
	if numAudioTokens == 0 {
		return nil, fmt.Errorf("multimodal: no audio token placeholders (token ID %d) found in TokenIDs", s.cfg.AudioTokenID)
	}
	if numAudioTokens != audioFrames {
		return nil, fmt.Errorf("multimodal: %d audio token positions but encoder produced %d frames (audio dim %d)", numAudioTokens, audioFrames, audioDim)
	}

	// Convert projected []T to []float32 for MergeEmbeddings.
	projectedF32 := make([]float32, len(projected))
	for i, v := range projected {
		projectedF32[i] = float32(v)
	}

	// Step 6: Merge audio embeddings with text embeddings.
	mergeCfg := MergeConfig{
		ImageTokenID:   s.cfg.AudioTokenID,
		MaxImageTokens: s.cfg.MaxAudioTokens,
		EmbedDim:       s.cfg.EmbedDim,
	}

	merged, err := MergeEmbeddings(input.TextEmbeddings, projectedF32, input.TokenIDs, mergeCfg)
	if err != nil {
		return nil, fmt.Errorf("multimodal: merge embeddings: %w", err)
	}

	return &AudioTextOutput{
		MergedEmbeddings: merged.Embeddings,
		SeqLen:           merged.SeqLen,
		EmbedDim:         merged.EmbedDim,
		AudioFrames:      audioFrames,
	}, nil
}

// prepareMelTensor transposes mel spectrogram from [NumFrames, NumMels] to
// [1, NumMels, NumFrames] for the Whisper encoder input.
func (s *AudioTextSession[T]) prepareMelTensor(mel *MelSpectrogram) (*tensor.TensorNumeric[T], error) {
	transposed := make([]T, mel.NumFrames*mel.NumMels)
	for f := 0; f < mel.NumFrames; f++ {
		for m := 0; m < mel.NumMels; m++ {
			// Source: [f, m] -> Dest: [m, f]
			transposed[m*mel.NumFrames+f] = T(mel.Data[f*mel.NumMels+m])
		}
	}
	return tensor.New[T]([]int{1, mel.NumMels, mel.NumFrames}, transposed)
}

// NumAudioTokens counts how many entries in tokenIDs equal audioTokenID.
func NumAudioTokens(tokenIDs []int, audioTokenID int) int {
	n := 0
	for _, id := range tokenIDs {
		if id == audioTokenID {
			n++
		}
	}
	return n
}
