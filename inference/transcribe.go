package inference

import (
	"context"
	"fmt"
	"strings"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/audio"
)

// Transcribe converts audio bytes (WAV format) to text using a speech-to-text
// model such as Voxtral. It extracts a mel spectrogram from the audio, runs
// a single forward pass through the model graph, and greedily decodes the
// output logits at each temporal position until an EOS token is encountered.
//
// This is a parallel (non-autoregressive) decode: the mel spectrogram is
// processed once through the full encoder+adapter+decoder stack, and each
// temporal position independently predicts the most likely token. For a
// causal decoder model this is an approximation — full autoregressive
// transcription requires a two-phase graph (encode once, decode autoregressively)
// which is not yet wired.
func (m *Model) Transcribe(ctx context.Context, audioBytes []byte, opts ...GenerateOption) (string, error) {
	samples, sampleRate, err := audio.ParseWAV(audioBytes)
	if err != nil {
		return "", fmt.Errorf("parse WAV: %w", err)
	}

	numMels := m.config.AudioNumMels
	if numMels == 0 {
		// Defaults: Voxtral uses 128, Whisper uses 80.
		if m.config.Architecture == "whisper" {
			numMels = 80
		} else {
			numMels = 128
		}
	}

	ext := audio.NewMelExtractor(audio.MelConfig{
		SampleRate: sampleRate,
		NumMels:    numMels,
	})

	chunks := audio.ChunkAudio(samples, sampleRate, 30.0)
	if len(chunks) == 0 {
		return "", fmt.Errorf("no audio data")
	}

	sc := buildSamplingConfig(opts)
	eosID := m.generator.Config().EOSTokenID

	g := m.generator.Graph()

	var sb strings.Builder
	for i, chunk := range chunks {
		mel2d, err := ext.Extract(chunk)
		if err != nil {
			return "", fmt.Errorf("mel extraction chunk %d: %w", i, err)
		}

		// MelExtractor returns [numMels, T]; the WhisperEncoder expects [1, numMels, T].
		s := mel2d.Shape()
		mel, reshapeErr := tensor.New[float32]([]int{1, s[0], s[1]}, mel2d.Data())
		if reshapeErr != nil {
			return "", fmt.Errorf("reshape mel chunk %d: %w", i, reshapeErr)
		}

		logits, err := g.Forward(ctx, mel)
		if err != nil {
			return "", fmt.Errorf("graph forward chunk %d: %w", i, err)
		}

		// Decode logits: shape [1, T, vocab].
		shape := logits.Shape()
		if len(shape) != 3 {
			return "", fmt.Errorf("unexpected logits shape %v", shape)
		}
		T, vocab := shape[1], shape[2]
		data := logits.Data()

		ids := make([]int, 0, T)
		for t := 0; t < T; t++ {
			offset := t * vocab
			maxIdx := 0
			maxVal := data[offset]
			for v := 1; v < vocab; v++ {
				if data[offset+v] > maxVal {
					maxVal = data[offset+v]
					maxIdx = v
				}
			}
			if eosID > 0 && maxIdx == eosID {
				break
			}
			ids = append(ids, maxIdx)
		}
		_ = sc // sampling config reserved for future temperature/top-k support

		if len(ids) == 0 {
			continue
		}

		text, err := m.tokenizer.Decode(ids)
		if err != nil {
			return "", fmt.Errorf("decode chunk %d: %w", i, err)
		}
		sb.WriteString(text)
	}

	return sb.String(), nil
}
