package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/layers/audio"
)

// TranscribeCommand implements the "transcribe" CLI command for
// speech-to-text transcription using Voxtral or Whisper models.
type TranscribeCommand struct {
	out io.Writer
}

// NewTranscribeCommand creates a new TranscribeCommand.
func NewTranscribeCommand(out io.Writer) *TranscribeCommand {
	if out == nil {
		out = os.Stdout
	}
	return &TranscribeCommand{out: out}
}

// Name implements Command.Name.
func (c *TranscribeCommand) Name() string { return "transcribe" }

// Description implements Command.Description.
func (c *TranscribeCommand) Description() string {
	return "Transcribe audio to text using a speech-to-text model"
}

// Usage implements Command.Usage.
func (c *TranscribeCommand) Usage() string {
	return "transcribe [OPTIONS]"
}

// Examples implements Command.Examples.
func (c *TranscribeCommand) Examples() []string {
	return []string{
		"transcribe --model voxtral --audio recording.wav",
		"transcribe --model voxtral --audio interview.wav --mels 128",
	}
}

// Run implements Command.Run.
func (c *TranscribeCommand) Run(ctx context.Context, args []string) error {
	var modelPath, audioPath string
	numMels := 128

	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--model":
			v, err := nextVal("--model")
			if err != nil {
				return err
			}
			modelPath = v
		case "--audio":
			v, err := nextVal("--audio")
			if err != nil {
				return err
			}
			audioPath = v
		case "--mels":
			v, err := nextVal("--mels")
			if err != nil {
				return err
			}
			n := 0
			for _, ch := range v {
				if ch < '0' || ch > '9' {
					return fmt.Errorf("--mels must be a positive integer, got %q", v)
				}
				n = n*10 + int(ch-'0')
			}
			numMels = n
		default:
			if strings.HasPrefix(arg, "--") {
				return fmt.Errorf("unknown flag: %s", arg)
			}
		}
	}

	if modelPath == "" {
		return fmt.Errorf("--model is required")
	}
	if audioPath == "" {
		return fmt.Errorf("--audio is required")
	}

	// Read audio file.
	wavData, err := os.ReadFile(audioPath)
	if err != nil {
		return fmt.Errorf("read audio: %w", err)
	}

	// Parse WAV.
	samples, sampleRate, err := audio.ParseWAV(wavData)
	if err != nil {
		return fmt.Errorf("parse WAV: %w", err)
	}
	fmt.Fprintf(c.out, "Audio: %d samples at %d Hz (%.1fs)\n",
		len(samples), sampleRate, float64(len(samples))/float64(sampleRate))

	// Extract mel spectrogram.
	ext := audio.NewMelExtractor(audio.MelConfig{
		SampleRate: sampleRate,
		NumMels:    numMels,
	})

	// Chunk audio into 30-second segments.
	chunks := audio.ChunkAudio(samples, sampleRate, 30.0)
	fmt.Fprintf(c.out, "Chunks: %d (30s each)\n", len(chunks))

	for i, chunk := range chunks {
		mel, err := ext.Extract(chunk)
		if err != nil {
			return fmt.Errorf("mel extraction chunk %d: %w", i, err)
		}
		fmt.Fprintf(c.out, "Chunk %d: mel shape %v\n", i, mel.Shape())
	}

	// Load model.
	fmt.Fprintf(c.out, "Loading model: %s\n", modelPath)
	model, err := inference.LoadFile(modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	defer model.Close()

	// Transcribe audio.
	text, err := model.Transcribe(ctx, wavData,
		inference.WithMaxTokens(256),
		inference.WithTemperature(0),
	)
	if err != nil {
		return fmt.Errorf("transcribe: %w", err)
	}

	fmt.Fprintln(c.out, text)
	return nil
}
