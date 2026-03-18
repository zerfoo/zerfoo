// Command audio-transcription demonstrates speech-to-text using the Zerfoo
// OpenAI-compatible API server.
//
// It starts an in-process API server with a Whisper model, then sends an audio
// file to the /v1/audio/transcriptions endpoint -- the same API that OpenAI
// clients use. This shows how to embed a full transcription service inside a
// Go application.
//
// Usage:
//
//	go build -o audio-transcription ./examples/audio-transcription/
//	./audio-transcription --model path/to/whisper.gguf --audio recording.wav
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	modelPath := flag.String("model", "", "path to a Whisper GGUF model file")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	audioPath := flag.String("audio", "", "path to an audio file (WAV, MP3, FLAC, or OGG)")
	language := flag.String("language", "", "optional language hint (e.g., 'en', 'fr')")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <whisper.gguf> --audio <recording.wav>\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" || *audioPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Read the audio file.
	audioData, err := os.ReadFile(*audioPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read audio: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Audio: %s (%d bytes)\n", *audioPath, len(audioData))

	// Load the Whisper model.
	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Create an API server with the Whisper transcriber backend.
	// In production you would implement serve.Transcriber to run the Whisper
	// inference graph; here we show the wiring pattern.
	transcriber := &whisperTranscriber{model: model}
	srv := serve.NewServer(model, serve.WithTranscriber(transcriber))

	// Start the server on a random port.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Fprintf(os.Stderr, "listen: %v\n", err)
		os.Exit(1)
	}
	defer listener.Close()

	baseURL := fmt.Sprintf("http://%s", listener.Addr().String())
	fmt.Fprintf(os.Stderr, "Server listening on %s\n", baseURL)

	httpServer := &http.Server{Handler: srv.Handler()}
	go httpServer.Serve(listener) //nolint:errcheck
	defer httpServer.Shutdown(context.Background())

	// Give the server a moment to start.
	time.Sleep(50 * time.Millisecond)

	// Send the audio file to the transcription endpoint using multipart form upload.
	transcript, err := transcribe(baseURL, audioData, *language)
	if err != nil {
		fmt.Fprintf(os.Stderr, "transcribe: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(transcript)
}

// transcribe sends an audio file to the OpenAI-compatible transcription endpoint.
func transcribe(baseURL string, audioData []byte, language string) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Write the audio file part.
	part, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}
	if _, err := part.Write(audioData); err != nil {
		return "", fmt.Errorf("write audio: %w", err)
	}

	// Add the model field (required by the OpenAI API spec).
	if err := w.WriteField("model", "whisper-1"); err != nil {
		return "", fmt.Errorf("write model field: %w", err)
	}

	if language != "" {
		if err := w.WriteField("language", language); err != nil {
			return "", fmt.Errorf("write language field: %w", err)
		}
	}
	w.Close()

	resp, err := http.Post(baseURL+"/v1/audio/transcriptions", w.FormDataContentType(), &buf)
	if err != nil {
		return "", fmt.Errorf("POST: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}

	var result struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}
	return result.Text, nil
}

// whisperTranscriber implements serve.Transcriber using the loaded model.
// In a real application, this would run the Whisper encoder-decoder graph.
// Here it demonstrates the interface wiring.
type whisperTranscriber struct {
	model *inference.Model
}

func (t *whisperTranscriber) Transcribe(ctx context.Context, audio []byte, language string) (string, error) {
	// Build a prompt that describes the transcription task.
	// A full Whisper implementation would process the audio spectrogram through
	// the encoder and decode text autoregressively. This stub demonstrates
	// the serve.Transcriber interface contract.
	prompt := "Transcribe the following audio."
	if language != "" {
		prompt = fmt.Sprintf("Transcribe the following audio in %s.", language)
	}

	result, err := t.model.Generate(ctx, prompt)
	if err != nil {
		return "", err
	}
	return result, nil
}
