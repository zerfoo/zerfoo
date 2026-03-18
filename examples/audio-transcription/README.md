# Audio Transcription

Speech-to-text transcription using the Zerfoo OpenAI-compatible API server.

## How it works

1. Loads a Whisper GGUF model via `inference.LoadFile`
2. Implements the `serve.Transcriber` interface to bridge the model to the API server
3. Starts an in-process HTTP server using `serve.NewServer` with `serve.WithTranscriber`
4. Sends the audio file to `/v1/audio/transcriptions` using multipart form upload
5. Prints the transcription result

This demonstrates embedding a full OpenAI-compatible transcription service inside a Go application. The same endpoint works with any OpenAI client library.

## Prerequisites

Requires a Whisper-architecture GGUF model file.

## Usage

```bash
go build -o audio-transcription ./examples/audio-transcription/

# Transcribe an audio file
./audio-transcription --model path/to/whisper.gguf --audio recording.wav

# With language hint
./audio-transcription --model path/to/whisper.gguf --audio recording.mp3 --language en

# With GPU
./audio-transcription --model path/to/whisper.gguf --device cuda --audio recording.wav
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to a Whisper GGUF model file |
| `--audio` | (required) | Path to an audio file (WAV, MP3, FLAC, OGG) |
| `--device` | cpu | Compute device: "cpu", "cuda" |
| `--language` | (auto) | Optional language hint (e.g., "en", "fr") |
