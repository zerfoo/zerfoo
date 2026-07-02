package serve

import (
	"context"
	"fmt"
	"io"
	"net/http"
)

// maxAudioSize is the maximum allowed size for uploaded audio files (25 MB).
const maxAudioSize = 25 * 1024 * 1024

// Transcriber converts raw audio bytes into a text transcript.
type Transcriber interface {
	Transcribe(ctx context.Context, audio []byte, language string) (string, error)
}

// WithTranscriber sets the audio transcription backend for the
// /v1/audio/transcriptions endpoint.
func WithTranscriber(t Transcriber) ServerOption {
	return func(s *Server) {
		s.transcriber = t
	}
}

// TranscriptionResponse is the /v1/audio/transcriptions JSON response.
type TranscriptionResponse struct {
	Text string `json:"text"`
}

func (s *Server) handleAudioTranscriptions(w http.ResponseWriter, r *http.Request) {
	if s.transcriber == nil {
		writeError(w, http.StatusNotImplemented, "audio transcription is not configured")
		return
	}

	// Limit request body size.
	r.Body = http.MaxBytesReader(w, r.Body, maxAudioSize+1024)

	if err := r.ParseMultipartForm(maxAudioSize); err != nil {
		s.logger.Debug("invalid multipart form", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid multipart form")
		return
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		s.logger.Debug("missing required field 'file'", "error", err.Error())
		writeError(w, http.StatusBadRequest, "missing required field 'file'")
		return
	}
	defer file.Close()

	audioData, err := io.ReadAll(io.LimitReader(file, maxAudioSize+1))
	if err != nil {
		s.logger.Debug("failed to read audio file", "error", err.Error())
		writeError(w, http.StatusBadRequest, "failed to read audio file")
		return
	}
	if len(audioData) > maxAudioSize {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("audio file exceeds maximum size of %d bytes", maxAudioSize))
		return
	}
	if len(audioData) == 0 {
		writeError(w, http.StatusBadRequest, "audio file is empty")
		return
	}

	language := r.FormValue("language")
	responseFormat := r.FormValue("response_format")
	if responseFormat == "" {
		responseFormat = "json"
	}

	transcript, err := s.transcriber.Transcribe(r.Context(), audioData, language)
	if err != nil {
		writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
		return
	}

	switch responseFormat {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(transcript)) //nolint:errcheck
	default:
		writeJSON(w, http.StatusOK, TranscriptionResponse{Text: transcript})
	}
}

// detectAudioFormat returns a human-readable format name from magic bytes.
func detectAudioFormat(data []byte) string {
	if len(data) < 4 {
		return "unknown"
	}
	// WAV: "RIFF"
	if data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' {
		return "wav"
	}
	// FLAC: "fLaC"
	if data[0] == 'f' && data[1] == 'L' && data[2] == 'a' && data[3] == 'C' {
		return "flac"
	}
	// OGG: "OggS"
	if data[0] == 'O' && data[1] == 'g' && data[2] == 'g' && data[3] == 'S' {
		return "ogg"
	}
	// MP3: ID3 tag or sync word
	if data[0] == 'I' && data[1] == 'D' && data[2] == '3' {
		return "mp3"
	}
	if data[0] == 0xFF && (data[1]&0xE0) == 0xE0 {
		return "mp3"
	}
	// M4A/MP4: check for ftyp box
	if len(data) >= 8 && data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p' {
		return "m4a"
	}
	return "unknown"
}

