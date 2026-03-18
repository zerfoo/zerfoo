package serve

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// mockTranscriber implements Transcriber for testing.
type mockTranscriber struct {
	text string
	err  error
}

func (m *mockTranscriber) Transcribe(_ context.Context, _ []byte, _ string) (string, error) {
	return m.text, m.err
}

// createMultipartAudio creates a multipart/form-data request body with an audio file.
func createMultipartAudio(t *testing.T, audioData []byte, fields map[string]string) (*bytes.Buffer, string) {
	t.Helper()
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", "test.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(audioData); err != nil {
		t.Fatal(err)
	}

	for k, v := range fields {
		if err := writer.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return &buf, writer.FormDataContentType()
}

// testWAVData returns minimal WAV-like bytes (just magic header for format detection).
func testWAVData() []byte {
	return []byte("RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00")
}

func TestAudioAPIEndpoint(t *testing.T) {
	mdl := buildTestModel(t)
	transcriber := &mockTranscriber{text: "hello world"}
	srv := NewServer(mdl, WithTranscriber(transcriber))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, contentType := createMultipartAudio(t, testWAVData(), map[string]string{
		"model":    "whisper-1",
		"language": "en",
	})

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		ts.URL+"/v1/audio/transcriptions", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result TranscriptionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Text != "hello world" {
		t.Errorf("Text = %q, want %q", result.Text, "hello world")
	}
}

func TestAudioAPIEndpoint_TextFormat(t *testing.T) {
	mdl := buildTestModel(t)
	transcriber := &mockTranscriber{text: "transcribed text"}
	srv := NewServer(mdl, WithTranscriber(transcriber))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, contentType := createMultipartAudio(t, testWAVData(), map[string]string{
		"response_format": "text",
	})

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		ts.URL+"/v1/audio/transcriptions", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type = %q, want text/plain", ct)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "transcribed text" {
		t.Errorf("body = %q, want %q", string(data), "transcribed text")
	}
}

func TestAudioAPIEndpoint_MissingFile(t *testing.T) {
	mdl := buildTestModel(t)
	transcriber := &mockTranscriber{text: "ok"}
	srv := NewServer(mdl, WithTranscriber(transcriber))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Send multipart without a "file" field.
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	writer.WriteField("model", "whisper-1")
	writer.Close()

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		ts.URL+"/v1/audio/transcriptions", &buf)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestAudioAPIEndpoint_NoTranscriber(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl) // no transcriber
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, contentType := createMultipartAudio(t, testWAVData(), nil)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		ts.URL+"/v1/audio/transcriptions", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501", resp.StatusCode)
	}
}

func TestAudioAPIEndpoint_TranscriberError(t *testing.T) {
	mdl := buildTestModel(t)
	transcriber := &mockTranscriber{err: errors.New("transcription failed")}
	srv := NewServer(mdl, WithTranscriber(transcriber))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, contentType := createMultipartAudio(t, testWAVData(), nil)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		ts.URL+"/v1/audio/transcriptions", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestAudioAPIEndpoint_EmptyFile(t *testing.T) {
	mdl := buildTestModel(t)
	transcriber := &mockTranscriber{text: "ok"}
	srv := NewServer(mdl, WithTranscriber(transcriber))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, contentType := createMultipartAudio(t, []byte{}, nil)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		ts.URL+"/v1/audio/transcriptions", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", contentType)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestDetectAudioFormat(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want string
	}{
		{"WAV", []byte("RIFF\x00\x00\x00\x00WAVE"), "wav"},
		{"FLAC", []byte("fLaC\x00\x00\x00\x22"), "flac"},
		{"OGG", []byte("OggS\x00\x00\x00\x00"), "ogg"},
		{"MP3 ID3", []byte("ID3\x04\x00\x00"), "mp3"},
		{"MP3 sync", []byte{0xFF, 0xFB, 0x90, 0x00}, "mp3"},
		{"M4A", []byte("\x00\x00\x00\x1cftypisom"), "m4a"},
		{"unknown", []byte{0x00, 0x01, 0x02, 0x03}, "unknown"},
		{"short", []byte{0x00}, "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := detectAudioFormat(tt.data)
			if got != tt.want {
				t.Errorf("detectAudioFormat() = %q, want %q", got, tt.want)
			}
		})
	}
}
