package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sha256hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func TestDownload_Fresh(t *testing.T) {
	body := []byte("hello world — this is a GGUF file")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rng := r.Header.Get("Range"); rng != "" {
			t.Errorf("unexpected Range header on fresh download: %s", rng)
		}
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
		w.Write(body)
	}))
	t.Cleanup(srv.Close)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	dl := NewDownloader(srv.Client())

	if err := dl.Download(context.Background(), srv.URL+"/file.gguf", dest); err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read dest: %v", err)
	}
	if string(got) != string(body) {
		t.Errorf("got %q, want %q", got, body)
	}

	// Partial file should not exist after successful download.
	if _, err := os.Stat(dest + ".partial"); !os.IsNotExist(err) {
		t.Error("partial file still exists after download")
	}
}

func TestDownload_Resume(t *testing.T) {
	fullBody := []byte("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	partialLen := 10

	var gotRange string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRange = r.Header.Get("Range")
		if gotRange != "" {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(fullBody)-partialLen))
			w.WriteHeader(http.StatusPartialContent)
			w.Write(fullBody[partialLen:])
		} else {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(fullBody)))
			w.Write(fullBody)
		}
	}))
	t.Cleanup(srv.Close)

	dir := t.TempDir()
	dest := filepath.Join(dir, "model.gguf")
	partialPath := dest + ".partial"

	// Write the first 10 bytes as a partial file.
	if err := os.WriteFile(partialPath, fullBody[:partialLen], 0o644); err != nil {
		t.Fatalf("write partial: %v", err)
	}

	dl := NewDownloader(srv.Client())
	if err := dl.Download(context.Background(), srv.URL+"/file.gguf", dest); err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	wantRange := "bytes=10-"
	if gotRange != wantRange {
		t.Errorf("Range header = %q, want %q", gotRange, wantRange)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read dest: %v", err)
	}
	if string(got) != string(fullBody) {
		t.Errorf("got %q, want %q", got, fullBody)
	}
}

func TestDownload_SHA256_Success(t *testing.T) {
	body := []byte("checksum test data")
	digest := sha256hex(body)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	}))
	t.Cleanup(srv.Close)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	dl := NewDownloader(srv.Client())

	if err := dl.Download(context.Background(), srv.URL+"/file.gguf", dest, WithSHA256(digest)); err != nil {
		t.Fatalf("Download failed: %v", err)
	}
}

func TestDownload_SHA256_Mismatch(t *testing.T) {
	body := []byte("checksum test data")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	}))
	t.Cleanup(srv.Close)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	dl := NewDownloader(srv.Client())

	err := dl.Download(context.Background(), srv.URL+"/file.gguf", dest, WithSHA256("0000000000000000000000000000000000000000000000000000000000000000"))
	if err == nil {
		t.Fatal("expected SHA256 mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Errorf("error = %q, want sha256 mismatch", err)
	}

	// Dest should not exist on SHA256 failure (partial file remains).
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Error("dest file should not exist after SHA256 mismatch")
	}
}

func TestDownload_Progress(t *testing.T) {
	body := []byte("progress tracking bytes")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
		w.Write(body)
	}))
	t.Cleanup(srv.Close)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	dl := NewDownloader(srv.Client())

	var calls []struct{ downloaded, total int64 }
	if err := dl.Download(context.Background(), srv.URL+"/file.gguf", dest,
		WithProgress(func(downloaded, total int64) {
			calls = append(calls, struct{ downloaded, total int64 }{downloaded, total})
		}),
	); err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	if len(calls) == 0 {
		t.Fatal("progress callback was never called")
	}

	last := calls[len(calls)-1]
	if last.downloaded != int64(len(body)) {
		t.Errorf("final downloaded = %d, want %d", last.downloaded, len(body))
	}
	if last.total != int64(len(body)) {
		t.Errorf("final total = %d, want %d", last.total, len(body))
	}

	// Verify monotonically increasing.
	for i := 1; i < len(calls); i++ {
		if calls[i].downloaded < calls[i-1].downloaded {
			t.Errorf("progress not monotonic at index %d: %d < %d", i, calls[i].downloaded, calls[i-1].downloaded)
		}
	}
}

func TestDownload_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	dest := filepath.Join(t.TempDir(), "model.gguf")
	dl := NewDownloader(srv.Client())

	err := dl.Download(context.Background(), srv.URL+"/file.gguf", dest)
	if err == nil {
		t.Fatal("expected error on 500, got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want mention of 500", err)
	}
}

func TestDownload_CreatesDirectory(t *testing.T) {
	body := []byte("data")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	}))
	t.Cleanup(srv.Close)

	dest := filepath.Join(t.TempDir(), "deep", "nested", "dir", "model.gguf")
	dl := NewDownloader(srv.Client())

	if err := dl.Download(context.Background(), srv.URL+"/file.gguf", dest); err != nil {
		t.Fatalf("Download failed: %v", err)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(got) != "data" {
		t.Errorf("got %q, want %q", got, "data")
	}
}

func TestFormatProgress(t *testing.T) {
	tests := []struct {
		name       string
		filename   string
		downloaded int64
		total      int64
		wantSub    string
	}{
		{"with total", "model.gguf", 50 << 20, 200 << 20, "25.0%"},
		{"no total", "model.gguf", 1024, 0, "1.0 KB"},
		{"gb range", "big.gguf", 2 << 30, 4 << 30, "50.0%"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FormatProgress(tt.filename, tt.downloaded, tt.total)
			if !strings.Contains(got, tt.wantSub) {
				t.Errorf("FormatProgress() = %q, want substring %q", got, tt.wantSub)
			}
		})
	}
}
