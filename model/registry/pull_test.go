package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewHFPullFunc_MockServer(t *testing.T) {
	// Set up a mock HuggingFace server.
	mux := http.NewServeMux()

	// API endpoint: model file listing.
	mux.HandleFunc("/api/models/test-org/test-model", func(w http.ResponseWriter, r *http.Request) {
		info := HFModelInfo{
			ID: "test-org/test-model",
			Siblings: []HFSibling{
				{Filename: "model.onnx"},
				{Filename: "tokenizer.json"},
				{Filename: "config.json"},
				{Filename: "README.md"}, // Should be skipped.
			},
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(info); err != nil {
			t.Logf("encode error: %v", err)
		}
	})

	// CDN endpoints: file downloads.
	mux.HandleFunc("/test-org/test-model/resolve/main/model.onnx", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("fake-onnx-data"))
	})
	mux.HandleFunc("/test-org/test-model/resolve/main/tokenizer.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"model": {"type": "BPE"}}`))
	})
	mux.HandleFunc("/test-org/test-model/resolve/main/config.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"model_type": "gemma"}`))
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	// Create registry with HF pull function.
	dir := t.TempDir()
	reg, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})
	reg.SetPullFunc(pullFn)

	// Pull the model.
	info, err := reg.Pull(context.Background(), "test-org/test-model")
	if err != nil {
		t.Fatalf("Pull error: %v", err)
	}

	if info.ID != "test-org/test-model" {
		t.Errorf("ID = %q, want %q", info.ID, "test-org/test-model")
	}
	if info.Size == 0 {
		t.Error("Size should be > 0")
	}

	// Verify files were downloaded.
	for _, name := range []string{"model.onnx", "tokenizer.json", "config.json"} {
		path := filepath.Join(info.Path, name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("expected file %s to exist", name)
		}
	}

	// README.md should NOT be downloaded.
	readmePath := filepath.Join(info.Path, "README.md")
	if _, err := os.Stat(readmePath); !os.IsNotExist(err) {
		t.Error("README.md should not be downloaded")
	}
}

func TestNewHFPullFunc_WithToken(t *testing.T) {
	var gotAuth string
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		info := HFModelInfo{ID: "org/model", Siblings: []HFSibling{{Filename: "config.json"}}}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/config.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{}`))
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	dir := t.TempDir()
	reg, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
		Token:  "hf_test_token_123",
	})
	reg.SetPullFunc(pullFn)

	if _, err := reg.Pull(context.Background(), "org/model"); err != nil {
		t.Fatal(err)
	}

	if gotAuth != "Bearer hf_test_token_123" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer hf_test_token_123")
	}
}

func TestNewHFPullFunc_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	dir := t.TempDir()
	reg, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})
	reg.SetPullFunc(pullFn)

	_, err = reg.Pull(context.Background(), "nonexistent/model")
	if err == nil {
		t.Error("Pull should fail for 404 response")
	}
}

func TestNewHFPullFunc_Progress(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, _ *http.Request) {
		info := HFModelInfo{ID: "org/model", Siblings: []HFSibling{{Filename: "config.json"}}}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/config.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "11")
		w.Write([]byte(`{"test": 1}`))
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	var progressCalls int
	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
		OnProgress: func(downloaded, total int64) {
			progressCalls++
		},
	})

	dir := t.TempDir()
	_, err := pullFn(context.Background(), "org/model", dir)
	if err != nil {
		t.Fatal(err)
	}

	if progressCalls == 0 {
		t.Error("progress callback should have been called")
	}
}

func TestShouldDownload(t *testing.T) {
	tests := []struct {
		filename string
		want     bool
	}{
		{"model.gguf", true},
		{"Model-Q4_K_M.gguf", true},
		{"model.onnx", true},
		{"tokenizer.json", true},
		{"tokenizer_config.json", true},
		{"config.json", true},
		{"generation_config.json", true},
		{"model.onnx_data", true},
		{"tokenizer.model", true},
		{"README.md", false},
		{"pytorch_model.bin", false},
		{".gitattributes", false},
	}

	for _, tc := range tests {
		if got := shouldDownload(tc.filename); got != tc.want {
			t.Errorf("shouldDownload(%q) = %v, want %v", tc.filename, got, tc.want)
		}
	}
}

func TestGetEnvOr(t *testing.T) {
	// Test with unset env var.
	if got := getEnvOr("ZERFOO_TEST_NONEXISTENT_VAR_12345", "fallback"); got != "fallback" {
		t.Errorf("getEnvOr = %q, want %q", got, "fallback")
	}

	// Test with set env var.
	t.Setenv("ZERFOO_TEST_ENV_VAR", "custom_value")
	if got := getEnvOr("ZERFOO_TEST_ENV_VAR", "fallback"); got != "custom_value" {
		t.Errorf("getEnvOr = %q, want %q", got, "custom_value")
	}
}

func TestNewHFPullFunc_DownloadError(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, _ *http.Request) {
		info := HFModelInfo{ID: "org/model", Siblings: []HFSibling{{Filename: "model.onnx"}}}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/model.onnx", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})

	dir := t.TempDir()
	_, err := pullFn(context.Background(), "org/model", dir)
	if err == nil {
		t.Error("Pull should fail for 403 download response")
	}
}

func TestNewHFPullFunc_EnvFallback(t *testing.T) {
	// Test that NewHFPullFunc uses env vars when set.
	t.Setenv("HUGGINGFACE_API_URL", "https://custom-api.example.com")
	t.Setenv("HUGGINGFACE_CDN_URL", "https://custom-cdn.example.com")
	t.Setenv("HF_TOKEN", "env_token")

	pullFn := NewHFPullFunc(HFPullOptions{})
	// We can't easily introspect the returned function, but at least verify it's non-nil.
	if pullFn == nil {
		t.Error("NewHFPullFunc should return non-nil function")
	}
}

func TestNewHFPullFunc_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("not json"))
	}))
	defer server.Close()

	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})

	dir := t.TempDir()
	_, err := pullFn(context.Background(), "org/model", dir)
	if err == nil {
		t.Error("Pull should fail for invalid JSON response")
	}
}

func TestResolveGGUFByQuant(t *testing.T) {
	files := []HFSibling{
		{Filename: "model-Q4_K_M.gguf"},
		{Filename: "model-Q8_0.gguf"},
		{Filename: "model-F16.gguf"},
		{Filename: "tokenizer.json"},
		{Filename: "config.json"},
	}

	tests := []struct {
		quant   string
		want    string
		wantErr bool
	}{
		{"Q4_K_M", "model-Q4_K_M.gguf", false},
		{"Q8_0", "model-Q8_0.gguf", false},
		{"F16", "model-F16.gguf", false},
		{"q4_k_m", "model-Q4_K_M.gguf", false}, // case insensitive
		{"Q5_K_S", "", true},                   // not found
	}

	for _, tc := range tests {
		got, err := resolveGGUFByQuant(files, tc.quant)
		if tc.wantErr {
			if err == nil {
				t.Errorf("resolveGGUFByQuant(%q) expected error, got %q", tc.quant, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("resolveGGUFByQuant(%q) error: %v", tc.quant, err)
			continue
		}
		if got != tc.want {
			t.Errorf("resolveGGUFByQuant(%q) = %q, want %q", tc.quant, got, tc.want)
		}
	}
}

func TestResolveGGUFByQuant_NoGGUFFiles(t *testing.T) {
	files := []HFSibling{
		{Filename: "config.json"},
		{Filename: "tokenizer.json"},
	}
	_, err := resolveGGUFByQuant(files, "Q4_K_M")
	if err == nil {
		t.Error("expected error when no GGUF files exist")
	}
}

func TestNewHFPullFunc_WithQuant(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, _ *http.Request) {
		info := HFModelInfo{
			ID: "org/model",
			Siblings: []HFSibling{
				{Filename: "model-Q4_K_M.gguf"},
				{Filename: "model-Q8_0.gguf"},
				{Filename: "config.json"},
			},
		}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/model-Q8_0.gguf", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("fake-q8-data"))
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	dir := t.TempDir()
	reg, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
		Quant:  "Q8_0",
	})
	reg.SetPullFunc(pullFn)

	info, err := reg.Pull(context.Background(), "org/model")
	if err != nil {
		t.Fatalf("Pull error: %v", err)
	}

	// Verify only the Q8_0 file was downloaded.
	ggufPath := filepath.Join(info.Path, "model-Q8_0.gguf")
	if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
		t.Error("expected Q8_0 GGUF file to be downloaded")
	}

	// Q4_K_M should NOT be downloaded.
	q4Path := filepath.Join(info.Path, "model-Q4_K_M.gguf")
	if _, statErr := os.Stat(q4Path); !os.IsNotExist(statErr) {
		t.Error("Q4_K_M GGUF should not be downloaded when Quant=Q8_0")
	}
}

func TestDownloadFile_ChecksumMatch(t *testing.T) {
	content := []byte("known content for checksum verification")
	hash := sha256.Sum256(content)
	hexHash := hex.EncodeToString(hash[:])

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, _ *http.Request) {
		info := HFModelInfo{ID: "org/model", Siblings: []HFSibling{{Filename: "config.json"}}}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/config.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("ETag", fmt.Sprintf("%q", hexHash))
		w.Write(content)
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	dir := t.TempDir()
	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})

	_, err := pullFn(context.Background(), "org/model", dir)
	if err != nil {
		t.Fatalf("Pull should succeed with matching checksum: %v", err)
	}

	// Verify the file exists at the final path with correct content.
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != string(content) {
		t.Errorf("file content = %q, want %q", data, content)
	}

	// Verify no temp file remains.
	tmpPath := filepath.Join(dir, "config.json.tmp")
	if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
		t.Error("temp file should not exist after successful download")
	}
}

func TestDownloadFile_ChecksumMismatch(t *testing.T) {
	content := []byte("actual content")
	wrongHash := strings.Repeat("ab", 32) // 64 hex chars, wrong hash

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, _ *http.Request) {
		info := HFModelInfo{ID: "org/model", Siblings: []HFSibling{{Filename: "config.json"}}}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/config.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("ETag", fmt.Sprintf("%q", wrongHash))
		w.Write(content)
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	dir := t.TempDir()
	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})

	_, err := pullFn(context.Background(), "org/model", dir)
	if err == nil {
		t.Fatal("Pull should fail with mismatched checksum")
	}
	if !strings.Contains(err.Error(), "checksum mismatch") {
		t.Errorf("error should mention checksum mismatch, got: %v", err)
	}

	// Verify neither final nor temp file exists.
	finalPath := filepath.Join(dir, "config.json")
	if _, err := os.Stat(finalPath); !os.IsNotExist(err) {
		t.Error("final file should not exist after checksum mismatch")
	}
	tmpPath := filepath.Join(dir, "config.json.tmp")
	if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
		t.Error("temp file should not exist after checksum mismatch")
	}
}

func TestDownloadFile_InterruptedDownload_NoPartialFile(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, _ *http.Request) {
		info := HFModelInfo{ID: "org/model", Siblings: []HFSibling{{Filename: "model.gguf"}}}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/model.gguf", func(w http.ResponseWriter, _ *http.Request) {
		// Write partial data then close connection abruptly by hijacking.
		w.Header().Set("Content-Length", "1000000")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			w.Write([]byte("partial"))
			f.Flush()
		}
		// Hijack the connection to simulate an interrupted download.
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, _ := hj.Hijack()
			if conn != nil {
				conn.Close()
			}
		}
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	dir := t.TempDir()
	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})

	_, err := pullFn(context.Background(), "org/model", dir)
	if err == nil {
		t.Fatal("Pull should fail for interrupted download")
	}

	// The final file must not exist.
	finalPath := filepath.Join(dir, "model.gguf")
	if _, err := os.Stat(finalPath); !os.IsNotExist(err) {
		t.Error("final file should not exist after interrupted download")
	}

	// The temp file must be cleaned up.
	tmpPath := filepath.Join(dir, "model.gguf.tmp")
	if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
		t.Error("temp file should be cleaned up after interrupted download")
	}
}

func TestExtractSHA256(t *testing.T) {
	tests := []struct {
		name   string
		header string
		value  string
		want   string
	}{
		{"quoted ETag", "ETag", `"abc123def456abc123def456abc123def456abc123def456abc123def456abcd"`, "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"},
		{"Content-Sha256", "Content-Sha256", "abc123def456abc123def456abc123def456abc123def456abc123def456abcd", "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"},
		{"non-hex ETag", "ETag", `"not-a-sha256-hash"`, ""},
		{"short ETag", "ETag", `"abcdef"`, ""},
		{"empty", "ETag", "", ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := &http.Response{Header: http.Header{}}
			if tc.value != "" {
				resp.Header.Set(tc.header, tc.value)
			}
			got := extractSHA256(resp)
			if got != tc.want {
				t.Errorf("extractSHA256() = %q, want %q", got, tc.want)
			}
		})
	}
}
