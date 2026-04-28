package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/model/registry"
)

// mockPullRegistry is a test double for registry.ModelRegistry.
type mockPullRegistry struct {
	models  map[string]*registry.ModelInfo
	pulled  bool
	pullErr error
}

func (r *mockPullRegistry) Get(id string) (*registry.ModelInfo, bool) {
	info, ok := r.models[id]
	return info, ok
}

func (r *mockPullRegistry) Pull(_ context.Context, id string) (*registry.ModelInfo, error) {
	r.pulled = true
	if r.pullErr != nil {
		return nil, r.pullErr
	}
	if info, ok := r.models[id]; ok {
		return info, nil
	}
	return &registry.ModelInfo{ID: id, Path: "/cache/" + id, Size: 1024}, nil
}

func (r *mockPullRegistry) List() []registry.ModelInfo {
	var out []registry.ModelInfo
	for _, info := range r.models {
		out = append(out, *info)
	}
	return out
}
func (r *mockPullRegistry) Delete(id string) error {
	if _, ok := r.models[id]; !ok {
		return fmt.Errorf("model %q not found", id)
	}
	delete(r.models, id)
	return nil
}

func TestPullCommand_Name(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if cmd.Name() != "pull" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "pull")
	}
}

func TestPullCommand_Description(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestPullCommand_Usage(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "pull") {
		t.Error("Usage() should contain 'pull'")
	}
}

func TestPullCommand_Examples(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestPullCommand_MissingModelID(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), nil)
	if err == nil {
		t.Error("expected error for missing model ID")
	}
	if !strings.Contains(err.Error(), "model ID is required") {
		t.Errorf("error = %q, want 'model ID is required'", err.Error())
	}
}

func TestPullCommand_AlreadyCached(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models: map[string]*registry.ModelInfo{
			"test-model": {ID: "test-model", Path: "/cache/test-model"},
		},
	}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(buf.String(), "Already up to date") {
		t.Errorf("output = %q, want 'Already up to date'", buf.String())
	}
}

func TestPullCommand_PullsNewModel(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"new-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !reg.pulled {
		t.Error("expected Pull to be called")
	}
	if !strings.Contains(buf.String(), "Model saved to") {
		t.Errorf("output = %q, want 'Model saved to'", buf.String())
	}
}

func TestPullCommand_CacheDirFlag(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"--cache-dir", "/tmp/cache", "test"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

func TestPullCommand_CacheDirMissingValue(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), []string{"--cache-dir"})
	if err == nil {
		t.Error("expected error for --cache-dir without value")
	}
}

func TestPullCommand_UnexpectedArgument(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), []string{"model1", "model2"})
	if err == nil {
		t.Error("expected error for extra argument")
	}
}

func TestPullCommand_NilRegistry(t *testing.T) {
	// When no registry is provided, it creates a default one.
	var buf bytes.Buffer
	cmd := NewPullCommand(nil, &buf)
	// This will fail to pull (no real model) but exercises the nil-registry code path.
	err := cmd.Run(context.Background(), []string{"nonexistent"})
	if err == nil {
		t.Error("expected error")
	}
}

func TestPullCommand_PullError(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models:  map[string]*registry.ModelInfo{},
		pullErr: errors.New("network timeout"),
	}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"fail-model"})
	if err == nil {
		t.Fatal("expected error when Pull fails")
	}
	if !strings.Contains(err.Error(), "network timeout") {
		t.Errorf("error = %q, want it to contain 'network timeout'", err.Error())
	}
	if !reg.pulled {
		t.Error("expected Pull to be called")
	}
}

func TestPullCommand_PullOutputIncludesSize(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"size-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "Pulling size-model") {
		t.Errorf("output = %q, want it to contain 'Pulling size-model'", output)
	}
	if !strings.Contains(output, "Size: 1024 bytes") {
		t.Errorf("output = %q, want it to contain 'Size: 1024 bytes'", output)
	}
}

func TestPullCommand_AlreadyCachedOutputIncludesPath(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models: map[string]*registry.ModelInfo{
			"cached": {ID: "cached", Path: "/models/cached"},
		},
	}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"cached"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(buf.String(), "/models/cached") {
		t.Errorf("output = %q, want it to contain path '/models/cached'", buf.String())
	}
	if reg.pulled {
		t.Error("Pull should not be called for cached model")
	}
}

func TestPullCommand_NilRegistryCreatesLocalRegistry(t *testing.T) {
	tmp := t.TempDir()
	var buf bytes.Buffer
	cmd := NewPullCommand(nil, &buf)
	// Use --cache-dir so the LocalRegistry targets a temp directory.
	// Pull will attempt to reach HuggingFace and fail (no real server).
	// We verify we get an error (not a nil-pointer panic).
	err := cmd.Run(context.Background(), []string{"--cache-dir", tmp, "org/test-model"})
	if err == nil {
		t.Fatal("expected error from LocalRegistry.Pull (HF unreachable)")
	}
}

// Ensure *PullCommand satisfies Command interface at compile time.
func TestPullCommand_Interface(t *testing.T) {
	var _ Command = (*PullCommand)(nil)
	_ = os.Stderr // use os to avoid unused import
}

// newMockHFServer creates an httptest server that simulates the HuggingFace
// API (model listing) and CDN (file download). It returns the server and
// separate API/CDN base URLs (they share the same server, differentiated by path prefix).
func newMockHFServer(t *testing.T, modelID string, files map[string][]byte) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	// API endpoint: GET /api/models/<org>/<model>
	mux.HandleFunc("/api/models/"+modelID, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var siblings []registry.HFSibling
		for name := range files {
			siblings = append(siblings, registry.HFSibling{Filename: name})
		}
		resp := registry.HFModelInfo{ID: modelID, Siblings: siblings}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Errorf("encode response: %v", err)
		}
	})

	// CDN endpoint: GET /cdn/<org>/<model>/resolve/main/<filename>
	mux.HandleFunc("/cdn/"+modelID+"/resolve/main/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Extract filename from path after /cdn/<org>/<model>/resolve/main/
		prefix := "/cdn/" + modelID + "/resolve/main/"
		filename := strings.TrimPrefix(r.URL.Path, prefix)
		data, ok := files[filename]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		if _, err := w.Write(data); err != nil {
			t.Errorf("write response: %v", err)
		}
	})

	return httptest.NewServer(mux)
}

func TestPullCommand_Integration(t *testing.T) {
	modelID := "testorg/testmodel"
	modelConfig := []byte(`{"architectures":["TestArch"]}`)
	modelONNX := []byte("fake-onnx-model-data-for-testing")
	tokenizerJSON := []byte(`{"version":"1.0"}`)

	tests := []struct {
		name       string
		modelID    string
		files      map[string][]byte
		wantOutput []string
		wantFiles  []string
	}{
		{
			name:    "pulls model with onnx and tokenizer",
			modelID: modelID,
			files: map[string][]byte{
				"config.json":    modelConfig,
				"model.onnx":     modelONNX,
				"tokenizer.json": tokenizerJSON,
				"README.md":      []byte("# readme"), // should be skipped
			},
			wantOutput: []string{
				"Pulling " + modelID,
				"Model saved to:",
				"Size:",
			},
			wantFiles: []string{
				"config.json",
				"model.onnx",
				"tokenizer.json",
			},
		},
		{
			name:    "pulls model with gguf format",
			modelID: modelID,
			files: map[string][]byte{
				"model.gguf":  []byte("fake-gguf-data"),
				"LICENSE":     []byte("MIT"), // should be skipped
				"config.json": modelConfig,
			},
			wantOutput: []string{
				"Pulling " + modelID,
				"Model saved to:",
			},
			wantFiles: []string{
				"model.gguf",
				"config.json",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newMockHFServer(t, tt.modelID, tt.files)
			defer srv.Close()

			cacheDir := t.TempDir()

			lr, err := registry.NewLocalRegistry(cacheDir)
			if err != nil {
				t.Fatalf("NewLocalRegistry: %v", err)
			}
			lr.SetPullFunc(registry.NewHFPullFunc(registry.HFPullOptions{
				APIURL: srv.URL + "/api/models",
				CDNURL: srv.URL + "/cdn",
				Client: srv.Client(),
			}))

			var buf bytes.Buffer
			cmd := NewPullCommand(lr, &buf)
			err = cmd.Run(context.Background(), []string{tt.modelID})
			if err != nil {
				t.Fatalf("Run error: %v", err)
			}

			output := buf.String()
			for _, want := range tt.wantOutput {
				if !strings.Contains(output, want) {
					t.Errorf("output missing %q, got: %s", want, output)
				}
			}

			// Verify expected files were downloaded.
			modelDir := filepath.Join(cacheDir, "testorg", "testmodel")
			for _, wantFile := range tt.wantFiles {
				path := filepath.Join(modelDir, wantFile)
				data, readErr := os.ReadFile(path)
				if readErr != nil {
					t.Errorf("expected file %s to exist: %v", wantFile, readErr)
					continue
				}
				if len(data) == 0 {
					t.Errorf("file %s is empty", wantFile)
				}
			}

			// Verify skipped files were not downloaded.
			skippedFiles := []string{"README.md", "LICENSE"}
			for _, skip := range skippedFiles {
				if _, ok := tt.files[skip]; !ok {
					continue
				}
				path := filepath.Join(modelDir, skip)
				if _, statErr := os.Stat(path); statErr == nil {
					t.Errorf("file %s should not have been downloaded", skip)
				}
			}
		})
	}
}

func TestPullCommand_IntegrationAlreadyCached(t *testing.T) {
	modelID := "testorg/cachedmodel"
	files := map[string][]byte{
		"config.json": []byte(`{"architectures":["TestArch"]}`),
		"model.onnx":  []byte("onnx-data"),
	}

	srv := newMockHFServer(t, modelID, files)
	defer srv.Close()

	cacheDir := t.TempDir()

	lr, err := registry.NewLocalRegistry(cacheDir)
	if err != nil {
		t.Fatalf("NewLocalRegistry: %v", err)
	}
	lr.SetPullFunc(registry.NewHFPullFunc(registry.HFPullOptions{
		APIURL: srv.URL + "/api/models",
		CDNURL: srv.URL + "/cdn",
		Client: srv.Client(),
	}))

	var buf bytes.Buffer
	cmd := NewPullCommand(lr, &buf)

	// First pull.
	if err := cmd.Run(context.Background(), []string{modelID}); err != nil {
		t.Fatalf("first pull: %v", err)
	}
	if !strings.Contains(buf.String(), "Model saved to:") {
		t.Errorf("first pull output = %q, want 'Model saved to:'", buf.String())
	}

	// Second pull should report already cached.
	buf.Reset()
	if err := cmd.Run(context.Background(), []string{modelID}); err != nil {
		t.Fatalf("second pull: %v", err)
	}
	if !strings.Contains(buf.String(), "Already up to date") {
		t.Errorf("second pull output = %q, want 'Already up to date'", buf.String())
	}
}

func TestPullCommand_IntegrationServerError(t *testing.T) {
	// Server that always returns 500.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	cacheDir := t.TempDir()
	lr, err := registry.NewLocalRegistry(cacheDir)
	if err != nil {
		t.Fatalf("NewLocalRegistry: %v", err)
	}
	lr.SetPullFunc(registry.NewHFPullFunc(registry.HFPullOptions{
		APIURL: srv.URL + "/api/models",
		CDNURL: srv.URL + "/cdn",
		Client: srv.Client(),
	}))

	var buf bytes.Buffer
	cmd := NewPullCommand(lr, &buf)
	err = cmd.Run(context.Background(), []string{"org/model"})
	if err == nil {
		t.Fatal("expected error for server error response")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want it to contain '500'", err.Error())
	}
}

func TestPullCommand_QuantFlag(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"--quant", "Q8_0", "org/model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "quant: Q8_0") {
		t.Errorf("output = %q, want it to contain 'quant: Q8_0'", output)
	}
}

func TestPullCommand_QuantFlagMissingValue(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), []string{"--quant"})
	if err == nil {
		t.Error("expected error for --quant without value")
	}
}

func TestPullCommand_DefaultQuant(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"org/model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "quant: Q4_K_M") {
		t.Errorf("output = %q, want it to contain 'quant: Q4_K_M'", output)
	}
}

// TestPullListRm_Integration tests the full pull/list/rm cycle using
// a mock HuggingFace server and a real LocalRegistry.
func TestPullListRm_Integration(t *testing.T) {
	modelID := "testorg/testmodel"
	ggufData := []byte("fake-gguf-model-data-for-testing")

	srv := newMockHFServer(t, modelID, map[string][]byte{
		"model-Q4_K_M.gguf": ggufData,
		"config.json":       []byte(`{"architectures":["TestArch"]}`),
	})
	defer srv.Close()

	cacheDir := t.TempDir()
	lr, err := registry.NewLocalRegistry(cacheDir)
	if err != nil {
		t.Fatalf("NewLocalRegistry: %v", err)
	}
	lr.SetPullFunc(registry.NewHFPullFunc(registry.HFPullOptions{
		APIURL: srv.URL + "/api/models",
		CDNURL: srv.URL + "/cdn",
		Client: srv.Client(),
	}))

	// Step 1: List — should be empty.
	var buf bytes.Buffer
	listCmd := NewListCommand(lr, &buf)
	if err := listCmd.Run(context.Background(), nil); err != nil {
		t.Fatalf("list (empty): %v", err)
	}
	if !strings.Contains(buf.String(), "No cached models") {
		t.Errorf("expected 'No cached models', got: %s", buf.String())
	}

	// Step 2: Pull.
	buf.Reset()
	pullCmd := NewPullCommand(lr, &buf)
	if err := pullCmd.Run(context.Background(), []string{modelID}); err != nil {
		t.Fatalf("pull: %v", err)
	}
	if !strings.Contains(buf.String(), "Model saved to") {
		t.Errorf("pull output missing 'Model saved to', got: %s", buf.String())
	}

	// Step 3: List — should show the model.
	buf.Reset()
	if err := listCmd.Run(context.Background(), nil); err != nil {
		t.Fatalf("list (after pull): %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "REPO") {
		t.Error("list output should contain header")
	}
	if !strings.Contains(output, modelID) {
		t.Errorf("list output should contain %q, got: %s", modelID, output)
	}

	// Step 4: Pull again — should report already cached.
	buf.Reset()
	if err := pullCmd.Run(context.Background(), []string{modelID}); err != nil {
		t.Fatalf("pull (cached): %v", err)
	}
	if !strings.Contains(buf.String(), "Already up to date") {
		t.Errorf("expected 'Already up to date', got: %s", buf.String())
	}

	// Step 5: Remove.
	buf.Reset()
	rmCmd := NewRmCommand(lr, &buf)
	if err := rmCmd.Run(context.Background(), []string{modelID}); err != nil {
		t.Fatalf("rm: %v", err)
	}
	if !strings.Contains(buf.String(), "Removed") {
		t.Errorf("rm output missing 'Removed', got: %s", buf.String())
	}

	// Step 6: List — should be empty again.
	buf.Reset()
	if err := listCmd.Run(context.Background(), nil); err != nil {
		t.Fatalf("list (after rm): %v", err)
	}
	if !strings.Contains(buf.String(), "No cached models") {
		t.Errorf("expected 'No cached models' after rm, got: %s", buf.String())
	}

	// Step 7: Remove again — should error.
	buf.Reset()
	if err := rmCmd.Run(context.Background(), []string{modelID}); err == nil {
		t.Error("expected error when removing already-deleted model")
	}
}
