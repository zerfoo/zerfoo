package registry

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// List: walk encounters errors (unreadable subdirectory)
// ---------------------------------------------------------------------------

func TestLocalRegistry_List_WalkError(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Create a subdirectory that cannot be read.
	badDir := filepath.Join(dir, "org", "broken")
	if err := os.MkdirAll(badDir, 0o750); err != nil {
		t.Fatal(err)
	}
	// Write a valid config.json so it would normally be found.
	info := ModelInfo{ID: "org/broken", Architecture: "test"}
	data, _ := json.Marshal(info)
	if err := os.WriteFile(filepath.Join(badDir, "config.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}

	// Make the parent unreadable so Walk reports errors.
	orgDir := filepath.Join(dir, "org")
	if err := os.Chmod(orgDir, 0o000); err != nil {
		t.Skip("cannot change permissions on this platform")
	}
	defer os.Chmod(orgDir, 0o750)
	// List should handle the walk error gracefully.
	models := r.List()
	// The error branch skips entries, so 0 models expected.
	if len(models) != 0 {
		t.Errorf("List() = %d items, want 0 (directory unreadable)", len(models))
	}
}

// ---------------------------------------------------------------------------
// downloadFile: target directory is read-only (os.Create fails)
// ---------------------------------------------------------------------------

func TestHFPull_DownloadFileCreateError(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/org/model", func(w http.ResponseWriter, _ *http.Request) {
		info := HFModelInfo{ID: "org/model", Siblings: []HFSibling{{Filename: "model.onnx"}}}
		json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("/org/model/resolve/main/model.onnx", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("data"))
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	pullFn := NewHFPullFunc(HFPullOptions{
		APIURL: server.URL + "/api/models",
		CDNURL: server.URL,
		Client: server.Client(),
	})

	// Create a read-only target dir so os.Create fails.
	dir := t.TempDir()
	readOnlyDir := filepath.Join(dir, "readonly")
	if err := os.MkdirAll(readOnlyDir, 0o500); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(readOnlyDir, 0o750)
	_, err := pullFn(context.Background(), "org/model", readOnlyDir)
	if err == nil {
		t.Error("expected error when target dir is read-only")
	}
}

// ---------------------------------------------------------------------------
// Pull: MkdirAll error for model directory
// ---------------------------------------------------------------------------

func TestLocalRegistry_Pull_MkdirError(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	r.SetPullFunc(func(_ context.Context, _ string, _ string) (*ModelInfo, error) {
		return &ModelInfo{ID: "org/model"}, nil
	})

	// Create a file where the model directory would be, so MkdirAll fails.
	orgDir := filepath.Join(dir, "org")
	if err := os.MkdirAll(orgDir, 0o750); err != nil {
		t.Fatal(err)
	}
	// Create a regular file named "model" to block directory creation.
	if err := os.WriteFile(filepath.Join(orgDir, "model"), []byte("blocker"), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err = r.Pull(context.Background(), "org/model")
	if err == nil {
		t.Error("expected error when model directory cannot be created")
	}
}
