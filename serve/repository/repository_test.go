package repository

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func newTestRepo(t *testing.T) *FileSystemRepository {
	t.Helper()
	dir := t.TempDir()
	repo, err := NewFileSystemRepository(dir)
	if err != nil {
		t.Fatalf("NewFileSystemRepository: %v", err)
	}
	return repo
}

func uploadTestModel(t *testing.T, repo *FileSystemRepository, id, name string) {
	t.Helper()
	meta := ModelMetadata{ID: id, Name: name, Version: "1.0", Format: "gguf"}
	if err := repo.Upload(meta, strings.NewReader("fake-gguf-data")); err != nil {
		t.Fatalf("Upload(%s): %v", id, err)
	}
}

func TestRepository_CRUD(t *testing.T) {
	tests := []struct {
		name string
		fn   func(t *testing.T, repo *FileSystemRepository)
	}{
		{
			name: "list empty",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				models, err := repo.List()
				if err != nil {
					t.Fatalf("List: %v", err)
				}
				if len(models) != 0 {
					t.Fatalf("expected 0 models, got %d", len(models))
				}
			},
		},
		{
			name: "upload and get",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "llama-3b", "Llama 3B")

				meta, err := repo.Get("llama-3b")
				if err != nil {
					t.Fatalf("Get: %v", err)
				}
				if meta.ID != "llama-3b" {
					t.Errorf("ID = %q, want %q", meta.ID, "llama-3b")
				}
				if meta.Name != "Llama 3B" {
					t.Errorf("Name = %q, want %q", meta.Name, "Llama 3B")
				}
				if meta.Format != "gguf" {
					t.Errorf("Format = %q, want %q", meta.Format, "gguf")
				}
				if meta.Size != int64(len("fake-gguf-data")) {
					t.Errorf("Size = %d, want %d", meta.Size, len("fake-gguf-data"))
				}
				if meta.SHA256 == "" {
					t.Error("SHA256 should not be empty")
				}
				if meta.CreatedAt.IsZero() {
					t.Error("CreatedAt should not be zero")
				}
			},
		},
		{
			name: "upload duplicate",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "model-a", "Model A")
				meta := ModelMetadata{ID: "model-a", Name: "Model A v2"}
				err := repo.Upload(meta, strings.NewReader("data"))
				if err != ErrAlreadyExists {
					t.Fatalf("expected ErrAlreadyExists, got %v", err)
				}
			},
		},
		{
			name: "list multiple",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "model-1", "Model 1")
				uploadTestModel(t, repo, "model-2", "Model 2")

				models, err := repo.List()
				if err != nil {
					t.Fatalf("List: %v", err)
				}
				if len(models) != 2 {
					t.Fatalf("expected 2 models, got %d", len(models))
				}
			},
		},
		{
			name: "delete",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "to-delete", "Delete Me")

				if err := repo.Delete("to-delete"); err != nil {
					t.Fatalf("Delete: %v", err)
				}

				_, err := repo.Get("to-delete")
				if err != ErrNotFound {
					t.Fatalf("expected ErrNotFound after delete, got %v", err)
				}
			},
		},
		{
			name: "delete not found",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				err := repo.Delete("nonexistent")
				if err != ErrNotFound {
					t.Fatalf("expected ErrNotFound, got %v", err)
				}
			},
		},
		{
			name: "get not found",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				_, err := repo.Get("nonexistent")
				if err != ErrNotFound {
					t.Fatalf("expected ErrNotFound, got %v", err)
				}
			},
		},
		{
			name: "model file written to disk",
			fn: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "disk-check", "Disk Check")

				data, err := os.ReadFile(filepath.Join(repo.baseDir, "disk-check", "model.gguf"))
				if err != nil {
					t.Fatalf("ReadFile: %v", err)
				}
				if string(data) != "fake-gguf-data" {
					t.Errorf("model file content = %q, want %q", data, "fake-gguf-data")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := newTestRepo(t)
			tt.fn(t, repo)
		})
	}
}

// multipartUpload builds a multipart request body with metadata JSON and a file.
func multipartUpload(t *testing.T, meta uploadRequest, fileContent string) (*bytes.Buffer, string) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	metaJSON, err := json.Marshal(meta)
	if err != nil {
		t.Fatalf("marshal metadata: %v", err)
	}
	if err := w.WriteField("metadata", string(metaJSON)); err != nil {
		t.Fatalf("write metadata field: %v", err)
	}

	fw, err := w.CreateFormFile("file", "model.gguf")
	if err != nil {
		t.Fatalf("create form file: %v", err)
	}
	if _, err := io.WriteString(fw, fileContent); err != nil {
		t.Fatalf("write file content: %v", err)
	}
	w.Close()
	return &buf, w.FormDataContentType()
}

func TestRepository_PathTraversal(t *testing.T) {
	tests := []struct {
		name    string
		id      string
		wantErr bool
	}{
		{name: "parent escape", id: "../../etc", wantErr: true},
		{name: "deep escape", id: "../../../tmp/evil", wantErr: true},
		{name: "normal id", id: "gemma-3-1b", wantErr: false},
		{name: "nested path", id: "org/model", wantErr: false},
		{name: "dot only", id: ".", wantErr: true},
		{name: "dot dot only", id: "..", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := newTestRepo(t)
			_, err := repo.modelDir(tt.id)
			if tt.wantErr {
				if err != ErrPathTraversal {
					t.Fatalf("modelDir(%q) = %v, want ErrPathTraversal", tt.id, err)
				}
				// Verify Get, Upload, and Delete also return the error.
				if _, err := repo.Get(tt.id); err != ErrPathTraversal {
					t.Errorf("Get(%q) = %v, want ErrPathTraversal", tt.id, err)
				}
				if err := repo.Upload(ModelMetadata{ID: tt.id, Name: "x"}, strings.NewReader("data")); err != ErrPathTraversal {
					t.Errorf("Upload(%q) = %v, want ErrPathTraversal", tt.id, err)
				}
				if err := repo.Delete(tt.id); err != ErrPathTraversal {
					t.Errorf("Delete(%q) = %v, want ErrPathTraversal", tt.id, err)
				}
			} else {
				if err != nil {
					t.Fatalf("modelDir(%q) = %v, want nil", tt.id, err)
				}
			}
		})
	}
}

func TestRepository_PathTraversal_HTTP(t *testing.T) {
	repo := newTestRepo(t)
	handler := NewHandler(repo)
	mux := http.NewServeMux()
	handler.RegisterRoutes(mux)

	// GET with traversal ID should return 400.
	req := httptest.NewRequest("GET", "/v1/models/..%2F..%2Fetc%2Fpasswd", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("GET traversal: status = %d, want %d", rec.Code, http.StatusBadRequest)
	}

	// DELETE with traversal ID should return 400.
	req = httptest.NewRequest("DELETE", "/v1/models/..%2F..%2Fetc%2Fpasswd", nil)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("DELETE traversal: status = %d, want %d", rec.Code, http.StatusBadRequest)
	}

	// POST with traversal ID should return 400.
	buf, ct := multipartUpload(t, uploadRequest{
		ID: "../../etc/evil", Name: "Evil",
	}, "data")
	req = httptest.NewRequest("POST", "/v1/models", buf)
	req.Header.Set("Content-Type", ct)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("POST traversal: status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestRepository_HTTPHandlers(t *testing.T) {
	tests := []struct {
		name       string
		setup      func(t *testing.T, repo *FileSystemRepository)
		method     string
		path       string
		body       func(t *testing.T) (io.Reader, string) // returns body and content-type
		wantStatus int
		check      func(t *testing.T, resp *http.Response, body []byte)
	}{
		{
			name:       "list empty",
			method:     "GET",
			path:       "/v1/models",
			wantStatus: http.StatusOK,
			check: func(t *testing.T, _ *http.Response, body []byte) {
				var resp listResponse
				if err := json.Unmarshal(body, &resp); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if resp.Object != "list" {
					t.Errorf("object = %q, want %q", resp.Object, "list")
				}
				if len(resp.Data) != 0 {
					t.Errorf("expected 0 models, got %d", len(resp.Data))
				}
			},
		},
		{
			name:   "upload model",
			method: "POST",
			path:   "/v1/models",
			body: func(t *testing.T) (io.Reader, string) {
				buf, ct := multipartUpload(t, uploadRequest{
					ID: "test-model", Name: "Test Model", Version: "1.0", Format: "gguf",
				}, "gguf-binary-data")
				return buf, ct
			},
			wantStatus: http.StatusCreated,
			check: func(t *testing.T, _ *http.Response, body []byte) {
				var meta ModelMetadata
				if err := json.Unmarshal(body, &meta); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if meta.ID != "test-model" {
					t.Errorf("ID = %q, want %q", meta.ID, "test-model")
				}
				if meta.SHA256 == "" {
					t.Error("SHA256 should not be empty")
				}
				if meta.Size != int64(len("gguf-binary-data")) {
					t.Errorf("Size = %d, want %d", meta.Size, len("gguf-binary-data"))
				}
			},
		},
		{
			name: "get model",
			setup: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "get-me", "Get Me")
			},
			method:     "GET",
			path:       "/v1/models/get-me",
			wantStatus: http.StatusOK,
			check: func(t *testing.T, _ *http.Response, body []byte) {
				var meta ModelMetadata
				if err := json.Unmarshal(body, &meta); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if meta.ID != "get-me" {
					t.Errorf("ID = %q, want %q", meta.ID, "get-me")
				}
			},
		},
		{
			name:       "get model not found",
			method:     "GET",
			path:       "/v1/models/nonexistent",
			wantStatus: http.StatusNotFound,
		},
		{
			name: "delete model",
			setup: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "delete-me", "Delete Me")
			},
			method:     "DELETE",
			path:       "/v1/models/delete-me",
			wantStatus: http.StatusOK,
			check: func(t *testing.T, _ *http.Response, body []byte) {
				var resp deleteResponse
				if err := json.Unmarshal(body, &resp); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if !resp.Deleted {
					t.Error("expected deleted=true")
				}
				if resp.ID != "delete-me" {
					t.Errorf("ID = %q, want %q", resp.ID, "delete-me")
				}
			},
		},
		{
			name:       "delete model not found",
			method:     "DELETE",
			path:       "/v1/models/nonexistent",
			wantStatus: http.StatusNotFound,
		},
		{
			name: "upload duplicate",
			setup: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "dup-model", "Dup Model")
			},
			method: "POST",
			path:   "/v1/models",
			body: func(t *testing.T) (io.Reader, string) {
				buf, ct := multipartUpload(t, uploadRequest{
					ID: "dup-model", Name: "Dup Model", Version: "2.0",
				}, "data")
				return buf, ct
			},
			wantStatus: http.StatusConflict,
		},
		{
			name:   "upload missing id",
			method: "POST",
			path:   "/v1/models",
			body: func(t *testing.T) (io.Reader, string) {
				buf, ct := multipartUpload(t, uploadRequest{
					Name: "No ID",
				}, "data")
				return buf, ct
			},
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "list after upload and delete",
			setup: func(t *testing.T, repo *FileSystemRepository) {
				uploadTestModel(t, repo, "m1", "Model 1")
				uploadTestModel(t, repo, "m2", "Model 2")
				if err := repo.Delete("m1"); err != nil {
					t.Fatalf("Delete: %v", err)
				}
			},
			method:     "GET",
			path:       "/v1/models",
			wantStatus: http.StatusOK,
			check: func(t *testing.T, _ *http.Response, body []byte) {
				var resp listResponse
				if err := json.Unmarshal(body, &resp); err != nil {
					t.Fatalf("unmarshal: %v", err)
				}
				if len(resp.Data) != 1 {
					t.Fatalf("expected 1 model, got %d", len(resp.Data))
				}
				if resp.Data[0].ID != "m2" {
					t.Errorf("remaining model ID = %q, want %q", resp.Data[0].ID, "m2")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := newTestRepo(t)
			if tt.setup != nil {
				tt.setup(t, repo)
			}

			handler := NewHandler(repo)
			mux := http.NewServeMux()
			handler.RegisterRoutes(mux)

			var body io.Reader
			contentType := ""
			if tt.body != nil {
				body, contentType = tt.body(t)
			}

			req := httptest.NewRequest(tt.method, tt.path, body)
			if contentType != "" {
				req.Header.Set("Content-Type", contentType)
			}
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, req)

			resp := rec.Result()
			defer resp.Body.Close()
			respBody, _ := io.ReadAll(resp.Body)

			if resp.StatusCode != tt.wantStatus {
				t.Errorf("status = %d, want %d; body: %s", resp.StatusCode, tt.wantStatus, respBody)
			}
			if tt.check != nil {
				tt.check(t, resp, respBody)
			}
		})
	}
}
