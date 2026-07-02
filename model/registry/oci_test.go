package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// mockOCIRegistry simulates an OCI distribution spec registry for testing.
type mockOCIRegistry struct {
	mu        sync.Mutex
	blobs     map[string][]byte   // digest -> data
	manifests map[string][]byte   // repo/tag -> manifest JSON
	tags      map[string][]string // repo -> tags
}

func newMockOCIRegistry() *mockOCIRegistry {
	return &mockOCIRegistry{
		blobs:     make(map[string][]byte),
		manifests: make(map[string][]byte),
		tags:      make(map[string][]string),
	}
}

func (m *mockOCIRegistry) handler() http.Handler {
	mux := http.NewServeMux()

	// Blob upload: POST /v2/{repo}/blobs/uploads/?digest=...
	mux.HandleFunc("/v2/", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path

		switch {
		// Blob upload.
		case r.Method == http.MethodPost && strings.Contains(path, "/blobs/uploads/"):
			digest := r.URL.Query().Get("digest")
			if digest == "" {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			data, err := readBody(r)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			m.mu.Lock()
			m.blobs[digest] = data
			m.mu.Unlock()
			w.WriteHeader(http.StatusCreated)

		// Blob download: GET /v2/{repo}/blobs/{digest}
		case r.Method == http.MethodGet && strings.Contains(path, "/blobs/"):
			parts := strings.Split(path, "/blobs/")
			if len(parts) != 2 {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			digest := parts[1]
			m.mu.Lock()
			data, ok := m.blobs[digest]
			m.mu.Unlock()
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/octet-stream")
			w.Write(data) //nolint:errcheck

		// Manifest PUT: PUT /v2/{repo}/manifests/{tag}
		case r.Method == http.MethodPut && strings.Contains(path, "/manifests/"):
			parts := strings.SplitN(path, "/manifests/", 2)
			if len(parts) != 2 {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			repo := strings.TrimPrefix(parts[0], "/v2/")
			tag := parts[1]
			data, err := readBody(r)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			key := repo + "/" + tag
			m.mu.Lock()
			m.manifests[key] = data
			// Track tags.
			found := false
			for _, t := range m.tags[repo] {
				if t == tag {
					found = true
					break
				}
			}
			if !found {
				m.tags[repo] = append(m.tags[repo], tag)
			}
			m.mu.Unlock()
			w.WriteHeader(http.StatusCreated)

		// Manifest GET: GET /v2/{repo}/manifests/{ref}
		case r.Method == http.MethodGet && strings.Contains(path, "/manifests/"):
			parts := strings.SplitN(path, "/manifests/", 2)
			if len(parts) != 2 {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			repo := strings.TrimPrefix(parts[0], "/v2/")
			ref := parts[1]
			key := repo + "/" + ref
			m.mu.Lock()
			data, ok := m.manifests[key]
			m.mu.Unlock()
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", MediaTypeOCIManifest)
			w.Write(data) //nolint:errcheck

		// Tags list: GET /v2/{repo}/tags/list
		case r.Method == http.MethodGet && strings.HasSuffix(path, "/tags/list"):
			repo := strings.TrimPrefix(strings.TrimSuffix(path, "/tags/list"), "/v2/")
			m.mu.Lock()
			tags := m.tags[repo]
			m.mu.Unlock()
			if tags == nil {
				tags = []string{}
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(TagList{Name: repo, Tags: tags}) //nolint:errcheck

		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})

	return mux
}

func readBody(r *http.Request) ([]byte, error) {
	defer r.Body.Close() //nolint:errcheck
	data := make([]byte, 0, 1024)
	buf := make([]byte, 512)
	for {
		n, err := r.Body.Read(buf)
		if n > 0 {
			data = append(data, buf[:n]...)
		}
		if err != nil {
			break
		}
	}
	return data, nil
}

func TestRegistry_Push(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	// Create a fake GGUF model file.
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	modelData := []byte("fake-gguf-model-data-for-testing")
	if err := os.WriteFile(modelPath, modelData, 0o600); err != nil {
		t.Fatal(err)
	}

	err := reg.Push(context.Background(), "registry.example.com/myrepo:v1.0", modelPath)
	if err != nil {
		t.Fatalf("Push error: %v", err)
	}

	// Verify the blob was stored.
	expectedDigest := sha256Digest(modelData)
	mock.mu.Lock()
	_, blobExists := mock.blobs[expectedDigest]
	mock.mu.Unlock()
	if !blobExists {
		t.Error("model blob not found in registry")
	}

	// Verify the manifest was stored.
	mock.mu.Lock()
	manifestData, manifestExists := mock.manifests["myrepo/v1.0"]
	mock.mu.Unlock()
	if !manifestExists {
		t.Fatal("manifest not found in registry")
	}

	var m Manifest
	if err := json.Unmarshal(manifestData, &m); err != nil {
		t.Fatalf("unmarshal manifest: %v", err)
	}

	if m.SchemaVersion != 2 {
		t.Errorf("SchemaVersion = %d, want 2", m.SchemaVersion)
	}
	if m.MediaType != MediaTypeOCIManifest {
		t.Errorf("MediaType = %q, want %q", m.MediaType, MediaTypeOCIManifest)
	}
	if m.Config.MediaType != MediaTypeModelConfig {
		t.Errorf("Config.MediaType = %q, want %q", m.Config.MediaType, MediaTypeModelConfig)
	}
	if len(m.Layers) != 1 {
		t.Fatalf("Layers count = %d, want 1", len(m.Layers))
	}
	if m.Layers[0].MediaType != MediaTypeGGUF {
		t.Errorf("Layer.MediaType = %q, want %q", m.Layers[0].MediaType, MediaTypeGGUF)
	}
	if m.Layers[0].Digest != expectedDigest {
		t.Errorf("Layer.Digest = %q, want %q", m.Layers[0].Digest, expectedDigest)
	}
	if m.Layers[0].Size != int64(len(modelData)) {
		t.Errorf("Layer.Size = %d, want %d", m.Layers[0].Size, len(modelData))
	}
}

func TestRegistry_Pull(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	// First push a model.
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	modelData := []byte("gguf-data-for-pull-test")
	if err := os.WriteFile(modelPath, modelData, 0o600); err != nil {
		t.Fatal(err)
	}

	if err := reg.Push(context.Background(), "registry.example.com/models/llama:v2", modelPath); err != nil {
		t.Fatalf("Push error: %v", err)
	}

	// Now pull it.
	destPath := filepath.Join(dir, "pulled-model.gguf")
	if err := reg.Pull(context.Background(), "registry.example.com/models/llama:v2", destPath); err != nil {
		t.Fatalf("Pull error: %v", err)
	}

	// Verify the pulled file matches.
	pulled, err := os.ReadFile(destPath)
	if err != nil {
		t.Fatalf("read pulled file: %v", err)
	}
	if string(pulled) != string(modelData) {
		t.Errorf("pulled data = %q, want %q", string(pulled), string(modelData))
	}
}

func TestRegistry_Tags(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	// Push two versions.
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("data"), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, tag := range []string{"v1.0", "v2.0", "latest"} {
		ref := "registry.example.com/models/gemma:" + tag
		if err := reg.Push(context.Background(), ref, modelPath); err != nil {
			t.Fatalf("Push(%s) error: %v", tag, err)
		}
	}

	// List tags.
	tags, err := reg.Tags(context.Background(), "models/gemma")
	if err != nil {
		t.Fatalf("Tags error: %v", err)
	}

	if len(tags) != 3 {
		t.Fatalf("Tags count = %d, want 3", len(tags))
	}

	want := map[string]bool{"v1.0": true, "v2.0": true, "latest": true}
	for _, tag := range tags {
		if !want[tag] {
			t.Errorf("unexpected tag %q", tag)
		}
	}
}

func TestRegistry_Resolve(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	modelData := []byte("resolve-test-data")
	if err := os.WriteFile(modelPath, modelData, 0o600); err != nil {
		t.Fatal(err)
	}

	if err := reg.Push(context.Background(), "registry.example.com/repo:v1", modelPath); err != nil {
		t.Fatalf("Push error: %v", err)
	}

	m, err := reg.Resolve(context.Background(), "registry.example.com/repo:v1")
	if err != nil {
		t.Fatalf("Resolve error: %v", err)
	}

	if m.SchemaVersion != 2 {
		t.Errorf("SchemaVersion = %d, want 2", m.SchemaVersion)
	}
	if len(m.Layers) != 1 {
		t.Fatalf("Layers count = %d, want 1", len(m.Layers))
	}
	if m.Layers[0].MediaType != MediaTypeGGUF {
		t.Errorf("Layer.MediaType = %q, want %q", m.Layers[0].MediaType, MediaTypeGGUF)
	}
}

func TestRegistry_Push_FileNotFound(t *testing.T) {
	reg := NewRegistry("http://localhost:0")
	err := reg.Push(context.Background(), "registry.example.com/repo:v1", "/nonexistent/model.gguf")
	if err == nil {
		t.Error("Push should fail for nonexistent file")
	}
}

func TestRegistry_Push_InvalidRef(t *testing.T) {
	reg := NewRegistry("http://localhost:0")
	err := reg.Push(context.Background(), "invalidref", "/nonexistent/model.gguf")
	if err == nil {
		t.Error("Push should fail for invalid reference")
	}
}

func TestRegistry_Pull_ManifestNotFound(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	err := reg.Pull(context.Background(), "registry.example.com/repo:nonexistent", "/tmp/out.gguf")
	if err == nil {
		t.Error("Pull should fail for nonexistent manifest")
	}
}

func TestRegistry_Tags_EmptyRepo(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	tags, err := reg.Tags(context.Background(), "empty/repo")
	if err != nil {
		t.Fatalf("Tags error: %v", err)
	}
	if len(tags) != 0 {
		t.Errorf("Tags count = %d, want 0", len(tags))
	}
}

func TestRegistry_WithCredentials(t *testing.T) {
	var gotUser, gotPass string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUser, gotPass, _ = r.BasicAuth()
		json.NewEncoder(w).Encode(TagList{Name: "repo", Tags: []string{"v1"}}) //nolint:errcheck
	}))
	defer server.Close()

	reg := NewRegistry(server.URL,
		WithCredentials("myuser", "mypass"),
		WithHTTPClient(server.Client()),
	)

	_, err := reg.Tags(context.Background(), "repo")
	if err != nil {
		t.Fatalf("Tags error: %v", err)
	}

	if gotUser != "myuser" {
		t.Errorf("username = %q, want %q", gotUser, "myuser")
	}
	if gotPass != "mypass" {
		t.Errorf("password = %q, want %q", gotPass, "mypass")
	}
}

func TestRegistry_DefaultTag(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("data"), 0o600); err != nil {
		t.Fatal(err)
	}

	// Push without tag — should default to "latest".
	if err := reg.Push(context.Background(), "registry.example.com/repo", modelPath); err != nil {
		t.Fatalf("Push error: %v", err)
	}

	// Resolve using explicit "latest".
	_, err := reg.Resolve(context.Background(), "registry.example.com/repo:latest")
	if err != nil {
		t.Fatalf("Resolve(latest) error: %v", err)
	}
}

func TestRegistry_DigestReference(t *testing.T) {
	mock := newMockOCIRegistry()
	server := httptest.NewServer(mock.handler())
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	modelData := []byte("digest-ref-test")
	if err := os.WriteFile(modelPath, modelData, 0o600); err != nil {
		t.Fatal(err)
	}

	if err := reg.Push(context.Background(), "registry.example.com/repo:v1", modelPath); err != nil {
		t.Fatalf("Push error: %v", err)
	}

	// Store the manifest under the digest key too (simulating registry behavior).
	h := sha256.Sum256([]byte("digest-ref-test"))
	digest := "sha256:" + hex.EncodeToString(h[:])
	mock.mu.Lock()
	if data, ok := mock.manifests["repo/v1"]; ok {
		mock.manifests["repo/"+digest] = data
	}
	mock.mu.Unlock()

	// Resolve by digest.
	m, err := reg.Resolve(context.Background(), "registry.example.com/repo@"+digest)
	if err != nil {
		t.Fatalf("Resolve by digest error: %v", err)
	}
	if m.SchemaVersion != 2 {
		t.Errorf("SchemaVersion = %d, want 2", m.SchemaVersion)
	}
}

func TestParseReference(t *testing.T) {
	tests := []struct {
		input   string
		want    Reference
		wantErr bool
	}{
		{
			input: "registry.example.com/repo:v1",
			want:  Reference{Registry: "registry.example.com", Repository: "repo", Tag: "v1"},
		},
		{
			input: "registry.example.com/org/repo:latest",
			want:  Reference{Registry: "registry.example.com", Repository: "org/repo", Tag: "latest"},
		},
		{
			input: "registry.example.com/repo@sha256:abc123",
			want:  Reference{Registry: "registry.example.com", Repository: "repo", Digest: "sha256:abc123"},
		},
		{
			input: "registry.example.com/repo",
			want:  Reference{Registry: "registry.example.com", Repository: "repo", Tag: "latest"},
		},
		{
			input:   "invalidref",
			wantErr: true,
		},
		{
			input:   "registry.example.com/../../etc/passwd:latest",
			wantErr: true,
		},
		{
			input:   "registry.example.com/repo/../secret:v1",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		got, err := parseReference(tc.input)
		if tc.wantErr {
			if err == nil {
				t.Errorf("parseReference(%q) expected error", tc.input)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseReference(%q) error: %v", tc.input, err)
			continue
		}
		if got.Registry != tc.want.Registry {
			t.Errorf("parseReference(%q).Registry = %q, want %q", tc.input, got.Registry, tc.want.Registry)
		}
		if got.Repository != tc.want.Repository {
			t.Errorf("parseReference(%q).Repository = %q, want %q", tc.input, got.Repository, tc.want.Repository)
		}
		if got.Tag != tc.want.Tag {
			t.Errorf("parseReference(%q).Tag = %q, want %q", tc.input, got.Tag, tc.want.Tag)
		}
		if got.Digest != tc.want.Digest {
			t.Errorf("parseReference(%q).Digest = %q, want %q", tc.input, got.Digest, tc.want.Digest)
		}
	}
}

func TestRegistry_GetBlob_OversizedResponse(t *testing.T) {
	// Temporarily lower maxBlobSize so we can test the rejection without
	// allocating 20 GB of memory.
	orig := maxBlobSize
	maxBlobSize = 1024
	defer func() { maxBlobSize = orig }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		// Write maxBlobSize+1 bytes to exceed the limit.
		w.Write(make([]byte, 1025)) //nolint:errcheck
	}))
	defer server.Close()

	reg := NewRegistry(server.URL, WithHTTPClient(server.Client()))

	_, err := reg.getBlob(context.Background(), "repo", "sha256:fake")
	if err == nil {
		t.Fatal("getBlob should reject oversized blob")
	}
	if !strings.Contains(err.Error(), "exceeds maximum size") {
		t.Errorf("unexpected error: %v", err)
	}

	// Verify the production default is 20 GB.
	if orig != 20<<30 {
		t.Errorf("default maxBlobSize = %d, want %d", orig, 20<<30)
	}
}

func TestSha256Digest(t *testing.T) {
	data := []byte("hello world")
	got := sha256Digest(data)
	h := sha256.Sum256(data)
	want := "sha256:" + hex.EncodeToString(h[:])
	if got != want {
		t.Errorf("sha256Digest = %q, want %q", got, want)
	}
}
