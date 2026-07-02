package huggingface

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestServer(t *testing.T, handler http.HandlerFunc) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c := &Client{
		httpClient: srv.Client(),
		baseURL:    srv.URL,
		token:      "test-token",
	}
	return c, srv
}

var testModelResponse = apiResponse{
	ID: "google/gemma-3-1b-it",
	Siblings: []apiSibling{
		{RFilename: "config.json", Size: 1024},
		{RFilename: "gemma-3-1b-it-Q4_K_M.gguf", Size: 800_000_000},
		{RFilename: "gemma-3-1b-it-Q8_0.gguf", Size: 1_200_000_000},
		{RFilename: "gemma-3-1b-it-F16.gguf", Size: 2_400_000_000},
		{RFilename: "README.md", Size: 512},
	},
}

func TestGetModel(t *testing.T) {
	tests := []struct {
		name      string
		id        string
		status    int
		resp      any
		wantErr   bool
		wantFiles int
	}{
		{
			name:      "success",
			id:        "google/gemma-3-1b-it",
			status:    200,
			resp:      testModelResponse,
			wantFiles: 5,
		},
		{
			name:    "not found",
			id:      "nonexistent/model",
			status:  404,
			resp:    map[string]string{"error": "not found"},
			wantErr: true,
		},
		{
			name:    "server error",
			id:      "broken/model",
			status:  500,
			resp:    map[string]string{"error": "internal"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				json.NewEncoder(w).Encode(tt.resp)
			})

			info, err := c.GetModel(tt.id)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if info.ID != "google/gemma-3-1b-it" {
				t.Errorf("got ID %q, want %q", info.ID, "google/gemma-3-1b-it")
			}
			if len(info.Files) != tt.wantFiles {
				t.Errorf("got %d files, want %d", len(info.Files), tt.wantFiles)
			}
		})
	}
}

func TestGetModel_AuthorizationHeader(t *testing.T) {
	var gotAuth string
	c, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		json.NewEncoder(w).Encode(testModelResponse)
	})

	_, err := c.GetModel("google/gemma-3-1b-it")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "Bearer test-token" {
		t.Errorf("got Authorization %q, want %q", gotAuth, "Bearer test-token")
	}
}

func TestGetModel_NoToken(t *testing.T) {
	var gotAuth string
	c, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		json.NewEncoder(w).Encode(testModelResponse)
	})
	c.token = ""

	_, err := c.GetModel("google/gemma-3-1b-it")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "" {
		t.Errorf("expected no Authorization header, got %q", gotAuth)
	}
}

func TestListGGUFFiles(t *testing.T) {
	c, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(testModelResponse)
	})

	files, err := c.ListGGUFFiles("google/gemma-3-1b-it")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(files) != 3 {
		t.Fatalf("got %d GGUF files, want 3", len(files))
	}
	for _, f := range files {
		if f.Filename == "config.json" || f.Filename == "README.md" {
			t.Errorf("non-GGUF file %q in results", f.Filename)
		}
	}
}

func TestResolveGGUF(t *testing.T) {
	tests := []struct {
		name     string
		quant    string
		wantFile string
		wantErr  bool
	}{
		{
			name:     "exact Q4_K_M",
			quant:    "Q4_K_M",
			wantFile: "gemma-3-1b-it-Q4_K_M.gguf",
		},
		{
			name:     "exact Q8_0",
			quant:    "Q8_0",
			wantFile: "gemma-3-1b-it-Q8_0.gguf",
		},
		{
			name:     "case insensitive",
			quant:    "q4_k_m",
			wantFile: "gemma-3-1b-it-Q4_K_M.gguf",
		},
		{
			name:     "F16",
			quant:    "F16",
			wantFile: "gemma-3-1b-it-F16.gguf",
		},
		{
			name:    "no match",
			quant:   "Q2_K",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(testModelResponse)
			})

			f, err := c.ResolveGGUF("google/gemma-3-1b-it", tt.quant)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if f.Filename != tt.wantFile {
				t.Errorf("got %q, want %q", f.Filename, tt.wantFile)
			}
		})
	}
}

func TestListGGUFFiles_NoGGUF(t *testing.T) {
	c, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(apiResponse{
			ID: "org/model",
			Siblings: []apiSibling{
				{RFilename: "config.json", Size: 100},
				{RFilename: "model.safetensors", Size: 5000},
			},
		})
	})

	files, err := c.ListGGUFFiles("org/model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(files) != 0 {
		t.Errorf("got %d files, want 0", len(files))
	}
}

func TestGetModel_RequestPath(t *testing.T) {
	var gotPath string
	c, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewEncoder(w).Encode(testModelResponse)
	})

	_, err := c.GetModel("google/gemma-3-1b-it")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "/api/models/google/gemma-3-1b-it"
	if gotPath != want {
		t.Errorf("got path %q, want %q", gotPath, want)
	}
}
