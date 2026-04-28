package serve

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/serve/security"
)

func TestScopeAuthorization(t *testing.T) {
	mdl := buildTestModel(t)
	ks := security.NewKeyStore()

	inferenceKey, _, err := ks.Create("inference-key", []security.Scope{security.ScopeInference, security.ScopeReadOnly}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	adminKey, _, err := ks.Create("admin-key", []security.Scope{security.ScopeAdmin, security.ScopeInference, security.ScopeReadOnly}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	readOnlyKey, _, err := ks.Create("readonly-key", []security.Scope{security.ScopeReadOnly}, time.Time{})
	if err != nil {
		t.Fatal(err)
	}

	srv := NewServer(mdl, WithKeyStore(ks))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	doReq := func(t *testing.T, method, url, token string) *http.Response {
		t.Helper()
		var body *strings.Reader
		if method == http.MethodPost {
			body = strings.NewReader(`{"model":"test","messages":[{"role":"user","content":"hi"}]}`)
		}
		var req *http.Request
		var reqErr error
		if body != nil {
			req, reqErr = http.NewRequestWithContext(context.Background(), method, url, body)
		} else {
			req, reqErr = http.NewRequestWithContext(context.Background(), method, url, http.NoBody)
		}
		if reqErr != nil {
			t.Fatal(reqErr)
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		if method == http.MethodPost {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, doErr := http.DefaultClient.Do(req)
		if doErr != nil {
			t.Fatal(doErr)
		}
		return resp
	}

	t.Run("no token returns 401", func(t *testing.T) {
		resp := doReq(t, http.MethodGet, ts.URL+"/v1/models", "")
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401", resp.StatusCode)
		}
	})

	t.Run("unknown token returns 401", func(t *testing.T) {
		resp := doReq(t, http.MethodGet, ts.URL+"/v1/models", "bad-key")
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401", resp.StatusCode)
		}
	})

	t.Run("inference key can POST /v1/chat/completions", func(t *testing.T) {
		resp := doReq(t, http.MethodPost, ts.URL+"/v1/chat/completions", inferenceKey)
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusUnauthorized {
			t.Fatalf("status = %d, want not 401/403", resp.StatusCode)
		}
	})

	t.Run("inference key cannot DELETE /v1/models/test", func(t *testing.T) {
		resp := doReq(t, http.MethodDelete, ts.URL+"/v1/models/test", inferenceKey)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("status = %d, want 403", resp.StatusCode)
		}
	})

	t.Run("admin key can DELETE /v1/models/test", func(t *testing.T) {
		resp := doReq(t, http.MethodDelete, ts.URL+"/v1/models/test", adminKey)
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusUnauthorized {
			t.Fatalf("status = %d, want not 401/403", resp.StatusCode)
		}
	})

	t.Run("admin key can POST /v1/chat/completions", func(t *testing.T) {
		resp := doReq(t, http.MethodPost, ts.URL+"/v1/chat/completions", adminKey)
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusUnauthorized {
			t.Fatalf("status = %d, want not 401/403", resp.StatusCode)
		}
	})

	t.Run("readonly key can GET /v1/models", func(t *testing.T) {
		resp := doReq(t, http.MethodGet, ts.URL+"/v1/models", readOnlyKey)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
	})

	t.Run("readonly key cannot POST /v1/chat/completions", func(t *testing.T) {
		resp := doReq(t, http.MethodPost, ts.URL+"/v1/chat/completions", readOnlyKey)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("status = %d, want 403", resp.StatusCode)
		}
	})

	t.Run("GET /v1/unknown-future-endpoint requires ScopeReadOnly", func(t *testing.T) {
		resp := doReq(t, http.MethodGet, ts.URL+"/v1/unknown-future-endpoint", readOnlyKey)
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusUnauthorized {
			t.Fatalf("status = %d, want not 401/403", resp.StatusCode)
		}
	})

	t.Run("no-scope key rejected for /v1/ paths", func(t *testing.T) {
		noScopeKey, _, err := ks.Create("no-scope-key", []security.Scope{}, time.Time{})
		if err != nil {
			t.Fatal(err)
		}
		resp := doReq(t, http.MethodGet, ts.URL+"/v1/models", noScopeKey)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("status = %d, want 403", resp.StatusCode)
		}
	})

	t.Run("metrics skips auth with keystore", func(t *testing.T) {
		resp := doReq(t, http.MethodGet, ts.URL+"/metrics", "")
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
	})
}

func TestStaticAPIKeySkipsScopeChecks(t *testing.T) {
	mdl := buildTestModel(t)
	const apiKey = "static-secret"
	srv := NewServer(mdl, WithAPIKey(apiKey))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, err := http.NewRequestWithContext(context.Background(), http.MethodDelete, ts.URL+"/v1/models/test", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusForbidden {
		t.Fatalf("status = 403, static key mode should not enforce scopes")
	}
}

func TestRequiredScope(t *testing.T) {
	tests := []struct {
		method string
		path   string
		want   security.Scope
	}{
		{http.MethodDelete, "/v1/models/test", security.ScopeAdmin},
		{http.MethodDelete, "/v1/models", security.ScopeAdmin},
		{http.MethodPost, "/v1/chat/completions", security.ScopeInference},
		{http.MethodPost, "/v1/completions", security.ScopeInference},
		{http.MethodPost, "/v1/embeddings", security.ScopeInference},
		{http.MethodPost, "/v1/audio/transcriptions", security.ScopeInference},
		{http.MethodGet, "/v1/models", security.ScopeReadOnly},
		{http.MethodGet, "/v1/models/test", security.ScopeReadOnly},
		{http.MethodGet, "/v1/unknown-future-endpoint", security.ScopeReadOnly},
		{http.MethodGet, "/metrics", ""},
		{http.MethodGet, "/healthz", ""},
	}
	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			got := requiredScope(tt.method, tt.path)
			if got != tt.want {
				t.Fatalf("requiredScope(%q, %q) = %q, want %q", tt.method, tt.path, got, tt.want)
			}
		})
	}
}
