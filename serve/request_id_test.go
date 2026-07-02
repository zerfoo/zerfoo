package serve

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"
)

// uuidPattern matches a standard UUID v4 format.
var uuidPattern = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

func TestRequestIDEchoed(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	const wantID = "my-custom-request-id-12345"

	req, err := http.NewRequest(http.MethodGet, ts.URL+"/v1/models", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("X-Request-Id", wantID)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	got := resp.Header.Get("X-Request-Id")
	if got != wantID {
		t.Fatalf("X-Request-Id = %q, want %q", got, wantID)
	}
}

func TestRequestIDGenerated(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, err := http.NewRequest(http.MethodGet, ts.URL+"/v1/models", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}
	// Do not set X-Request-Id header.

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	got := resp.Header.Get("X-Request-Id")
	if got == "" {
		t.Fatal("expected X-Request-Id in response, got empty string")
	}
	if !uuidPattern.MatchString(got) {
		t.Fatalf("generated X-Request-Id %q does not match UUID v4 pattern", got)
	}
}
