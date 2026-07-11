package serve

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/log"
)

func TestLogging_SuccessfulRequest(t *testing.T) {
	mdl := buildTestModel(t)
	var buf bytes.Buffer
	logger := log.New(&buf, log.LevelInfo, log.FormatText)
	srv := NewServer(mdl, WithLogger(logger))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models")
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	output := buf.String()
	for _, want := range []string{"INFO", "request completed", "method=GET", "path=/v1/models", "status_code=200", "latency_ms="} {
		if !strings.Contains(output, want) {
			t.Errorf("log output missing %q, got: %s", want, output)
		}
	}
}

func TestLogging_ClientError(t *testing.T) {
	mdl := buildTestModel(t)
	var buf bytes.Buffer
	logger := log.New(&buf, log.LevelInfo, log.FormatText)
	srv := NewServer(mdl, WithLogger(logger))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", "invalid")
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}

	output := buf.String()
	for _, want := range []string{"WARN", "status_code=400"} {
		if !strings.Contains(output, want) {
			t.Errorf("log output missing %q, got: %s", want, output)
		}
	}
}

func TestLogging_ServerError(t *testing.T) {
	mdl := buildErrorModel(t)
	var buf bytes.Buffer
	logger := log.New(&buf, log.LevelInfo, log.FormatText)
	srv := NewServer(mdl, WithLogger(logger))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", resp.StatusCode)
	}

	output := buf.String()
	for _, want := range []string{"ERROR", "status_code=500"} {
		if !strings.Contains(output, want) {
			t.Errorf("log output missing %q, got: %s", want, output)
		}
	}
}

func TestLogging_NotFound(t *testing.T) {
	mdl := buildTestModel(t)
	var buf bytes.Buffer
	logger := log.New(&buf, log.LevelInfo, log.FormatText)
	srv := NewServer(mdl, WithLogger(logger))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models/nonexistent")
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}

	output := buf.String()
	for _, want := range []string{"WARN", "status_code=404"} {
		if !strings.Contains(output, want) {
			t.Errorf("log output missing %q, got: %s", want, output)
		}
	}
}

func TestLogging_JSONFormat(t *testing.T) {
	mdl := buildTestModel(t)
	var buf bytes.Buffer
	logger := log.New(&buf, log.LevelInfo, log.FormatJSON)
	srv := NewServer(mdl, WithLogger(logger))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models")
	_ = resp.Body.Close()

	var entry map[string]string
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("JSON decode error: %v, output: %s", err, buf.String())
	}

	requiredKeys := []string{"method", "path", "model", "prompt_tokens", "completion_tokens", "latency_ms", "status_code"}
	for _, key := range requiredKeys {
		if _, ok := entry[key]; !ok {
			t.Errorf("JSON log missing key %q, got: %v", key, entry)
		}
	}
	if entry["status_code"] != "200" {
		t.Errorf("status_code = %q, want %q", entry["status_code"], "200")
	}
}

func TestLogging_NopLogger(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl) // no WithLogger, should use Nop
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models")
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	// No panic or error means Nop logger works correctly.
}

// TestLogging_EscapesControlCharsInPath guards against SERVE-6: logMiddleware
// must log the percent-encoded request path (r.URL.EscapedPath()), not the
// percent-decoded r.URL.Path, so a request path carrying CR/LF (or other
// control characters) cannot forge or split log lines.
func TestLogging_EscapesControlCharsInPath(t *testing.T) {
	mdl := buildTestModel(t)
	var buf bytes.Buffer
	logger := log.New(&buf, log.LevelInfo, log.FormatText)
	srv := NewServer(mdl, WithLogger(logger))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	const injected = "FAKE-INJECTED-LOG-LINE"
	resp := doGet(t, ts.URL+"/v1/models%0D%0A"+injected)
	_ = resp.Body.Close()

	output := buf.String()

	// The decoded form must never appear: a raw CR/LF immediately followed
	// by attacker-controlled text would fabricate what looks like a second,
	// independent log line.
	if strings.Contains(output, "\r\n"+injected) {
		t.Fatalf("raw control chars leaked into log output (log injection): %q", output)
	}

	// Exactly one log line should have been written; a raw CR/LF in the path
	// would otherwise split it into two.
	lines := strings.Split(strings.TrimRight(output, "\n"), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected exactly one log line, got %d: %q", len(lines), output)
	}

	// The escaped, percent-encoded form must be present in the path field.
	if !strings.Contains(lines[0], "path=/v1/models%0D%0A"+injected) {
		t.Errorf("expected escaped path in log line, got: %q", lines[0])
	}
}

func TestLogging_StructuredFields(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		path       string
		body       string
		wantFields map[string]string
	}{
		{
			name:   "chat completions",
			method: http.MethodPost,
			path:   "/v1/chat/completions",
			body:   `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`,
			wantFields: map[string]string{
				"method": "POST",
				"path":   "/v1/chat/completions",
			},
		},
		{
			name:   "completions",
			method: http.MethodPost,
			path:   "/v1/completions",
			body:   `{"prompt":"hello","max_tokens":5}`,
			wantFields: map[string]string{
				"method": "POST",
				"path":   "/v1/completions",
			},
		},
		{
			name:   "models list",
			method: http.MethodGet,
			path:   "/v1/models",
			wantFields: map[string]string{
				"method": "GET",
				"path":   "/v1/models",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mdl := buildTestModel(t)
			var buf bytes.Buffer
			logger := log.New(&buf, log.LevelInfo, log.FormatJSON)
			srv := NewServer(mdl, WithLogger(logger))
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			var resp *http.Response
			switch tt.method {
			case http.MethodGet:
				resp = doGet(t, ts.URL+tt.path)
			case http.MethodPost:
				resp = doPost(t, ts.URL+tt.path, "application/json", tt.body)
			default:
				req, err := http.NewRequestWithContext(context.Background(), tt.method, ts.URL+tt.path, http.NoBody)
				if err != nil {
					t.Fatal(err)
				}
				resp, err = http.DefaultClient.Do(req)
				if err != nil {
					t.Fatal(err)
				}
			}
			_ = resp.Body.Close()

			var entry map[string]string
			if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
				t.Fatalf("JSON decode: %v, output: %s", err, buf.String())
			}

			for k, want := range tt.wantFields {
				if got := entry[k]; got != want {
					t.Errorf("%s = %q, want %q", k, got, want)
				}
			}
		})
	}
}
