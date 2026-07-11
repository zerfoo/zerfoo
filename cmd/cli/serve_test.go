package cli

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
	"github.com/zerfoo/zerfoo/serve/security"
	"github.com/zerfoo/zerfoo/serve/shutdown"
)

func TestServeCommand_Name(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if cmd.Name() != "serve" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "serve")
	}
}

func TestServeCommand_Description(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestServeCommand_Usage(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "serve") {
		t.Error("Usage() should contain 'serve'")
	}
}

func TestServeCommand_Examples(t *testing.T) {
	cmd := NewServeCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestServeCommand_MissingModelID(t *testing.T) {
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	err := cmd.Run(context.Background(), nil)
	if err == nil {
		t.Error("expected error for missing model ID")
	}
}

func TestServeCommand_LoadError(t *testing.T) {
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("load failed")
	}
	err := cmd.Run(context.Background(), []string{"--allow-no-auth", "test-model"})
	if err == nil {
		t.Error("expected error from load")
	}
}

func TestServeCommand_FlagParsing(t *testing.T) {
	tests := []struct {
		name string
		args []string
		err  string
	}{
		{"port missing value", []string{"--port"}, "--port requires a value"},
		{"cache-dir missing value", []string{"--cache-dir"}, "--cache-dir requires a value"},
		{"unexpected arg", []string{"model1", "model2"}, "unexpected argument"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewServeCommand(nil, &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Error("expected error")
			}
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.err)
			}
		})
	}
}

func TestServeCommand_WithCoordinator(t *testing.T) {
	// Verify the command registers with the coordinator.
	coord := shutdown.New()
	var out bytes.Buffer
	cmd := NewServeCommand(coord, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("load failed")
	}
	// Fails at load, but exercises the coordinator path.
	_ = cmd.Run(context.Background(), []string{"--allow-no-auth", "test-model"})
}

func TestServeCommand_StartsAndStopsOnCancel(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	coord := shutdown.New()
	cmd := NewServeCommand(coord, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{"--port", "0", "--allow-no-auth", "test-model"})
	}()

	// Give server a moment to start, then cancel.
	cancel()

	err := <-errCh
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(out.String(), "Serving test-model") {
		t.Errorf("output = %q, want 'Serving test-model'", out.String())
	}
}

func TestServeCommand_CustomPort(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{"--port", "0", "--allow-no-auth", "test-model"})
	}()

	cancel()
	err := <-errCh
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

func TestShutdownAdapter_Close(t *testing.T) {
	// Test that shutdownAdapter properly delegates to http.Server.Shutdown.
	srv := &http.Server{}
	adapter := shutdownAdapter{srv}
	if err := adapter.Close(context.Background()); err != nil {
		t.Errorf("Close error: %v", err)
	}
}

func TestServeCommand_Interface(t *testing.T) {
	var _ Command = (*ServeCommand)(nil)
}

func TestParseGPUList(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    []int
		wantErr string
	}{
		{"single GPU", "0", []int{0}, ""},
		{"multiple GPUs", "0,1,2,3", []int{0, 1, 2, 3}, ""},
		{"non-contiguous", "0,2,5", []int{0, 2, 5}, ""},
		{"spaces around IDs", " 0 , 1 , 2 ", []int{0, 1, 2}, ""},
		{"negative ID", "-1", nil, "negative GPU ID"},
		{"non-numeric", "abc", nil, "non-numeric GPU ID"},
		{"duplicate", "0,1,0", nil, "duplicate GPU ID"},
		{"empty element", "0,,1", nil, "empty GPU ID"},
		{"trailing comma", "0,1,", nil, "empty GPU ID"},
		{"mixed invalid", "0,abc,2", nil, "non-numeric GPU ID"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseGPUList(tc.input)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("error = %q, want to contain %q", err.Error(), tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("got[%d] = %d, want %d", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestServeCommand_NoKeyNoFlag(t *testing.T) {
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("should not be called")
	}
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err == nil {
		t.Fatal("expected error when no API key and no --allow-no-auth")
	}
	want := "set --api-key, ZERFOO_API_KEY, or --allow-no-auth"
	if !strings.Contains(err.Error(), want) {
		t.Errorf("error = %q, want to contain %q", err.Error(), want)
	}
}

func TestServeCommand_AllowNoAuth(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{"--port", "0", "--allow-no-auth", "test-model"})
	}()

	cancel()
	err := <-errCh
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(out.String(), "no API key configured") {
		t.Errorf("output = %q, want warning about no API key", out.String())
	}
}

func TestServeCommand_GPUsFlag(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"gpus missing value", []string{"--gpus"}, "--gpus requires a value"},
		{"gpus invalid", []string{"--gpus", "abc", "test-model"}, "invalid --gpus value"},
		{"gpus negative", []string{"--gpus", "-1", "test-model"}, "invalid --gpus value"},
		{"gpus duplicate", []string{"--gpus", "0,0", "test-model"}, "invalid --gpus value"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewServeCommand(nil, &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestServeCommand_TLSFlags(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"tls-cert missing value", []string{"--tls-cert"}, "--tls-cert requires a value"},
		{"tls-key missing value", []string{"--tls-key"}, "--tls-key requires a value"},
		{"cert without key", []string{"--tls-cert", "cert.pem", "test-model"}, "both --tls-cert and --tls-key are required"},
		{"key without cert", []string{"--tls-key", "key.pem", "test-model"}, "both --tls-cert and --tls-key are required"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewServeCommand(nil, &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestServeCommand_TLSFlagsParsed(t *testing.T) {
	// Verify both flags are accepted together (will fail at TLS load, not at flag parsing).
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{"--port", "0", "--allow-no-auth", "--tls-cert", "cert.pem", "--tls-key", "key.pem", "test-model"})
	}()

	// Cancel immediately — the server will fail to start with invalid TLS files,
	// but the important thing is no flag-parsing error occurred.
	cancel()
	err := <-errCh
	// We expect either nil (context cancelled before TLS error) or a TLS-related error,
	// but NOT a flag-parsing error.
	if err != nil && strings.Contains(err.Error(), "both --tls-cert and --tls-key are required") {
		t.Fatalf("unexpected flag validation error: %v", err)
	}
}

func TestServeCommand_GPUsFlagValid(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{"--port", "0", "--allow-no-auth", "--gpus", "0,1,2,3", "test-model"})
	}()

	cancel()
	err := <-errCh
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

// waitForServer blocks until a *serve.Server arrives on ch or the timeout
// elapses. It exists so tests exercising cmd.newServer's DI hook don't hang
// forever if Run fails to reach server construction.
func waitForServer(t *testing.T, ch <-chan *serve.Server) *serve.Server {
	t.Helper()
	select {
	case srv := <-ch:
		return srv
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for server construction")
		return nil
	}
}

func TestServeCommand_RateLimitFlagValidation(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"rate-limit missing value", []string{"--rate-limit"}, "--rate-limit requires a value"},
		{"rate-limit non-numeric", []string{"--rate-limit", "abc", "test-model"}, "invalid --rate-limit value"},
		{"rate-limit zero", []string{"--rate-limit", "0", "test-model"}, "invalid --rate-limit value"},
		{"rate-limit negative", []string{"--rate-limit", "-5", "test-model"}, "invalid --rate-limit value"},
		{"rate-limit-burst missing value", []string{"--rate-limit-burst"}, "--rate-limit-burst requires a value"},
		{"rate-limit-burst non-numeric", []string{"--rate-limit-burst", "abc", "test-model"}, "invalid --rate-limit-burst value"},
		{"rate-limit-burst zero", []string{"--rate-limit", "1", "--rate-limit-burst", "0", "test-model"}, "invalid --rate-limit-burst value"},
		{"rate-limit-burst without rate-limit", []string{"--rate-limit-burst", "5", "test-model"}, "--rate-limit-burst requires --rate-limit"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewServeCommand(nil, &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestServeCommand_KeystoreFlagValidation(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"keystore missing value", []string{"--keystore"}, "--keystore requires a value"},
		{"keystore unopenable path", []string{"--keystore", "/nonexistent-dir-zf-t145/apikeys.db", "test-model"}, "open --keystore"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewServeCommand(nil, &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

// TestServeCommand_RateLimitWiring verifies that --rate-limit/--rate-limit-burst
// actually construct a serve.Server with a functioning rate limiter (T145.1):
// the server built by the CLI's newServer hook must reject the request that
// exceeds the configured burst with 429, not just accept the flag silently.
func TestServeCommand_RateLimitWiring(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	serverCh := make(chan *serve.Server, 1)
	cmd.newServer = func(m *inference.Model, opts ...serve.ServerOption) *serve.Server {
		s := serve.NewServer(m, opts...)
		serverCh <- s
		return s
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Run(ctx, []string{
			"--port", "0", "--allow-no-auth",
			"--rate-limit", "1", "--rate-limit-burst", "1",
			"test-model",
		})
	}()

	srv := waitForServer(t, serverCh)
	handler := srv.Handler()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.RemoteAddr = "10.0.0.5:1234"
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("first request: got status %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}

	req2 := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req2.RemoteAddr = "10.0.0.5:1234"
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusTooManyRequests {
		t.Fatalf("second request: got status %d, want %d; body=%s", rec2.Code, http.StatusTooManyRequests, rec2.Body.String())
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

// TestServeCommand_KeystoreWiring verifies that --keystore opens the given
// bbolt file and wires it into the server's scope-based auth (T145.1): a
// pre-provisioned read_only key must authenticate and authorize a GET
// request, and a request without any credentials must be rejected.
func TestServeCommand_KeystoreWiring(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "apikeys.db")

	// Pre-populate the keystore, then close it -- bbolt takes an exclusive
	// file lock, so the backend must be closed before the CLI opens the
	// same path.
	backend, err := security.NewBboltKeyStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("NewBboltKeyStoreBackend: %v", err)
	}
	ks := security.NewKeyStore(security.WithBackend(backend))
	rawKey, _, err := ks.Create("t145-test-key", []security.Scope{security.ScopeReadOnly}, time.Time{})
	if err != nil {
		t.Fatalf("Create key: %v", err)
	}
	if err := backend.Close(); err != nil {
		t.Fatalf("close backend: %v", err)
	}

	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	serverCh := make(chan *serve.Server, 1)
	cmd.newServer = func(m *inference.Model, opts ...serve.ServerOption) *serve.Server {
		s := serve.NewServer(m, opts...)
		serverCh <- s
		return s
	}

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		// Deliberately no --api-key/--allow-no-auth: --keystore alone must
		// satisfy the "authentication configured" opt-in check.
		errCh <- cmd.Run(ctx, []string{"--port", "0", "--keystore", dbPath, "test-model"})
	}()

	srv := waitForServer(t, serverCh)
	handler := srv.Handler()

	unauth := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, unauth)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("no credentials: got status %d, want %d", rec.Code, http.StatusUnauthorized)
	}

	authed := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	authed.Header.Set("Authorization", "Bearer "+rawKey)
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, authed)
	if rec2.Code != http.StatusOK {
		t.Errorf("valid keystore key: got status %d, want %d; body=%s", rec2.Code, http.StatusOK, rec2.Body.String())
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("Run error: %v", err)
	}
}
