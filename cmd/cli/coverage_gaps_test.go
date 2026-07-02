package cli

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
)

// ---------------------------------------------------------------------------
// ServeCommand: --cache-dir option path
// ---------------------------------------------------------------------------

func TestServeCommand_WithCacheDir(t *testing.T) {
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, opts ...inference.Option) (*inference.Model, error) {
		// Verify option was passed (we can't inspect it but at least exercise the path).
		return nil, errors.New("expected load fail")
	}
	err := cmd.Run(context.Background(), []string{"--allow-no-auth", "--cache-dir", "/tmp/test-cache", "test-model"})
	if err == nil {
		t.Error("expected error from load")
	}
	if !strings.Contains(err.Error(), "load model") {
		t.Errorf("error = %q, want load model error", err.Error())
	}
}

// ---------------------------------------------------------------------------
// ServeCommand: ListenAndServe returns a non-ErrServerClosed error
// ---------------------------------------------------------------------------

func TestServeCommand_ListenAndServeError(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewServeCommand(nil, &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}

	// Use an invalid port to force ListenAndServe to fail.
	err := cmd.Run(context.Background(), []string{"--allow-no-auth", "--port", "invalid-port", "test-model"})
	if err == nil {
		t.Error("expected error from invalid port")
	}
}
