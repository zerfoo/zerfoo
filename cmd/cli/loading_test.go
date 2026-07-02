package cli

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestLoadingIndicator_StartsAndStops(t *testing.T) {
	var buf bytes.Buffer
	li := startLoading(&buf)
	// Let it run briefly so it writes output.
	time.Sleep(50 * time.Millisecond)
	li.stop()

	out := buf.String()
	if !strings.Contains(out, "Loading model...") {
		t.Errorf("expected output to contain 'Loading model...', got %q", out)
	}
}

func TestLoadingIndicator_NonTTY(t *testing.T) {
	var buf bytes.Buffer
	li := startLoading(&buf)
	li.stop()

	out := buf.String()
	if out != "Loading model...\n" {
		t.Errorf("non-TTY output = %q, want %q", out, "Loading model...\n")
	}
}

