package cli

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestRenderBar(t *testing.T) {
	tests := []struct {
		pct  int
		want string
	}{
		{0, "[                              ]"},
		{50, "[==============>               ]"},
		{100, "[==============================]"},
	}
	for _, tc := range tests {
		got := renderBar(tc.pct, 30)
		if got != tc.want {
			t.Errorf("renderBar(%d, 30) = %q, want %q", tc.pct, got, tc.want)
		}
	}
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		bytes int64
		want  string
	}{
		{0, "0B"},
		{512, "512B"},
		{1024, "1.0KB"},
		{1536, "1.5KB"},
		{1048576, "1.0MB"},
		{1073741824, "1.0GB"},
		{2684354560, "2.5GB"},
	}
	for _, tc := range tests {
		got := formatBytes(tc.bytes)
		if got != tc.want {
			t.Errorf("formatBytes(%d) = %q, want %q", tc.bytes, got, tc.want)
		}
	}
}

func TestProgressDisplay_NonTTY(t *testing.T) {
	var buf bytes.Buffer
	p := newProgressDisplay(&buf, false)

	total := int64(1000)
	// Simulate progress at various points.
	for i := int64(0); i <= total; i += 100 {
		p.callback(i, total)
	}

	output := buf.String()
	// Should have lines for 0%, 10%, 20%, ... 100%.
	for _, pct := range []string{"0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"} {
		if !strings.Contains(output, pct) {
			t.Errorf("output missing %q, got:\n%s", pct, output)
		}
	}
}

func TestProgressDisplay_TTY(t *testing.T) {
	var buf bytes.Buffer
	p := newProgressDisplay(&buf, true)

	// Force immediate writes by resetting lastWritten.
	p.callback(500, 1000)
	p.callback(1000, 1000) // 100% always prints

	output := buf.String()
	if !strings.Contains(output, "[") || !strings.Contains(output, "]") {
		t.Errorf("TTY output should contain progress bar, got: %q", output)
	}
	if !strings.Contains(output, "50%") {
		t.Errorf("output missing 50%%, got: %q", output)
	}
	if !strings.Contains(output, "100%") {
		t.Errorf("output missing 100%%, got: %q", output)
	}
}

func TestProgressDisplay_UnknownTotal(t *testing.T) {
	var buf bytes.Buffer
	p := newProgressDisplay(&buf, false)

	p.callback(1024, -1)

	output := buf.String()
	if !strings.Contains(output, "Downloaded") {
		t.Errorf("output should contain 'Downloaded', got: %q", output)
	}
	if !strings.Contains(output, "1.0KB") {
		t.Errorf("output should contain '1.0KB', got: %q", output)
	}
}

func TestProgressDisplay_CallbackInvoked(t *testing.T) {
	var buf bytes.Buffer
	p := newProgressDisplay(&buf, false)

	// Verify the callback can be called without panic and produces output.
	p.callback(0, 100)
	p.callback(50, 100)
	p.callback(100, 100)

	if buf.Len() == 0 {
		t.Error("progress callback should produce output")
	}
}

func TestIsTTY_Buffer(t *testing.T) {
	var buf bytes.Buffer
	if isTTY(&buf) {
		t.Error("bytes.Buffer should not be a TTY")
	}
}

func TestIsTTY_RegularFile(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "tty-test")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if isTTY(f) {
		t.Error("regular file should not be a TTY")
	}
}
