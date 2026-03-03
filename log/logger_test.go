package log

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestLevelString(t *testing.T) {
	tests := []struct {
		level Level
		want  string
	}{
		{LevelDebug, "DEBUG"},
		{LevelInfo, "INFO"},
		{LevelWarn, "WARN"},
		{LevelError, "ERROR"},
		{Level(99), "UNKNOWN"},
	}

	for _, tc := range tests {
		t.Run(tc.want, func(t *testing.T) {
			if got := tc.level.String(); got != tc.want {
				t.Errorf("Level(%d).String() = %q, want %q", tc.level, got, tc.want)
			}
		})
	}
}

func TestNopLogger(t *testing.T) {
	l := Nop()
	// Must not panic.
	l.Debug("msg", "k", "v")
	l.Info("msg", "k", "v")
	l.Warn("msg", "k", "v")
	l.Error("msg", "k", "v")
}

func TestNopLogger_InterfaceCompliance(t *testing.T) {
	var _ = Nop()
}

func TestStdLogger_TextFormat(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatText)

	l.Info("hello world", "key", "val")

	out := buf.String()
	if !strings.Contains(out, "INFO") {
		t.Errorf("expected INFO in output, got %q", out)
	}
	if !strings.Contains(out, "hello world") {
		t.Errorf("expected message in output, got %q", out)
	}
	if !strings.Contains(out, "key=val") {
		t.Errorf("expected key=val in output, got %q", out)
	}
}

func TestStdLogger_JSONFormat(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatJSON)

	l.Info("test message", "count", "42")

	var entry map[string]string
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("failed to parse JSON: %v\nraw: %q", err, buf.String())
	}

	if entry["level"] != "INFO" {
		t.Errorf("level = %q, want INFO", entry["level"])
	}
	if entry["msg"] != "test message" {
		t.Errorf("msg = %q, want %q", entry["msg"], "test message")
	}
	if entry["count"] != "42" {
		t.Errorf("count = %q, want %q", entry["count"], "42")
	}
}

func TestStdLogger_LevelFiltering(t *testing.T) {
	tests := []struct {
		name      string
		minLevel  Level
		logLevel  string
		expectLog bool
	}{
		{"debug at debug level", LevelDebug, "debug", true},
		{"info at debug level", LevelDebug, "info", true},
		{"warn at debug level", LevelDebug, "warn", true},
		{"error at debug level", LevelDebug, "error", true},
		{"debug at info level", LevelInfo, "debug", false},
		{"info at info level", LevelInfo, "info", true},
		{"debug at warn level", LevelWarn, "debug", false},
		{"info at warn level", LevelWarn, "info", false},
		{"warn at warn level", LevelWarn, "warn", true},
		{"error at warn level", LevelWarn, "error", true},
		{"debug at error level", LevelError, "debug", false},
		{"warn at error level", LevelError, "warn", false},
		{"error at error level", LevelError, "error", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			l := New(&buf, tc.minLevel, FormatText)

			switch tc.logLevel {
			case "debug":
				l.Debug("msg")
			case "info":
				l.Info("msg")
			case "warn":
				l.Warn("msg")
			case "error":
				l.Error("msg")
			}

			hasOutput := buf.Len() > 0
			if hasOutput != tc.expectLog {
				t.Errorf("output=%v, want output=%v (buf=%q)", hasOutput, tc.expectLog, buf.String())
			}
		})
	}
}

func TestStdLogger_MultipleFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatText)

	l.Info("request", "method", "GET", "path", "/api", "status", "200")

	out := buf.String()
	for _, want := range []string{"method=GET", "path=/api", "status=200"} {
		if !strings.Contains(out, want) {
			t.Errorf("expected %q in output, got %q", want, out)
		}
	}
}

func TestStdLogger_OddFieldCount(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatText)

	// Odd number of fields: last key should show with a missing-value marker.
	l.Info("msg", "orphan")

	out := buf.String()
	if !strings.Contains(out, "orphan") {
		t.Errorf("expected orphan key in output, got %q", out)
	}
}

func TestStdLogger_NoFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatText)

	l.Info("simple message")

	out := buf.String()
	if !strings.Contains(out, "simple message") {
		t.Errorf("expected message in output, got %q", out)
	}
}

func TestStdLogger_JSONMultipleFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatJSON)

	l.Warn("alert", "code", "500", "detail", "internal error")

	var entry map[string]string
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if entry["level"] != "WARN" {
		t.Errorf("level = %q, want WARN", entry["level"])
	}
	if entry["code"] != "500" {
		t.Errorf("code = %q, want 500", entry["code"])
	}
	if entry["detail"] != "internal error" {
		t.Errorf("detail = %q, want %q", entry["detail"], "internal error")
	}
}

func TestStdLogger_JSONOddFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatJSON)

	l.Error("fail", "orphan")

	var entry map[string]string
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("failed to parse JSON: %v", err)
	}

	if entry["orphan"] != "MISSING" {
		t.Errorf("orphan = %q, want MISSING", entry["orphan"])
	}
}

func TestWithLevel(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, LevelDebug, FormatText)

	child := l.WithLevel(LevelError)
	child.Debug("should not appear")
	child.Error("should appear")

	out := buf.String()
	if strings.Contains(out, "should not appear") {
		t.Error("debug message should have been filtered")
	}
	if !strings.Contains(out, "should appear") {
		t.Error("error message should have appeared")
	}
}

func TestFormatConstants(t *testing.T) {
	if FormatText != 0 {
		t.Errorf("FormatText = %d, want 0", FormatText)
	}
	if FormatJSON != 1 {
		t.Errorf("FormatJSON = %d, want 1", FormatJSON)
	}
}
