// Package log provides a structured, leveled logging abstraction.
//
// It defines a Logger interface with Debug, Info, Warn, and Error methods.
// Two implementations are provided:
//   - StdLogger: writes to an io.Writer with configurable level filtering
//     and text or JSON output format.
//   - nopLogger: a zero-allocation no-op logger for when logging is disabled.
package log

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"
)

// Level represents a log severity level.
type Level int

const (
	// LevelDebug is the most verbose level.
	LevelDebug Level = iota
	// LevelInfo is for general operational messages.
	LevelInfo
	// LevelWarn is for warning conditions.
	LevelWarn
	// LevelError is for error conditions.
	LevelError
)

// String returns the human-readable name of a log level.
func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Format controls the output format of the logger.
type Format int

const (
	// FormatText outputs human-readable text lines.
	FormatText Format = iota
	// FormatJSON outputs one JSON object per line.
	FormatJSON
)

// Logger is the interface for structured, leveled logging.
// Each method accepts a message and optional key-value pairs.
// Keys and values are provided as alternating string arguments.
type Logger interface {
	// Debug logs a message at DEBUG level.
	Debug(msg string, fields ...string)
	// Info logs a message at INFO level.
	Info(msg string, fields ...string)
	// Warn logs a message at WARN level.
	Warn(msg string, fields ...string)
	// Error logs a message at ERROR level.
	Error(msg string, fields ...string)
	// WithLevel returns a new Logger with the given minimum level.
	WithLevel(level Level) Logger
}

// StdLogger writes structured log entries to an io.Writer.
type StdLogger struct {
	out    io.Writer
	level  Level
	format Format
	mu     sync.Mutex
}

// New creates a new StdLogger writing to w at the given minimum level
// and output format.
func New(w io.Writer, level Level, format Format) *StdLogger {
	return &StdLogger{
		out:    w,
		level:  level,
		format: format,
	}
}

// Debug logs at DEBUG level.
func (l *StdLogger) Debug(msg string, fields ...string) {
	l.log(LevelDebug, msg, fields)
}

// Info logs at INFO level.
func (l *StdLogger) Info(msg string, fields ...string) {
	l.log(LevelInfo, msg, fields)
}

// Warn logs at WARN level.
func (l *StdLogger) Warn(msg string, fields ...string) {
	l.log(LevelWarn, msg, fields)
}

// Error logs at ERROR level.
func (l *StdLogger) Error(msg string, fields ...string) {
	l.log(LevelError, msg, fields)
}

// WithLevel returns a new StdLogger sharing the same writer but with
// a different minimum log level.
func (l *StdLogger) WithLevel(level Level) Logger {
	return &StdLogger{
		out:    l.out,
		level:  level,
		format: l.format,
	}
}

func (l *StdLogger) log(level Level, msg string, fields []string) {
	if level < l.level {
		return
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	switch l.format {
	case FormatJSON:
		l.writeJSON(level, msg, fields)
	default:
		l.writeText(level, msg, fields)
	}
}

func (l *StdLogger) writeText(level Level, msg string, fields []string) {
	ts := time.Now().UTC().Format(time.RFC3339)
	_, _ = fmt.Fprintf(l.out, "%s %s %s", ts, level.String(), msg)

	for i := 0; i < len(fields); i += 2 {
		key := fields[i]
		val := "MISSING"
		if i+1 < len(fields) {
			val = fields[i+1]
		}
		_, _ = fmt.Fprintf(l.out, " %s=%s", key, val)
	}

	_, _ = fmt.Fprintln(l.out)
}

func (l *StdLogger) writeJSON(level Level, msg string, fields []string) {
	entry := make(map[string]string, 3+len(fields)/2)
	entry["time"] = time.Now().UTC().Format(time.RFC3339)
	entry["level"] = level.String()
	entry["msg"] = msg

	for i := 0; i < len(fields); i += 2 {
		key := fields[i]
		val := "MISSING"
		if i+1 < len(fields) {
			val = fields[i+1]
		}
		entry[key] = val
	}

	data, _ := json.Marshal(entry)
	data = append(data, '\n')
	_, _ = l.out.Write(data)
}

// nopLogger is a Logger that does nothing.
type nopLogger struct{}

// Nop returns a Logger that discards all log output with zero allocation.
func Nop() Logger {
	return nopLogger{}
}

func (nopLogger) Debug(_ string, _ ...string) {}
func (nopLogger) Info(_ string, _ ...string)  {}
func (nopLogger) Warn(_ string, _ ...string)  {}
func (nopLogger) Error(_ string, _ ...string) {}
func (nopLogger) WithLevel(_ Level) Logger    { return nopLogger{} }
