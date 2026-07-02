package online

import (
	"bufio"
	"encoding/json"
	"os"
	"sync"
	"time"
)

// EventType constants identify the kind of audit event.
const (
	EventTrigger    = "trigger"
	EventUpdate     = "update"
	EventRollback   = "rollback"
	EventValidation = "validation"
)

// AuditEvent records a single auditable action in the online learning pipeline.
type AuditEvent struct {
	Timestamp time.Time      `json:"timestamp"`
	EventType string         `json:"event_type"`
	Details   map[string]any `json:"details,omitempty"`
	Outcome   string         `json:"outcome"`
}

// AuditLog writes and reads AuditEvents as JSONL (one JSON object per line).
type AuditLog struct {
	mu   sync.Mutex
	file *os.File
	path string
}

// NewAuditLog opens or creates a JSONL audit log file at path.
func NewAuditLog(path string) (*AuditLog, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}
	return &AuditLog{file: f, path: path}, nil
}

// Log marshals the event to JSON, appends a newline, and flushes to disk.
func (a *AuditLog) Log(event AuditEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	data = append(data, '\n')

	a.mu.Lock()
	defer a.mu.Unlock()

	if _, err := a.file.Write(data); err != nil {
		return err
	}
	return a.file.Sync()
}

// ReadAll reads all events from the audit log file.
func (a *AuditLog) ReadAll() ([]AuditEvent, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	f, err := os.Open(a.path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var events []AuditEvent
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var ev AuditEvent
		if err := json.Unmarshal(scanner.Bytes(), &ev); err != nil {
			return nil, err
		}
		events = append(events, ev)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return events, nil
}

// Close closes the underlying file.
func (a *AuditLog) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.file.Close()
}
