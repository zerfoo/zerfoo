package cloud

import (
	"sync"
	"time"
)

// AuditAction identifies the type of API operation being logged.
type AuditAction string

const (
	AuditActionInference AuditAction = "inference"
	AuditActionCreate    AuditAction = "create"
	AuditActionUpdate    AuditAction = "update"
	AuditActionDelete    AuditAction = "delete"
	AuditActionList      AuditAction = "list"
	AuditActionAuth      AuditAction = "auth"
)

// AuditResult records the outcome of an API request.
type AuditResult string

const (
	AuditResultSuccess      AuditResult = "success"
	AuditResultDenied       AuditResult = "denied"
	AuditResultRateLimited  AuditResult = "rate_limited"
	AuditResultError        AuditResult = "error"
	AuditResultUnauthorized AuditResult = "unauthorized"
)

// AuditEntry records a single auditable event for SOC 2 compliance.
// Sensitive data (API keys, request bodies) is never stored.
type AuditEntry struct {
	Timestamp  time.Time   `json:"timestamp"`
	TenantID   string      `json:"tenant_id"`
	Action     AuditAction `json:"action"`
	Result     AuditResult `json:"result"`
	Resource   string      `json:"resource"`
	StatusCode int         `json:"status_code"`
	Method     string      `json:"method"`
	Path       string      `json:"path"`
	RemoteAddr string      `json:"remote_addr"`
}

// AuditStore is the persistence interface for audit entries.
type AuditStore interface {
	// Append persists an audit entry.
	Append(entry AuditEntry) error

	// Query returns audit entries for a tenant within the given time range.
	Query(tenantID string, from, to time.Time) ([]AuditEntry, error)
}

// AuditLogger records API requests for SOC 2 compliance.
// It deliberately omits sensitive fields (API keys, request/response bodies).
type AuditLogger struct {
	store AuditStore
}

// NewAuditLogger creates an AuditLogger backed by the given store.
func NewAuditLogger(store AuditStore) *AuditLogger {
	return &AuditLogger{store: store}
}

// Log records an audit entry.
func (a *AuditLogger) Log(entry AuditEntry) error {
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}
	return a.store.Append(entry)
}

// Query returns audit entries for a tenant within the given time range.
func (a *AuditLogger) Query(tenantID string, from, to time.Time) ([]AuditEntry, error) {
	return a.store.Query(tenantID, from, to)
}

// MemoryAuditStore is an in-memory AuditStore for testing and development.
type MemoryAuditStore struct {
	mu      sync.Mutex
	entries []AuditEntry
}

// NewMemoryAuditStore creates a new in-memory audit store.
func NewMemoryAuditStore() *MemoryAuditStore {
	return &MemoryAuditStore{}
}

// Append appends an entry to the in-memory store.
func (s *MemoryAuditStore) Append(entry AuditEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, entry)
	return nil
}

// Query returns entries matching the tenant and time range.
func (s *MemoryAuditStore) Query(tenantID string, from, to time.Time) ([]AuditEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var out []AuditEntry
	for _, e := range s.entries {
		if e.TenantID != tenantID {
			continue
		}
		if e.Timestamp.Before(from) || e.Timestamp.After(to) {
			continue
		}
		out = append(out, e)
	}
	return out, nil
}

// All returns a copy of all stored entries.
func (s *MemoryAuditStore) All() []AuditEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]AuditEntry, len(s.entries))
	copy(out, s.entries)
	return out
}
