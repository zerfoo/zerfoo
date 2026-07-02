package cloud

import (
	"sync"
	"time"
)

// BillingRecord captures token usage for a single inference request.
type BillingRecord struct {
	TenantID         string    `json:"tenant_id"`
	InputTokens      int       `json:"input_tokens"`
	OutputTokens     int       `json:"output_tokens"`
	Timestamp        time.Time `json:"timestamp"`
}

// BillingStore is the persistence interface for billing records.
type BillingStore interface {
	// Store persists a billing record.
	Store(record BillingRecord) error

	// Query returns all billing records for a tenant within the given time range.
	Query(tenantID string, from, to time.Time) ([]BillingRecord, error)
}

// TokenMeter tracks input and output token usage per tenant and emits
// billing records to a BillingStore.
type TokenMeter struct {
	store BillingStore
}

// NewTokenMeter creates a TokenMeter backed by the given BillingStore.
func NewTokenMeter(store BillingStore) *TokenMeter {
	return &TokenMeter{store: store}
}

// Record records token usage for a tenant and persists a BillingRecord.
func (m *TokenMeter) Record(tenantID string, inputTokens, outputTokens int) error {
	return m.store.Store(BillingRecord{
		TenantID:     tenantID,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		Timestamp:    time.Now().UTC(),
	})
}

// Query returns billing records for a tenant within the given time range.
func (m *TokenMeter) Query(tenantID string, from, to time.Time) ([]BillingRecord, error) {
	return m.store.Query(tenantID, from, to)
}

// MemoryBillingStore is an in-memory BillingStore for testing and development.
type MemoryBillingStore struct {
	mu      sync.Mutex
	records []BillingRecord
}

// NewMemoryBillingStore creates a new in-memory billing store.
func NewMemoryBillingStore() *MemoryBillingStore {
	return &MemoryBillingStore{}
}

// Store appends a record to the in-memory store.
func (s *MemoryBillingStore) Store(record BillingRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records = append(s.records, record)
	return nil
}

// Query returns records matching the tenant and time range.
func (s *MemoryBillingStore) Query(tenantID string, from, to time.Time) ([]BillingRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var out []BillingRecord
	for _, r := range s.records {
		if r.TenantID != tenantID {
			continue
		}
		if r.Timestamp.Before(from) || r.Timestamp.After(to) {
			continue
		}
		out = append(out, r)
	}
	return out, nil
}

// All returns a copy of all stored records.
func (s *MemoryBillingStore) All() []BillingRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]BillingRecord, len(s.records))
	copy(out, s.records)
	return out
}
