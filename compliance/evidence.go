package compliance

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// EvidenceType classifies evidence artifacts.
type EvidenceType string

const (
	EvidenceScreenshot   EvidenceType = "screenshot"
	EvidenceLog          EvidenceType = "log"
	EvidenceConfig       EvidenceType = "config"
	EvidenceTestResult   EvidenceType = "test_result"
	EvidencePolicy       EvidenceType = "policy"
	EvidenceAuditTrail   EvidenceType = "audit_trail"
)

// Evidence represents a timestamped evidence artifact collected for a control.
type Evidence struct {
	ID          string
	ControlID   string
	Type        EvidenceType
	Description string
	Collector   string
	CollectedAt time.Time
	Hash        string // SHA-256 of content
	Content     []byte
}

// EvidenceStore collects and indexes evidence artifacts.
type EvidenceStore struct {
	mu       sync.RWMutex
	items    map[string]*Evidence
	byCtrl   map[string][]string // controlID -> evidence IDs
}

// NewEvidenceStore creates an empty evidence store.
func NewEvidenceStore() *EvidenceStore {
	return &EvidenceStore{
		items:  make(map[string]*Evidence),
		byCtrl: make(map[string][]string),
	}
}

// Collect stores a new evidence artifact. The hash is computed automatically
// from the content. Returns an error if the evidence ID is empty.
func (s *EvidenceStore) Collect(e Evidence) error {
	if e.ID == "" {
		return fmt.Errorf("compliance: evidence ID is required")
	}
	if e.ControlID == "" {
		return fmt.Errorf("compliance: evidence ControlID is required")
	}

	h := sha256.Sum256(e.Content)
	e.Hash = hex.EncodeToString(h[:])
	if e.CollectedAt.IsZero() {
		e.CollectedAt = time.Now()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	cp := e
	s.items[e.ID] = &cp
	s.byCtrl[e.ControlID] = append(s.byCtrl[e.ControlID], e.ID)
	return nil
}

// Get returns an evidence artifact by ID.
func (s *EvidenceStore) Get(id string) (Evidence, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	e, ok := s.items[id]
	if !ok {
		return Evidence{}, fmt.Errorf("compliance: evidence %s not found", id)
	}
	return *e, nil
}

// ForControl returns all evidence artifacts for a given control ID.
func (s *EvidenceStore) ForControl(controlID string) []Evidence {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ids := s.byCtrl[controlID]
	out := make([]Evidence, 0, len(ids))
	for _, id := range ids {
		if e, ok := s.items[id]; ok {
			out = append(out, *e)
		}
	}
	return out
}

// All returns all evidence artifacts.
func (s *EvidenceStore) All() []Evidence {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Evidence, 0, len(s.items))
	for _, e := range s.items {
		out = append(out, *e)
	}
	return out
}
