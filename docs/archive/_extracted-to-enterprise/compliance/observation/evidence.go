package observation

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// EvidenceItem represents a timestamped evidence artifact collected during
// the observation period. Unlike the base compliance.Evidence, this is
// specifically scoped to a point in the observation timeline.
type EvidenceItem struct {
	ID          string
	ControlID   string
	Description string
	Collector   string
	CollectedAt time.Time
	Hash        string // SHA-256 of content
	Content     []byte
	Tags        []string
}

// EvidenceAccumulator collects timestamped evidence over the observation
// period. Evidence is indexed by control ID for efficient retrieval during
// report generation.
type EvidenceAccumulator struct {
	mu     sync.RWMutex
	items  []EvidenceItem
	byCtrl map[string][]int // controlID -> indices into items
}

// NewEvidenceAccumulator creates an empty evidence accumulator.
func NewEvidenceAccumulator() *EvidenceAccumulator {
	return &EvidenceAccumulator{
		byCtrl: make(map[string][]int),
	}
}

// Add stores a new evidence item. The hash is computed automatically from
// the content. Returns an error if the ID or ControlID is empty.
func (a *EvidenceAccumulator) Add(e EvidenceItem) error {
	if e.ID == "" {
		return fmt.Errorf("observation: evidence ID is required")
	}
	if e.ControlID == "" {
		return fmt.Errorf("observation: evidence ControlID is required")
	}

	h := sha256.Sum256(e.Content)
	e.Hash = hex.EncodeToString(h[:])
	if e.CollectedAt.IsZero() {
		e.CollectedAt = time.Now()
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	idx := len(a.items)
	a.items = append(a.items, e)
	a.byCtrl[e.ControlID] = append(a.byCtrl[e.ControlID], idx)
	return nil
}

// ForControl returns all evidence items for a given control ID.
func (a *EvidenceAccumulator) ForControl(controlID string) []EvidenceItem {
	a.mu.RLock()
	defer a.mu.RUnlock()

	indices := a.byCtrl[controlID]
	out := make([]EvidenceItem, 0, len(indices))
	for _, idx := range indices {
		out = append(out, a.items[idx])
	}
	return out
}

// All returns all evidence items in collection order.
func (a *EvidenceAccumulator) All() []EvidenceItem {
	a.mu.RLock()
	defer a.mu.RUnlock()

	out := make([]EvidenceItem, len(a.items))
	copy(out, a.items)
	return out
}

// Count returns the total number of evidence items.
func (a *EvidenceAccumulator) Count() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.items)
}

// CountForControl returns the number of evidence items for a given control.
func (a *EvidenceAccumulator) CountForControl(controlID string) int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.byCtrl[controlID])
}

// ControlIDs returns the set of control IDs that have evidence.
func (a *EvidenceAccumulator) ControlIDs() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()

	ids := make([]string, 0, len(a.byCtrl))
	for id := range a.byCtrl {
		ids = append(ids, id)
	}
	return ids
}
