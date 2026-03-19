package observation

import (
	"fmt"
	"sync"
	"time"
)

// Severity classifies the impact of a deviation.
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
)

// DeviationStatus tracks the remediation state of a deviation.
type DeviationStatus string

const (
	DeviationOpen       DeviationStatus = "open"
	DeviationMitigated  DeviationStatus = "mitigated"
	DeviationRemediated DeviationStatus = "remediated"
	DeviationAccepted   DeviationStatus = "accepted" // risk accepted
)

// Deviation records a control failure or exception observed during the
// observation period.
type Deviation struct {
	ID            string
	ControlID     string
	Severity      Severity
	Status        DeviationStatus
	Description   string
	RootCause     string
	Remediation   string
	DetectedAt    time.Time
	ResolvedAt    time.Time
	ResolvedBy    string
}

// DeviationTracker records and manages control deviations during the
// observation period.
type DeviationTracker struct {
	mu         sync.RWMutex
	deviations []Deviation
	byCtrl     map[string][]int // controlID -> indices
}

// NewDeviationTracker creates a new deviation tracker.
func NewDeviationTracker() *DeviationTracker {
	return &DeviationTracker{
		byCtrl: make(map[string][]int),
	}
}

// Record adds a new deviation. Returns an error if the ID or ControlID is
// empty.
func (t *DeviationTracker) Record(d Deviation) error {
	if d.ID == "" {
		return fmt.Errorf("observation: deviation ID is required")
	}
	if d.ControlID == "" {
		return fmt.Errorf("observation: deviation ControlID is required")
	}

	if d.DetectedAt.IsZero() {
		d.DetectedAt = time.Now()
	}
	if d.Status == "" {
		d.Status = DeviationOpen
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	idx := len(t.deviations)
	t.deviations = append(t.deviations, d)
	t.byCtrl[d.ControlID] = append(t.byCtrl[d.ControlID], idx)
	return nil
}

// Resolve updates the status and resolution details of a deviation. Returns
// an error if the deviation is not found.
func (t *DeviationTracker) Resolve(id string, status DeviationStatus, remediation string, resolvedBy string, resolvedAt time.Time) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	for i := range t.deviations {
		if t.deviations[i].ID == id {
			t.deviations[i].Status = status
			t.deviations[i].Remediation = remediation
			t.deviations[i].ResolvedBy = resolvedBy
			t.deviations[i].ResolvedAt = resolvedAt
			return nil
		}
	}
	return fmt.Errorf("observation: deviation %s not found", id)
}

// All returns all deviations.
func (t *DeviationTracker) All() []Deviation {
	t.mu.RLock()
	defer t.mu.RUnlock()

	out := make([]Deviation, len(t.deviations))
	copy(out, t.deviations)
	return out
}

// ForControl returns deviations for a specific control.
func (t *DeviationTracker) ForControl(controlID string) []Deviation {
	t.mu.RLock()
	defer t.mu.RUnlock()

	indices := t.byCtrl[controlID]
	out := make([]Deviation, 0, len(indices))
	for _, idx := range indices {
		out = append(out, t.deviations[idx])
	}
	return out
}

// Open returns all deviations that have not been resolved.
func (t *DeviationTracker) Open() []Deviation {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var out []Deviation
	for _, d := range t.deviations {
		if d.Status == DeviationOpen {
			out = append(out, d)
		}
	}
	return out
}

// Count returns the total number of deviations.
func (t *DeviationTracker) Count() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.deviations)
}

// OpenCount returns the number of open deviations.
func (t *DeviationTracker) OpenCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	n := 0
	for _, d := range t.deviations {
		if d.Status == DeviationOpen {
			n++
		}
	}
	return n
}
