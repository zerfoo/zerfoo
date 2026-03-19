// Package compliance provides SOC 2 Trust Services Criteria (TSC) control
// mappings and evidence collection for the zerfoo ML framework.
package compliance

import (
	"fmt"
	"sync"
	"time"
)

// Category represents a SOC 2 Trust Services Category.
type Category string

const (
	CategorySecurity      Category = "security"
	CategoryAvailability  Category = "availability"
	CategoryProcessing    Category = "processing_integrity"
	CategoryConfidential  Category = "confidentiality"
	CategoryPrivacy       Category = "privacy"
)

// ControlStatus indicates the implementation state of a control.
type ControlStatus string

const (
	StatusNotImplemented ControlStatus = "not_implemented"
	StatusPartial        ControlStatus = "partial"
	StatusImplemented    ControlStatus = "implemented"
	StatusEffective      ControlStatus = "effective"
)

// Control represents a single SOC 2 control mapped to a Trust Services Criterion.
type Control struct {
	ID          string
	Name        string
	Description string
	Category    Category
	Criteria    []string // e.g., ["CC6.1", "CC6.2"]
	Status      ControlStatus
	Owner       string
	LastTested  time.Time
}

// ControlRegistry maintains a set of SOC 2 controls.
type ControlRegistry struct {
	mu       sync.RWMutex
	controls map[string]*Control
}

// NewControlRegistry creates an empty control registry.
func NewControlRegistry() *ControlRegistry {
	return &ControlRegistry{
		controls: make(map[string]*Control),
	}
}

// Register adds a control to the registry. It returns an error if the
// control ID is empty or already registered.
func (r *ControlRegistry) Register(c Control) error {
	if c.ID == "" {
		return fmt.Errorf("compliance: control ID is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.controls[c.ID]; exists {
		return fmt.Errorf("compliance: control %s already registered", c.ID)
	}
	cp := c
	r.controls[c.ID] = &cp
	return nil
}

// Get returns a control by ID or an error if not found.
func (r *ControlRegistry) Get(id string) (Control, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	c, ok := r.controls[id]
	if !ok {
		return Control{}, fmt.Errorf("compliance: control %s not found", id)
	}
	return *c, nil
}

// All returns all registered controls.
func (r *ControlRegistry) All() []Control {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]Control, 0, len(r.controls))
	for _, c := range r.controls {
		out = append(out, *c)
	}
	return out
}

// UpdateStatus updates the status of a control. It returns an error if the
// control is not found.
func (r *ControlRegistry) UpdateStatus(id string, status ControlStatus) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	c, ok := r.controls[id]
	if !ok {
		return fmt.Errorf("compliance: control %s not found", id)
	}
	c.Status = status
	return nil
}
