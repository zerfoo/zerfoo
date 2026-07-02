package support

import (
	"sync"
	"time"
)

// SLAPolicy defines response and resolution time targets for a priority level.
type SLAPolicy struct {
	Priority       Priority
	ResponseTime   time.Duration // max time to first response (triage)
	ResolutionTime time.Duration // max time to resolution
}

// DefaultSLAPolicies returns standard enterprise SLA targets.
func DefaultSLAPolicies() []SLAPolicy {
	return []SLAPolicy{
		{Priority: P0Critical, ResponseTime: 15 * time.Minute, ResolutionTime: 4 * time.Hour},
		{Priority: P1High, ResponseTime: 1 * time.Hour, ResolutionTime: 24 * time.Hour},
		{Priority: P2Medium, ResponseTime: 4 * time.Hour, ResolutionTime: 72 * time.Hour},
		{Priority: P3Low, ResponseTime: 24 * time.Hour, ResolutionTime: 168 * time.Hour}, // 7 days
	}
}

// BreachType indicates which SLA target was breached.
type BreachType string

const (
	BreachResponse   BreachType = "response"
	BreachResolution BreachType = "resolution"
)

// Breach records an SLA violation.
type Breach struct {
	TicketID   string
	Type       BreachType
	Priority   Priority
	Deadline   time.Time
	BreachedAt time.Time
}

// BreachHandler is called when an SLA breach is detected.
type BreachHandler func(Breach)

// SLATracker monitors tickets against SLA policies and fires breach alerts.
type SLATracker struct {
	mu       sync.RWMutex
	policies map[Priority]SLAPolicy
	handlers []BreachHandler
}

// NewSLATracker creates a tracker with the given policies.
func NewSLATracker(policies []SLAPolicy) *SLATracker {
	m := make(map[Priority]SLAPolicy, len(policies))
	for _, p := range policies {
		m[p.Priority] = p
	}
	return &SLATracker{policies: m}
}

// OnBreach registers a handler that is called when a breach is detected.
func (s *SLATracker) OnBreach(h BreachHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.handlers = append(s.handlers, h)
}

// Check evaluates whether a ticket has breached its SLA at the given time.
// It returns any breaches found.
func (s *SLATracker) Check(t *Ticket, now time.Time) []Breach {
	s.mu.RLock()
	policy, ok := s.policies[t.Priority]
	handlers := s.handlers
	s.mu.RUnlock()

	if !ok {
		return nil
	}

	var breaches []Breach

	// Response SLA: ticket should be triaged within ResponseTime.
	if t.Status == StatusOpen {
		deadline := t.CreatedAt.Add(policy.ResponseTime)
		if now.After(deadline) {
			breaches = append(breaches, Breach{
				TicketID:   t.ID,
				Type:       BreachResponse,
				Priority:   t.Priority,
				Deadline:   deadline,
				BreachedAt: now,
			})
		}
	}

	// Resolution SLA: ticket should be resolved within ResolutionTime.
	if t.Status != StatusResolved && t.Status != StatusClosed {
		deadline := t.CreatedAt.Add(policy.ResolutionTime)
		if now.After(deadline) {
			breaches = append(breaches, Breach{
				TicketID:   t.ID,
				Type:       BreachResolution,
				Priority:   t.Priority,
				Deadline:   deadline,
				BreachedAt: now,
			})
		}
	}

	for _, b := range breaches {
		for _, h := range handlers {
			h(b)
		}
	}

	return breaches
}

// CheckAll evaluates all tickets in a store for SLA breaches.
func (s *SLATracker) CheckAll(store *Store, now time.Time) []Breach {
	tickets := store.All()
	var all []Breach
	for _, t := range tickets {
		all = append(all, s.Check(t, now)...)
	}
	return all
}
