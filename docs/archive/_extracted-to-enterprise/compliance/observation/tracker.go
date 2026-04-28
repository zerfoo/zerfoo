// Package observation implements the SOC 2 Type II observation period
// framework. It tracks the observation lifecycle (start, monitor, complete),
// accumulates timestamped evidence, records control deviations, and generates
// observation period reports with control effectiveness statistics.
package observation

import (
	"fmt"
	"sync"
	"time"
)

// MinimumDuration is the minimum observation period required for SOC 2 Type II
// (6 months).
const MinimumDuration = 6 * 30 * 24 * time.Hour // ~180 days

// Phase represents the current phase of the observation period.
type Phase string

const (
	PhaseNotStarted Phase = "not_started"
	PhaseActive     Phase = "active"
	PhaseCompleted  Phase = "completed"
)

// Milestone marks a significant event during the observation period.
type Milestone struct {
	Name        string
	Description string
	DueDate     time.Time
	CompletedAt time.Time
	Completed   bool
}

// Period tracks the SOC 2 Type II observation period lifecycle.
type Period struct {
	mu         sync.RWMutex
	startDate  time.Time
	endDate    time.Time
	phase      Phase
	milestones []Milestone
}

// NewPeriod creates a new observation period that has not yet started.
func NewPeriod() *Period {
	return &Period{
		phase: PhaseNotStarted,
	}
}

// Start begins the observation period. The end date is set to at least
// MinimumDuration from the start date. If duration is greater than
// MinimumDuration, the provided duration is used instead. Returns an error
// if the period has already started.
func (p *Period) Start(start time.Time, duration time.Duration) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.phase != PhaseNotStarted {
		return fmt.Errorf("observation: period already started (phase: %s)", p.phase)
	}

	if duration < MinimumDuration {
		duration = MinimumDuration
	}

	p.startDate = start
	p.endDate = start.Add(duration)
	p.phase = PhaseActive
	return nil
}

// Complete marks the observation period as completed. It returns an error if
// the period is not active or the minimum duration has not elapsed.
func (p *Period) Complete(now time.Time) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.phase != PhaseActive {
		return fmt.Errorf("observation: period is not active (phase: %s)", p.phase)
	}

	elapsed := now.Sub(p.startDate)
	if elapsed < MinimumDuration {
		return fmt.Errorf("observation: minimum duration not met (elapsed: %s, required: %s)",
			elapsed.Round(24*time.Hour), MinimumDuration)
	}

	p.phase = PhaseCompleted
	p.endDate = now
	return nil
}

// AddMilestone adds a milestone to the observation period. Returns an error
// if the period is not active.
func (p *Period) AddMilestone(m Milestone) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.phase != PhaseActive {
		return fmt.Errorf("observation: cannot add milestone when period is %s", p.phase)
	}

	p.milestones = append(p.milestones, m)
	return nil
}

// CompleteMilestone marks a milestone as completed by name. Returns an error
// if the milestone is not found.
func (p *Period) CompleteMilestone(name string, completedAt time.Time) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for i := range p.milestones {
		if p.milestones[i].Name == name {
			p.milestones[i].Completed = true
			p.milestones[i].CompletedAt = completedAt
			return nil
		}
	}
	return fmt.Errorf("observation: milestone %q not found", name)
}

// Phase returns the current phase.
func (p *Period) Phase() Phase {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.phase
}

// StartDate returns the observation start date.
func (p *Period) StartDate() time.Time {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.startDate
}

// EndDate returns the observation end date.
func (p *Period) EndDate() time.Time {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.endDate
}

// Milestones returns a copy of all milestones.
func (p *Period) Milestones() []Milestone {
	p.mu.RLock()
	defer p.mu.RUnlock()

	out := make([]Milestone, len(p.milestones))
	copy(out, p.milestones)
	return out
}

// Elapsed returns the time elapsed since the observation period started.
// Returns zero if the period has not started.
func (p *Period) Elapsed(now time.Time) time.Duration {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.phase == PhaseNotStarted {
		return 0
	}
	return now.Sub(p.startDate)
}

// RemainingDays returns the number of days remaining until the scheduled end
// date. Returns 0 if the period is completed or past the end date.
func (p *Period) RemainingDays(now time.Time) int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.phase != PhaseActive {
		return 0
	}

	remaining := p.endDate.Sub(now)
	if remaining <= 0 {
		return 0
	}
	return int(remaining / (24 * time.Hour))
}
