package observation

import (
	"fmt"
	"sync"
	"time"
)

// CheckResult captures the outcome of a single control monitoring check.
type CheckResult struct {
	ControlID string
	CheckedAt time.Time
	Passed    bool
	Details   string
}

// CheckFunc is a function that evaluates whether a control is operating
// effectively. It returns true if the control passed and an optional
// description of the result.
type CheckFunc func() (passed bool, details string)

// ScheduledCheck defines a recurring control check.
type ScheduledCheck struct {
	ControlID string
	Name      string
	Interval  time.Duration
	Check     CheckFunc
	LastRun   time.Time
}

// Monitor performs continuous control monitoring with scheduled checks during
// the observation period. It records all check results for the observation
// report.
type Monitor struct {
	mu      sync.RWMutex
	checks  []ScheduledCheck
	results []CheckResult
}

// NewMonitor creates a new control monitor.
func NewMonitor() *Monitor {
	return &Monitor{}
}

// Register adds a scheduled check for a control. Returns an error if the
// control ID or check function is nil.
func (m *Monitor) Register(sc ScheduledCheck) error {
	if sc.ControlID == "" {
		return fmt.Errorf("observation: check ControlID is required")
	}
	if sc.Check == nil {
		return fmt.Errorf("observation: check function is required for control %s", sc.ControlID)
	}
	if sc.Interval <= 0 {
		return fmt.Errorf("observation: check interval must be positive for control %s", sc.ControlID)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.checks = append(m.checks, sc)
	return nil
}

// RunDue executes all checks that are due (i.e., the interval has elapsed
// since their last run). It returns the results of executed checks.
func (m *Monitor) RunDue(now time.Time) []CheckResult {
	m.mu.Lock()
	defer m.mu.Unlock()

	var results []CheckResult

	for i := range m.checks {
		sc := &m.checks[i]
		if !sc.LastRun.IsZero() && now.Sub(sc.LastRun) < sc.Interval {
			continue
		}

		passed, details := sc.Check()
		result := CheckResult{
			ControlID: sc.ControlID,
			CheckedAt: now,
			Passed:    passed,
			Details:   details,
		}

		sc.LastRun = now
		m.results = append(m.results, result)
		results = append(results, result)
	}

	return results
}

// RunAll executes all registered checks regardless of schedule. Returns the
// results.
func (m *Monitor) RunAll(now time.Time) []CheckResult {
	m.mu.Lock()
	defer m.mu.Unlock()

	var results []CheckResult

	for i := range m.checks {
		sc := &m.checks[i]
		passed, details := sc.Check()
		result := CheckResult{
			ControlID: sc.ControlID,
			CheckedAt: now,
			Passed:    passed,
			Details:   details,
		}

		sc.LastRun = now
		m.results = append(m.results, result)
		results = append(results, result)
	}

	return results
}

// Results returns all check results recorded so far.
func (m *Monitor) Results() []CheckResult {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]CheckResult, len(m.results))
	copy(out, m.results)
	return out
}

// ResultsForControl returns check results for a specific control.
func (m *Monitor) ResultsForControl(controlID string) []CheckResult {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var out []CheckResult
	for _, r := range m.results {
		if r.ControlID == controlID {
			out = append(out, r)
		}
	}
	return out
}

// Checks returns a copy of all registered scheduled checks.
func (m *Monitor) Checks() []ScheduledCheck {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]ScheduledCheck, len(m.checks))
	copy(out, m.checks)
	return out
}
