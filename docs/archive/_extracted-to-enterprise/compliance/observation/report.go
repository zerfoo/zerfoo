package observation

import (
	"fmt"
	"strings"
	"time"
)

// ControlEffectiveness summarizes a control's effectiveness over the
// observation period.
type ControlEffectiveness struct {
	ControlID     string
	TotalChecks   int
	PassedChecks  int
	FailedChecks  int
	EffectivenessRate float64 // 0.0 to 1.0
	Deviations    int
	EvidenceCount int
}

// Report captures the full observation period summary, including control
// effectiveness statistics, deviations, and evidence coverage.
type Report struct {
	Title          string
	GeneratedAt    time.Time
	PeriodStart    time.Time
	PeriodEnd      time.Time
	Phase          Phase
	DurationDays   int
	Milestones     []Milestone
	Controls       []ControlEffectiveness
	TotalChecks    int
	TotalPassed    int
	TotalFailed    int
	TotalDeviations int
	OpenDeviations  int
	TotalEvidence   int
	OverallEffectiveness float64
}

// GenerateReport produces an observation period report from the tracker,
// monitor, evidence accumulator, and deviation tracker.
func GenerateReport(period *Period, monitor *Monitor, evidence *EvidenceAccumulator, deviations *DeviationTracker) Report {
	now := time.Now()

	r := Report{
		Title:       "SOC 2 Type II Observation Period Report",
		GeneratedAt: now,
		PeriodStart: period.StartDate(),
		PeriodEnd:   period.EndDate(),
		Phase:       period.Phase(),
		Milestones:  period.Milestones(),
	}

	if !r.PeriodStart.IsZero() {
		end := r.PeriodEnd
		if end.IsZero() {
			end = now
		}
		r.DurationDays = int(end.Sub(r.PeriodStart) / (24 * time.Hour))
	}

	// Build effectiveness per control from monitoring results.
	controlChecks := make(map[string]*ControlEffectiveness)
	for _, result := range monitor.Results() {
		ce, ok := controlChecks[result.ControlID]
		if !ok {
			ce = &ControlEffectiveness{ControlID: result.ControlID}
			controlChecks[result.ControlID] = ce
		}
		ce.TotalChecks++
		if result.Passed {
			ce.PassedChecks++
		} else {
			ce.FailedChecks++
		}
	}

	// Add deviation counts.
	for _, d := range deviations.All() {
		ce, ok := controlChecks[d.ControlID]
		if !ok {
			ce = &ControlEffectiveness{ControlID: d.ControlID}
			controlChecks[d.ControlID] = ce
		}
		ce.Deviations++
	}

	// Add evidence counts.
	for _, id := range evidence.ControlIDs() {
		ce, ok := controlChecks[id]
		if !ok {
			ce = &ControlEffectiveness{ControlID: id}
			controlChecks[id] = ce
		}
		ce.EvidenceCount = evidence.CountForControl(id)
	}

	// Compute effectiveness rates and aggregate totals.
	for _, ce := range controlChecks {
		if ce.TotalChecks > 0 {
			ce.EffectivenessRate = float64(ce.PassedChecks) / float64(ce.TotalChecks)
		}
		r.Controls = append(r.Controls, *ce)
		r.TotalChecks += ce.TotalChecks
		r.TotalPassed += ce.PassedChecks
		r.TotalFailed += ce.FailedChecks
	}

	r.TotalDeviations = deviations.Count()
	r.OpenDeviations = deviations.OpenCount()
	r.TotalEvidence = evidence.Count()

	if r.TotalChecks > 0 {
		r.OverallEffectiveness = float64(r.TotalPassed) / float64(r.TotalChecks)
	}

	return r
}

// String returns a human-readable summary of the observation report.
func (r Report) String() string {
	var b strings.Builder

	fmt.Fprintf(&b, "=== %s ===\n", r.Title)
	fmt.Fprintf(&b, "Generated: %s\n", r.GeneratedAt.Format(time.RFC3339))
	fmt.Fprintf(&b, "Period: %s to %s (%d days)\n",
		r.PeriodStart.Format("2006-01-02"),
		r.PeriodEnd.Format("2006-01-02"),
		r.DurationDays)
	fmt.Fprintf(&b, "Phase: %s\n\n", r.Phase)

	fmt.Fprintf(&b, "--- Milestones ---\n")
	for _, m := range r.Milestones {
		status := "pending"
		if m.Completed {
			status = "completed"
		}
		fmt.Fprintf(&b, "  [%s] %s (due: %s)\n", status, m.Name, m.DueDate.Format("2006-01-02"))
	}

	fmt.Fprintf(&b, "\n--- Control Effectiveness ---\n")
	for _, ce := range r.Controls {
		fmt.Fprintf(&b, "  %s: %d/%d checks passed (%.1f%%), %d deviations, %d evidence items\n",
			ce.ControlID, ce.PassedChecks, ce.TotalChecks,
			ce.EffectivenessRate*100, ce.Deviations, ce.EvidenceCount)
	}

	fmt.Fprintf(&b, "\n--- Summary ---\n")
	fmt.Fprintf(&b, "Total checks: %d (passed: %d, failed: %d)\n",
		r.TotalChecks, r.TotalPassed, r.TotalFailed)
	fmt.Fprintf(&b, "Overall effectiveness: %.1f%%\n", r.OverallEffectiveness*100)
	fmt.Fprintf(&b, "Deviations: %d total, %d open\n", r.TotalDeviations, r.OpenDeviations)
	fmt.Fprintf(&b, "Evidence items: %d\n", r.TotalEvidence)

	return b.String()
}
