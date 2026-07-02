// Package audit provides SOC 2 Type I audit tooling including readiness
// assessment, evidence collection automation, gap analysis, and report
// generation. It ties together the compliance and security packages to
// produce a complete audit picture.
package audit

import (
	"sort"
	"time"

	"github.com/zerfoo/zerfoo/compliance"
)

// ReadinessLevel indicates how ready a control is for audit.
type ReadinessLevel string

const (
	ReadinessReady    ReadinessLevel = "ready"
	ReadinessPartial  ReadinessLevel = "partial"
	ReadinessNotReady ReadinessLevel = "not_ready"
)

// ControlReadiness captures the readiness state of a single control.
type ControlReadiness struct {
	Control      compliance.Control
	Level        ReadinessLevel
	EvidenceIDs  []string
	Assessment   compliance.ControlAssessment
	HasAssessment bool
	Notes        string
}

// ReadinessAssessment maps all implemented controls to Trust Services Criteria
// and evaluates audit readiness.
type ReadinessAssessment struct {
	AssessedAt time.Time
	Controls   []ControlReadiness
}

// Ready returns controls that are fully ready for audit.
func (ra *ReadinessAssessment) Ready() []ControlReadiness {
	return ra.filterByLevel(ReadinessReady)
}

// Partial returns controls that are partially ready.
func (ra *ReadinessAssessment) Partial() []ControlReadiness {
	return ra.filterByLevel(ReadinessPartial)
}

// NotReady returns controls that are not ready for audit.
func (ra *ReadinessAssessment) NotReady() []ControlReadiness {
	return ra.filterByLevel(ReadinessNotReady)
}

// ReadinessRate returns the percentage of controls that are fully ready.
func (ra *ReadinessAssessment) ReadinessRate() float64 {
	if len(ra.Controls) == 0 {
		return 0
	}
	ready := len(ra.Ready())
	return float64(ready) / float64(len(ra.Controls)) * 100
}

func (ra *ReadinessAssessment) filterByLevel(level ReadinessLevel) []ControlReadiness {
	var out []ControlReadiness
	for _, cr := range ra.Controls {
		if cr.Level == level {
			out = append(out, cr)
		}
	}
	return out
}

// AssessReadiness performs a readiness assessment using the control mapping
// and evidence collector. A control is "ready" if it has both an assessment
// (compliant or partially compliant) and at least one piece of evidence.
// It is "partial" if it has either an assessment or evidence, but not both.
// It is "not ready" if it has neither.
func AssessReadiness(mapping *compliance.ControlMapping, collector *compliance.EvidenceCollector) *ReadinessAssessment {
	controls := mapping.Controls()
	sort.Slice(controls, func(i, j int) bool {
		return controls[i].ID < controls[j].ID
	})

	ra := &ReadinessAssessment{
		AssessedAt: time.Now(),
		Controls:   make([]ControlReadiness, 0, len(controls)),
	}

	for _, ctrl := range controls {
		cr := ControlReadiness{
			Control: ctrl,
			Level:   ReadinessNotReady,
		}

		assessment, hasAssessment := mapping.Assessment(ctrl.ID)
		if hasAssessment {
			cr.Assessment = assessment
			cr.HasAssessment = true
		}

		var evidenceIDs []string
		if collector != nil {
			for _, ev := range collector.EvidenceByControl(ctrl.ID) {
				evidenceIDs = append(evidenceIDs, ev.ID)
			}
		}
		cr.EvidenceIDs = evidenceIDs

		hasEvidence := len(evidenceIDs) > 0
		assessedCompliant := hasAssessment && (assessment.Status == compliance.StatusCompliant || assessment.Status == compliance.StatusPartial)

		switch {
		case assessedCompliant && hasEvidence:
			cr.Level = ReadinessReady
			cr.Notes = "Control assessed and evidence collected"
		case assessedCompliant || hasEvidence:
			cr.Level = ReadinessPartial
			if !hasEvidence {
				cr.Notes = "Assessment recorded but no evidence collected"
			} else {
				cr.Notes = "Evidence collected but no assessment recorded"
			}
		case hasAssessment && assessment.Status == compliance.StatusNotApplicable:
			cr.Level = ReadinessReady
			cr.Notes = "Control marked as not applicable"
		default:
			cr.Notes = "No assessment or evidence"
		}

		ra.Controls = append(ra.Controls, cr)
	}

	return ra
}
