// Package audit provides SOC 2 Type I audit readiness assessment, evidence
// collection, gap analysis, and report generation.
package audit

import (
	"fmt"
	"time"

	"github.com/zerfoo/zerfoo/compliance"
)

// ReadinessLevel indicates the readiness state of a control for audit.
type ReadinessLevel string

const (
	ReadinessNotReady ReadinessLevel = "not_ready"
	ReadinessPartial  ReadinessLevel = "partial"
	ReadinessReady    ReadinessLevel = "ready"
)

// ControlAssessment captures a Type I point-in-time assessment of a control.
type ControlAssessment struct {
	ControlID   string
	Readiness   ReadinessLevel
	Findings    []string
	EvidenceIDs []string
	AssessedAt  time.Time
	Assessor    string
}

// Gap represents a deficiency identified during gap analysis.
type Gap struct {
	ControlID   string
	Description string
	Severity    string // "high", "medium", "low"
	Remediation string
	IdentifiedAt time.Time
}

// Report is a Type I audit report capturing the point-in-time state.
type Report struct {
	Title       string
	GeneratedAt time.Time
	Assessments []ControlAssessment
	Gaps        []Gap
	Summary     string
}

// Assess performs a readiness assessment for a control given available evidence.
func Assess(ctrl compliance.Control, evidence []compliance.Evidence) ControlAssessment {
	a := ControlAssessment{
		ControlID:  ctrl.ID,
		AssessedAt: time.Now(),
	}

	if ctrl.Status == compliance.StatusNotImplemented {
		a.Readiness = ReadinessNotReady
		a.Findings = append(a.Findings, "control not implemented")
		return a
	}

	if len(evidence) == 0 {
		a.Readiness = ReadinessNotReady
		a.Findings = append(a.Findings, "no evidence collected")
		return a
	}

	for _, e := range evidence {
		a.EvidenceIDs = append(a.EvidenceIDs, e.ID)
	}

	switch ctrl.Status {
	case compliance.StatusEffective:
		a.Readiness = ReadinessReady
	case compliance.StatusImplemented:
		a.Readiness = ReadinessPartial
		a.Findings = append(a.Findings, "control implemented but effectiveness not yet verified")
	default:
		a.Readiness = ReadinessPartial
		a.Findings = append(a.Findings, fmt.Sprintf("control status: %s", ctrl.Status))
	}

	return a
}
