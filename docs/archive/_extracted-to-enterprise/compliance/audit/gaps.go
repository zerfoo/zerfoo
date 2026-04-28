package audit

import (
	"sort"

	"github.com/zerfoo/zerfoo/compliance"
)

// GapSeverity indicates the severity of a missing control.
type GapSeverity string

const (
	SeverityCritical GapSeverity = "critical"
	SeverityHigh     GapSeverity = "high"
	SeverityMedium   GapSeverity = "medium"
	SeverityLow      GapSeverity = "low"
)

// Gap represents a single control that is missing or incomplete.
type Gap struct {
	Control       compliance.Control
	Severity      GapSeverity
	MissingItems  []string // What is missing (e.g., "assessment", "evidence", "policy")
	Recommendation string
}

// GapAnalysis is the result of comparing implemented controls against
// the full set of required SOC 2 Trust Services Criteria.
type GapAnalysis struct {
	TotalControls      int
	ImplementedControls int
	GapCount           int
	Gaps               []Gap
}

// GapRate returns the percentage of controls that have gaps.
func (ga *GapAnalysis) GapRate() float64 {
	if ga.TotalControls == 0 {
		return 0
	}
	return float64(ga.GapCount) / float64(ga.TotalControls) * 100
}

// GapsByCategory returns gaps filtered by TSC category.
func (ga *GapAnalysis) GapsByCategory(cat compliance.Category) []Gap {
	var out []Gap
	for _, g := range ga.Gaps {
		if g.Control.Category == cat {
			out = append(out, g)
		}
	}
	return out
}

// GapsBySeverity returns gaps filtered by severity.
func (ga *GapAnalysis) GapsBySeverity(sev GapSeverity) []Gap {
	var out []Gap
	for _, g := range ga.Gaps {
		if g.Severity == sev {
			out = append(out, g)
		}
	}
	return out
}

// AnalyzeGaps examines the control mapping and evidence collector to identify
// controls that are missing assessments, evidence, or both. It returns a
// GapAnalysis with prioritized findings.
func AnalyzeGaps(mapping *compliance.ControlMapping, collector *compliance.EvidenceCollector) *GapAnalysis {
	controls := mapping.Controls()
	sort.Slice(controls, func(i, j int) bool {
		return controls[i].ID < controls[j].ID
	})

	ga := &GapAnalysis{
		TotalControls: len(controls),
	}

	for _, ctrl := range controls {
		assessment, hasAssessment := mapping.Assessment(ctrl.ID)

		var evidence []compliance.Evidence
		if collector != nil {
			evidence = collector.EvidenceByControl(ctrl.ID)
		}
		hasEvidence := len(evidence) > 0

		// Control is fully implemented if assessed compliant/NA with evidence.
		if hasAssessment {
			switch assessment.Status {
			case compliance.StatusCompliant:
				if hasEvidence {
					ga.ImplementedControls++
					continue
				}
			case compliance.StatusNotApplicable:
				ga.ImplementedControls++
				continue
			}
		}

		// Identify what's missing.
		var missing []string
		if !hasAssessment {
			missing = append(missing, "assessment")
		} else if assessment.Status == compliance.StatusNonCompliant {
			missing = append(missing, "compliant implementation")
		} else if assessment.Status == compliance.StatusPartial {
			missing = append(missing, "full implementation")
		}
		if !hasEvidence {
			missing = append(missing, "evidence")
		}

		if len(missing) == 0 {
			ga.ImplementedControls++
			continue
		}

		gap := Gap{
			Control:       ctrl,
			MissingItems:  missing,
			Severity:      categorizeSeverity(ctrl, missing),
			Recommendation: recommendAction(ctrl, missing),
		}

		ga.Gaps = append(ga.Gaps, gap)
		ga.GapCount++
	}

	return ga
}

// categorizeSeverity assigns severity based on the control category and
// number of missing items. Security controls with no assessment or evidence
// are critical; availability and confidentiality gaps are high.
func categorizeSeverity(ctrl compliance.Control, missing []string) GapSeverity {
	allMissing := len(missing) >= 2

	switch ctrl.Category {
	case compliance.CategorySecurity:
		if allMissing {
			return SeverityCritical
		}
		return SeverityHigh
	case compliance.CategoryAvailability, compliance.CategoryConfidentiality:
		if allMissing {
			return SeverityHigh
		}
		return SeverityMedium
	default:
		if allMissing {
			return SeverityMedium
		}
		return SeverityLow
	}
}

func recommendAction(ctrl compliance.Control, missing []string) string {
	hasNoAssessment := false
	hasNoEvidence := false
	for _, m := range missing {
		switch m {
		case "assessment":
			hasNoAssessment = true
		case "evidence":
			hasNoEvidence = true
		}
	}

	switch {
	case hasNoAssessment && hasNoEvidence:
		return "Implement control, perform assessment, and collect evidence for " + string(ctrl.ID) + " (" + ctrl.Title + ")"
	case hasNoAssessment:
		return "Perform formal assessment of " + string(ctrl.ID) + " (" + ctrl.Title + ")"
	case hasNoEvidence:
		return "Collect and document evidence for " + string(ctrl.ID) + " (" + ctrl.Title + ")"
	default:
		return "Review and remediate " + string(ctrl.ID) + " (" + ctrl.Title + ")"
	}
}
