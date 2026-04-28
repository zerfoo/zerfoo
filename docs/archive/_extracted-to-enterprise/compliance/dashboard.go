package compliance

import "time"

// DashboardSummary provides an overview of the compliance posture.
type DashboardSummary struct {
	GeneratedAt     time.Time
	TotalControls   int
	Compliant       int
	Partial         int
	NonCompliant    int
	NotAssessed     int
	NotApplicable   int
	EvidenceCount   int
	CategorySummary []CategorySummary
}

// ComplianceRate returns the percentage of assessed controls that are fully compliant.
// Returns 0 if no controls have been assessed.
func (ds *DashboardSummary) ComplianceRate() float64 {
	assessed := ds.Compliant + ds.Partial + ds.NonCompliant
	if assessed == 0 {
		return 0
	}
	return float64(ds.Compliant) / float64(assessed) * 100
}

// CategorySummary provides compliance status for a single category.
type CategorySummary struct {
	Category     Category
	Total        int
	Compliant    int
	Partial      int
	NonCompliant int
	NotAssessed  int
}

// Dashboard generates a compliance dashboard from a ControlMapping and EvidenceCollector.
type Dashboard struct {
	mapping   *ControlMapping
	collector *EvidenceCollector
}

// NewDashboard creates a Dashboard from the given mapping and collector.
func NewDashboard(mapping *ControlMapping, collector *EvidenceCollector) *Dashboard {
	return &Dashboard{
		mapping:   mapping,
		collector: collector,
	}
}

// Summary computes the current compliance dashboard summary.
func (d *Dashboard) Summary() DashboardSummary {
	controls := d.mapping.Controls()

	summary := DashboardSummary{
		GeneratedAt:   time.Now(),
		TotalControls: len(controls),
	}

	if d.collector != nil {
		summary.EvidenceCount = len(d.collector.Evidence())
	}

	catCounts := make(map[Category]*CategorySummary)
	for _, c := range controls {
		cs, ok := catCounts[c.Category]
		if !ok {
			cs = &CategorySummary{Category: c.Category}
			catCounts[c.Category] = cs
		}
		cs.Total++

		a, assessed := d.mapping.Assessment(c.ID)
		if !assessed {
			summary.NotAssessed++
			cs.NotAssessed++
			continue
		}
		switch a.Status {
		case StatusCompliant:
			summary.Compliant++
			cs.Compliant++
		case StatusPartial:
			summary.Partial++
			cs.Partial++
		case StatusNonCompliant:
			summary.NonCompliant++
			cs.NonCompliant++
		case StatusNotApplicable:
			summary.NotApplicable++
		default:
			summary.NotAssessed++
			cs.NotAssessed++
		}
	}

	for _, cs := range catCounts {
		summary.CategorySummary = append(summary.CategorySummary, *cs)
	}
	return summary
}

// ControlDetail provides detailed status for a single control.
type ControlDetail struct {
	Control    Control
	Assessment ControlAssessment
	Assessed   bool
	Evidence   []Evidence
}

// ControlDetails returns detailed information for every control.
func (d *Dashboard) ControlDetails() []ControlDetail {
	controls := d.mapping.Controls()
	details := make([]ControlDetail, 0, len(controls))
	for _, c := range controls {
		cd := ControlDetail{Control: c}
		if a, ok := d.mapping.Assessment(c.ID); ok {
			cd.Assessment = a
			cd.Assessed = true
		}
		if d.collector != nil {
			cd.Evidence = d.collector.EvidenceByControl(c.ID)
		}
		details = append(details, cd)
	}
	return details
}
