package audit

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/compliance"
)

// ReportConfig configures the audit report generation.
type ReportConfig struct {
	Organization string
	Auditor      string
	PeriodStart  time.Time
	PeriodEnd    time.Time
}

// Report is a complete SOC 2 Type I audit report.
type Report struct {
	Config     ReportConfig
	Readiness  *ReadinessAssessment
	Gaps       *GapAnalysis
	GeneratedAt time.Time
}

// GenerateReport creates a complete SOC 2 Type I audit report from the
// control mapping and evidence collector.
func GenerateReport(cfg ReportConfig, mapping *compliance.ControlMapping, collector *compliance.EvidenceCollector) *Report {
	return &Report{
		Config:      cfg,
		Readiness:   AssessReadiness(mapping, collector),
		Gaps:        AnalyzeGaps(mapping, collector),
		GeneratedAt: time.Now(),
	}
}

// Render formats the audit report as a markdown document.
func (r *Report) Render() string {
	var b strings.Builder

	r.renderHeader(&b)
	r.renderExecutiveSummary(&b)
	r.renderReadinessOverview(&b)
	r.renderGapAnalysis(&b)
	r.renderControlDetails(&b)
	r.renderTestingProcedures(&b)
	r.renderConclusion(&b)

	return b.String()
}

func (r *Report) renderHeader(b *strings.Builder) {
	b.WriteString(fmt.Sprintf("# SOC 2 Type I Audit Report\n\n"))
	b.WriteString(fmt.Sprintf("**Organization:** %s\n", r.Config.Organization))
	b.WriteString(fmt.Sprintf("**Auditor:** %s\n", r.Config.Auditor))
	b.WriteString(fmt.Sprintf("**Audit Period:** %s to %s\n",
		r.Config.PeriodStart.Format("2006-01-02"),
		r.Config.PeriodEnd.Format("2006-01-02")))
	b.WriteString(fmt.Sprintf("**Report Generated:** %s\n\n",
		r.GeneratedAt.Format("2006-01-02 15:04:05")))
	b.WriteString("---\n\n")
}

func (r *Report) renderExecutiveSummary(b *strings.Builder) {
	b.WriteString("## Executive Summary\n\n")
	b.WriteString(fmt.Sprintf("This SOC 2 Type I report evaluates the design effectiveness of %s's controls ",
		r.Config.Organization))
	b.WriteString("relevant to the Trust Services Criteria as of the audit date.\n\n")

	b.WriteString("### Key Metrics\n\n")
	b.WriteString(fmt.Sprintf("| Metric | Value |\n"))
	b.WriteString(fmt.Sprintf("|--------|-------|\n"))
	b.WriteString(fmt.Sprintf("| Total Controls | %d |\n", r.Gaps.TotalControls))
	b.WriteString(fmt.Sprintf("| Implemented Controls | %d |\n", r.Gaps.ImplementedControls))
	b.WriteString(fmt.Sprintf("| Controls with Gaps | %d |\n", r.Gaps.GapCount))
	b.WriteString(fmt.Sprintf("| Readiness Rate | %.1f%% |\n", r.Readiness.ReadinessRate()))
	b.WriteString(fmt.Sprintf("| Gap Rate | %.1f%% |\n\n", r.Gaps.GapRate()))
}

func (r *Report) renderReadinessOverview(b *strings.Builder) {
	b.WriteString("## Readiness Assessment\n\n")

	ready := r.Readiness.Ready()
	partial := r.Readiness.Partial()
	notReady := r.Readiness.NotReady()

	b.WriteString(fmt.Sprintf("| Status | Count |\n"))
	b.WriteString(fmt.Sprintf("|--------|-------|\n"))
	b.WriteString(fmt.Sprintf("| Ready | %d |\n", len(ready)))
	b.WriteString(fmt.Sprintf("| Partial | %d |\n", len(partial)))
	b.WriteString(fmt.Sprintf("| Not Ready | %d |\n\n", len(notReady)))

	if len(ready) > 0 {
		b.WriteString("### Ready Controls\n\n")
		for _, cr := range ready {
			b.WriteString(fmt.Sprintf("- **%s** — %s\n", cr.Control.ID, cr.Control.Title))
		}
		b.WriteString("\n")
	}

	if len(partial) > 0 {
		b.WriteString("### Partially Ready Controls\n\n")
		for _, cr := range partial {
			b.WriteString(fmt.Sprintf("- **%s** — %s: %s\n", cr.Control.ID, cr.Control.Title, cr.Notes))
		}
		b.WriteString("\n")
	}

	if len(notReady) > 0 {
		b.WriteString("### Not Ready Controls\n\n")
		for _, cr := range notReady {
			b.WriteString(fmt.Sprintf("- **%s** — %s: %s\n", cr.Control.ID, cr.Control.Title, cr.Notes))
		}
		b.WriteString("\n")
	}
}

func (r *Report) renderGapAnalysis(b *strings.Builder) {
	b.WriteString("## Gap Analysis\n\n")

	if r.Gaps.GapCount == 0 {
		b.WriteString("No gaps identified. All controls are fully implemented with evidence.\n\n")
		return
	}

	b.WriteString(fmt.Sprintf("**%d gaps identified across %d controls.**\n\n",
		r.Gaps.GapCount, r.Gaps.TotalControls))

	// Group by severity.
	for _, sev := range []GapSeverity{SeverityCritical, SeverityHigh, SeverityMedium, SeverityLow} {
		gaps := r.Gaps.GapsBySeverity(sev)
		if len(gaps) == 0 {
			continue
		}
		b.WriteString(fmt.Sprintf("### %s Severity (%d)\n\n", titleCase(string(sev)), len(gaps)))
		b.WriteString("| Control | Title | Missing | Recommendation |\n")
		b.WriteString("|---------|-------|---------|----------------|\n")
		for _, g := range gaps {
			b.WriteString(fmt.Sprintf("| %s | %s | %s | %s |\n",
				g.Control.ID, g.Control.Title,
				strings.Join(g.MissingItems, ", "),
				g.Recommendation))
		}
		b.WriteString("\n")
	}
}

func (r *Report) renderControlDetails(b *strings.Builder) {
	b.WriteString("## Control Details by Category\n\n")

	categories := []compliance.Category{
		compliance.CategorySecurity,
		compliance.CategoryAvailability,
		compliance.CategoryConfidentiality,
		compliance.CategoryProcessingIntegrity,
		compliance.CategoryPrivacy,
	}

	for _, cat := range categories {
		catControls := filterControlsByCategory(r.Readiness.Controls, cat)
		if len(catControls) == 0 {
			continue
		}
		b.WriteString(fmt.Sprintf("### %s\n\n", cat))
		b.WriteString("| Control | Title | Readiness | Evidence Count | Notes |\n")
		b.WriteString("|---------|-------|-----------|----------------|-------|\n")
		for _, cr := range catControls {
			b.WriteString(fmt.Sprintf("| %s | %s | %s | %d | %s |\n",
				cr.Control.ID, cr.Control.Title,
				cr.Level, len(cr.EvidenceIDs), cr.Notes))
		}
		b.WriteString("\n")
	}
}

func (r *Report) renderTestingProcedures(b *strings.Builder) {
	b.WriteString("## Control Testing Procedures\n\n")

	procedures := testingProcedures()
	categories := sortedCategories(procedures)

	for _, cat := range categories {
		procs := procedures[cat]
		b.WriteString(fmt.Sprintf("### %s\n\n", cat))
		for _, p := range procs {
			b.WriteString(fmt.Sprintf("**%s: %s**\n\n", p.ControlID, p.Title))
			b.WriteString(fmt.Sprintf("- **Objective:** %s\n", p.Objective))
			b.WriteString(fmt.Sprintf("- **Procedure:** %s\n", p.Procedure))
			b.WriteString(fmt.Sprintf("- **Expected Evidence:** %s\n\n", p.ExpectedEvidence))
		}
	}
}

func (r *Report) renderConclusion(b *strings.Builder) {
	b.WriteString("## Conclusion\n\n")

	rate := r.Readiness.ReadinessRate()
	switch {
	case rate >= 90:
		b.WriteString(fmt.Sprintf("The organization demonstrates strong control design with a %.1f%% readiness rate. ", rate))
		b.WriteString("Minor gaps should be addressed before the Type II examination.\n")
	case rate >= 70:
		b.WriteString(fmt.Sprintf("The organization has made significant progress with a %.1f%% readiness rate. ", rate))
		b.WriteString("Several gaps require remediation before proceeding to a Type II examination.\n")
	case rate >= 50:
		b.WriteString(fmt.Sprintf("The organization has moderate readiness at %.1f%%. ", rate))
		b.WriteString("Substantial work is needed to address identified gaps before a Type II examination.\n")
	default:
		b.WriteString(fmt.Sprintf("The organization's readiness rate of %.1f%% indicates early-stage compliance maturity. ", rate))
		b.WriteString("A comprehensive remediation plan is recommended before pursuing SOC 2 certification.\n")
	}
}

func filterControlsByCategory(controls []ControlReadiness, cat compliance.Category) []ControlReadiness {
	var out []ControlReadiness
	for _, cr := range controls {
		if cr.Control.Category == cat {
			out = append(out, cr)
		}
	}
	return out
}

// TestingProcedure defines how to test a specific control.
type TestingProcedure struct {
	ControlID        compliance.ControlID
	Category         compliance.Category
	Title            string
	Objective        string
	Procedure        string
	ExpectedEvidence string
}

func testingProcedures() map[compliance.Category][]TestingProcedure {
	return map[compliance.Category][]TestingProcedure{
		compliance.CategorySecurity: {
			{ControlID: "CC1.1", Category: compliance.CategorySecurity, Title: "Integrity and Ethics", Objective: "Verify commitment to integrity and ethical values", Procedure: "Review code of conduct, ethics policies, and acknowledgment records", ExpectedEvidence: "Signed code of conduct, ethics policy document"},
			{ControlID: "CC2.1", Category: compliance.CategorySecurity, Title: "Information Quality", Objective: "Verify use of relevant, quality information", Procedure: "Review information governance policies and data quality procedures", ExpectedEvidence: "Data governance policy, quality review records"},
			{ControlID: "CC3.1", Category: compliance.CategorySecurity, Title: "Risk Objectives", Objective: "Verify objectives are specified with sufficient clarity", Procedure: "Review risk assessment documentation and objective statements", ExpectedEvidence: "Risk assessment report, documented objectives"},
			{ControlID: "CC5.1", Category: compliance.CategorySecurity, Title: "Control Activities", Objective: "Verify control activities mitigate risks", Procedure: "Review control documentation and test control effectiveness", ExpectedEvidence: "Control matrix, test results"},
			{ControlID: "CC6.1", Category: compliance.CategorySecurity, Title: "Logical Access", Objective: "Verify logical access security implementation", Procedure: "Review API key management, authentication mechanisms, and access controls", ExpectedEvidence: "API key policies, authentication configuration, access control lists"},
			{ControlID: "CC7.1", Category: compliance.CategorySecurity, Title: "Infrastructure Monitoring", Objective: "Verify detection and monitoring procedures", Procedure: "Review monitoring configuration, alerting rules, and CI/CD pipeline logs", ExpectedEvidence: "Monitoring dashboards, alert configuration, CI/CD logs"},
			{ControlID: "CC8.1", Category: compliance.CategorySecurity, Title: "Change Management", Objective: "Verify change management process", Procedure: "Review pull request history, CI/CD checks, and release procedures", ExpectedEvidence: "PR reviews, CI/CD pipeline results, release notes"},
			{ControlID: "CC9.1", Category: compliance.CategorySecurity, Title: "Risk Mitigation", Objective: "Verify risk mitigation activities", Procedure: "Review vulnerability scan results and remediation tracking", ExpectedEvidence: "Scan reports, remediation tickets"},
		},
		compliance.CategoryAvailability: {
			{ControlID: "A1.1", Category: compliance.CategoryAvailability, Title: "Capacity Planning", Objective: "Verify capacity planning and monitoring", Procedure: "Review capacity plans, resource utilization metrics, and scaling policies", ExpectedEvidence: "Capacity plan document, utilization dashboards"},
			{ControlID: "A1.2", Category: compliance.CategoryAvailability, Title: "Recovery Procedures", Objective: "Verify disaster recovery infrastructure", Procedure: "Review backup procedures, recovery infrastructure, and DR documentation", ExpectedEvidence: "Backup logs, DR plan, infrastructure diagram"},
			{ControlID: "A1.3", Category: compliance.CategoryAvailability, Title: "Recovery Testing", Objective: "Verify recovery testing procedures", Procedure: "Review recovery test results and post-test reports", ExpectedEvidence: "Test execution records, recovery time metrics"},
		},
		compliance.CategoryConfidentiality: {
			{ControlID: "C1.1", Category: compliance.CategoryConfidentiality, Title: "Confidential Info ID", Objective: "Verify identification of confidential information", Procedure: "Review data classification policies and secrets management implementation", ExpectedEvidence: "Classification policy, secrets store configuration"},
			{ControlID: "C1.2", Category: compliance.CategoryConfidentiality, Title: "Confidential Info Disposal", Objective: "Verify secure disposal procedures", Procedure: "Review data retention and disposal policies and procedures", ExpectedEvidence: "Retention policy, disposal certificates"},
		},
		compliance.CategoryProcessingIntegrity: {
			{ControlID: "PI1.1", Category: compliance.CategoryProcessingIntegrity, Title: "Processing Accuracy", Objective: "Verify processing accuracy controls", Procedure: "Review input validation, model parity tests, and output verification", ExpectedEvidence: "Test results, validation rules, parity reports"},
			{ControlID: "PI1.2", Category: compliance.CategoryProcessingIntegrity, Title: "Input Validation", Objective: "Verify input validation controls", Procedure: "Review input sanitization and validation code and test coverage", ExpectedEvidence: "Validation code, test coverage report"},
		},
		compliance.CategoryPrivacy: {
			{ControlID: "P1.1", Category: compliance.CategoryPrivacy, Title: "Privacy Notice", Objective: "Verify privacy notice is provided", Procedure: "Review privacy policy and notice distribution mechanism", ExpectedEvidence: "Privacy policy document, distribution records"},
		},
	}
}

func titleCase(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

func sortedCategories(m map[compliance.Category][]TestingProcedure) []compliance.Category {
	order := []compliance.Category{
		compliance.CategorySecurity,
		compliance.CategoryAvailability,
		compliance.CategoryConfidentiality,
		compliance.CategoryProcessingIntegrity,
		compliance.CategoryPrivacy,
	}
	var out []compliance.Category
	for _, cat := range order {
		if _, ok := m[cat]; ok {
			out = append(out, cat)
		}
	}
	return out
}

// SortGapsBySeverity sorts gaps with critical first, low last.
func SortGapsBySeverity(gaps []Gap) {
	order := map[GapSeverity]int{
		SeverityCritical: 0,
		SeverityHigh:     1,
		SeverityMedium:   2,
		SeverityLow:      3,
	}
	sort.Slice(gaps, func(i, j int) bool {
		return order[gaps[i].Severity] < order[gaps[j].Severity]
	})
}
