package audit

import (
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/compliance"
)

func setupMapping(t *testing.T) *compliance.ControlMapping {
	t.Helper()
	return compliance.NewControlMapping()
}

func assessSome(t *testing.T, mapping *compliance.ControlMapping) {
	t.Helper()
	now := time.Now()
	// Assess some controls as compliant.
	for _, id := range []compliance.ControlID{"CC6.1", "CC6.7", "CC7.1", "CC7.3", "CC7.4", "CC9.1", "C1.1"} {
		if err := mapping.Assess(compliance.ControlAssessment{
			ControlID:  id,
			Status:     compliance.StatusCompliant,
			AssessedAt: now,
			AssessedBy: "test-auditor",
		}); err != nil {
			t.Fatalf("unexpected error assessing %s: %v", id, err)
		}
	}
	// Assess one as partial.
	if err := mapping.Assess(compliance.ControlAssessment{
		ControlID:  "CC8.1",
		Status:     compliance.StatusPartial,
		AssessedAt: now,
		AssessedBy: "test-auditor",
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAssessReadiness_NoAssessments(t *testing.T) {
	mapping := setupMapping(t)
	ra := AssessReadiness(mapping, nil)

	if len(ra.Controls) == 0 {
		t.Fatal("expected controls in readiness assessment")
	}
	// With no assessments and no evidence, all should be not ready.
	for _, cr := range ra.Controls {
		if cr.Level != ReadinessNotReady {
			t.Errorf("control %s: expected not_ready, got %s", cr.Control.ID, cr.Level)
		}
	}
	if rate := ra.ReadinessRate(); rate != 0 {
		t.Errorf("expected 0%% readiness, got %.1f%%", rate)
	}
}

func TestAssessReadiness_WithAssessmentsAndEvidence(t *testing.T) {
	mapping := setupMapping(t)
	assessSome(t, mapping)

	collector, _, _ := CollectAllEvidence(nil)
	ra := AssessReadiness(mapping, collector)

	ready := ra.Ready()
	if len(ready) == 0 {
		t.Error("expected some ready controls")
	}

	// CC6.1 is assessed compliant and has security evidence — should be ready.
	found := false
	for _, cr := range ready {
		if cr.Control.ID == "CC6.1" {
			found = true
			break
		}
	}
	if !found {
		t.Error("CC6.1 should be ready (assessed compliant + security evidence)")
	}

	if rate := ra.ReadinessRate(); rate <= 0 {
		t.Error("expected positive readiness rate")
	}
}

func TestAssessReadiness_NotApplicable(t *testing.T) {
	mapping := setupMapping(t)
	if err := mapping.Assess(compliance.ControlAssessment{
		ControlID:  "P1.2",
		Status:     compliance.StatusNotApplicable,
		AssessedAt: time.Now(),
		AssessedBy: "test-auditor",
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ra := AssessReadiness(mapping, nil)
	for _, cr := range ra.Controls {
		if cr.Control.ID == "P1.2" {
			if cr.Level != ReadinessReady {
				t.Errorf("P1.2 marked N/A should be ready, got %s", cr.Level)
			}
			return
		}
	}
	t.Error("P1.2 not found in readiness assessment")
}

func TestAnalyzeGaps_AllUnassessed(t *testing.T) {
	mapping := setupMapping(t)
	ga := AnalyzeGaps(mapping, nil)

	if ga.TotalControls == 0 {
		t.Fatal("expected controls")
	}
	if ga.GapCount == 0 {
		t.Error("expected gaps when nothing is assessed")
	}
	if ga.ImplementedControls != 0 {
		t.Errorf("expected 0 implemented, got %d", ga.ImplementedControls)
	}
	if ga.GapRate() == 0 {
		t.Error("expected non-zero gap rate")
	}
}

func TestAnalyzeGaps_WithImplementation(t *testing.T) {
	mapping := setupMapping(t)
	assessSome(t, mapping)

	collector, _, _ := CollectAllEvidence(nil)
	ga := AnalyzeGaps(mapping, collector)

	if ga.ImplementedControls == 0 {
		t.Error("expected some implemented controls")
	}
	if ga.GapCount >= ga.TotalControls {
		t.Error("expected fewer gaps than total controls")
	}
}

func TestAnalyzeGaps_SeverityClassification(t *testing.T) {
	mapping := setupMapping(t)
	ga := AnalyzeGaps(mapping, nil)

	critical := ga.GapsBySeverity(SeverityCritical)
	if len(critical) == 0 {
		t.Error("expected critical gaps for unassessed security controls")
	}

	// All critical gaps should be security controls.
	for _, g := range critical {
		if g.Control.Category != compliance.CategorySecurity {
			t.Errorf("critical gap %s has category %s, expected Security", g.Control.ID, g.Control.Category)
		}
	}
}

func TestAnalyzeGaps_ByCategory(t *testing.T) {
	mapping := setupMapping(t)
	ga := AnalyzeGaps(mapping, nil)

	secGaps := ga.GapsByCategory(compliance.CategorySecurity)
	if len(secGaps) == 0 {
		t.Error("expected security gaps")
	}

	availGaps := ga.GapsByCategory(compliance.CategoryAvailability)
	if len(availGaps) == 0 {
		t.Error("expected availability gaps")
	}
}

func TestSecurityControlSource(t *testing.T) {
	src := &SecurityControlSource{}
	if name := src.Name(); name != "security-controls" {
		t.Errorf("unexpected name: %s", name)
	}

	controls := src.SupportedControls()
	if len(controls) == 0 {
		t.Error("expected supported controls")
	}

	// Collect evidence for a known control.
	ev, err := src.Collect("CC6.1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ev) != 1 {
		t.Fatalf("expected 1 evidence item, got %d", len(ev))
	}
	if ev[0].ID != "sec-apikey-access" {
		t.Errorf("unexpected evidence ID: %s", ev[0].ID)
	}

	// Unsupported control returns nil.
	ev, err = src.Collect("CC1.1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ev != nil {
		t.Errorf("expected nil for unsupported control, got %v", ev)
	}
}

func TestPolicyEvidenceSource(t *testing.T) {
	pol := compliance.PolicyDocument{
		Type:       compliance.PolicyAccessControl,
		Title:      "Test Policy",
		Version:    "1.0",
		ControlIDs: []compliance.ControlID{"CC6.1", "CC6.2"},
	}
	src := &PolicyEvidenceSource{Policies: []compliance.PolicyDocument{pol}}

	supported := src.SupportedControls()
	if len(supported) != 2 {
		t.Fatalf("expected 2 supported controls, got %d", len(supported))
	}

	ev, err := src.Collect("CC6.1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ev) != 1 {
		t.Fatalf("expected 1 evidence, got %d", len(ev))
	}
	if ev[0].ControlID != "CC6.1" {
		t.Errorf("unexpected control ID: %s", ev[0].ControlID)
	}
}

func TestCollectAllEvidence(t *testing.T) {
	collector, count, errs := CollectAllEvidence(nil)
	if len(errs) > 0 {
		t.Errorf("unexpected errors: %v", errs)
	}
	if count == 0 {
		t.Error("expected some evidence collected")
	}
	if len(collector.Evidence()) == 0 {
		t.Error("expected evidence in collector")
	}
}

func TestGenerateReport(t *testing.T) {
	mapping := setupMapping(t)
	assessSome(t, mapping)

	collector, _, _ := CollectAllEvidence(nil)

	cfg := ReportConfig{
		Organization: "Feza, Inc",
		Auditor:      "Test Auditor",
		PeriodStart:  time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC),
		PeriodEnd:    time.Date(2028, 12, 31, 0, 0, 0, 0, time.UTC),
	}

	report := GenerateReport(cfg, mapping, collector)

	if report.Readiness == nil {
		t.Fatal("report missing readiness assessment")
	}
	if report.Gaps == nil {
		t.Fatal("report missing gap analysis")
	}
}

func TestReportRender(t *testing.T) {
	mapping := setupMapping(t)
	assessSome(t, mapping)
	collector, _, _ := CollectAllEvidence(nil)

	cfg := ReportConfig{
		Organization: "Feza, Inc",
		Auditor:      "Test Auditor",
		PeriodStart:  time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC),
		PeriodEnd:    time.Date(2028, 12, 31, 0, 0, 0, 0, time.UTC),
	}

	report := GenerateReport(cfg, mapping, collector)
	md := report.Render()

	required := []string{
		"SOC 2 Type I Audit Report",
		"Feza, Inc",
		"Test Auditor",
		"Executive Summary",
		"Readiness Assessment",
		"Gap Analysis",
		"Control Details by Category",
		"Control Testing Procedures",
		"Conclusion",
		"Security",
	}
	for _, s := range required {
		if !strings.Contains(md, s) {
			t.Errorf("report missing expected content: %q", s)
		}
	}

	// Verify markdown table formatting.
	if !strings.Contains(md, "| Metric | Value |") {
		t.Error("report missing metrics table")
	}
}

func TestSortGapsBySeverity(t *testing.T) {
	gaps := []Gap{
		{Severity: SeverityLow},
		{Severity: SeverityCritical},
		{Severity: SeverityMedium},
		{Severity: SeverityHigh},
	}
	SortGapsBySeverity(gaps)

	expected := []GapSeverity{SeverityCritical, SeverityHigh, SeverityMedium, SeverityLow}
	for i, g := range gaps {
		if g.Severity != expected[i] {
			t.Errorf("position %d: expected %s, got %s", i, expected[i], g.Severity)
		}
	}
}

func TestGapAnalysis_ZeroControls(t *testing.T) {
	ga := &GapAnalysis{}
	if rate := ga.GapRate(); rate != 0 {
		t.Errorf("expected 0 gap rate for empty analysis, got %f", rate)
	}
}

func TestReadinessAssessment_ZeroControls(t *testing.T) {
	ra := &ReadinessAssessment{}
	if rate := ra.ReadinessRate(); rate != 0 {
		t.Errorf("expected 0 readiness rate for empty assessment, got %f", rate)
	}
}
