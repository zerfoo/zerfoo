package compliance

import (
	"strings"
	"testing"
	"time"
)

func TestNewControlMapping(t *testing.T) {
	cm := NewControlMapping()
	controls := cm.Controls()
	if len(controls) == 0 {
		t.Fatal("expected default controls, got none")
	}

	// Verify key controls exist.
	for _, id := range []ControlID{"CC1.1", "CC6.1", "CC8.1", "A1.1", "C1.1", "PI1.1", "P1.1"} {
		if _, ok := cm.Control(id); !ok {
			t.Errorf("expected control %s to exist", id)
		}
	}
}

func TestControlsByCategory(t *testing.T) {
	cm := NewControlMapping()

	avail := cm.ControlsByCategory(CategoryAvailability)
	if len(avail) == 0 {
		t.Fatal("expected availability controls")
	}
	for _, c := range avail {
		if c.Category != CategoryAvailability {
			t.Errorf("control %s has category %s, want %s", c.ID, c.Category, CategoryAvailability)
		}
	}

	privacy := cm.ControlsByCategory(CategoryPrivacy)
	if len(privacy) == 0 {
		t.Fatal("expected privacy controls")
	}
}

func TestAssessControl(t *testing.T) {
	cm := NewControlMapping()

	a := ControlAssessment{
		ControlID:   "CC6.1",
		Status:      StatusCompliant,
		AssessedAt:  time.Now(),
		AssessedBy:  "auditor",
		EvidenceIDs: []string{"ev-1"},
		Notes:       "MFA enabled",
	}
	if err := cm.Assess(a); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got, ok := cm.Assessment("CC6.1")
	if !ok {
		t.Fatal("expected assessment for CC6.1")
	}
	if got.Status != StatusCompliant {
		t.Errorf("got status %s, want %s", got.Status, StatusCompliant)
	}
}

func TestAssessUnknownControl(t *testing.T) {
	cm := NewControlMapping()
	err := cm.Assess(ControlAssessment{ControlID: "INVALID"})
	if err == nil {
		t.Fatal("expected error for unknown control")
	}
}

func TestEvidenceCollector(t *testing.T) {
	ghSrc := &GitHubCISource{Owner: "zerfoo", Repo: "zerfoo"}
	crSrc := &CodeReviewSource{Owner: "zerfoo", Repo: "zerfoo"}
	alSrc := &AccessLogSource{SystemName: "prod-iam", LogPath: "/var/log/access.log"}

	collector := NewEvidenceCollector(ghSrc, crSrc)
	collector.AddSource(alSrc)

	count, errs := collector.CollectAll()
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if count == 0 {
		t.Fatal("expected collected evidence")
	}

	all := collector.Evidence()
	if len(all) != count {
		t.Errorf("Evidence() returned %d items, expected %d", len(all), count)
	}
}

func TestCollectForControl(t *testing.T) {
	ghSrc := &GitHubCISource{Owner: "zerfoo", Repo: "zerfoo"}
	collector := NewEvidenceCollector(ghSrc)

	items, errs := collector.CollectForControl("CC8.1")
	if len(errs) > 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(items) == 0 {
		t.Fatal("expected evidence for CC8.1")
	}
	for _, e := range items {
		if e.ControlID != "CC8.1" {
			t.Errorf("evidence %s has control %s, want CC8.1", e.ID, e.ControlID)
		}
	}
}

func TestEvidenceByControl(t *testing.T) {
	ghSrc := &GitHubCISource{Owner: "zerfoo", Repo: "zerfoo"}
	collector := NewEvidenceCollector(ghSrc)
	collector.CollectAll()

	items := collector.EvidenceByControl("CC7.1")
	if len(items) == 0 {
		t.Fatal("expected evidence for CC7.1")
	}
}

func TestGeneratePolicy(t *testing.T) {
	effective := time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC)
	types := []PolicyType{
		PolicyAccessControl,
		PolicyChangeManagement,
		PolicyIncidentResponse,
		PolicyDataClassification,
		PolicyRiskAssessment,
		PolicyVendorManagement,
	}

	for _, pt := range types {
		doc, err := GeneratePolicy(pt, "Feza", "CTO", "CEO", effective)
		if err != nil {
			t.Fatalf("GeneratePolicy(%s): %v", pt, err)
		}
		if doc.Title == "" {
			t.Errorf("policy %s has empty title", pt)
		}
		if doc.Owner != "CTO" {
			t.Errorf("policy %s owner = %q, want CTO", pt, doc.Owner)
		}
		if len(doc.Sections) == 0 {
			t.Errorf("policy %s has no sections", pt)
		}
		if len(doc.ControlIDs) == 0 {
			t.Errorf("policy %s has no control IDs", pt)
		}
		if !doc.ReviewBy.After(effective) {
			t.Errorf("policy %s review date %s should be after effective date %s", pt, doc.ReviewBy, effective)
		}
	}
}

func TestGeneratePolicyUnknownType(t *testing.T) {
	_, err := GeneratePolicy("invalid", "Feza", "CTO", "CEO", time.Now())
	if err == nil {
		t.Fatal("expected error for unknown policy type")
	}
}

func TestPolicyRender(t *testing.T) {
	doc, _ := GeneratePolicy(PolicyAccessControl, "Feza", "CTO", "CEO", time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC))
	rendered := doc.Render()
	if !strings.Contains(rendered, "Feza Access Control Policy") {
		t.Error("rendered policy missing title")
	}
	if !strings.Contains(rendered, "2028-07-01") {
		t.Error("rendered policy missing effective date")
	}
	if !strings.Contains(rendered, "## Purpose") {
		t.Error("rendered policy missing Purpose section")
	}
}

func TestDashboardSummary(t *testing.T) {
	cm := NewControlMapping()

	// Assess a few controls.
	cm.Assess(ControlAssessment{ControlID: "CC6.1", Status: StatusCompliant, AssessedAt: time.Now()})
	cm.Assess(ControlAssessment{ControlID: "CC6.2", Status: StatusPartial, AssessedAt: time.Now()})
	cm.Assess(ControlAssessment{ControlID: "CC8.1", Status: StatusNonCompliant, AssessedAt: time.Now()})
	cm.Assess(ControlAssessment{ControlID: "A1.1", Status: StatusNotApplicable, AssessedAt: time.Now()})

	collector := NewEvidenceCollector(&GitHubCISource{Owner: "zerfoo", Repo: "zerfoo"})
	collector.CollectAll()

	dashboard := NewDashboard(cm, collector)
	s := dashboard.Summary()

	if s.TotalControls == 0 {
		t.Fatal("expected total controls > 0")
	}
	if s.Compliant != 1 {
		t.Errorf("compliant = %d, want 1", s.Compliant)
	}
	if s.Partial != 1 {
		t.Errorf("partial = %d, want 1", s.Partial)
	}
	if s.NonCompliant != 1 {
		t.Errorf("non-compliant = %d, want 1", s.NonCompliant)
	}
	if s.NotApplicable != 1 {
		t.Errorf("not-applicable = %d, want 1", s.NotApplicable)
	}
	if s.NotAssessed == 0 {
		t.Error("expected some not-assessed controls")
	}
	if s.EvidenceCount == 0 {
		t.Error("expected evidence count > 0")
	}
}

func TestDashboardComplianceRate(t *testing.T) {
	s := DashboardSummary{Compliant: 3, Partial: 1, NonCompliant: 1}
	rate := s.ComplianceRate()
	// 3 / (3+1+1) = 60%
	if rate < 59.9 || rate > 60.1 {
		t.Errorf("compliance rate = %f, want ~60", rate)
	}

	empty := DashboardSummary{}
	if empty.ComplianceRate() != 0 {
		t.Errorf("empty compliance rate = %f, want 0", empty.ComplianceRate())
	}
}

func TestControlDetails(t *testing.T) {
	cm := NewControlMapping()
	cm.Assess(ControlAssessment{ControlID: "CC6.1", Status: StatusCompliant, AssessedAt: time.Now()})

	collector := NewEvidenceCollector(&AccessLogSource{SystemName: "iam", LogPath: "/var/log/iam.log"})
	collector.CollectAll()

	dashboard := NewDashboard(cm, collector)
	details := dashboard.ControlDetails()

	if len(details) == 0 {
		t.Fatal("expected control details")
	}

	// Find CC6.1 — should have assessment and evidence.
	found := false
	for _, d := range details {
		if d.Control.ID == "CC6.1" {
			found = true
			if !d.Assessed {
				t.Error("CC6.1 should be assessed")
			}
			if len(d.Evidence) == 0 {
				t.Error("CC6.1 should have evidence from access log source")
			}
		}
	}
	if !found {
		t.Error("CC6.1 not found in control details")
	}
}
