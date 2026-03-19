package security

import (
	"testing"
	"time"
)

func TestAuditReportCountBySeverity(t *testing.T) {
	r := &AuditReport{
		Findings: []Finding{
			{Severity: SeverityCritical, CVE: "CVE-2024-0001"},
			{Severity: SeverityHigh, CVE: "CVE-2024-0002"},
			{Severity: SeverityMedium, CVE: "CVE-2024-0003"},
			{Severity: SeverityLow, CVE: "CVE-2024-0004"},
			{Severity: SeverityInfo, CVE: "CVE-2024-0005"},
		},
		ScannedAt: time.Now(),
		TotalDeps: 42,
	}

	if got := r.CountBySeverity(SeverityCritical); got != 1 {
		t.Fatalf("critical: expected 1, got %d", got)
	}
	if got := r.CountBySeverity(SeverityHigh); got != 2 {
		t.Fatalf("high+: expected 2, got %d", got)
	}
	if got := r.CountBySeverity(SeverityMedium); got != 3 {
		t.Fatalf("medium+: expected 3, got %d", got)
	}
	if got := r.CountBySeverity(SeverityInfo); got != 5 {
		t.Fatalf("info+: expected 5, got %d", got)
	}
}

func TestAuditReportHasCritical(t *testing.T) {
	r := &AuditReport{
		Findings: []Finding{
			{Severity: SeverityHigh},
		},
	}
	if r.HasCritical() {
		t.Fatal("expected no critical")
	}

	r.Findings = append(r.Findings, Finding{Severity: SeverityCritical})
	if !r.HasCritical() {
		t.Fatal("expected critical")
	}
}

func TestAuditReportEmpty(t *testing.T) {
	r := &AuditReport{}
	if r.HasCritical() {
		t.Fatal("empty report should not have critical")
	}
	if got := r.CountBySeverity(SeverityInfo); got != 0 {
		t.Fatalf("expected 0, got %d", got)
	}
}
