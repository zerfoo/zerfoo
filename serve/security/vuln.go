package security

import (
	"context"
	"time"
)

// Severity indicates the severity of a vulnerability finding.
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
	SeverityInfo     Severity = "info"
)

// Finding represents a single vulnerability or audit finding.
type Finding struct {
	ID          string
	Severity    Severity
	Package     string
	Version     string
	FixVersion  string // empty if no fix available
	Description string
	CVE         string // e.g., "CVE-2024-1234"
	FoundAt     time.Time
}

// DependencyAuditor scans project dependencies for known vulnerabilities.
type DependencyAuditor interface {
	// Audit returns all known vulnerability findings for the project.
	Audit(ctx context.Context) ([]Finding, error)
}

// CVEChecker queries a CVE database for a specific package and version.
type CVEChecker interface {
	// Check returns findings for the given package at the given version.
	Check(ctx context.Context, pkg, version string) ([]Finding, error)
}

// AuditReport summarizes a dependency audit.
type AuditReport struct {
	Findings  []Finding
	ScannedAt time.Time
	TotalDeps int
}

// CountBySeverity returns the number of findings at or above the given severity.
func (r *AuditReport) CountBySeverity(minSeverity Severity) int {
	order := map[Severity]int{
		SeverityCritical: 4,
		SeverityHigh:     3,
		SeverityMedium:   2,
		SeverityLow:      1,
		SeverityInfo:     0,
	}
	threshold := order[minSeverity]
	count := 0
	for _, f := range r.Findings {
		if order[f.Severity] >= threshold {
			count++
		}
	}
	return count
}

// HasCritical reports whether the report contains any critical findings.
func (r *AuditReport) HasCritical() bool {
	for _, f := range r.Findings {
		if f.Severity == SeverityCritical {
			return true
		}
	}
	return false
}
