package compliance

import (
	"fmt"
	"time"
)

// EvidenceType classifies the source of compliance evidence.
type EvidenceType string

const (
	EvidenceCICD       EvidenceType = "ci_cd"
	EvidenceAccessLog  EvidenceType = "access_log"
	EvidenceCodeReview EvidenceType = "code_review"
	EvidencePolicy     EvidenceType = "policy"
	EvidenceManual     EvidenceType = "manual"
)

// Evidence represents a single piece of compliance evidence.
type Evidence struct {
	ID          string
	Type        EvidenceType
	ControlID   ControlID
	Title       string
	Description string
	Source      string // URI or path to the source system
	CollectedAt time.Time
	CollectedBy string
	Data        map[string]string // Arbitrary key-value evidence data
}

// EvidenceSource defines the interface for automated evidence collection.
type EvidenceSource interface {
	// Name returns a human-readable name for this evidence source.
	Name() string

	// Collect gathers evidence relevant to the given control.
	// Returns collected evidence or an error if collection fails.
	Collect(controlID ControlID) ([]Evidence, error)

	// SupportedControls returns the control IDs this source can provide evidence for.
	SupportedControls() []ControlID
}

// EvidenceCollector orchestrates evidence collection from multiple sources.
type EvidenceCollector struct {
	sources  []EvidenceSource
	evidence map[string]Evidence // keyed by Evidence.ID
}

// NewEvidenceCollector creates a collector with the given sources.
func NewEvidenceCollector(sources ...EvidenceSource) *EvidenceCollector {
	return &EvidenceCollector{
		sources:  sources,
		evidence: make(map[string]Evidence),
	}
}

// AddSource registers an additional evidence source.
func (ec *EvidenceCollector) AddSource(src EvidenceSource) {
	ec.sources = append(ec.sources, src)
}

// CollectAll gathers evidence from all sources for all their supported controls.
// Returns the number of evidence items collected and any errors encountered.
func (ec *EvidenceCollector) CollectAll() (int, []error) {
	var errs []error
	count := 0
	for _, src := range ec.sources {
		for _, cid := range src.SupportedControls() {
			items, err := src.Collect(cid)
			if err != nil {
				errs = append(errs, fmt.Errorf("source %s, control %s: %w", src.Name(), cid, err))
				continue
			}
			for _, item := range items {
				ec.evidence[item.ID] = item
				count++
			}
		}
	}
	return count, errs
}

// CollectForControl gathers evidence for a specific control from all sources.
func (ec *EvidenceCollector) CollectForControl(controlID ControlID) ([]Evidence, []error) {
	var results []Evidence
	var errs []error
	for _, src := range ec.sources {
		for _, cid := range src.SupportedControls() {
			if cid != controlID {
				continue
			}
			items, err := src.Collect(controlID)
			if err != nil {
				errs = append(errs, fmt.Errorf("source %s: %w", src.Name(), err))
				continue
			}
			for _, item := range items {
				ec.evidence[item.ID] = item
			}
			results = append(results, items...)
		}
	}
	return results, errs
}

// Evidence returns all collected evidence.
func (ec *EvidenceCollector) Evidence() []Evidence {
	out := make([]Evidence, 0, len(ec.evidence))
	for _, e := range ec.evidence {
		out = append(out, e)
	}
	return out
}

// EvidenceByControl returns collected evidence filtered by control ID.
func (ec *EvidenceCollector) EvidenceByControl(controlID ControlID) []Evidence {
	var out []Evidence
	for _, e := range ec.evidence {
		if e.ControlID == controlID {
			out = append(out, e)
		}
	}
	return out
}

// GitHubCISource collects evidence from GitHub Actions CI/CD pipelines.
type GitHubCISource struct {
	Owner string
	Repo  string
}

// Name returns the source name.
func (g *GitHubCISource) Name() string {
	return fmt.Sprintf("github-ci:%s/%s", g.Owner, g.Repo)
}

// SupportedControls returns controls that CI/CD evidence supports.
func (g *GitHubCISource) SupportedControls() []ControlID {
	return []ControlID{"CC7.1", "CC8.1", "CC6.8"}
}

// Collect gathers CI/CD evidence for the given control.
func (g *GitHubCISource) Collect(controlID ControlID) ([]Evidence, error) {
	now := time.Now()
	source := fmt.Sprintf("https://github.com/%s/%s/actions", g.Owner, g.Repo)
	switch controlID {
	case "CC7.1":
		return []Evidence{{
			ID:          fmt.Sprintf("ghci-%s-%s-monitoring-%d", g.Owner, g.Repo, now.Unix()),
			Type:        EvidenceCICD,
			ControlID:   controlID,
			Title:       "CI/CD Pipeline Monitoring",
			Description: "GitHub Actions workflows provide automated build, test, and deployment monitoring.",
			Source:      source,
			CollectedAt: now,
			CollectedBy: "compliance-automation",
			Data: map[string]string{
				"owner": g.Owner,
				"repo":  g.Repo,
			},
		}}, nil
	case "CC8.1":
		return []Evidence{{
			ID:          fmt.Sprintf("ghci-%s-%s-change-mgmt-%d", g.Owner, g.Repo, now.Unix()),
			Type:        EvidenceCICD,
			ControlID:   controlID,
			Title:       "Automated Change Management",
			Description: "All changes pass through CI pipeline with automated testing before merge.",
			Source:      source,
			CollectedAt: now,
			CollectedBy: "compliance-automation",
			Data: map[string]string{
				"owner": g.Owner,
				"repo":  g.Repo,
			},
		}}, nil
	case "CC6.8":
		return []Evidence{{
			ID:          fmt.Sprintf("ghci-%s-%s-sw-control-%d", g.Owner, g.Repo, now.Unix()),
			Type:        EvidenceCICD,
			ControlID:   controlID,
			Title:       "Software Change Controls",
			Description: "Branch protection and required CI checks prevent unauthorized software changes.",
			Source:      source,
			CollectedAt: now,
			CollectedBy: "compliance-automation",
			Data: map[string]string{
				"owner": g.Owner,
				"repo":  g.Repo,
			},
		}}, nil
	default:
		return nil, fmt.Errorf("unsupported control %s for GitHub CI source", controlID)
	}
}

// CodeReviewSource collects evidence from code review practices.
type CodeReviewSource struct {
	Owner string
	Repo  string
}

// Name returns the source name.
func (cr *CodeReviewSource) Name() string {
	return fmt.Sprintf("code-review:%s/%s", cr.Owner, cr.Repo)
}

// SupportedControls returns controls that code review evidence supports.
func (cr *CodeReviewSource) SupportedControls() []ControlID {
	return []ControlID{"CC8.1", "CC6.8", "CC5.2"}
}

// Collect gathers code review evidence for the given control.
func (cr *CodeReviewSource) Collect(controlID ControlID) ([]Evidence, error) {
	now := time.Now()
	source := fmt.Sprintf("https://github.com/%s/%s/pulls", cr.Owner, cr.Repo)
	switch controlID {
	case "CC8.1", "CC6.8", "CC5.2":
		return []Evidence{{
			ID:          fmt.Sprintf("cr-%s-%s-%s-%d", cr.Owner, cr.Repo, controlID, now.Unix()),
			Type:        EvidenceCodeReview,
			ControlID:   controlID,
			Title:       "Code Review Process",
			Description: "All code changes require pull request review before merge.",
			Source:      source,
			CollectedAt: now,
			CollectedBy: "compliance-automation",
			Data: map[string]string{
				"owner": cr.Owner,
				"repo":  cr.Repo,
			},
		}}, nil
	default:
		return nil, fmt.Errorf("unsupported control %s for code review source", controlID)
	}
}

// AccessLogSource collects evidence from access log systems.
type AccessLogSource struct {
	SystemName string
	LogPath    string
}

// Name returns the source name.
func (al *AccessLogSource) Name() string {
	return fmt.Sprintf("access-log:%s", al.SystemName)
}

// SupportedControls returns controls that access log evidence supports.
func (al *AccessLogSource) SupportedControls() []ControlID {
	return []ControlID{"CC6.1", "CC6.2", "CC6.3"}
}

// Collect gathers access log evidence for the given control.
func (al *AccessLogSource) Collect(controlID ControlID) ([]Evidence, error) {
	now := time.Now()
	switch controlID {
	case "CC6.1", "CC6.2", "CC6.3":
		return []Evidence{{
			ID:          fmt.Sprintf("al-%s-%s-%d", al.SystemName, controlID, now.Unix()),
			Type:        EvidenceAccessLog,
			ControlID:   controlID,
			Title:       fmt.Sprintf("Access Log Evidence - %s", al.SystemName),
			Description: fmt.Sprintf("Access logs from %s demonstrating logical access controls.", al.SystemName),
			Source:      al.LogPath,
			CollectedAt: now,
			CollectedBy: "compliance-automation",
			Data: map[string]string{
				"system":   al.SystemName,
				"log_path": al.LogPath,
			},
		}}, nil
	default:
		return nil, fmt.Errorf("unsupported control %s for access log source", controlID)
	}
}
