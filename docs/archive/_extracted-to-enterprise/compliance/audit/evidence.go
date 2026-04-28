package audit

import (
	"sort"

	"github.com/zerfoo/zerfoo/compliance"
)

// SecurityControlSource collects evidence from the security package's
// implemented controls. It maps security features (API key management,
// encryption, network security, secrets management, vulnerability scanning,
// incident response) to the SOC 2 controls they satisfy.
type SecurityControlSource struct{}

// Name returns the source name.
func (s *SecurityControlSource) Name() string {
	return "security-controls"
}

// SupportedControls returns the SOC 2 control IDs that the security package
// provides evidence for.
func (s *SecurityControlSource) SupportedControls() []compliance.ControlID {
	return []compliance.ControlID{
		"CC6.1", // Logical access — API key management
		"CC6.2", // User access — API key provisioning
		"CC6.7", // Transmission security — encryption
		"CC7.1", // Infrastructure monitoring — network security
		"CC7.2", // Anomaly detection — rate limiting, IP filtering
		"CC7.3", // Security incident response — incident hooks
		"CC7.4", // Incident response plan — incident management
		"CC9.1", // Risk mitigation — vulnerability scanning
		"C1.1",  // Confidential info — secrets management
	}
}

// Collect returns evidence for a specific control based on security package capabilities.
func (s *SecurityControlSource) Collect(controlID compliance.ControlID) ([]compliance.Evidence, error) {
	m := securityEvidenceMap()
	ev, ok := m[controlID]
	if !ok {
		return nil, nil
	}
	return []compliance.Evidence{ev}, nil
}

func securityEvidenceMap() map[compliance.ControlID]compliance.Evidence {
	return map[compliance.ControlID]compliance.Evidence{
		"CC6.1": {
			ID: "sec-apikey-access", Type: compliance.EvidencePolicy, ControlID: "CC6.1",
			Title: "API Key Access Control", Description: "security.APIKeyManager implements logical access controls with key generation, validation, and revocation.",
			Source: "security/apikey.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "APIKeyManager", "package": "security"},
		},
		"CC6.2": {
			ID: "sec-apikey-provisioning", Type: compliance.EvidencePolicy, ControlID: "CC6.2",
			Title: "API Key Provisioning", Description: "API keys are provisioned with scoped permissions and expiration. Keys are validated before granting system access.",
			Source: "security/apikey.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "APIKeyManager", "package": "security"},
		},
		"CC6.7": {
			ID: "sec-encryption", Type: compliance.EvidencePolicy, ControlID: "CC6.7",
			Title: "Encryption Controls", Description: "security.Encryption provides AES-256-GCM encryption for data at rest and TLS configuration for data in transit.",
			Source: "security/encryption.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "Encryption", "package": "security"},
		},
		"CC7.1": {
			ID: "sec-network-monitoring", Type: compliance.EvidencePolicy, ControlID: "CC7.1",
			Title: "Network Security Monitoring", Description: "security.NetworkSecurity provides rate limiting and IP filtering for infrastructure monitoring.",
			Source: "security/network.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "NetworkSecurity", "package": "security"},
		},
		"CC7.2": {
			ID: "sec-anomaly-detection", Type: compliance.EvidencePolicy, ControlID: "CC7.2",
			Title: "Rate Limiting and IP Filtering", Description: "Rate limiter and IP filter detect and block anomalous access patterns.",
			Source: "security/network.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "RateLimiter,IPFilter", "package": "security"},
		},
		"CC7.3": {
			ID: "sec-incident-response", Type: compliance.EvidencePolicy, ControlID: "CC7.3",
			Title: "Incident Detection and Response", Description: "security.IncidentManager provides incident creation, classification, and tracking.",
			Source: "security/incident.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "IncidentManager", "package": "security"},
		},
		"CC7.4": {
			ID: "sec-incident-plan", Type: compliance.EvidencePolicy, ControlID: "CC7.4",
			Title: "Incident Response Procedures", Description: "Incident response procedures with severity classification and structured lifecycle management.",
			Source: "security/incident.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "IncidentManager", "package": "security"},
		},
		"CC9.1": {
			ID: "sec-vuln-scanning", Type: compliance.EvidencePolicy, ControlID: "CC9.1",
			Title: "Vulnerability Scanning", Description: "security.VulnScanner provides automated vulnerability scanning and finding tracking.",
			Source: "security/vuln.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "VulnScanner", "package": "security"},
		},
		"C1.1": {
			ID: "sec-secrets-mgmt", Type: compliance.EvidencePolicy, ControlID: "C1.1",
			Title: "Secrets Management", Description: "security.SecretStore provides secure storage and retrieval of confidential information.",
			Source: "security/secrets.go", CollectedBy: "audit-automation",
			Data: map[string]string{"component": "SecretStore", "package": "security"},
		},
	}
}

// PolicyEvidenceSource collects evidence from compliance policy documents.
type PolicyEvidenceSource struct {
	Policies []compliance.PolicyDocument
}

// Name returns the source name.
func (p *PolicyEvidenceSource) Name() string {
	return "policy-documents"
}

// SupportedControls returns the control IDs covered by the registered policies.
func (p *PolicyEvidenceSource) SupportedControls() []compliance.ControlID {
	seen := make(map[compliance.ControlID]bool)
	for _, pol := range p.Policies {
		for _, cid := range pol.ControlIDs {
			seen[cid] = true
		}
	}
	out := make([]compliance.ControlID, 0, len(seen))
	for cid := range seen {
		out = append(out, cid)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Collect returns evidence for a control based on matching policy documents.
func (p *PolicyEvidenceSource) Collect(controlID compliance.ControlID) ([]compliance.Evidence, error) {
	var results []compliance.Evidence
	for _, pol := range p.Policies {
		for _, cid := range pol.ControlIDs {
			if cid == controlID {
				results = append(results, compliance.Evidence{
					ID:          "pol-" + string(pol.Type) + "-" + string(controlID),
					Type:        compliance.EvidencePolicy,
					ControlID:   controlID,
					Title:       pol.Title,
					Description: "Policy document addressing " + string(controlID),
					Source:      "compliance/policy:" + string(pol.Type),
					CollectedBy: "audit-automation",
					Data: map[string]string{
						"policy_type":    string(pol.Type),
						"policy_version": pol.Version,
					},
				})
				break
			}
		}
	}
	return results, nil
}

// CollectAllEvidence creates an EvidenceCollector pre-configured with all
// audit evidence sources: security controls, policies, CI/CD, and code review.
func CollectAllEvidence(policies []compliance.PolicyDocument) (*compliance.EvidenceCollector, int, []error) {
	collector := compliance.NewEvidenceCollector(
		&SecurityControlSource{},
		&PolicyEvidenceSource{Policies: policies},
		&compliance.GitHubCISource{Owner: "zerfoo", Repo: "zerfoo"},
		&compliance.CodeReviewSource{Owner: "zerfoo", Repo: "zerfoo"},
	)
	count, errs := collector.CollectAll()
	return collector, count, errs
}
