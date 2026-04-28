package compliance

import (
	"fmt"
	"strings"
	"time"
)

// PolicyType classifies the kind of policy document.
type PolicyType string

const (
	PolicyAccessControl      PolicyType = "access_control"
	PolicyChangeManagement   PolicyType = "change_management"
	PolicyIncidentResponse   PolicyType = "incident_response"
	PolicyDataClassification PolicyType = "data_classification"
	PolicyRiskAssessment     PolicyType = "risk_assessment"
	PolicyVendorManagement   PolicyType = "vendor_management"
)

// PolicyDocument represents a compliance policy document.
type PolicyDocument struct {
	Type        PolicyType
	Title       string
	Version     string
	Owner       string
	ApprovedBy  string
	EffectiveAt time.Time
	ReviewBy    time.Time
	Sections    []PolicySection
	ControlIDs  []ControlID // Controls this policy satisfies
}

// PolicySection represents a section within a policy document.
type PolicySection struct {
	Heading string
	Body    string
}

// Render formats the policy document as a human-readable text document.
func (pd *PolicyDocument) Render() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("# %s\n\n", pd.Title))
	b.WriteString(fmt.Sprintf("**Version:** %s\n", pd.Version))
	b.WriteString(fmt.Sprintf("**Owner:** %s\n", pd.Owner))
	b.WriteString(fmt.Sprintf("**Approved By:** %s\n", pd.ApprovedBy))
	b.WriteString(fmt.Sprintf("**Effective Date:** %s\n", pd.EffectiveAt.Format("2006-01-02")))
	b.WriteString(fmt.Sprintf("**Next Review:** %s\n\n", pd.ReviewBy.Format("2006-01-02")))
	for _, s := range pd.Sections {
		b.WriteString(fmt.Sprintf("## %s\n\n%s\n\n", s.Heading, s.Body))
	}
	return b.String()
}

// PolicyTemplate defines a function that generates a PolicyDocument.
type PolicyTemplate func(org, owner, approver string, effective time.Time) PolicyDocument

// PolicyTemplates returns all available policy templates keyed by PolicyType.
func PolicyTemplates() map[PolicyType]PolicyTemplate {
	return map[PolicyType]PolicyTemplate{
		PolicyAccessControl:      accessControlTemplate,
		PolicyChangeManagement:   changeManagementTemplate,
		PolicyIncidentResponse:   incidentResponseTemplate,
		PolicyDataClassification: dataClassificationTemplate,
		PolicyRiskAssessment:     riskAssessmentTemplate,
		PolicyVendorManagement:   vendorManagementTemplate,
	}
}

// GeneratePolicy creates a policy document from a template.
func GeneratePolicy(ptype PolicyType, org, owner, approver string, effective time.Time) (PolicyDocument, error) {
	templates := PolicyTemplates()
	tmpl, ok := templates[ptype]
	if !ok {
		return PolicyDocument{}, fmt.Errorf("compliance: unknown policy type %q", ptype)
	}
	return tmpl(org, owner, approver, effective), nil
}

func accessControlTemplate(org, owner, approver string, effective time.Time) PolicyDocument {
	return PolicyDocument{
		Type:        PolicyAccessControl,
		Title:       fmt.Sprintf("%s Access Control Policy", org),
		Version:     "1.0",
		Owner:       owner,
		ApprovedBy:  approver,
		EffectiveAt: effective,
		ReviewBy:    effective.AddDate(1, 0, 0),
		ControlIDs:  []ControlID{"CC6.1", "CC6.2", "CC6.3"},
		Sections: []PolicySection{
			{Heading: "Purpose", Body: "This policy establishes requirements for controlling access to organizational information systems and data to protect against unauthorized access."},
			{Heading: "Scope", Body: "This policy applies to all employees, contractors, and third parties with access to organizational systems."},
			{Heading: "User Access Management", Body: "All access must be authorized prior to provisioning. Access rights are granted based on the principle of least privilege and need-to-know basis. Access reviews are conducted quarterly."},
			{Heading: "Authentication Requirements", Body: "Multi-factor authentication is required for all remote access and privileged accounts. Passwords must meet minimum complexity requirements and be rotated periodically."},
			{Heading: "Access Revocation", Body: "Access is revoked within 24 hours of employment termination. Access changes due to role changes are processed within 48 hours."},
			{Heading: "Monitoring and Review", Body: "Access logs are reviewed monthly. Privileged access is monitored in real-time. Anomalous access patterns trigger automated alerts."},
		},
	}
}

func changeManagementTemplate(org, owner, approver string, effective time.Time) PolicyDocument {
	return PolicyDocument{
		Type:        PolicyChangeManagement,
		Title:       fmt.Sprintf("%s Change Management Policy", org),
		Version:     "1.0",
		Owner:       owner,
		ApprovedBy:  approver,
		EffectiveAt: effective,
		ReviewBy:    effective.AddDate(1, 0, 0),
		ControlIDs:  []ControlID{"CC8.1", "CC6.8"},
		Sections: []PolicySection{
			{Heading: "Purpose", Body: "This policy defines the process for managing changes to information systems to minimize disruption and ensure changes are authorized, tested, and documented."},
			{Heading: "Scope", Body: "This policy applies to all changes to production systems, including code, infrastructure, and configuration changes."},
			{Heading: "Change Process", Body: "All changes require a pull request with peer review. Changes must pass automated CI/CD testing. Emergency changes require post-implementation review within 48 hours."},
			{Heading: "Testing Requirements", Body: "All changes must pass unit tests, integration tests, and security scans before deployment. Rollback procedures must be documented for each change."},
			{Heading: "Release Management", Body: "Releases follow semantic versioning with automated release-please tooling. Production deployments require approval from a code owner."},
		},
	}
}

func incidentResponseTemplate(org, owner, approver string, effective time.Time) PolicyDocument {
	return PolicyDocument{
		Type:        PolicyIncidentResponse,
		Title:       fmt.Sprintf("%s Incident Response Policy", org),
		Version:     "1.0",
		Owner:       owner,
		ApprovedBy:  approver,
		EffectiveAt: effective,
		ReviewBy:    effective.AddDate(1, 0, 0),
		ControlIDs:  []ControlID{"CC7.3", "CC7.4"},
		Sections: []PolicySection{
			{Heading: "Purpose", Body: "This policy establishes procedures for identifying, responding to, and recovering from security incidents."},
			{Heading: "Scope", Body: "This policy applies to all security events affecting organizational systems and data."},
			{Heading: "Incident Classification", Body: "Incidents are classified as Critical (P1), High (P2), Medium (P3), or Low (P4) based on impact and urgency. Classification determines response times and escalation paths."},
			{Heading: "Response Procedures", Body: "P1 incidents require immediate response with 15-minute acknowledgment SLA. All incidents are tracked from detection through resolution. Post-incident reviews are required for P1 and P2 incidents."},
			{Heading: "Communication", Body: "Affected stakeholders are notified within 1 hour for P1 incidents. Status updates are provided at regular intervals. Post-incident reports are published within 5 business days."},
			{Heading: "Recovery and Lessons Learned", Body: "Recovery procedures follow documented runbooks. Post-incident reviews identify root causes and preventive measures. Action items from reviews are tracked to completion."},
		},
	}
}

func dataClassificationTemplate(org, owner, approver string, effective time.Time) PolicyDocument {
	return PolicyDocument{
		Type:        PolicyDataClassification,
		Title:       fmt.Sprintf("%s Data Classification Policy", org),
		Version:     "1.0",
		Owner:       owner,
		ApprovedBy:  approver,
		EffectiveAt: effective,
		ReviewBy:    effective.AddDate(1, 0, 0),
		ControlIDs:  []ControlID{"C1.1", "C1.2"},
		Sections: []PolicySection{
			{Heading: "Purpose", Body: "This policy defines data classification levels and handling requirements to protect confidential information."},
			{Heading: "Classification Levels", Body: "Data is classified as Public, Internal, Confidential, or Restricted. Classification is based on the sensitivity and impact of unauthorized disclosure."},
			{Heading: "Handling Requirements", Body: "Confidential and Restricted data must be encrypted at rest and in transit. Access to Restricted data requires explicit authorization and is logged."},
			{Heading: "Disposal", Body: "Data is disposed of according to its classification level. Confidential and Restricted data requires secure deletion with verification."},
		},
	}
}

func riskAssessmentTemplate(org, owner, approver string, effective time.Time) PolicyDocument {
	return PolicyDocument{
		Type:        PolicyRiskAssessment,
		Title:       fmt.Sprintf("%s Risk Assessment Policy", org),
		Version:     "1.0",
		Owner:       owner,
		ApprovedBy:  approver,
		EffectiveAt: effective,
		ReviewBy:    effective.AddDate(1, 0, 0),
		ControlIDs:  []ControlID{"CC3.1", "CC3.2", "CC3.3", "CC3.4"},
		Sections: []PolicySection{
			{Heading: "Purpose", Body: "This policy establishes the framework for identifying, assessing, and managing risks to the organization."},
			{Heading: "Risk Assessment Process", Body: "Risk assessments are conducted annually and when significant changes occur. Risks are evaluated for likelihood and impact using a standardized scoring matrix."},
			{Heading: "Risk Treatment", Body: "Identified risks are treated through acceptance, mitigation, transfer, or avoidance. Risk treatment plans are documented and tracked."},
			{Heading: "Fraud Risk", Body: "The organization considers the potential for fraud in its risk assessments, including incentives, opportunities, and rationalizations."},
		},
	}
}

func vendorManagementTemplate(org, owner, approver string, effective time.Time) PolicyDocument {
	return PolicyDocument{
		Type:        PolicyVendorManagement,
		Title:       fmt.Sprintf("%s Vendor Management Policy", org),
		Version:     "1.0",
		Owner:       owner,
		ApprovedBy:  approver,
		EffectiveAt: effective,
		ReviewBy:    effective.AddDate(1, 0, 0),
		ControlIDs:  []ControlID{"CC9.2"},
		Sections: []PolicySection{
			{Heading: "Purpose", Body: "This policy establishes requirements for assessing and managing risks associated with third-party vendors."},
			{Heading: "Vendor Assessment", Body: "All vendors with access to sensitive data or critical systems undergo security assessment before engagement. Assessments evaluate the vendor's security posture, compliance certifications, and data handling practices."},
			{Heading: "Ongoing Monitoring", Body: "Vendor compliance is reviewed annually. Vendors must notify the organization of material security incidents within 24 hours. Service level agreements include security requirements."},
		},
	}
}
