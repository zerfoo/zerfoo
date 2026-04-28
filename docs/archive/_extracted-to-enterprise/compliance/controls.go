// Package compliance provides SOC 2 compliance automation tooling including
// Trust Services Criteria control mapping, evidence collection, policy
// document generation, and control status tracking.
package compliance

import (
	"fmt"
	"time"
)

// Category represents a SOC 2 Trust Services Category.
type Category string

const (
	CategorySecurity       Category = "Security"
	CategoryAvailability   Category = "Availability"
	CategoryConfidentiality Category = "Confidentiality"
	CategoryProcessingIntegrity Category = "Processing Integrity"
	CategoryPrivacy        Category = "Privacy"
)

// ControlID uniquely identifies a SOC 2 control (e.g., "CC1.1", "A1.2").
type ControlID string

// Control represents a single SOC 2 Trust Services Criteria control.
type Control struct {
	ID          ControlID
	Category    Category
	Title       string
	Description string
	Criteria    string // Trust Services Criteria reference (e.g., "CC1.1")
}

// ControlStatus represents the compliance status of a control.
type ControlStatus string

const (
	StatusNotAssessed ControlStatus = "not_assessed"
	StatusCompliant   ControlStatus = "compliant"
	StatusPartial     ControlStatus = "partially_compliant"
	StatusNonCompliant ControlStatus = "non_compliant"
	StatusNotApplicable ControlStatus = "not_applicable"
)

// ControlAssessment records the assessment of a control at a point in time.
type ControlAssessment struct {
	ControlID   ControlID
	Status      ControlStatus
	AssessedAt  time.Time
	AssessedBy  string
	EvidenceIDs []string
	Notes       string
}

// ControlMapping maps SOC 2 Trust Services Criteria to implementation evidence.
type ControlMapping struct {
	controls    map[ControlID]Control
	assessments map[ControlID]ControlAssessment
}

// NewControlMapping returns a ControlMapping pre-populated with the standard
// SOC 2 Type II Trust Services Criteria controls.
func NewControlMapping() *ControlMapping {
	cm := &ControlMapping{
		controls:    make(map[ControlID]Control),
		assessments: make(map[ControlID]ControlAssessment),
	}
	for _, c := range defaultControls() {
		cm.controls[c.ID] = c
	}
	return cm
}

// Control returns the control definition for the given ID.
func (cm *ControlMapping) Control(id ControlID) (Control, bool) {
	c, ok := cm.controls[id]
	return c, ok
}

// Controls returns all registered controls.
func (cm *ControlMapping) Controls() []Control {
	out := make([]Control, 0, len(cm.controls))
	for _, c := range cm.controls {
		out = append(out, c)
	}
	return out
}

// ControlsByCategory returns controls filtered by category.
func (cm *ControlMapping) ControlsByCategory(cat Category) []Control {
	var out []Control
	for _, c := range cm.controls {
		if c.Category == cat {
			out = append(out, c)
		}
	}
	return out
}

// Assess records an assessment for a control.
func (cm *ControlMapping) Assess(a ControlAssessment) error {
	if _, ok := cm.controls[a.ControlID]; !ok {
		return fmt.Errorf("compliance: unknown control %q", a.ControlID)
	}
	cm.assessments[a.ControlID] = a
	return nil
}

// Assessment returns the most recent assessment for a control.
func (cm *ControlMapping) Assessment(id ControlID) (ControlAssessment, bool) {
	a, ok := cm.assessments[id]
	return a, ok
}

// defaultControls returns the standard SOC 2 Trust Services Criteria controls.
func defaultControls() []Control {
	return []Control{
		// CC1: Control Environment
		{ID: "CC1.1", Category: CategorySecurity, Title: "COSO Principle 1", Criteria: "CC1.1", Description: "The entity demonstrates a commitment to integrity and ethical values."},
		{ID: "CC1.2", Category: CategorySecurity, Title: "COSO Principle 2", Criteria: "CC1.2", Description: "The board of directors demonstrates independence from management and exercises oversight."},
		{ID: "CC1.3", Category: CategorySecurity, Title: "COSO Principle 3", Criteria: "CC1.3", Description: "Management establishes structures, reporting lines, and authorities."},
		{ID: "CC1.4", Category: CategorySecurity, Title: "COSO Principle 4", Criteria: "CC1.4", Description: "The entity demonstrates a commitment to attract, develop, and retain competent individuals."},
		{ID: "CC1.5", Category: CategorySecurity, Title: "COSO Principle 5", Criteria: "CC1.5", Description: "The entity holds individuals accountable for their internal control responsibilities."},

		// CC2: Communication and Information
		{ID: "CC2.1", Category: CategorySecurity, Title: "COSO Principle 13", Criteria: "CC2.1", Description: "The entity obtains or generates and uses relevant, quality information."},
		{ID: "CC2.2", Category: CategorySecurity, Title: "COSO Principle 14", Criteria: "CC2.2", Description: "The entity internally communicates information necessary to support the functioning of internal control."},
		{ID: "CC2.3", Category: CategorySecurity, Title: "COSO Principle 15", Criteria: "CC2.3", Description: "The entity communicates with external parties regarding matters affecting the functioning of internal control."},

		// CC3: Risk Assessment
		{ID: "CC3.1", Category: CategorySecurity, Title: "COSO Principle 6", Criteria: "CC3.1", Description: "The entity specifies objectives with sufficient clarity."},
		{ID: "CC3.2", Category: CategorySecurity, Title: "COSO Principle 7", Criteria: "CC3.2", Description: "The entity identifies risks to the achievement of its objectives."},
		{ID: "CC3.3", Category: CategorySecurity, Title: "COSO Principle 8", Criteria: "CC3.3", Description: "The entity considers the potential for fraud."},
		{ID: "CC3.4", Category: CategorySecurity, Title: "COSO Principle 9", Criteria: "CC3.4", Description: "The entity identifies and assesses changes that could significantly impact internal control."},

		// CC4: Monitoring Activities
		{ID: "CC4.1", Category: CategorySecurity, Title: "COSO Principle 16", Criteria: "CC4.1", Description: "The entity selects, develops, and performs ongoing and/or separate evaluations."},
		{ID: "CC4.2", Category: CategorySecurity, Title: "COSO Principle 17", Criteria: "CC4.2", Description: "The entity evaluates and communicates internal control deficiencies."},

		// CC5: Control Activities
		{ID: "CC5.1", Category: CategorySecurity, Title: "COSO Principle 10", Criteria: "CC5.1", Description: "The entity selects and develops control activities that mitigate risks."},
		{ID: "CC5.2", Category: CategorySecurity, Title: "COSO Principle 11", Criteria: "CC5.2", Description: "The entity selects and develops general control activities over technology."},
		{ID: "CC5.3", Category: CategorySecurity, Title: "COSO Principle 12", Criteria: "CC5.3", Description: "The entity deploys control activities through policies."},

		// CC6: Logical and Physical Access Controls
		{ID: "CC6.1", Category: CategorySecurity, Title: "Logical Access Security", Criteria: "CC6.1", Description: "The entity implements logical access security software, infrastructure, and architectures."},
		{ID: "CC6.2", Category: CategorySecurity, Title: "User Access Management", Criteria: "CC6.2", Description: "Prior to issuing system credentials, the entity registers and authorizes new users."},
		{ID: "CC6.3", Category: CategorySecurity, Title: "Access Removal", Criteria: "CC6.3", Description: "The entity authorizes, modifies, or removes access to data and assets."},
		{ID: "CC6.6", Category: CategorySecurity, Title: "External Threats", Criteria: "CC6.6", Description: "The entity implements controls to prevent or detect and act upon the introduction of unauthorized or malicious software."},
		{ID: "CC6.7", Category: CategorySecurity, Title: "Transmission Security", Criteria: "CC6.7", Description: "The entity restricts the transmission of data to authorized channels."},
		{ID: "CC6.8", Category: CategorySecurity, Title: "Unauthorized Software", Criteria: "CC6.8", Description: "The entity implements controls to prevent or detect unauthorized changes to software."},

		// CC7: System Operations
		{ID: "CC7.1", Category: CategorySecurity, Title: "Infrastructure Monitoring", Criteria: "CC7.1", Description: "To meet its objectives, the entity uses detection and monitoring procedures."},
		{ID: "CC7.2", Category: CategorySecurity, Title: "Anomaly Detection", Criteria: "CC7.2", Description: "The entity monitors system components for anomalies indicative of malicious acts."},
		{ID: "CC7.3", Category: CategorySecurity, Title: "Security Incident Response", Criteria: "CC7.3", Description: "The entity evaluates security events to determine whether they constitute security incidents."},
		{ID: "CC7.4", Category: CategorySecurity, Title: "Incident Response Plan", Criteria: "CC7.4", Description: "The entity responds to identified security incidents."},

		// CC8: Change Management
		{ID: "CC8.1", Category: CategorySecurity, Title: "Change Management Process", Criteria: "CC8.1", Description: "The entity authorizes, designs, develops, configures, documents, tests, approves, and implements changes."},

		// CC9: Risk Mitigation
		{ID: "CC9.1", Category: CategorySecurity, Title: "Risk Mitigation", Criteria: "CC9.1", Description: "The entity identifies, selects, and develops risk mitigation activities."},
		{ID: "CC9.2", Category: CategorySecurity, Title: "Vendor Risk Management", Criteria: "CC9.2", Description: "The entity assesses and manages risks associated with vendors and business partners."},

		// A1: Availability
		{ID: "A1.1", Category: CategoryAvailability, Title: "Capacity Planning", Criteria: "A1.1", Description: "The entity maintains, monitors, and evaluates current processing capacity and use."},
		{ID: "A1.2", Category: CategoryAvailability, Title: "Recovery Procedures", Criteria: "A1.2", Description: "The entity authorizes, designs, develops, implements, operates, and maintains environmental protections and recovery infrastructure."},
		{ID: "A1.3", Category: CategoryAvailability, Title: "Recovery Testing", Criteria: "A1.3", Description: "The entity tests recovery plan procedures supporting system recovery."},

		// C1: Confidentiality
		{ID: "C1.1", Category: CategoryConfidentiality, Title: "Confidential Information Identification", Criteria: "C1.1", Description: "The entity identifies and maintains confidential information."},
		{ID: "C1.2", Category: CategoryConfidentiality, Title: "Confidential Information Disposal", Criteria: "C1.2", Description: "The entity disposes of confidential information to meet objectives."},

		// PI1: Processing Integrity
		{ID: "PI1.1", Category: CategoryProcessingIntegrity, Title: "Processing Accuracy", Criteria: "PI1.1", Description: "The entity implements policies and procedures over system processing to ensure accuracy and completeness."},
		{ID: "PI1.2", Category: CategoryProcessingIntegrity, Title: "Input Validation", Criteria: "PI1.2", Description: "The entity implements policies and procedures over system inputs."},
		{ID: "PI1.3", Category: CategoryProcessingIntegrity, Title: "Output Review", Criteria: "PI1.3", Description: "The entity implements policies and procedures over system outputs."},

		// P1: Privacy
		{ID: "P1.1", Category: CategoryPrivacy, Title: "Privacy Notice", Criteria: "P1.1", Description: "The entity provides notice to data subjects about its privacy practices."},
		{ID: "P1.2", Category: CategoryPrivacy, Title: "Choice and Consent", Criteria: "P1.2", Description: "The entity communicates choices available regarding the collection and use of personal information."},
	}
}
