package cloud

import (
	"encoding/xml"
	"errors"
	"time"
)

var (
	errEmptyEntityID  = errors.New("cloud: SSO entity ID must not be empty")
	errEmptySignOnURL = errors.New("cloud: SSO sign-on URL must not be empty")
	errEmptyCert      = errors.New("cloud: SSO certificate must not be empty")
	errExpiredSAML    = errors.New("cloud: SAML assertion has expired")
	errNoSubject      = errors.New("cloud: SAML assertion missing subject")
	errNoConditions   = errors.New("cloud: SAML assertion missing conditions")
)

// SSOProvider defines the interface for SSO authentication.
// Implementations handle protocol-specific details (SAML 2.0, OIDC, etc.).
type SSOProvider interface {
	// EntityID returns the identity provider's entity ID.
	EntityID() string

	// ValidateAssertion validates an assertion and returns the authenticated identity.
	ValidateAssertion(assertion []byte) (*SSOIdentity, error)
}

// SSOIdentity represents an authenticated user from an SSO provider.
type SSOIdentity struct {
	Subject    string            `json:"subject"`
	TenantID   string            `json:"tenant_id"`
	Email      string            `json:"email,omitempty"`
	Attributes map[string]string `json:"attributes,omitempty"`
	ExpiresAt  time.Time         `json:"expires_at"`
}

// SAMLMetadata holds identity provider configuration parsed from SAML 2.0 metadata XML.
type SAMLMetadata struct {
	EntityID         string `json:"entity_id"`
	SignOnURL        string `json:"sign_on_url"`
	Certificate      string `json:"certificate"`
	NameIDFormat     string `json:"name_id_format,omitempty"`
	WantAuthnSigned  bool   `json:"want_authn_signed"`
}

// samlEntityDescriptor is the XML structure for SAML 2.0 EntityDescriptor.
type samlEntityDescriptor struct {
	XMLName  xml.Name           `xml:"EntityDescriptor"`
	EntityID string             `xml:"entityID,attr"`
	IDPSSODescriptor samlIDPSSODescriptor `xml:"IDPSSODescriptor"`
}

type samlIDPSSODescriptor struct {
	WantAuthnRequestsSigned bool                    `xml:"WantAuthnRequestsSigned,attr"`
	KeyDescriptors          []samlKeyDescriptor     `xml:"KeyDescriptor"`
	SingleSignOnServices    []samlSingleSignOnService `xml:"SingleSignOnService"`
	NameIDFormats           []samlNameIDFormat      `xml:"NameIDFormat"`
}

type samlKeyDescriptor struct {
	Use     string      `xml:"use,attr"`
	KeyInfo samlKeyInfo `xml:"KeyInfo"`
}

type samlKeyInfo struct {
	X509Data samlX509Data `xml:"X509Data"`
}

type samlX509Data struct {
	Certificate string `xml:"X509Certificate"`
}

type samlSingleSignOnService struct {
	Binding  string `xml:"Binding,attr"`
	Location string `xml:"Location,attr"`
}

type samlNameIDFormat struct {
	Value string `xml:",chardata"`
}

// ParseSAMLMetadata parses SAML 2.0 IdP metadata XML into a SAMLMetadata struct.
func ParseSAMLMetadata(data []byte) (*SAMLMetadata, error) {
	var desc samlEntityDescriptor
	if err := xml.Unmarshal(data, &desc); err != nil {
		return nil, err
	}

	meta := &SAMLMetadata{
		EntityID:        desc.EntityID,
		WantAuthnSigned: desc.IDPSSODescriptor.WantAuthnRequestsSigned,
	}

	// Extract signing certificate.
	for _, kd := range desc.IDPSSODescriptor.KeyDescriptors {
		if kd.Use == "signing" || kd.Use == "" {
			meta.Certificate = kd.KeyInfo.X509Data.Certificate
			break
		}
	}

	// Extract SSO URL (prefer HTTP-POST binding).
	for _, sso := range desc.IDPSSODescriptor.SingleSignOnServices {
		meta.SignOnURL = sso.Location
		if sso.Binding == "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" {
			break
		}
	}

	// Extract NameID format.
	if len(desc.IDPSSODescriptor.NameIDFormats) > 0 {
		meta.NameIDFormat = desc.IDPSSODescriptor.NameIDFormats[0].Value
	}

	if meta.EntityID == "" {
		return nil, errEmptyEntityID
	}
	if meta.SignOnURL == "" {
		return nil, errEmptySignOnURL
	}
	if meta.Certificate == "" {
		return nil, errEmptyCert
	}

	return meta, nil
}

// SAMLProvider implements SSOProvider for SAML 2.0.
type SAMLProvider struct {
	metadata *SAMLMetadata
	tenantID string
}

// NewSAMLProvider creates a SAML 2.0 SSO provider from parsed metadata,
// bound to a specific tenant.
func NewSAMLProvider(metadata *SAMLMetadata, tenantID string) *SAMLProvider {
	return &SAMLProvider{
		metadata: metadata,
		tenantID: tenantID,
	}
}

// EntityID returns the identity provider's entity ID.
func (p *SAMLProvider) EntityID() string {
	return p.metadata.EntityID
}

// samlResponse is a minimal SAML 2.0 Response structure for assertion parsing.
type samlResponse struct {
	XMLName   xml.Name       `xml:"Response"`
	Assertion samlAssertion  `xml:"Assertion"`
}

type samlAssertion struct {
	Subject    samlSubject    `xml:"Subject"`
	Conditions samlConditions `xml:"Conditions"`
	Attributes []samlAttribute `xml:"AttributeStatement>Attribute"`
}

type samlSubject struct {
	NameID string `xml:"NameID"`
}

type samlConditions struct {
	NotBefore    string `xml:"NotBefore,attr"`
	NotOnOrAfter string `xml:"NotOnOrAfter,attr"`
}

type samlAttribute struct {
	Name   string `xml:"Name,attr"`
	Values []samlAttributeValue `xml:"AttributeValue"`
}

type samlAttributeValue struct {
	Value string `xml:",chardata"`
}

// ValidateAssertion parses and validates a SAML 2.0 assertion.
// In production, this would also verify the XML signature against the IdP certificate.
func (p *SAMLProvider) ValidateAssertion(assertion []byte) (*SSOIdentity, error) {
	var resp samlResponse
	if err := xml.Unmarshal(assertion, &resp); err != nil {
		return nil, err
	}

	if resp.Assertion.Subject.NameID == "" {
		return nil, errNoSubject
	}

	cond := resp.Assertion.Conditions
	if cond.NotBefore == "" || cond.NotOnOrAfter == "" {
		return nil, errNoConditions
	}

	notOnOrAfter, err := time.Parse(time.RFC3339, cond.NotOnOrAfter)
	if err != nil {
		return nil, err
	}

	if time.Now().After(notOnOrAfter) {
		return nil, errExpiredSAML
	}

	identity := &SSOIdentity{
		Subject:    resp.Assertion.Subject.NameID,
		TenantID:   p.tenantID,
		ExpiresAt:  notOnOrAfter,
		Attributes: make(map[string]string),
	}

	for _, attr := range resp.Assertion.Attributes {
		if len(attr.Values) > 0 {
			identity.Attributes[attr.Name] = attr.Values[0].Value
			if attr.Name == "email" {
				identity.Email = attr.Values[0].Value
			}
		}
	}

	return identity, nil
}
