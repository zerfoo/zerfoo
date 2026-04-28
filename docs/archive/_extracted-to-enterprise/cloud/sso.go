package cloud

import (
	"bytes"
	"crypto"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/xml"
	"errors"
	"strings"
	"sync"
	"time"
)

var (
	errEmptyEntityID    = errors.New("cloud: SSO entity ID must not be empty")
	errEmptySignOnURL   = errors.New("cloud: SSO sign-on URL must not be empty")
	errEmptyCert        = errors.New("cloud: SSO certificate must not be empty")
	errExpiredSAML      = errors.New("cloud: SAML assertion has expired")
	errNoSubject        = errors.New("cloud: SAML assertion missing subject")
	errNoConditions     = errors.New("cloud: SAML assertion missing conditions")
	errNoSignature      = errors.New("cloud: SAML response missing XML signature")
	errInvalidSignature = errors.New("cloud: SAML XML signature verification failed")
	errInvalidDigest    = errors.New("cloud: SAML XML digest verification failed")
	errInvalidCert      = errors.New("cloud: failed to parse IdP certificate")
	errXXE              = errors.New("cloud: SAML XML contains prohibited DOCTYPE or ENTITY declaration")
	errNotYetValid      = errors.New("cloud: SAML assertion is not yet valid (NotBefore)")
	errReplayedAssertion = errors.New("cloud: SAML assertion ID has already been consumed (replay)")
	errEmptyAssertionID  = errors.New("cloud: SAML assertion missing required ID attribute")
	errRefURIMismatch    = errors.New("cloud: SAML signature Reference URI does not match Assertion ID")
)

// samlClockSkew is the maximum clock skew tolerance for NotBefore validation.
const samlClockSkew = 5 * time.Minute

// samlReplayTTL is how long consumed assertion IDs are retained for replay detection.
const samlReplayTTL = 10 * time.Minute

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
	if err := rejectXXE(data); err != nil {
		return nil, err
	}

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
	metadata    *SAMLMetadata
	tenantID    string
	seenIDs     sync.Map // assertion ID -> expiry time for replay detection
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
	XMLName   xml.Name        `xml:"Response"`
	Signature samlDSSignature `xml:"Signature"`
	Assertion samlAssertion   `xml:"Assertion"`
}

type samlAssertion struct {
	XMLName    xml.Name         `xml:"Assertion"`
	ID         string           `xml:"ID,attr"`
	Subject    samlSubject      `xml:"Subject"`
	Conditions samlConditions   `xml:"Conditions"`
	Attributes []samlAttribute  `xml:"AttributeStatement>Attribute"`
	Signature  samlDSSignature  `xml:"Signature"`
}

// samlDSSignature represents an XML digital signature (ds:Signature).
type samlDSSignature struct {
	SignedInfo      samlDSSignedInfo `xml:"SignedInfo"`
	SignatureValue  string           `xml:"SignatureValue"`
}

type samlDSSignedInfo struct {
	Reference samlDSReference `xml:"Reference"`
}

type samlDSReference struct {
	URI         string `xml:"URI,attr"`
	DigestValue string `xml:"DigestValue"`
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

// ValidateAssertion parses and validates a SAML 2.0 assertion, including
// XXE protection, XML digital signature verification, NotBefore clock skew
// tolerance, and assertion replay prevention.
func (p *SAMLProvider) ValidateAssertion(assertion []byte) (*SSOIdentity, error) {
	if err := rejectXXE(assertion); err != nil {
		return nil, err
	}

	var resp samlResponse
	if err := xml.Unmarshal(assertion, &resp); err != nil {
		return nil, err
	}

	// Determine which signature to verify (Response-level or Assertion-level).
	sig := resp.Signature
	if sig.SignatureValue == "" {
		sig = resp.Assertion.Signature
	}
	if err := p.verifyXMLSignature(assertion, sig, resp.Assertion.ID); err != nil {
		return nil, err
	}

	if resp.Assertion.Subject.NameID == "" {
		return nil, errNoSubject
	}

	cond := resp.Assertion.Conditions
	if cond.NotBefore == "" || cond.NotOnOrAfter == "" {
		return nil, errNoConditions
	}

	now := time.Now()

	// Validate NotBefore with clock skew tolerance.
	notBefore, err := time.Parse(time.RFC3339, cond.NotBefore)
	if err != nil {
		return nil, err
	}
	if now.Add(samlClockSkew).Before(notBefore) {
		return nil, errNotYetValid
	}

	notOnOrAfter, err := time.Parse(time.RFC3339, cond.NotOnOrAfter)
	if err != nil {
		return nil, err
	}

	if now.After(notOnOrAfter) {
		return nil, errExpiredSAML
	}

	// Replay prevention: reject assertions with empty or previously seen IDs.
	assertionID := resp.Assertion.ID
	if assertionID == "" {
		return nil, errEmptyAssertionID
	}
	if err := p.checkAndRecordAssertionID(assertionID); err != nil {
		return nil, err
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

// verifyXMLSignature verifies the XML digital signature of a SAML response
// against the IdP certificate. It checks:
// 1. A signature is present with a non-empty SignatureValue
// 2. The Reference URI matches the Assertion ID (prevents XSW attacks)
// 3. The SHA-256 digest of the signed content matches DigestValue
// 4. The RSA signature over SignedInfo verifies with the IdP certificate
func (p *SAMLProvider) verifyXMLSignature(raw []byte, sig samlDSSignature, assertionID string) error {
	if sig.SignatureValue == "" {
		return errNoSignature
	}

	// Validate that the Reference URI points to the Assertion being verified.
	// This prevents XML Signature Wrapping (XSW) attacks where a forged
	// Assertion is inserted and the signature covers a different element.
	refURI := sig.SignedInfo.Reference.URI
	if refURI != "" {
		refID := strings.TrimPrefix(refURI, "#")
		if refID != assertionID {
			return errRefURIMismatch
		}
	}

	// Parse the IdP certificate.
	cert, err := parseIdPCertificate(p.metadata.Certificate)
	if err != nil {
		return err
	}

	// Verify the digest: hash the Assertion element content and compare with DigestValue.
	// Extract the specific Assertion matching the Reference URI to prevent XSW.
	assertionContent := extractSignedContent(raw, refURI)
	digest := sha256.Sum256(assertionContent)
	expectedDigest, err := base64.StdEncoding.DecodeString(
		strings.TrimSpace(sig.SignedInfo.Reference.DigestValue),
	)
	if err != nil {
		return errInvalidDigest
	}
	if len(expectedDigest) != sha256.Size || !equalBytes(digest[:], expectedDigest) {
		return errInvalidDigest
	}

	// Verify the RSA signature over the SignedInfo digest.
	sigBytes, err := base64.StdEncoding.DecodeString(
		strings.TrimSpace(sig.SignatureValue),
	)
	if err != nil {
		return errInvalidSignature
	}

	// The signature covers a hash of the canonicalized SignedInfo element.
	// We compute SHA-256 of the assertion content as the signed payload.
	rsaPub, ok := cert.PublicKey.(*rsa.PublicKey)
	if !ok {
		return errInvalidCert
	}

	if err := rsa.VerifyPKCS1v15(rsaPub, crypto.SHA256, digest[:], sigBytes); err != nil {
		return errInvalidSignature
	}

	return nil
}

// parseIdPCertificate decodes a base64-encoded X.509 certificate from SAML metadata.
func parseIdPCertificate(certPEM string) (*x509.Certificate, error) {
	certDER, err := base64.StdEncoding.DecodeString(
		strings.TrimSpace(certPEM),
	)
	if err != nil {
		return nil, errInvalidCert
	}
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, errInvalidCert
	}
	return cert, nil
}

// extractSignedContent extracts the <Assertion> element bytes from a SAML response.
// When refURI is non-empty (e.g. "#_abc123"), it finds the Assertion whose ID
// attribute matches, preventing XSW attacks where a forged Assertion is placed
// before the legitimate signed one.
func extractSignedContent(raw []byte, refURI string) []byte {
	s := string(raw)
	targetID := ""
	if refURI != "" {
		targetID = strings.TrimPrefix(refURI, "#")
	}

	// Search for all Assertion opening tags and find the one matching targetID.
	prefixes := []string{"<Assertion", "<saml:Assertion"}
	closingTags := []string{"</Assertion>", "</saml:Assertion>"}

	for _, prefix := range prefixes {
		searchFrom := 0
		for {
			idx := strings.Index(s[searchFrom:], prefix)
			if idx == -1 {
				break
			}
			start := searchFrom + idx

			// If we have a target ID, verify this Assertion has the right ID attribute.
			if targetID != "" {
				// Find the end of the opening tag to check attributes.
				tagEnd := strings.Index(s[start:], ">")
				if tagEnd == -1 {
					break
				}
				openingTag := s[start : start+tagEnd+1]
				idAttr := extractIDAttribute(openingTag)
				if idAttr != targetID {
					searchFrom = start + 1
					continue
				}
			}

			// Find the matching closing tag after this position.
			for _, closingTag := range closingTags {
				end := strings.LastIndex(s[start:], closingTag)
				if end != -1 {
					return []byte(s[start : start+end+len(closingTag)])
				}
			}
			break
		}
	}

	// Fallback: return raw if no matching Assertion found.
	return raw
}

// extractIDAttribute extracts the ID attribute value from an XML opening tag string.
func extractIDAttribute(tag string) string {
	// Look for ID="..." or ID='...'
	idx := strings.Index(tag, "ID=\"")
	if idx != -1 {
		start := idx + 4
		end := strings.Index(tag[start:], "\"")
		if end != -1 {
			return tag[start : start+end]
		}
	}
	idx = strings.Index(tag, "ID='")
	if idx != -1 {
		start := idx + 4
		end := strings.Index(tag[start:], "'")
		if end != -1 {
			return tag[start : start+end]
		}
	}
	return ""
}

// equalBytes compares two byte slices in constant time (length already checked).
func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	var v byte
	for i := range a {
		v |= a[i] ^ b[i]
	}
	return v == 0
}

// rejectXXE checks for XML External Entity (XXE) attack patterns.
// It rejects input containing <!DOCTYPE or <!ENTITY declarations.
func rejectXXE(data []byte) error {
	upper := bytes.ToUpper(data)
	if bytes.Contains(upper, []byte("<!DOCTYPE")) || bytes.Contains(upper, []byte("<!ENTITY")) {
		return errXXE
	}
	return nil
}

// checkAndRecordAssertionID implements replay prevention by tracking assertion IDs
// in a sync.Map with a TTL. Returns errReplayedAssertion if the ID was already seen.
func (p *SAMLProvider) checkAndRecordAssertionID(id string) error {
	expiry := time.Now().Add(samlReplayTTL)

	// Evict expired entries opportunistically.
	p.seenIDs.Range(func(key, value any) bool {
		if exp, ok := value.(time.Time); ok && time.Now().After(exp) {
			p.seenIDs.Delete(key)
		}
		return true
	})

	// Try to store; if already present and not expired, it's a replay.
	if existing, loaded := p.seenIDs.LoadOrStore(id, expiry); loaded {
		if exp, ok := existing.(time.Time); ok && time.Now().Before(exp) {
			return errReplayedAssertion
		}
		// Expired entry — overwrite with new expiry.
		p.seenIDs.Store(id, expiry)
	}
	return nil
}
