package cloud

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

var testAssertionCounter atomic.Int64

// uniqueAssertionID returns a unique assertion ID for testing.
func uniqueAssertionID() string {
	n := testAssertionCounter.Add(1)
	b := make([]byte, 8)
	// Use counter + random bytes for uniqueness.
	b[0] = byte(n >> 56)
	b[1] = byte(n >> 48)
	b[2] = byte(n >> 40)
	b[3] = byte(n >> 32)
	b[4] = byte(n >> 24)
	b[5] = byte(n >> 16)
	b[6] = byte(n >> 8)
	b[7] = byte(n)
	return "_" + hex.EncodeToString(b)
}

// testIdP holds a test identity provider's key pair and certificate.
type testIdP struct {
	key     *rsa.PrivateKey
	cert    *x509.Certificate
	certB64 string // base64-encoded DER certificate
}

// newTestIdP generates a self-signed test IdP certificate and key pair.
func newTestIdP(t *testing.T) *testIdP {
	t.Helper()

	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate RSA key: %v", err)
	}

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "test-idp"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create certificate: %v", err)
	}

	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		t.Fatalf("parse certificate: %v", err)
	}

	return &testIdP{
		key:     key,
		cert:    cert,
		certB64: base64.StdEncoding.EncodeToString(certDER),
	}
}

// signedSAMLResponse builds a SAML Response XML with a valid XML digital signature.
func (idp *testIdP) signedSAMLResponse(subject, notBefore, notOnOrAfter string, attrs map[string]string) string {
	// Build the Assertion element.
	var attrXML string
	for k, v := range attrs {
		attrXML += fmt.Sprintf(`      <Attribute Name="%s"><AttributeValue>%s</AttributeValue></Attribute>
`, k, v)
	}

	var attrStmt string
	if attrXML != "" {
		attrStmt = fmt.Sprintf(`    <AttributeStatement>
%s    </AttributeStatement>
`, attrXML)
	}

	assertionID := uniqueAssertionID()
	assertion := fmt.Sprintf(`<Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion" ID="%s">
    <Subject><NameID>%s</NameID></Subject>
    <Conditions NotBefore="%s" NotOnOrAfter="%s"/>
%s  </Assertion>`, assertionID, subject, notBefore, notOnOrAfter, attrStmt)

	// Compute digest of the Assertion element.
	digest := sha256.Sum256([]byte(assertion))
	digestB64 := base64.StdEncoding.EncodeToString(digest[:])

	// Sign the digest with the IdP key.
	sig, err := rsa.SignPKCS1v15(rand.Reader, idp.key, crypto.SHA256, digest[:])
	if err != nil {
		panic(fmt.Sprintf("sign: %v", err))
	}
	sigB64 := base64.StdEncoding.EncodeToString(sig)

	return fmt.Sprintf(`<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
    <SignedInfo>
      <Reference URI="#%s">
        <DigestValue>%s</DigestValue>
      </Reference>
    </SignedInfo>
    <SignatureValue>%s</SignatureValue>
  </Signature>
  %s
</Response>`, assertionID, digestB64, sigB64, assertion)
}

// signedSAMLResponseNoID builds a SAML Response XML where the Assertion has no ID attribute.
func (idp *testIdP) signedSAMLResponseNoID(subject, notBefore, notOnOrAfter string) string {
	assertion := fmt.Sprintf(`<Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion">
    <Subject><NameID>%s</NameID></Subject>
    <Conditions NotBefore="%s" NotOnOrAfter="%s"/>
  </Assertion>`, subject, notBefore, notOnOrAfter)

	digest := sha256.Sum256([]byte(assertion))
	digestB64 := base64.StdEncoding.EncodeToString(digest[:])

	sig, err := rsa.SignPKCS1v15(rand.Reader, idp.key, crypto.SHA256, digest[:])
	if err != nil {
		panic(fmt.Sprintf("sign: %v", err))
	}
	sigB64 := base64.StdEncoding.EncodeToString(sig)

	return fmt.Sprintf(`<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
    <SignedInfo>
      <Reference>
        <DigestValue>%s</DigestValue>
      </Reference>
    </SignedInfo>
    <SignatureValue>%s</SignatureValue>
  </Signature>
  %s
</Response>`, digestB64, sigB64, assertion)
}

// testSAMLMetadata returns valid SAML metadata XML using the test IdP certificate.
func (idp *testIdP) testSAMLMetadata() []byte {
	return []byte(fmt.Sprintf(`<EntityDescriptor entityID="https://idp.example.com/saml" xmlns="urn:oasis:names:tc:SAML:2.0:metadata">
  <IDPSSODescriptor WantAuthnRequestsSigned="true">
    <KeyDescriptor use="signing">
      <KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#">
        <X509Data>
          <X509Certificate>%s</X509Certificate>
        </X509Data>
      </KeyInfo>
    </KeyDescriptor>
    <SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="https://idp.example.com/sso"/>
    <NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</NameIDFormat>
  </IDPSSODescriptor>
</EntityDescriptor>`, idp.certB64))
}

func TestSAML_SignatureVerification(t *testing.T) {
	idp := newTestIdP(t)
	meta, err := ParseSAMLMetadata(idp.testSAMLMetadata())
	if err != nil {
		t.Fatalf("ParseSAMLMetadata: %v", err)
	}
	provider := NewSAMLProvider(meta, "tenant-sig")

	past := time.Now().Add(-time.Hour).Format(time.RFC3339)
	future := time.Now().Add(time.Hour).Format(time.RFC3339)

	t.Run("valid signature passes", func(t *testing.T) {
		xml := idp.signedSAMLResponse("user@example.com", past, future, map[string]string{
			"email": "user@example.com",
			"role":  "admin",
		})
		identity, err := provider.ValidateAssertion([]byte(xml))
		if err != nil {
			t.Fatalf("ValidateAssertion: %v", err)
		}
		if identity.Subject != "user@example.com" {
			t.Errorf("Subject = %q, want %q", identity.Subject, "user@example.com")
		}
		if identity.TenantID != "tenant-sig" {
			t.Errorf("TenantID = %q, want %q", identity.TenantID, "tenant-sig")
		}
		if identity.Email != "user@example.com" {
			t.Errorf("Email = %q, want %q", identity.Email, "user@example.com")
		}
		if identity.Attributes["role"] != "admin" {
			t.Errorf("role = %q, want %q", identity.Attributes["role"], "admin")
		}
	})

	t.Run("tampered assertion fails", func(t *testing.T) {
		xml := idp.signedSAMLResponse("user@example.com", past, future, nil)
		// Tamper with the subject after signing.
		tampered := strings.Replace(xml, "user@example.com", "attacker@evil.com", 1)
		_, err := provider.ValidateAssertion([]byte(tampered))
		if err == nil {
			t.Fatal("expected error for tampered assertion")
		}
		if err != errInvalidDigest && err != errInvalidSignature {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("missing signature fails", func(t *testing.T) {
		xml := fmt.Sprintf(`<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion">
    <Subject><NameID>user@example.com</NameID></Subject>
    <Conditions NotBefore="%s" NotOnOrAfter="%s"/>
  </Assertion>
</Response>`, past, future)
		_, err := provider.ValidateAssertion([]byte(xml))
		if err != errNoSignature {
			t.Errorf("got error %v, want errNoSignature", err)
		}
	})

	t.Run("wrong key fails", func(t *testing.T) {
		// Sign with a different key than what the metadata has.
		otherIdP := newTestIdP(t)
		xml := otherIdP.signedSAMLResponse("user@example.com", past, future, nil)
		_, err := provider.ValidateAssertion([]byte(xml))
		if err == nil {
			t.Fatal("expected error for wrong signing key")
		}
		if err != errInvalidSignature {
			t.Errorf("got error %v, want errInvalidSignature", err)
		}
	})

	t.Run("invalid base64 signature value", func(t *testing.T) {
		xml := idp.signedSAMLResponse("user@example.com", past, future, nil)
		// Corrupt the signature value.
		xml = strings.Replace(xml, "<SignatureValue>", "<SignatureValue>!!!invalid!!!", 1)
		_, err := provider.ValidateAssertion([]byte(xml))
		if err == nil {
			t.Fatal("expected error for invalid base64 signature")
		}
	})
}

func TestSAML_XXEProtection(t *testing.T) {
	t.Run("DOCTYPE rejected in assertion", func(t *testing.T) {
		xml := `<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion">
    <Subject><NameID>&xxe;</NameID></Subject>
    <Conditions NotBefore="2020-01-01T00:00:00Z" NotOnOrAfter="2030-01-01T00:00:00Z"/>
  </Assertion>
</Response>`
		idp := newTestIdP(t)
		meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
		provider := NewSAMLProvider(meta, "t1")
		_, err := provider.ValidateAssertion([]byte(xml))
		if err != errXXE {
			t.Errorf("got error %v, want errXXE", err)
		}
	})

	t.Run("ENTITY rejected in assertion", func(t *testing.T) {
		xml := `<?xml version="1.0"?>
<!ENTITY xxe "malicious">
<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion">
    <Subject><NameID>user@example.com</NameID></Subject>
  </Assertion>
</Response>`
		idp := newTestIdP(t)
		meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
		provider := NewSAMLProvider(meta, "t1")
		_, err := provider.ValidateAssertion([]byte(xml))
		if err != errXXE {
			t.Errorf("got error %v, want errXXE", err)
		}
	})

	t.Run("DOCTYPE rejected in metadata", func(t *testing.T) {
		xml := `<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<EntityDescriptor entityID="https://evil.com" xmlns="urn:oasis:names:tc:SAML:2.0:metadata">
  <IDPSSODescriptor>
    <KeyDescriptor use="signing"><KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#"><X509Data><X509Certificate>cert</X509Certificate></X509Data></KeyInfo></KeyDescriptor>
    <SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="https://evil.com/sso"/>
  </IDPSSODescriptor>
</EntityDescriptor>`
		_, err := ParseSAMLMetadata([]byte(xml))
		if err != errXXE {
			t.Errorf("got error %v, want errXXE", err)
		}
	})

	t.Run("case insensitive DOCTYPE detection", func(t *testing.T) {
		xml := `<?xml version="1.0"?>
<!doctype foo>
<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion">
    <Subject><NameID>user@example.com</NameID></Subject>
  </Assertion>
</Response>`
		idp := newTestIdP(t)
		meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
		provider := NewSAMLProvider(meta, "t1")
		_, err := provider.ValidateAssertion([]byte(xml))
		if err != errXXE {
			t.Errorf("got error %v, want errXXE", err)
		}
	})
}

func TestSAML_ReplayPrevention(t *testing.T) {
	idp := newTestIdP(t)
	meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
	provider := NewSAMLProvider(meta, "tenant-replay")

	past := time.Now().Add(-time.Hour).Format(time.RFC3339)
	future := time.Now().Add(time.Hour).Format(time.RFC3339)

	// Generate the assertion once and reuse it for both calls.
	assertionXML := idp.signedSAMLResponse("user@example.com", past, future, nil)

	t.Run("first assertion accepted", func(t *testing.T) {
		_, err := provider.ValidateAssertion([]byte(assertionXML))
		if err != nil {
			t.Fatalf("first assertion should succeed: %v", err)
		}
	})

	t.Run("replayed assertion ID rejected", func(t *testing.T) {
		_, err := provider.ValidateAssertion([]byte(assertionXML))
		if err != errReplayedAssertion {
			t.Errorf("got error %v, want errReplayedAssertion", err)
		}
	})

	t.Run("different assertion ID accepted", func(t *testing.T) {
		// A new assertion with a different ID should still succeed.
		newXML := idp.signedSAMLResponse("user@example.com", past, future, nil)
		_, err := provider.ValidateAssertion([]byte(newXML))
		if err != nil {
			t.Fatalf("different assertion ID should succeed: %v", err)
		}
	})
}

func TestSAML_NotBeforeValidation(t *testing.T) {
	idp := newTestIdP(t)
	meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
	provider := NewSAMLProvider(meta, "tenant-notbefore")

	future := time.Now().Add(time.Hour).Format(time.RFC3339)

	t.Run("assertion before NotBefore minus skew rejected", func(t *testing.T) {
		// NotBefore is 10 minutes in the future -- beyond the 5-minute skew tolerance.
		farFuture := time.Now().Add(10 * time.Minute).Format(time.RFC3339)
		xml := idp.signedSAMLResponse("user@example.com", farFuture, future, nil)
		_, err := provider.ValidateAssertion([]byte(xml))
		if err != errNotYetValid {
			t.Errorf("got error %v, want errNotYetValid", err)
		}
	})

	t.Run("assertion within clock skew accepted", func(t *testing.T) {
		// NotBefore is 3 minutes in the future -- within the 5-minute skew tolerance.
		nearFuture := time.Now().Add(3 * time.Minute).Format(time.RFC3339)
		xml := idp.signedSAMLResponse("user@example.com", nearFuture, future, nil)
		_, err := provider.ValidateAssertion([]byte(xml))
		if err != nil {
			t.Fatalf("assertion within skew should succeed: %v", err)
		}
	})

	t.Run("assertion with past NotBefore accepted", func(t *testing.T) {
		past := time.Now().Add(-time.Hour).Format(time.RFC3339)
		xml := idp.signedSAMLResponse("user@example.com", past, future, nil)
		_, err := provider.ValidateAssertion([]byte(xml))
		if err != nil {
			t.Fatalf("assertion with past NotBefore should succeed: %v", err)
		}
	})
}

func TestSAML_EmptyAssertionID(t *testing.T) {
	idp := newTestIdP(t)
	meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
	provider := NewSAMLProvider(meta, "tenant-empty-id")

	past := time.Now().Add(-time.Hour).Format(time.RFC3339)
	future := time.Now().Add(time.Hour).Format(time.RFC3339)

	xml := idp.signedSAMLResponseNoID("user@example.com", past, future)
	_, err := provider.ValidateAssertion([]byte(xml))
	if err != errEmptyAssertionID {
		t.Errorf("got error %v, want errEmptyAssertionID", err)
	}
}

func TestSAML_XSWProtection(t *testing.T) {
	idp := newTestIdP(t)
	meta, err := ParseSAMLMetadata(idp.testSAMLMetadata())
	if err != nil {
		t.Fatalf("ParseSAMLMetadata: %v", err)
	}

	past := time.Now().Add(-time.Hour).Format(time.RFC3339)
	future := time.Now().Add(time.Hour).Format(time.RFC3339)

	tests := []struct {
		name    string
		build   func() string
		wantErr error
	}{
		{
			name: "valid signature passes",
			build: func() string {
				return idp.signedSAMLResponse("user@example.com", past, future, map[string]string{
					"email": "user@example.com",
				})
			},
			wantErr: nil,
		},
		{
			name: "tampered assertion content fails",
			build: func() string {
				xml := idp.signedSAMLResponse("user@example.com", past, future, nil)
				return strings.Replace(xml, "user@example.com", "attacker@evil.com", 1)
			},
			wantErr: errInvalidDigest,
		},
		{
			name: "XSW attack with forged assertion after signed one",
			build: func() string {
				// Build a legitimately signed response.
				signedXML := idp.signedSAMLResponse("legit@example.com", past, future, nil)

				// Insert a forged Assertion (with a different ID) AFTER the signed one.
				// Go's xml.Unmarshal overwrites struct fields with the last matching
				// element, so the forged assertion would be parsed as resp.Assertion.
				// The Reference URI validation catches this because the signature
				// references the original assertion ID, not the forged one.
				forgedAssertion := fmt.Sprintf(`<Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion" ID="_forged_id">
    <Subject><NameID>attacker@evil.com</NameID></Subject>
    <Conditions NotBefore="%s" NotOnOrAfter="%s"/>
  </Assertion>`, past, future)

				// Insert the forged assertion before the closing </Response> tag.
				return strings.Replace(signedXML,
					"</Response>",
					forgedAssertion+"\n</Response>", 1)
			},
			wantErr: errRefURIMismatch,
		},
		{
			name: "mismatched Reference URI fails",
			build: func() string {
				// Build a valid assertion and sign it, but manually alter the
				// Reference URI to point to a different ID.
				assertionID := uniqueAssertionID()
				assertion := fmt.Sprintf(`<Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion" ID="%s">
    <Subject><NameID>user@example.com</NameID></Subject>
    <Conditions NotBefore="%s" NotOnOrAfter="%s"/>
  </Assertion>`, assertionID, past, future)

				digest := sha256.Sum256([]byte(assertion))
				digestB64 := base64.StdEncoding.EncodeToString(digest[:])

				sig, err := rsa.SignPKCS1v15(rand.Reader, idp.key, crypto.SHA256, digest[:])
				if err != nil {
					panic(fmt.Sprintf("sign: %v", err))
				}
				sigB64 := base64.StdEncoding.EncodeToString(sig)

				// Use a wrong URI that doesn't match the Assertion ID.
				return fmt.Sprintf(`<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
    <SignedInfo>
      <Reference URI="#_wrong_id">
        <DigestValue>%s</DigestValue>
      </Reference>
    </SignedInfo>
    <SignatureValue>%s</SignatureValue>
  </Signature>
  %s
</Response>`, digestB64, sigB64, assertion)
			},
			wantErr: errRefURIMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewSAMLProvider(meta, "tenant-xsw")
			xmlData := tt.build()
			_, err := provider.ValidateAssertion([]byte(xmlData))
			if tt.wantErr == nil {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error %v, got nil", tt.wantErr)
			}
			if err != tt.wantErr {
				t.Errorf("got error %v, want %v", err, tt.wantErr)
			}
		})
	}
}
