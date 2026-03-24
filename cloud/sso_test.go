package cloud

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"fmt"
	"math/big"
	"strings"
	"testing"
	"time"
)

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

	assertion := fmt.Sprintf(`<Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion" ID="_assertion1">
    <Subject><NameID>%s</NameID></Subject>
    <Conditions NotBefore="%s" NotOnOrAfter="%s"/>
%s  </Assertion>`, subject, notBefore, notOnOrAfter, attrStmt)

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
      <Reference URI="#_assertion1">
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
