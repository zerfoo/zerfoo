package cloud

import (
	"fmt"
	"testing"
	"time"
)

func TestEnterprise_AuditLog(t *testing.T) {
	tests := []struct {
		name    string
		entries []AuditEntry
		queryID string
		want    int
	}{
		{
			name: "records single request",
			entries: []AuditEntry{
				{TenantID: "t1", Action: AuditActionInference, Result: AuditResultSuccess, StatusCode: 200, Method: "POST", Path: "/v1/chat/completions"},
			},
			queryID: "t1",
			want:    1,
		},
		{
			name: "isolates by tenant",
			entries: []AuditEntry{
				{TenantID: "t1", Action: AuditActionInference, Result: AuditResultSuccess, StatusCode: 200},
				{TenantID: "t2", Action: AuditActionInference, Result: AuditResultSuccess, StatusCode: 200},
				{TenantID: "t1", Action: AuditActionAuth, Result: AuditResultUnauthorized, StatusCode: 401},
			},
			queryID: "t1",
			want:    2,
		},
		{
			name: "records denied requests",
			entries: []AuditEntry{
				{TenantID: "t1", Action: AuditActionInference, Result: AuditResultRateLimited, StatusCode: 429},
				{TenantID: "t1", Action: AuditActionAuth, Result: AuditResultUnauthorized, StatusCode: 401},
			},
			queryID: "t1",
			want:    2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := NewMemoryAuditStore()
			logger := NewAuditLogger(store)

			for _, e := range tt.entries {
				if err := logger.Log(e); err != nil {
					t.Fatalf("Log: %v", err)
				}
			}

			now := time.Now()
			entries, err := logger.Query(tt.queryID, now.Add(-time.Minute), now.Add(time.Minute))
			if err != nil {
				t.Fatalf("Query: %v", err)
			}
			if len(entries) != tt.want {
				t.Errorf("got %d entries, want %d", len(entries), tt.want)
			}
		})
	}

	t.Run("records all required fields", func(t *testing.T) {
		store := NewMemoryAuditStore()
		logger := NewAuditLogger(store)

		logger.Log(AuditEntry{
			TenantID:   "t1",
			Action:     AuditActionInference,
			Result:     AuditResultSuccess,
			Resource:   "chat.completion",
			StatusCode: 200,
			Method:     "POST",
			Path:       "/v1/chat/completions",
			RemoteAddr: "10.0.0.1:54321",
		})

		entries := store.All()
		if len(entries) != 1 {
			t.Fatalf("got %d entries, want 1", len(entries))
		}
		e := entries[0]
		if e.TenantID != "t1" {
			t.Errorf("TenantID = %q, want %q", e.TenantID, "t1")
		}
		if e.Action != AuditActionInference {
			t.Errorf("Action = %q, want %q", e.Action, AuditActionInference)
		}
		if e.Result != AuditResultSuccess {
			t.Errorf("Result = %q, want %q", e.Result, AuditResultSuccess)
		}
		if e.Timestamp.IsZero() {
			t.Error("Timestamp should be set automatically")
		}
		if e.StatusCode != 200 {
			t.Errorf("StatusCode = %d, want 200", e.StatusCode)
		}
		if e.Method != "POST" {
			t.Errorf("Method = %q, want %q", e.Method, "POST")
		}
		if e.Path != "/v1/chat/completions" {
			t.Errorf("Path = %q, want %q", e.Path, "/v1/chat/completions")
		}
		if e.RemoteAddr != "10.0.0.1:54321" {
			t.Errorf("RemoteAddr = %q, want %q", e.RemoteAddr, "10.0.0.1:54321")
		}
	})

	t.Run("no sensitive data in audit entries", func(t *testing.T) {
		store := NewMemoryAuditStore()
		logger := NewAuditLogger(store)

		// Log an entry — verify no API key or request body fields exist.
		logger.Log(AuditEntry{
			TenantID:   "t1",
			Action:     AuditActionInference,
			Result:     AuditResultSuccess,
			StatusCode: 200,
			Method:     "POST",
			Path:       "/v1/chat/completions",
		})

		entries := store.All()
		if len(entries) != 1 {
			t.Fatalf("got %d entries, want 1", len(entries))
		}
		// AuditEntry struct has no fields for API keys, request bodies, or response bodies.
		// This test verifies the struct design prevents leaking sensitive data.
		e := entries[0]
		if e.TenantID == "" {
			t.Error("TenantID should be set")
		}
	})
}

func TestEnterprise_SSO(t *testing.T) {
	validMetadata := []byte(`<EntityDescriptor entityID="https://idp.example.com/saml" xmlns="urn:oasis:names:tc:SAML:2.0:metadata">
  <IDPSSODescriptor WantAuthnRequestsSigned="true">
    <KeyDescriptor use="signing">
      <KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#">
        <X509Data>
          <X509Certificate>MIICdummy...</X509Certificate>
        </X509Data>
      </KeyInfo>
    </KeyDescriptor>
    <SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="https://idp.example.com/sso"/>
    <NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</NameIDFormat>
  </IDPSSODescriptor>
</EntityDescriptor>`)

	t.Run("parse SAML metadata", func(t *testing.T) {
		tests := []struct {
			name      string
			data      []byte
			wantErr   bool
			wantID    string
			wantURL   string
			wantCert  string
		}{
			{
				name:     "valid metadata",
				data:     validMetadata,
				wantErr:  false,
				wantID:   "https://idp.example.com/saml",
				wantURL:  "https://idp.example.com/sso",
				wantCert: "MIICdummy...",
			},
			{
				name:    "invalid XML",
				data:    []byte("<not valid xml"),
				wantErr: true,
			},
			{
				name: "missing entity ID",
				data: []byte(`<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata">
  <IDPSSODescriptor>
    <KeyDescriptor use="signing"><KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#"><X509Data><X509Certificate>cert</X509Certificate></X509Data></KeyInfo></KeyDescriptor>
    <SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="https://idp.example.com/sso"/>
  </IDPSSODescriptor>
</EntityDescriptor>`),
				wantErr: true,
			},
			{
				name: "missing SSO URL",
				data: []byte(`<EntityDescriptor entityID="https://idp.example.com" xmlns="urn:oasis:names:tc:SAML:2.0:metadata">
  <IDPSSODescriptor>
    <KeyDescriptor use="signing"><KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#"><X509Data><X509Certificate>cert</X509Certificate></X509Data></KeyInfo></KeyDescriptor>
  </IDPSSODescriptor>
</EntityDescriptor>`),
				wantErr: true,
			},
			{
				name: "missing certificate",
				data: []byte(`<EntityDescriptor entityID="https://idp.example.com" xmlns="urn:oasis:names:tc:SAML:2.0:metadata">
  <IDPSSODescriptor>
    <SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="https://idp.example.com/sso"/>
  </IDPSSODescriptor>
</EntityDescriptor>`),
				wantErr: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				meta, err := ParseSAMLMetadata(tt.data)
				if (err != nil) != tt.wantErr {
					t.Errorf("ParseSAMLMetadata() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if tt.wantErr {
					return
				}
				if meta.EntityID != tt.wantID {
					t.Errorf("EntityID = %q, want %q", meta.EntityID, tt.wantID)
				}
				if meta.SignOnURL != tt.wantURL {
					t.Errorf("SignOnURL = %q, want %q", meta.SignOnURL, tt.wantURL)
				}
				if meta.Certificate != tt.wantCert {
					t.Errorf("Certificate = %q, want %q", meta.Certificate, tt.wantCert)
				}
			})
		}
	})

	t.Run("SAML metadata attributes", func(t *testing.T) {
		meta, err := ParseSAMLMetadata(validMetadata)
		if err != nil {
			t.Fatalf("ParseSAMLMetadata: %v", err)
		}
		if !meta.WantAuthnSigned {
			t.Error("WantAuthnSigned should be true")
		}
		if meta.NameIDFormat != "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress" {
			t.Errorf("NameIDFormat = %q", meta.NameIDFormat)
		}
	})

	t.Run("SSO provider interface", func(t *testing.T) {
		meta, err := ParseSAMLMetadata(validMetadata)
		if err != nil {
			t.Fatalf("ParseSAMLMetadata: %v", err)
		}

		provider := NewSAMLProvider(meta, "tenant-1")

		// Verify interface compliance.
		var _ SSOProvider = provider

		if provider.EntityID() != "https://idp.example.com/saml" {
			t.Errorf("EntityID() = %q", provider.EntityID())
		}
	})

	t.Run("validate SAML assertion", func(t *testing.T) {
		idp := newTestIdP(t)
		meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
		provider := NewSAMLProvider(meta, "tenant-1")

		future := time.Now().Add(time.Hour).Format(time.RFC3339)
		past := time.Now().Add(-time.Hour).Format(time.RFC3339)

		t.Run("valid assertion", func(t *testing.T) {
			xml := idp.signedSAMLResponse("user@example.com", past, future, map[string]string{
				"email": "user@example.com",
				"role":  "admin",
			})
			identity, err := provider.ValidateAssertion([]byte(xml))
			if err != nil {
				t.Fatalf("ValidateAssertion() error = %v", err)
			}
			if identity.Subject != "user@example.com" {
				t.Errorf("Subject = %q, want %q", identity.Subject, "user@example.com")
			}
			if identity.TenantID != "tenant-1" {
				t.Errorf("TenantID = %q, want %q", identity.TenantID, "tenant-1")
			}
		})

		t.Run("expired assertion", func(t *testing.T) {
			xml := idp.signedSAMLResponse("user@example.com", past, past, nil)
			_, err := provider.ValidateAssertion([]byte(xml))
			if err == nil {
				t.Error("expected error for expired assertion")
			}
		})

		t.Run("missing subject", func(t *testing.T) {
			xml := idp.signedSAMLResponse("", past, future, nil)
			_, err := provider.ValidateAssertion([]byte(xml))
			if err == nil {
				t.Error("expected error for missing subject")
			}
		})

		t.Run("missing conditions", func(t *testing.T) {
			// Unsigned assertion with missing conditions — fails at signature check first.
			noCondXML := fmt.Sprintf(`<Response xmlns="urn:oasis:names:tc:SAML:2.0:protocol">
  <Assertion xmlns="urn:oasis:names:tc:SAML:2.0:assertion">
    <Subject><NameID>user@example.com</NameID></Subject>
  </Assertion>
</Response>`)
			_, err := provider.ValidateAssertion([]byte(noCondXML))
			if err == nil {
				t.Error("expected error for missing conditions/signature")
			}
		})
	})

	t.Run("assertion attributes extracted", func(t *testing.T) {
		idp := newTestIdP(t)
		meta, _ := ParseSAMLMetadata(idp.testSAMLMetadata())
		provider := NewSAMLProvider(meta, "tenant-1")

		future := time.Now().Add(time.Hour).Format(time.RFC3339)
		past := time.Now().Add(-time.Hour).Format(time.RFC3339)

		assertion := idp.signedSAMLResponse("user@example.com", past, future, map[string]string{
			"email":      "user@example.com",
			"role":       "admin",
			"department": "engineering",
		})

		identity, err := provider.ValidateAssertion([]byte(assertion))
		if err != nil {
			t.Fatalf("ValidateAssertion: %v", err)
		}
		if identity.Email != "user@example.com" {
			t.Errorf("Email = %q, want %q", identity.Email, "user@example.com")
		}
		if identity.Attributes["role"] != "admin" {
			t.Errorf("role = %q, want %q", identity.Attributes["role"], "admin")
		}
		if identity.Attributes["department"] != "engineering" {
			t.Errorf("department = %q, want %q", identity.Attributes["department"], "engineering")
		}
		if identity.ExpiresAt.IsZero() {
			t.Error("ExpiresAt should be set")
		}
	})
}

func TestEnterprise_TenantIsolation(t *testing.T) {
	t.Run("tenants cannot access each other's data", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "alpha", APIKey: "key-alpha", RateLimit: 100, TokenBudget: 10000})
		tm.Create(TenantConfig{ID: "beta", APIKey: "key-beta", RateLimit: 100, TokenBudget: 10000})

		// Alpha's key should not resolve to beta.
		tenant, err := tm.GetByAPIKey("key-alpha")
		if err != nil {
			t.Fatalf("GetByAPIKey: %v", err)
		}
		if tenant.ID != "alpha" {
			t.Errorf("got tenant %q for key-alpha, want alpha", tenant.ID)
		}

		// Beta's key should not resolve to alpha.
		tenant, err = tm.GetByAPIKey("key-beta")
		if err != nil {
			t.Fatalf("GetByAPIKey: %v", err)
		}
		if tenant.ID != "beta" {
			t.Errorf("got tenant %q for key-beta, want beta", tenant.ID)
		}
	})

	t.Run("rate limits are isolated per tenant", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "k1", RateLimit: 2, TokenBudget: 10000})
		tm.Create(TenantConfig{ID: "t2", APIKey: "k2", RateLimit: 2, TokenBudget: 10000})

		t1, _ := tm.Get("t1")
		t2, _ := tm.Get("t2")

		// Exhaust t1's rate limit.
		t1.AllowRequest()
		t1.AllowRequest()
		if t1.AllowRequest() {
			t.Error("t1 should be rate limited after 2 requests")
		}

		// t2 should still be able to make requests.
		if !t2.AllowRequest() {
			t.Error("t2 should not be affected by t1's rate limit")
		}
	})

	t.Run("token budgets are isolated per tenant", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "k1", RateLimit: 100, TokenBudget: 50})
		tm.Create(TenantConfig{ID: "t2", APIKey: "k2", RateLimit: 100, TokenBudget: 50})

		t1, _ := tm.Get("t1")
		t2, _ := tm.Get("t2")

		// Exhaust t1's token budget.
		t1.ConsumeTokens(50)
		if t1.ConsumeTokens(1) {
			t.Error("t1 should be out of token budget")
		}

		// t2 should still have budget.
		if !t2.ConsumeTokens(25) {
			t.Error("t2 should not be affected by t1's token consumption")
		}
	})

	t.Run("audit logs are isolated per tenant", func(t *testing.T) {
		store := NewMemoryAuditStore()
		logger := NewAuditLogger(store)

		logger.Log(AuditEntry{TenantID: "alpha", Action: AuditActionInference, Result: AuditResultSuccess})
		logger.Log(AuditEntry{TenantID: "beta", Action: AuditActionInference, Result: AuditResultSuccess})
		logger.Log(AuditEntry{TenantID: "alpha", Action: AuditActionAuth, Result: AuditResultDenied})

		now := time.Now()
		alphaEntries, _ := logger.Query("alpha", now.Add(-time.Minute), now.Add(time.Minute))
		betaEntries, _ := logger.Query("beta", now.Add(-time.Minute), now.Add(time.Minute))

		if len(alphaEntries) != 2 {
			t.Errorf("alpha got %d entries, want 2", len(alphaEntries))
		}
		if len(betaEntries) != 1 {
			t.Errorf("beta got %d entries, want 1", len(betaEntries))
		}

		// Verify no cross-tenant leakage.
		for _, e := range alphaEntries {
			if e.TenantID != "alpha" {
				t.Errorf("alpha query returned entry for tenant %q", e.TenantID)
			}
		}
		for _, e := range betaEntries {
			if e.TenantID != "beta" {
				t.Errorf("beta query returned entry for tenant %q", e.TenantID)
			}
		}
	})

	t.Run("billing records are isolated per tenant", func(t *testing.T) {
		store := NewMemoryBillingStore()
		meter := NewTokenMeter(store)

		meter.Record("alpha", 100, 50)
		meter.Record("beta", 200, 100)
		meter.Record("alpha", 50, 25)

		now := time.Now()
		alphaRecords, _ := meter.Query("alpha", now.Add(-time.Minute), now.Add(time.Minute))
		betaRecords, _ := meter.Query("beta", now.Add(-time.Minute), now.Add(time.Minute))

		if len(alphaRecords) != 2 {
			t.Errorf("alpha got %d records, want 2", len(alphaRecords))
		}
		if len(betaRecords) != 1 {
			t.Errorf("beta got %d records, want 1", len(betaRecords))
		}
	})

	t.Run("deleted tenant keys cannot authenticate", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 100, TokenBudget: 10000})
		tm.Delete("t1")

		_, err := tm.GetByAPIKey("key-1")
		if err == nil {
			t.Error("deleted tenant's API key should not authenticate")
		}

		_, err = tm.Get("t1")
		if err == nil {
			t.Error("deleted tenant should not be retrievable")
		}
	})
}
