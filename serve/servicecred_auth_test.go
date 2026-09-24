//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package serve

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ajent-social/go/servicecred"
	"github.com/ajent-social/go/servicecred/boltstore"
	"github.com/zerfoo/zerfoo/serve/security"
)

func openTestServiceCred(t *testing.T) (*servicecred.Service, string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.Chmod(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "credentials.db")
	store, err := boltstore.Open(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	svc, err := servicecred.New(store)
	if err != nil {
		t.Fatal(err)
	}
	return svc, path
}

func issueTestCred(t *testing.T, svc *servicecred.Service, scopes []string) (raw string, meta servicecred.Metadata) {
	t.Helper()
	grant := servicecred.Grant{
		Access: servicecred.Access{
			Owner:    DefaultServiceCredOwner,
			Resource: DefaultServiceCredResource,
			Scopes:   scopes,
		},
		ExpiresAt: time.Now().Add(time.Hour),
	}
	secret, meta, err := svc.Issue(context.Background(), grant, grant)
	if err != nil {
		t.Fatal(err)
	}
	return secret.Reveal(), meta
}

func TestServiceCredIssueAuthorizeRevoke(t *testing.T) {
	mdl := buildTestModel(t)
	svc, _ := openTestServiceCred(t)
	raw, meta := issueTestCred(t, svc, []string{
		string(security.ScopeReadOnly),
		string(security.ScopeInference),
	})

	srv := NewServer(mdl, WithServiceCred(svc, DefaultServiceCredOwner, DefaultServiceCredResource))
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	doGET := func(t *testing.T, token string) *http.Response {
		t.Helper()
		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"/v1/models", nil)
		if err != nil {
			t.Fatal(err)
		}
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	t.Run("authorize", func(t *testing.T) {
		resp := doGET(t, raw)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
	})

	t.Run("insufficient scope returns 403", func(t *testing.T) {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodDelete, ts.URL+"/v1/models/test", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+raw)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusForbidden {
			t.Fatalf("status = %d, want 403", resp.StatusCode)
		}
	})

	if err := svc.Revoke(context.Background(), DefaultServiceCredOwner, DefaultServiceCredResource, meta.ID); err != nil {
		t.Fatal(err)
	}

	t.Run("revoke denies", func(t *testing.T) {
		resp := doGET(t, raw)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401 after revoke", resp.StatusCode)
		}
	})
}

func TestServiceCredLegacyZFKeyRejected(t *testing.T) {
	mdl := buildTestModel(t)
	svc, _ := openTestServiceCred(t)
	srv := NewServer(mdl, WithServiceCred(svc, DefaultServiceCredOwner, DefaultServiceCredResource))

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer zf_"+strings.Repeat("ab", 32))
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401 for legacy zf_ key", rec.Code)
	}
}
