package security

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRateLimiterAllow(t *testing.T) {
	rl := NewRateLimiter(10, 5)

	// First 5 requests should succeed (burst)
	for i := 0; i < 5; i++ {
		if !rl.Allow("1.2.3.4") {
			t.Fatalf("request %d should be allowed", i)
		}
	}
	// 6th should be denied
	if rl.Allow("1.2.3.4") {
		t.Fatal("should be rate limited")
	}

	// Different IP should still work
	if !rl.Allow("5.6.7.8") {
		t.Fatal("different IP should be allowed")
	}
}

func TestRateLimiterCleanup(t *testing.T) {
	rl := NewRateLimiter(10, 5)
	rl.cleanTTL = 0 // everything is stale immediately
	rl.Allow("1.2.3.4")
	rl.Cleanup()

	rl.mu.Lock()
	n := len(rl.buckets)
	rl.mu.Unlock()
	if n != 0 {
		t.Fatalf("expected 0 buckets after cleanup, got %d", n)
	}
}

func TestIPFilterAllowAll(t *testing.T) {
	f := NewIPFilter(nil, nil)
	if !f.Allowed("1.2.3.4") {
		t.Fatal("should allow all when no allowlist")
	}
}

func TestIPFilterDenyList(t *testing.T) {
	f := NewIPFilter(nil, []string{"10.0.0.1"})
	if f.Allowed("10.0.0.1") {
		t.Fatal("denied IP should be blocked")
	}
	if !f.Allowed("10.0.0.2") {
		t.Fatal("non-denied IP should be allowed")
	}
}

func TestIPFilterAllowList(t *testing.T) {
	f := NewIPFilter([]string{"10.0.0.1"}, nil)
	if !f.Allowed("10.0.0.1") {
		t.Fatal("allowlisted IP should be allowed")
	}
	if f.Allowed("10.0.0.2") {
		t.Fatal("non-allowlisted IP should be blocked")
	}
}

func TestIPFilterDenyTakesPrecedence(t *testing.T) {
	f := NewIPFilter([]string{"10.0.0.1"}, []string{"10.0.0.1"})
	if f.Allowed("10.0.0.1") {
		t.Fatal("deny should take precedence over allow")
	}
}

func TestIPFilterAddRemoveDeny(t *testing.T) {
	f := NewIPFilter(nil, nil)
	f.AddDeny("1.2.3.4")
	if f.Allowed("1.2.3.4") {
		t.Fatal("should be denied after AddDeny")
	}
	f.RemoveDeny("1.2.3.4")
	if !f.Allowed("1.2.3.4") {
		t.Fatal("should be allowed after RemoveDeny")
	}
}

func TestCORSMiddleware(t *testing.T) {
	policy := &CORSPolicy{
		AllowedOrigins: []string{"https://example.com"},
		AllowedMethods: []string{"GET", "POST"},
		AllowedHeaders: []string{"Authorization"},
		MaxAge:         3600,
	}

	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	handler := policy.Middleware(inner)

	// Regular request with matching origin
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Origin", "https://example.com")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "https://example.com" {
		t.Fatalf("expected origin header, got %q", got)
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	// Preflight OPTIONS request
	req = httptest.NewRequest(http.MethodOptions, "/", nil)
	req.Header.Set("Origin", "https://example.com")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("expected 204 for OPTIONS, got %d", rec.Code)
	}

	// Non-matching origin
	req = httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Origin", "https://evil.com")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("expected no origin header for non-matching origin, got %q", got)
	}
}

func TestCORSWildcard(t *testing.T) {
	policy := &CORSPolicy{
		AllowedOrigins: []string{"*"},
		AllowedMethods: []string{"GET"},
	}
	handler := policy.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Origin", "https://anything.com")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "https://anything.com" {
		t.Fatalf("expected wildcard to match, got %q", got)
	}
}

func TestClientIP(t *testing.T) {
	tests := []struct {
		name       string
		xff        string
		xri        string
		remoteAddr string
		want       string
	}{
		{"XFF first", "1.1.1.1, 2.2.2.2", "", "3.3.3.3:8080", "1.1.1.1"},
		{"XRI", "", "4.4.4.4", "3.3.3.3:8080", "4.4.4.4"},
		{"RemoteAddr", "", "", "5.5.5.5:9090", "5.5.5.5"},
		{"RemoteAddr no port", "", "", "6.6.6.6", "6.6.6.6"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			req.RemoteAddr = tt.remoteAddr
			if tt.xff != "" {
				req.Header.Set("X-Forwarded-For", tt.xff)
			}
			if tt.xri != "" {
				req.Header.Set("X-Real-IP", tt.xri)
			}
			if got := ClientIP(req); got != tt.want {
				t.Fatalf("ClientIP() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestValidateListenAddr(t *testing.T) {
	if err := ValidateListenAddr(":8080"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateListenAddr("0.0.0.0:443"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateListenAddr(""); err == nil {
		t.Fatal("expected error for empty addr")
	}
	if err := ValidateListenAddr("noport"); err == nil {
		t.Fatal("expected error for invalid addr")
	}
}
