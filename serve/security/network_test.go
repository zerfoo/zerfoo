package security

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"testing"
	"time"
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

// TestRateLimiterSizeCapBackstop is a regression test for CONC-M1: without a
// cap, rl.buckets grows one permanent entry per distinct client IP ever
// seen. It simulates sustained traffic from far more distinct IPs than the
// configured cap and asserts the map never grows past it, even though no
// cleanup ever runs (no Start, no Cleanup call) -- the cap is a standalone
// backstop, not merely a side effect of scheduled cleanup.
func TestRateLimiterSizeCapBackstop(t *testing.T) {
	rl := NewRateLimiter(10, 5)
	rl.SetMaxBuckets(50)

	for i := 0; i < 1000; i++ {
		rl.Allow(fmt.Sprintf("10.%d.%d.%d", i/65536, (i/256)%256, i%256))
	}

	rl.mu.Lock()
	n := len(rl.buckets)
	rl.mu.Unlock()

	if n > 50 {
		t.Fatalf("bucket map grew past the configured cap: got %d entries, want <= 50", n)
	}
	if n == 0 {
		t.Fatal("expected some buckets to remain after traffic")
	}
}

// TestRateLimiterBackgroundCleanupBounded is a regression test for CONC-M1:
// Cleanup existed but was never scheduled, so rl.buckets grew unbounded
// under sustained traffic from many distinct client IPs. It drives traffic
// from many distinct IPs, starts the background cleanup loop, then advances
// past cleanTTL and confirms the bucket count shrinks back down instead of
// growing without bound.
func TestRateLimiterBackgroundCleanupBounded(t *testing.T) {
	rl := NewRateLimiter(10, 5)
	rl.cleanTTL = 30 * time.Millisecond
	rl.Start()
	defer rl.Stop()

	// Sustained traffic from many distinct client IPs.
	for i := 0; i < 500; i++ {
		rl.Allow(fmt.Sprintf("192.168.%d.%d", i/256, i%256))
	}

	rl.mu.Lock()
	grown := len(rl.buckets)
	rl.mu.Unlock()
	if grown == 0 {
		t.Fatal("expected buckets to be populated after traffic")
	}

	// With no further traffic, every existing bucket becomes stale once
	// cleanTTL elapses; the background ticker should sweep them all without
	// any manual Cleanup call.
	deadline := time.Now().Add(2 * time.Second)
	for {
		rl.mu.Lock()
		n := len(rl.buckets)
		rl.mu.Unlock()
		if n == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("bucket map did not shrink after cleanTTL elapsed: still has %d entries (grew to %d)", n, grown)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestRateLimiterStartStopNoGoroutineLeak confirms the background cleanup
// ticker goroutine started by Start is fully reaped by Stop, that Start and
// Stop are each idempotent, and that Start/Stop can be safely repeated
// (e.g. across server restarts in tests) without accumulating goroutines.
func TestRateLimiterStartStopNoGoroutineLeak(t *testing.T) {
	before := runtime.NumGoroutine()

	rl := NewRateLimiter(10, 5)
	rl.cleanTTL = 5 * time.Millisecond
	rl.Start()
	rl.Start() // idempotent: must not spawn a second ticker goroutine

	time.Sleep(20 * time.Millisecond) // let a few ticks fire

	rl.Stop()
	rl.Stop() // idempotent: must not panic or block

	if !waitForGoroutineCountAtMost(before, 2*time.Second) {
		t.Fatalf("cleanup goroutine leaked after Stop: before=%d after=%d", before, runtime.NumGoroutine())
	}

	// Repeat the cycle to confirm no accumulation across restarts.
	rl.Start()
	time.Sleep(10 * time.Millisecond)
	rl.Stop()

	if !waitForGoroutineCountAtMost(before, 2*time.Second) {
		t.Fatalf("cleanup goroutine leaked after restart: before=%d after=%d", before, runtime.NumGoroutine())
	}
}

// waitForGoroutineCountAtMost polls runtime.NumGoroutine until it is <= want
// or timeout elapses, to avoid flaking on transient scheduler goroutines.
func waitForGoroutineCountAtMost(want int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if runtime.NumGoroutine() <= want {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(5 * time.Millisecond)
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
	if got := rec.Header().Get("Vary"); got != "Origin" {
		t.Fatalf("expected Vary: Origin when CORS headers are set, got %q", got)
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
	if got := rec.Header().Get("Vary"); got != "" {
		t.Fatalf("expected no Vary header for non-matching origin, got %q", got)
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

	// Wildcard config must return literal "*", not the reflected origin.
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("expected literal \"*\", got %q", got)
	}
	// Vary: Origin must NOT be set with a literal wildcard.
	if got := rec.Header().Get("Vary"); got != "" {
		t.Fatalf("expected no Vary header with wildcard origin, got %q", got)
	}
}

func TestCORSNonWildcard(t *testing.T) {
	policy := &CORSPolicy{
		AllowedOrigins: []string{"https://example.com"},
		AllowedMethods: []string{"GET", "POST"},
		AllowedHeaders: []string{"Authorization"},
	}
	handler := policy.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Origin", "https://example.com")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	// Non-wildcard config reflects the matching origin.
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "https://example.com" {
		t.Fatalf("expected reflected origin, got %q", got)
	}
	// Vary: Origin must be set for non-wildcard origins.
	if got := rec.Header().Get("Vary"); got != "Origin" {
		t.Fatalf("expected Vary: Origin, got %q", got)
	}
}

func TestClientIP(t *testing.T) {
	// ClientIP passes nil trustedProxies, so forwarding headers are never
	// trusted — it always returns RemoteAddr.
	tests := []struct {
		name       string
		xff        string
		xri        string
		remoteAddr string
		want       string
	}{
		{"XFF ignored without trusted proxies", "1.1.1.1, 2.2.2.2", "", "3.3.3.3:8080", "3.3.3.3"},
		{"XRI ignored without trusted proxies", "", "4.4.4.4", "3.3.3.3:8080", "3.3.3.3"},
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

func TestClientIPTrusted(t *testing.T) {
	tests := []struct {
		name           string
		xff            string
		xri            string
		remoteAddr     string
		trustedProxies map[string]bool
		want           string
	}{
		{
			name:           "XFF trusted proxy",
			xff:            "1.1.1.1, 2.2.2.2",
			remoteAddr:     "10.0.0.1:8080",
			trustedProxies: map[string]bool{"10.0.0.1": true},
			want:           "1.1.1.1",
		},
		{
			name:           "XFF untrusted proxy ignored",
			xff:            "1.1.1.1, 2.2.2.2",
			remoteAddr:     "9.9.9.9:8080",
			trustedProxies: map[string]bool{"10.0.0.1": true},
			want:           "9.9.9.9",
		},
		{
			name:           "XRI trusted proxy",
			xri:            "4.4.4.4",
			remoteAddr:     "10.0.0.1:8080",
			trustedProxies: map[string]bool{"10.0.0.1": true},
			want:           "4.4.4.4",
		},
		{
			name:           "XRI untrusted proxy ignored",
			xri:            "4.4.4.4",
			remoteAddr:     "9.9.9.9:8080",
			trustedProxies: map[string]bool{"10.0.0.1": true},
			want:           "9.9.9.9",
		},
		{
			name:           "nil trusted proxies ignores headers",
			xff:            "1.1.1.1",
			remoteAddr:     "3.3.3.3:8080",
			trustedProxies: nil,
			want:           "3.3.3.3",
		},
		{
			name:           "empty trusted proxies ignores headers",
			xff:            "1.1.1.1",
			remoteAddr:     "3.3.3.3:8080",
			trustedProxies: map[string]bool{},
			want:           "3.3.3.3",
		},
		{
			name:           "no headers returns RemoteAddr",
			remoteAddr:     "5.5.5.5:9090",
			trustedProxies: map[string]bool{"5.5.5.5": true},
			want:           "5.5.5.5",
		},
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
			if got := ClientIPTrusted(req, tt.trustedProxies); got != tt.want {
				t.Fatalf("ClientIPTrusted() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestRateLimiterSetTrustedProxies(t *testing.T) {
	rl := NewRateLimiter(10, 5)

	// Initially nil.
	if tp := rl.TrustedProxies(); tp != nil {
		t.Fatalf("expected nil trusted proxies, got %v", tp)
	}

	rl.SetTrustedProxies([]string{"10.0.0.1", "10.0.0.2"})
	tp := rl.TrustedProxies()
	if len(tp) != 2 || !tp["10.0.0.1"] || !tp["10.0.0.2"] {
		t.Fatalf("unexpected trusted proxies: %v", tp)
	}

	// Clear.
	rl.SetTrustedProxies(nil)
	if tp := rl.TrustedProxies(); tp != nil {
		t.Fatalf("expected nil after clear, got %v", tp)
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
