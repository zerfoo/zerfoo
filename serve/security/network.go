package security

import (
	"errors"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// defaultMaxBuckets caps the number of distinct client IPs a RateLimiter
// tracks at once. It is a backstop against unbounded memory growth (CONC-M1)
// for the case where the periodic Cleanup ticker cannot keep pace with the
// rate of distinct new IPs (e.g. a spoofed-source flood). Once the cap is
// reached, the least-recently-seen entry is evicted to make room for a new
// one; see evictOldestLocked.
const defaultMaxBuckets = 100_000

// RateLimiter implements a token-bucket rate limiter keyed by client IP.
//
// Left unmanaged, rl.buckets grows one permanent entry per distinct client
// IP ever seen (CONC-M1). Two mitigations are applied:
//
//  1. Start launches a background goroutine that calls Cleanup every
//     cleanTTL, evicting entries that have been idle for at least cleanTTL.
//  2. maxBuckets hard-caps the map size as a backstop: once reached, the
//     oldest entry is evicted before a new one is inserted, so growth stays
//     bounded even if Cleanup cannot keep up.
//
// Server wires Start/Stop automatically via WithRateLimiter/Close. Callers
// that use RateLimiter directly must call Start to enable scheduled cleanup
// and Stop during shutdown to avoid leaking the ticker goroutine.
type RateLimiter struct {
	mu             sync.Mutex
	buckets        map[string]*bucket
	rate           float64 // tokens per second
	burst          int     // maximum tokens
	cleanTTL       time.Duration
	maxBuckets     int             // hard cap on tracked IPs; 0 disables the cap
	trustedProxies map[string]bool // IPs allowed to set X-Forwarded-For / X-Real-IP

	started bool
	stopCh  chan struct{}
	wg      sync.WaitGroup
}

type bucket struct {
	tokens  float64
	lastSee time.Time
}

// NewRateLimiter creates a rate limiter allowing rate requests/second
// with a burst capacity of burst. Call Start to begin the background
// cleanup loop that keeps rl.buckets from growing unbounded (CONC-M1).
func NewRateLimiter(rate float64, burst int) *RateLimiter {
	return &RateLimiter{
		buckets:    make(map[string]*bucket),
		rate:       rate,
		burst:      burst,
		cleanTTL:   10 * time.Minute,
		maxBuckets: defaultMaxBuckets,
	}
}

// SetMaxBuckets overrides the hard cap on the number of distinct client IPs
// tracked at once. A value <= 0 disables the cap entirely (not recommended
// in production; see CONC-M1).
func (rl *RateLimiter) SetMaxBuckets(n int) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.maxBuckets = n
}

// Start begins the background cleanup loop, which calls Cleanup every
// cleanTTL so rl.buckets does not grow unbounded under sustained traffic
// from many distinct client IPs (CONC-M1). Start is idempotent: calling it
// on an already-started limiter is a no-op. Call Stop during shutdown to
// stop the loop and release its goroutine.
func (rl *RateLimiter) Start() {
	rl.mu.Lock()
	if rl.started {
		rl.mu.Unlock()
		return
	}
	interval := rl.cleanTTL
	if interval <= 0 {
		interval = time.Minute
	}
	stopCh := make(chan struct{})
	rl.stopCh = stopCh
	rl.started = true
	rl.mu.Unlock()

	rl.wg.Add(1)
	go rl.cleanupLoop(interval, stopCh)
}

// cleanupLoop periodically calls Cleanup until stopCh is closed.
func (rl *RateLimiter) cleanupLoop(interval time.Duration, stopCh chan struct{}) {
	defer rl.wg.Done()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			rl.Cleanup()
		case <-stopCh:
			return
		}
	}
}

// Stop halts the background cleanup loop started by Start and waits for its
// goroutine to exit. It is safe to call even if Start was never called, and
// safe to call multiple times. After Stop returns, Start may be called
// again to resume scheduled cleanup.
func (rl *RateLimiter) Stop() {
	rl.mu.Lock()
	if !rl.started {
		rl.mu.Unlock()
		return
	}
	rl.started = false
	stopCh := rl.stopCh
	rl.mu.Unlock()

	close(stopCh)
	rl.wg.Wait()
}

// SetTrustedProxies configures the set of proxy IPs whose
// X-Forwarded-For and X-Real-IP headers are trusted. When empty,
// forwarding headers are never trusted and ClientIPFromRequest always
// returns RemoteAddr.
func (rl *RateLimiter) SetTrustedProxies(proxies []string) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	if len(proxies) == 0 {
		rl.trustedProxies = nil
		return
	}
	m := make(map[string]bool, len(proxies))
	for _, p := range proxies {
		m[p] = true
	}
	rl.trustedProxies = m
}

// TrustedProxies returns a copy of the current trusted proxy set.
func (rl *RateLimiter) TrustedProxies() map[string]bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	if rl.trustedProxies == nil {
		return nil
	}
	cp := make(map[string]bool, len(rl.trustedProxies))
	for k, v := range rl.trustedProxies {
		cp[k] = v
	}
	return cp
}

// Allow reports whether a request from the given IP should be allowed.
func (rl *RateLimiter) Allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	b, ok := rl.buckets[ip]
	if !ok {
		if rl.maxBuckets > 0 && len(rl.buckets) >= rl.maxBuckets {
			rl.evictOldestLocked()
		}
		b = &bucket{tokens: float64(rl.burst), lastSee: now}
		rl.buckets[ip] = b
	}

	elapsed := now.Sub(b.lastSee).Seconds()
	b.tokens += elapsed * rl.rate
	if b.tokens > float64(rl.burst) {
		b.tokens = float64(rl.burst)
	}
	b.lastSee = now

	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// evictOldestLocked removes the single least-recently-seen bucket entry.
// It is the size-cap backstop against unbounded growth (CONC-M1) for when
// the periodic Cleanup ticker cannot keep pace with the rate of distinct
// new client IPs. Callers must hold rl.mu.
func (rl *RateLimiter) evictOldestLocked() {
	var oldestIP string
	var oldestTime time.Time
	found := false
	for ip, b := range rl.buckets {
		if !found || b.lastSee.Before(oldestTime) {
			oldestIP = ip
			oldestTime = b.lastSee
			found = true
		}
	}
	if found {
		delete(rl.buckets, oldestIP)
	}
}

// Cleanup removes stale entries older than the configured TTL.
func (rl *RateLimiter) Cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	cutoff := time.Now().Add(-rl.cleanTTL)
	for ip, b := range rl.buckets {
		if b.lastSee.Before(cutoff) {
			delete(rl.buckets, ip)
		}
	}
}

// IPFilter manages IP allowlists and denylists.
type IPFilter struct {
	mu        sync.RWMutex
	allowList map[string]bool // nil means allow all
	denyList  map[string]bool
}

// NewIPFilter creates an IP filter. If allowList is non-empty, only those IPs
// are permitted. denyList is always checked first.
func NewIPFilter(allowList, denyList []string) *IPFilter {
	f := &IPFilter{
		denyList: make(map[string]bool),
	}
	if len(allowList) > 0 {
		f.allowList = make(map[string]bool)
		for _, ip := range allowList {
			f.allowList[ip] = true
		}
	}
	for _, ip := range denyList {
		f.denyList[ip] = true
	}
	return f
}

// Allowed reports whether the given IP is permitted.
func (f *IPFilter) Allowed(ip string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()

	if f.denyList[ip] {
		return false
	}
	if f.allowList == nil {
		return true
	}
	return f.allowList[ip]
}

// AddDeny adds an IP to the deny list.
func (f *IPFilter) AddDeny(ip string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.denyList[ip] = true
}

// RemoveDeny removes an IP from the deny list.
func (f *IPFilter) RemoveDeny(ip string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.denyList, ip)
}

// CORSPolicy defines allowed CORS origins, methods, and headers.
type CORSPolicy struct {
	AllowedOrigins []string
	AllowedMethods []string
	AllowedHeaders []string
	MaxAge         int // seconds
}

// Middleware returns an HTTP middleware that applies the CORS policy.
func (p *CORSPolicy) Middleware(next http.Handler) http.Handler {
	origins := make(map[string]bool, len(p.AllowedOrigins))
	for _, o := range p.AllowedOrigins {
		origins[o] = true
	}
	methods := strings.Join(p.AllowedMethods, ", ")
	headers := strings.Join(p.AllowedHeaders, ", ")

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" {
			if origins["*"] {
				w.Header().Set("Access-Control-Allow-Origin", "*")
				w.Header().Set("Access-Control-Allow-Methods", methods)
				w.Header().Set("Access-Control-Allow-Headers", headers)
				if p.MaxAge > 0 {
					w.Header().Set("Access-Control-Max-Age", strconv.Itoa(p.MaxAge))
				}
			} else if origins[origin] {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Methods", methods)
				w.Header().Set("Access-Control-Allow-Headers", headers)
				w.Header().Set("Vary", "Origin")
				if p.MaxAge > 0 {
					w.Header().Set("Access-Control-Max-Age", strconv.Itoa(p.MaxAge))
				}
			}
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ClientIP extracts the client IP from a request, checking X-Forwarded-For
// and X-Real-IP headers before falling back to RemoteAddr.
//
// Deprecated: ClientIP trusts forwarding headers from any source. Use
// ClientIPTrusted with an explicit set of trusted proxy IPs instead.
func ClientIP(r *http.Request) string {
	return ClientIPTrusted(r, nil)
}

// ClientIPTrusted extracts the client IP from a request. Forwarding headers
// (X-Forwarded-For, X-Real-IP) are only honoured when RemoteAddr is in
// trustedProxies. When trustedProxies is nil or RemoteAddr is not listed,
// the function returns the IP from RemoteAddr directly.
func ClientIPTrusted(r *http.Request, trustedProxies map[string]bool) string {
	remoteIP := remoteAddrIP(r)

	if len(trustedProxies) > 0 && trustedProxies[remoteIP] {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			parts := strings.SplitN(xff, ",", 2)
			ip := strings.TrimSpace(parts[0])
			if ip != "" {
				return ip
			}
		}
		if xri := r.Header.Get("X-Real-IP"); xri != "" {
			return strings.TrimSpace(xri)
		}
	}

	return remoteIP
}

// remoteAddrIP extracts the IP portion from r.RemoteAddr.
func remoteAddrIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// ValidateListenAddr checks that addr is a valid host:port for binding.
func ValidateListenAddr(addr string) error {
	if addr == "" {
		return errors.New("security: listen address must not be empty")
	}
	_, _, err := net.SplitHostPort(addr)
	if err != nil {
		return errors.New("security: invalid listen address: " + err.Error())
	}
	return nil
}
