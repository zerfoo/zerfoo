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

// RateLimiter implements a token-bucket rate limiter keyed by client IP.
type RateLimiter struct {
	mu             sync.Mutex
	buckets        map[string]*bucket
	rate           float64 // tokens per second
	burst          int     // maximum tokens
	cleanTTL       time.Duration
	trustedProxies map[string]bool // IPs allowed to set X-Forwarded-For / X-Real-IP
}

type bucket struct {
	tokens  float64
	lastSee time.Time
}

// NewRateLimiter creates a rate limiter allowing rate requests/second
// with a burst capacity of burst.
func NewRateLimiter(rate float64, burst int) *RateLimiter {
	return &RateLimiter{
		buckets:  make(map[string]*bucket),
		rate:     rate,
		burst:    burst,
		cleanTTL: 10 * time.Minute,
	}
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
