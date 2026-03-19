package security

import (
	"context"
	"sync"
	"time"
)

// IncidentSeverity classifies the severity of a security incident.
type IncidentSeverity string

const (
	IncidentCritical IncidentSeverity = "critical"
	IncidentHigh     IncidentSeverity = "high"
	IncidentMedium   IncidentSeverity = "medium"
	IncidentLow      IncidentSeverity = "low"
)

// Incident represents a security event that requires attention.
type Incident struct {
	ID        string
	Severity  IncidentSeverity
	Source    string // e.g., "rate_limiter", "auth", "ip_filter"
	Message   string
	ClientIP  string
	Timestamp time.Time
}

// AlertHook is called when a security incident is detected.
type AlertHook func(ctx context.Context, inc Incident) error

// IncidentResponder monitors security events and triggers alert hooks.
// It supports automatic lockout after repeated suspicious activity.
type IncidentResponder struct {
	mu           sync.Mutex
	hooks        []AlertHook
	failCounts   map[string]int // keyed by client IP
	lockoutLimit int
	lockoutDur   time.Duration
	lockouts     map[string]time.Time
	ipFilter     *IPFilter // optional; used for automatic lockout
}

// IncidentResponderOption configures an IncidentResponder.
type IncidentResponderOption func(*IncidentResponder)

// WithLockout enables automatic IP lockout after limit incidents within
// the lockout window. Requires an IPFilter to enforce the lockout.
func WithLockout(limit int, duration time.Duration, filter *IPFilter) IncidentResponderOption {
	return func(ir *IncidentResponder) {
		ir.lockoutLimit = limit
		ir.lockoutDur = duration
		ir.ipFilter = filter
	}
}

// NewIncidentResponder creates an incident responder with the given alert hooks.
func NewIncidentResponder(hooks []AlertHook, opts ...IncidentResponderOption) *IncidentResponder {
	ir := &IncidentResponder{
		hooks:      hooks,
		failCounts: make(map[string]int),
		lockouts:   make(map[string]time.Time),
	}
	for _, opt := range opts {
		opt(ir)
	}
	return ir
}

// Report records an incident and fires all alert hooks. If automatic lockout
// is configured and the threshold is exceeded, the source IP is denied.
func (ir *IncidentResponder) Report(ctx context.Context, inc Incident) error {
	ir.mu.Lock()
	if inc.ClientIP != "" {
		ir.failCounts[inc.ClientIP]++
		if ir.lockoutLimit > 0 && ir.failCounts[inc.ClientIP] >= ir.lockoutLimit {
			ir.lockouts[inc.ClientIP] = time.Now()
			if ir.ipFilter != nil {
				ir.ipFilter.AddDeny(inc.ClientIP)
			}
			ir.failCounts[inc.ClientIP] = 0
		}
	}
	hooks := make([]AlertHook, len(ir.hooks))
	copy(hooks, ir.hooks)
	ir.mu.Unlock()

	var firstErr error
	for _, h := range hooks {
		if err := h(ctx, inc); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// IsLockedOut reports whether the given IP is currently locked out.
func (ir *IncidentResponder) IsLockedOut(ip string) bool {
	ir.mu.Lock()
	defer ir.mu.Unlock()

	t, ok := ir.lockouts[ip]
	if !ok {
		return false
	}
	if ir.lockoutDur > 0 && time.Since(t) > ir.lockoutDur {
		delete(ir.lockouts, ip)
		if ir.ipFilter != nil {
			ir.ipFilter.RemoveDeny(ip)
		}
		return false
	}
	return true
}

// ResetLockout clears the lockout and failure count for the given IP.
func (ir *IncidentResponder) ResetLockout(ip string) {
	ir.mu.Lock()
	defer ir.mu.Unlock()

	delete(ir.lockouts, ip)
	delete(ir.failCounts, ip)
	if ir.ipFilter != nil {
		ir.ipFilter.RemoveDeny(ip)
	}
}
