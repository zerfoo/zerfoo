// Package cloud provides multi-tenant namespace isolation for the serving layer.
package cloud

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	errEmptyAPIKey    = errors.New("cloud: API key must not be empty")
	errInvalidConfig  = errors.New("cloud: invalid tenant config")
	errAlreadyExists  = errors.New("cloud: tenant already registered")
	errTenantNotFound = errors.New("cloud: tenant not found")
)

// TenantConfig holds per-tenant quota configuration.
type TenantConfig struct {
	MaxConcurrentRequests int      `json:"max_concurrent_requests"`
	MaxTokensPerMinute    int64    `json:"max_tokens_per_minute"`
	ModelAllowList        []string `json:"model_allow_list,omitempty"`
}

// Validate checks that the configuration has valid values.
func (c TenantConfig) Validate() error {
	if c.MaxConcurrentRequests <= 0 {
		return errors.New("cloud: MaxConcurrentRequests must be positive")
	}
	if c.MaxTokensPerMinute <= 0 {
		return errors.New("cloud: MaxTokensPerMinute must be positive")
	}
	return nil
}

// Tenant represents a registered tenant with its quota state.
type Tenant struct {
	Config    TenantConfig
	inflight  atomic.Int32
	tokens    atomic.Int64
	lastReset atomic.Int64 // unix nano of last token reset
}

// ConsumeTokens attempts to consume n tokens from the tenant's per-minute budget.
// It returns true if the tokens were consumed, false if the budget is exhausted.
func (t *Tenant) ConsumeTokens(n int64) bool {
	t.maybeResetTokens()
	for {
		cur := t.tokens.Load()
		if cur < n {
			return false
		}
		if t.tokens.CompareAndSwap(cur, cur-n) {
			return true
		}
	}
}

// maybeResetTokens resets the token bucket if a minute has elapsed since last reset.
func (t *Tenant) maybeResetTokens() {
	now := time.Now().UnixNano()
	last := t.lastReset.Load()
	if now-last >= int64(time.Minute) {
		if t.lastReset.CompareAndSwap(last, now) {
			t.tokens.Store(t.Config.MaxTokensPerMinute)
		}
	}
}

// ModelAllowed returns true if the model is in the tenant's allow list.
// An empty allow list permits all models.
func (t *Tenant) ModelAllowed(model string) bool {
	if len(t.Config.ModelAllowList) == 0 {
		return true
	}
	for _, m := range t.Config.ModelAllowList {
		if m == model {
			return true
		}
	}
	return false
}

type contextKey struct{}

// TenantFromContext extracts the Tenant from the request context.
func TenantFromContext(ctx context.Context) *Tenant {
	t, _ := ctx.Value(contextKey{}).(*Tenant)
	return t
}

// TenantRegistry manages per-API-key tenant registrations and quotas.
type TenantRegistry struct {
	mu      sync.RWMutex
	tenants map[string]*Tenant
}

// NewTenantRegistry creates a new empty registry.
func NewTenantRegistry() *TenantRegistry {
	return &TenantRegistry{
		tenants: make(map[string]*Tenant),
	}
}

// Register adds a tenant with the given API key and configuration.
// The API key is SHA-256 hashed before storage so raw keys never reside in memory.
func (r *TenantRegistry) Register(apiKey string, cfg TenantConfig) error {
	if apiKey == "" {
		return errEmptyAPIKey
	}
	if err := cfg.Validate(); err != nil {
		return err
	}

	keyHash := hashAPIKey(apiKey)

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tenants[keyHash]; exists {
		return errAlreadyExists
	}

	t := &Tenant{Config: cfg}
	t.tokens.Store(cfg.MaxTokensPerMinute)
	t.lastReset.Store(time.Now().UnixNano())
	r.tenants[keyHash] = t
	return nil
}

// Get retrieves the tenant for the given API key.
func (r *TenantRegistry) Get(apiKey string) (*Tenant, error) {
	keyHash := hashAPIKey(apiKey)

	r.mu.RLock()
	defer r.mu.RUnlock()

	t, ok := r.tenants[keyHash]
	if !ok {
		return nil, errTenantNotFound
	}
	return t, nil
}

// Remove deletes a tenant registration.
func (r *TenantRegistry) Remove(apiKey string) error {
	keyHash := hashAPIKey(apiKey)

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.tenants[keyHash]; !ok {
		return errTenantNotFound
	}
	delete(r.tenants, keyHash)
	return nil
}

// hashAPIKey returns the hex-encoded SHA-256 hash of the given key.
func hashAPIKey(key string) string {
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:])
}

// Middleware returns an HTTP middleware that enforces tenant isolation.
// It extracts the API key from the Authorization header (Bearer <key>),
// enforces concurrency limits, token rate limits, and injects the Tenant
// into the request context.
func (r *TenantRegistry) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		apiKey := extractBearerToken(req)
		if apiKey == "" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		tenant, err := r.Get(apiKey)
		if err != nil {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		// Check token budget before admitting the request.
		if tenant.tokens.Load() <= 0 {
			tenant.maybeResetTokens()
			if tenant.tokens.Load() <= 0 {
				http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
				return
			}
		}

		// Enforce concurrency limit.
		cur := tenant.inflight.Add(1)
		if int(cur) > tenant.Config.MaxConcurrentRequests {
			tenant.inflight.Add(-1)
			http.Error(w, "too many concurrent requests", http.StatusTooManyRequests)
			return
		}
		defer tenant.inflight.Add(-1)

		ctx := context.WithValue(req.Context(), contextKey{}, tenant)
		next.ServeHTTP(w, req.WithContext(ctx))
	})
}

func extractBearerToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return ""
	}
	const prefix = "Bearer "
	if !strings.HasPrefix(auth, prefix) {
		return ""
	}
	return auth[len(prefix):]
}
