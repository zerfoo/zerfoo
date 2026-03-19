// Package cloud provides a multi-tenant managed inference service for Zerfoo.
//
// It wraps the serve.Server with tenant isolation, token-based billing,
// rate limiting, and health checking for cloud deployments.
//
// Stability: alpha
package cloud

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

var (
	errEmptyTenantID  = errors.New("cloud: tenant ID must not be empty")
	errEmptyAPIKey    = errors.New("cloud: API key must not be empty")
	errInvalidLimits  = errors.New("cloud: rate limits must be positive")
	errTenantExists   = errors.New("cloud: tenant already exists")
	errTenantNotFound = errors.New("cloud: tenant not found")
)

// TenantConfig is the input for creating or describing a tenant.
// It contains no atomic fields and is safe to copy.
type TenantConfig struct {
	ID          string `json:"id"`
	APIKey      string `json:"api_key"`
	RateLimit   int64  `json:"rate_limit"`   // max requests per minute
	TokenBudget int64  `json:"token_budget"` // max tokens per minute
}

// Tenant represents a registered cloud tenant with runtime rate-limit state.
// Always accessed via pointer; must not be copied.
type Tenant struct {
	ID          string
	APIKey      string
	RateLimit   int64
	TokenBudget int64

	// runtime state
	requestCount atomic.Int64
	tokenCount   atomic.Int64
	lastReset    atomic.Int64 // unix nano
}

// Config returns a copyable snapshot of the tenant's configuration.
func (t *Tenant) Config() TenantConfig {
	return TenantConfig{
		ID:          t.ID,
		APIKey:      t.APIKey,
		RateLimit:   t.RateLimit,
		TokenBudget: t.TokenBudget,
	}
}

// maybeReset resets the per-minute counters if a minute has elapsed.
func (t *Tenant) maybeReset() {
	now := time.Now().UnixNano()
	last := t.lastReset.Load()
	if now-last >= int64(time.Minute) {
		if t.lastReset.CompareAndSwap(last, now) {
			t.requestCount.Store(0)
			t.tokenCount.Store(0)
		}
	}
}

// AllowRequest checks whether the tenant can make another request this minute.
// Returns true and increments the counter if allowed.
func (t *Tenant) AllowRequest() bool {
	t.maybeReset()
	cur := t.requestCount.Add(1)
	if cur > t.RateLimit {
		t.requestCount.Add(-1)
		return false
	}
	return true
}

// ConsumeTokens attempts to consume n tokens from the per-minute budget.
// Returns true if the tokens were consumed.
func (t *Tenant) ConsumeTokens(n int64) bool {
	t.maybeReset()
	for {
		cur := t.tokenCount.Load()
		next := cur + n
		if next > t.TokenBudget {
			return false
		}
		if t.tokenCount.CompareAndSwap(cur, next) {
			return true
		}
	}
}

// TenantManager provides CRUD operations on tenants, keyed by both tenant ID
// and API key for O(1) lookups in either direction.
type TenantManager struct {
	mu       sync.RWMutex
	byID     map[string]*Tenant
	byAPIKey map[string]*Tenant
}

// NewTenantManager creates a new empty TenantManager.
func NewTenantManager() *TenantManager {
	return &TenantManager{
		byID:     make(map[string]*Tenant),
		byAPIKey: make(map[string]*Tenant),
	}
}

// Create registers a new tenant. The tenant ID and API key must be unique.
func (m *TenantManager) Create(cfg TenantConfig) error {
	if cfg.ID == "" {
		return errEmptyTenantID
	}
	if cfg.APIKey == "" {
		return errEmptyAPIKey
	}
	if cfg.RateLimit <= 0 || cfg.TokenBudget <= 0 {
		return errInvalidLimits
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.byID[cfg.ID]; ok {
		return errTenantExists
	}
	if _, ok := m.byAPIKey[cfg.APIKey]; ok {
		return errTenantExists
	}

	tenant := &Tenant{
		ID:          cfg.ID,
		APIKey:      cfg.APIKey,
		RateLimit:   cfg.RateLimit,
		TokenBudget: cfg.TokenBudget,
	}
	tenant.lastReset.Store(time.Now().UnixNano())

	m.byID[cfg.ID] = tenant
	m.byAPIKey[cfg.APIKey] = tenant
	return nil
}

// Get retrieves a tenant by ID.
func (m *TenantManager) Get(id string) (*Tenant, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	t, ok := m.byID[id]
	if !ok {
		return nil, errTenantNotFound
	}
	return t, nil
}

// GetByAPIKey retrieves a tenant by API key.
func (m *TenantManager) GetByAPIKey(apiKey string) (*Tenant, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	t, ok := m.byAPIKey[apiKey]
	if !ok {
		return nil, errTenantNotFound
	}
	return t, nil
}

// Update modifies a tenant's rate limits and token budget.
func (m *TenantManager) Update(id string, rateLimit, tokenBudget int64) error {
	if rateLimit <= 0 || tokenBudget <= 0 {
		return errInvalidLimits
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	t, ok := m.byID[id]
	if !ok {
		return errTenantNotFound
	}
	t.RateLimit = rateLimit
	t.TokenBudget = tokenBudget
	return nil
}

// Delete removes a tenant by ID.
func (m *TenantManager) Delete(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	t, ok := m.byID[id]
	if !ok {
		return errTenantNotFound
	}
	delete(m.byID, id)
	delete(m.byAPIKey, t.APIKey)
	return nil
}

// List returns a copyable snapshot of all tenant configurations.
func (m *TenantManager) List() []TenantConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]TenantConfig, 0, len(m.byID))
	for _, t := range m.byID {
		out = append(out, t.Config())
	}
	return out
}
