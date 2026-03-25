// Package cloud provides a multi-tenant managed inference service for Zerfoo.
//
// It wraps the serve.Server with tenant isolation, token-based billing,
// rate limiting, and health checking for cloud deployments.
//
// Stability: alpha
package cloud

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
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
	ID                    string   `json:"id"`
	APIKey                string   `json:"api_key"`
	RateLimit             int64    `json:"rate_limit"`               // max requests per minute
	TokenBudget           int64    `json:"token_budget"`             // max tokens per minute
	MaxConcurrentRequests int      `json:"max_concurrent_requests,omitempty"`
	ModelAllowList        []string `json:"model_allow_list,omitempty"`
}

// hashAPIKey returns the hex-encoded SHA-256 hash of an API key.
func hashAPIKey(key string) string {
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:])
}

// Tenant represents a registered cloud tenant with runtime rate-limit state.
// Always accessed via pointer; must not be copied.
type Tenant struct {
	ID         string
	apiKeyHash string // SHA-256 hash of the API key; raw key is never stored

	// rateLimit and tokenBudget are accessed atomically to avoid data races
	// between concurrent AllowRequest/ConsumeTokens reads and Update writes.
	rateLimit             atomic.Int64
	tokenBudget           atomic.Int64
	maxConcurrentRequests int      // 0 means unlimited
	modelAllowList        []string // empty means all models allowed

	// runtime state
	requestCount atomic.Int64
	tokenCount   atomic.Int64
	lastReset    atomic.Int64 // unix nano
	inflight     atomic.Int32 // current in-flight requests for concurrency limiting
}

// redactedAPIKey is the placeholder returned instead of real API keys.
const redactedAPIKey = "***redacted***"

// Config returns a copyable snapshot of the tenant's configuration.
// The APIKey field is redacted to prevent accidental credential leakage.
func (t *Tenant) Config() TenantConfig {
	var allowList []string
	if len(t.modelAllowList) > 0 {
		allowList = make([]string, len(t.modelAllowList))
		copy(allowList, t.modelAllowList)
	}
	return TenantConfig{
		ID:                    t.ID,
		APIKey:                redactedAPIKey,
		RateLimit:             t.rateLimit.Load(),
		TokenBudget:           t.tokenBudget.Load(),
		MaxConcurrentRequests: t.maxConcurrentRequests,
		ModelAllowList:        allowList,
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
	if cur > t.rateLimit.Load() {
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
		if next > t.tokenBudget.Load() {
			return false
		}
		if t.tokenCount.CompareAndSwap(cur, next) {
			return true
		}
	}
}

// RefundTokens returns n tokens to the per-minute budget, used to reconcile
// pre-authorized estimates with actual usage after inference completes.
func (t *Tenant) RefundTokens(n int64) {
	t.tokenCount.Add(-n)
}

// DeductTokens unconditionally adds n tokens to the consumed count without
// checking the budget. This is used to charge excess usage when actual token
// generation exceeds the pre-authorized estimate (e.g. max_tokens=1 but the
// model produced more tokens). Unlike ConsumeTokens, it never fails.
func (t *Tenant) DeductTokens(n int64) {
	t.tokenCount.Add(n)
}

// ModelAllowed returns true if the model is in the tenant's allow list.
// An empty allow list permits all models.
func (t *Tenant) ModelAllowed(model string) bool {
	if len(t.modelAllowList) == 0 {
		return true
	}
	for _, m := range t.modelAllowList {
		if m == model {
			return true
		}
	}
	return false
}

// AllowConcurrent checks whether the tenant can accept another concurrent
// request. If MaxConcurrentRequests is 0 (unset), concurrency is unlimited.
// Returns true and increments the in-flight counter if allowed.
func (t *Tenant) AllowConcurrent() bool {
	if t.maxConcurrentRequests <= 0 {
		return true
	}
	cur := t.inflight.Add(1)
	if int(cur) > t.maxConcurrentRequests {
		t.inflight.Add(-1)
		return false
	}
	return true
}

// ReleaseConcurrent decrements the in-flight counter after a request completes.
func (t *Tenant) ReleaseConcurrent() {
	t.inflight.Add(-1)
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
	keyHash := hashAPIKey(cfg.APIKey)
	if _, ok := m.byAPIKey[keyHash]; ok {
		return errTenantExists
	}

	var allowList []string
	if len(cfg.ModelAllowList) > 0 {
		allowList = make([]string, len(cfg.ModelAllowList))
		copy(allowList, cfg.ModelAllowList)
	}
	tenant := &Tenant{
		ID:                    cfg.ID,
		apiKeyHash:            keyHash,
		maxConcurrentRequests: cfg.MaxConcurrentRequests,
		modelAllowList:        allowList,
	}
	tenant.rateLimit.Store(cfg.RateLimit)
	tenant.tokenBudget.Store(cfg.TokenBudget)
	tenant.lastReset.Store(time.Now().UnixNano())

	m.byID[cfg.ID] = tenant
	m.byAPIKey[keyHash] = tenant
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

// GetByAPIKey retrieves a tenant by API key. The input key is hashed with
// SHA-256 for O(1) map lookup, then verified with constant-time comparison
// on the hashes to prevent timing side-channel attacks.
func (m *TenantManager) GetByAPIKey(apiKey string) (*Tenant, error) {
	keyHash := hashAPIKey(apiKey)

	m.mu.RLock()
	defer m.mu.RUnlock()

	t, ok := m.byAPIKey[keyHash]
	if !ok {
		return nil, errTenantNotFound
	}
	// Constant-time comparison on the hashes to prevent timing attacks.
	if subtle.ConstantTimeCompare([]byte(t.apiKeyHash), []byte(keyHash)) != 1 {
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
	t.rateLimit.Store(rateLimit)
	t.tokenBudget.Store(tokenBudget)
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
	delete(m.byAPIKey, t.apiKeyHash)
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
