package aws

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Entitlement represents a customer's entitlement to use the product.
type Entitlement struct {
	CustomerIdentifier string    `json:"customerIdentifier"`
	ProductCode        string    `json:"productCode"`
	Dimension          string    `json:"dimension"`
	Value              int       `json:"value"`
	ExpiresAt          time.Time `json:"expiresAt"`
}

// EntitlementStore persists entitlement records.
type EntitlementStore interface {
	Put(ctx context.Context, ent Entitlement) error
	Get(ctx context.Context, customerIdentifier, dimension string) (*Entitlement, error)
	List(ctx context.Context, customerIdentifier string) ([]Entitlement, error)
	Delete(ctx context.Context, customerIdentifier, dimension string) error
}

// EntitlementChecker verifies customer entitlements with caching.
type EntitlementChecker struct {
	store    EntitlementStore
	cacheTTL time.Duration

	mu    sync.RWMutex
	cache map[string]cachedEntitlement
}

type cachedEntitlement struct {
	entitled  bool
	fetchedAt time.Time
}

// NewEntitlementChecker creates an EntitlementChecker with the given store and cache TTL.
func NewEntitlementChecker(store EntitlementStore, cacheTTL time.Duration) *EntitlementChecker {
	return &EntitlementChecker{
		store:    store,
		cacheTTL: cacheTTL,
		cache:    make(map[string]cachedEntitlement),
	}
}

// IsEntitled checks whether a customer is entitled to use a given dimension.
// Results are cached for the configured TTL.
func (c *EntitlementChecker) IsEntitled(ctx context.Context, customerIdentifier, dimension string) (bool, error) {
	key := customerIdentifier + ":" + dimension

	c.mu.RLock()
	if cached, ok := c.cache[key]; ok {
		if time.Since(cached.fetchedAt) < c.cacheTTL {
			c.mu.RUnlock()
			return cached.entitled, nil
		}
	}
	c.mu.RUnlock()

	ent, err := c.store.Get(ctx, customerIdentifier, dimension)
	if err != nil {
		return false, fmt.Errorf("check entitlement: %w", err)
	}

	entitled := ent != nil && time.Now().Before(ent.ExpiresAt) && ent.Value > 0

	c.mu.Lock()
	c.cache[key] = cachedEntitlement{entitled: entitled, fetchedAt: time.Now()}
	c.mu.Unlock()

	return entitled, nil
}

// Invalidate removes a cached entitlement entry, forcing the next check
// to query the store.
func (c *EntitlementChecker) Invalidate(customerIdentifier, dimension string) {
	key := customerIdentifier + ":" + dimension
	c.mu.Lock()
	delete(c.cache, key)
	c.mu.Unlock()
}

// GrantEntitlement stores an entitlement for a customer.
func (c *EntitlementChecker) GrantEntitlement(ctx context.Context, ent Entitlement) error {
	if err := c.store.Put(ctx, ent); err != nil {
		return fmt.Errorf("grant entitlement: %w", err)
	}
	c.Invalidate(ent.CustomerIdentifier, ent.Dimension)
	return nil
}

// RevokeEntitlement removes an entitlement and invalidates the cache.
func (c *EntitlementChecker) RevokeEntitlement(ctx context.Context, customerIdentifier, dimension string) error {
	if err := c.store.Delete(ctx, customerIdentifier, dimension); err != nil {
		return fmt.Errorf("revoke entitlement: %w", err)
	}
	c.Invalidate(customerIdentifier, dimension)
	return nil
}

// MemoryEntitlementStore is an in-memory EntitlementStore for testing.
type MemoryEntitlementStore struct {
	mu   sync.Mutex
	ents map[string]Entitlement
}

// NewMemoryEntitlementStore creates a new in-memory entitlement store.
func NewMemoryEntitlementStore() *MemoryEntitlementStore {
	return &MemoryEntitlementStore{
		ents: make(map[string]Entitlement),
	}
}

func entKey(customerIdentifier, dimension string) string {
	return customerIdentifier + ":" + dimension
}

// Put stores an entitlement.
func (s *MemoryEntitlementStore) Put(_ context.Context, ent Entitlement) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ents[entKey(ent.CustomerIdentifier, ent.Dimension)] = ent
	return nil
}

// Get retrieves an entitlement.
func (s *MemoryEntitlementStore) Get(_ context.Context, customerIdentifier, dimension string) (*Entitlement, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ent, ok := s.ents[entKey(customerIdentifier, dimension)]
	if !ok {
		return nil, nil
	}
	return &ent, nil
}

// List returns all entitlements for a customer.
func (s *MemoryEntitlementStore) List(_ context.Context, customerIdentifier string) ([]Entitlement, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Entitlement
	for _, ent := range s.ents {
		if ent.CustomerIdentifier == customerIdentifier {
			out = append(out, ent)
		}
	}
	return out, nil
}

// Delete removes an entitlement.
func (s *MemoryEntitlementStore) Delete(_ context.Context, customerIdentifier, dimension string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.ents, entKey(customerIdentifier, dimension))
	return nil
}
