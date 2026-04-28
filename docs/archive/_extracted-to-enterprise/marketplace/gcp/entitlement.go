package gcp

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// EntitlementManager handles the full lifecycle of GCP Marketplace entitlements
// using the Partner Procurement API. It supports approve, reject, suspend,
// and reinstate operations with local state tracking.
type EntitlementManager struct {
	procurement ProcurementAPI
	store       EntitlementStore
	cacheTTL    time.Duration

	mu    sync.RWMutex
	cache map[string]cachedState
}

// EntitlementStore persists local entitlement state.
type EntitlementStore interface {
	Put(ctx context.Context, ent LocalEntitlement) error
	Get(ctx context.Context, entitlementName string) (*LocalEntitlement, error)
	List(ctx context.Context, accountName string) ([]LocalEntitlement, error)
	Delete(ctx context.Context, entitlementName string) error
}

// LocalEntitlement tracks the local state of a GCP Marketplace entitlement.
type LocalEntitlement struct {
	Name       string           `json:"name"`
	Account    string           `json:"account"`
	Product    string           `json:"product"`
	Plan       string           `json:"plan"`
	State      EntitlementState `json:"state"`
	CreateTime time.Time        `json:"createTime"`
	UpdateTime time.Time        `json:"updateTime"`
}

type cachedState struct {
	state     EntitlementState
	fetchedAt time.Time
}

// NewEntitlementManager creates an EntitlementManager backed by the given
// procurement client, store, and cache TTL.
func NewEntitlementManager(procurement ProcurementAPI, store EntitlementStore, cacheTTL time.Duration) *EntitlementManager {
	return &EntitlementManager{
		procurement: procurement,
		store:       store,
		cacheTTL:    cacheTTL,
		cache:       make(map[string]cachedState),
	}
}

// Approve approves a pending entitlement and stores it locally.
func (m *EntitlementManager) Approve(ctx context.Context, entitlementName string) error {
	if err := m.procurement.ApproveEntitlement(ctx, entitlementName); err != nil {
		return fmt.Errorf("approve entitlement: %w", err)
	}

	ent, err := m.procurement.GetEntitlement(ctx, entitlementName)
	if err != nil {
		return fmt.Errorf("get entitlement after approve: %w", err)
	}

	local := LocalEntitlement{
		Name:       ent.Name,
		Account:    ent.Account,
		Product:    ent.Product,
		Plan:       ent.Plan,
		State:      EntitlementActive,
		CreateTime: ent.CreateTime,
		UpdateTime: time.Now(),
	}

	if err := m.store.Put(ctx, local); err != nil {
		return fmt.Errorf("store entitlement: %w", err)
	}

	m.invalidate(entitlementName)
	return nil
}

// Reject rejects a pending entitlement with the given reason.
func (m *EntitlementManager) Reject(ctx context.Context, entitlementName, reason string) error {
	if err := m.procurement.RejectEntitlement(ctx, entitlementName, reason); err != nil {
		return fmt.Errorf("reject entitlement: %w", err)
	}

	local := LocalEntitlement{
		Name:       entitlementName,
		State:      EntitlementCancelled,
		UpdateTime: time.Now(),
	}

	if err := m.store.Put(ctx, local); err != nil {
		return fmt.Errorf("store rejected entitlement: %w", err)
	}

	m.invalidate(entitlementName)
	return nil
}

// Suspend suspends an active entitlement with the given reason.
func (m *EntitlementManager) Suspend(ctx context.Context, entitlementName, reason string) error {
	if err := m.procurement.SuspendEntitlement(ctx, entitlementName, reason); err != nil {
		return fmt.Errorf("suspend entitlement: %w", err)
	}

	existing, err := m.store.Get(ctx, entitlementName)
	if err != nil {
		return fmt.Errorf("get entitlement for suspend: %w", err)
	}

	local := LocalEntitlement{
		Name:       entitlementName,
		State:      EntitlementSuspended,
		UpdateTime: time.Now(),
	}
	if existing != nil {
		local.Account = existing.Account
		local.Product = existing.Product
		local.Plan = existing.Plan
		local.CreateTime = existing.CreateTime
	}

	if err := m.store.Put(ctx, local); err != nil {
		return fmt.Errorf("store suspended entitlement: %w", err)
	}

	m.invalidate(entitlementName)
	return nil
}

// Reinstate reinstates a suspended entitlement.
func (m *EntitlementManager) Reinstate(ctx context.Context, entitlementName string) error {
	if err := m.procurement.ReinstateEntitlement(ctx, entitlementName); err != nil {
		return fmt.Errorf("reinstate entitlement: %w", err)
	}

	existing, err := m.store.Get(ctx, entitlementName)
	if err != nil {
		return fmt.Errorf("get entitlement for reinstate: %w", err)
	}

	local := LocalEntitlement{
		Name:       entitlementName,
		State:      EntitlementActive,
		UpdateTime: time.Now(),
	}
	if existing != nil {
		local.Account = existing.Account
		local.Product = existing.Product
		local.Plan = existing.Plan
		local.CreateTime = existing.CreateTime
	}

	if err := m.store.Put(ctx, local); err != nil {
		return fmt.Errorf("store reinstated entitlement: %w", err)
	}

	m.invalidate(entitlementName)
	return nil
}

// IsActive checks whether an entitlement is currently active.
// Results are cached for the configured TTL.
func (m *EntitlementManager) IsActive(ctx context.Context, entitlementName string) (bool, error) {
	m.mu.RLock()
	if cached, ok := m.cache[entitlementName]; ok {
		if time.Since(cached.fetchedAt) < m.cacheTTL {
			m.mu.RUnlock()
			return cached.state == EntitlementActive, nil
		}
	}
	m.mu.RUnlock()

	ent, err := m.store.Get(ctx, entitlementName)
	if err != nil {
		return false, fmt.Errorf("check entitlement: %w", err)
	}

	active := ent != nil && ent.State == EntitlementActive

	state := EntitlementCancelled
	if active {
		state = EntitlementActive
	}

	m.mu.Lock()
	m.cache[entitlementName] = cachedState{state: state, fetchedAt: time.Now()}
	m.mu.Unlock()

	return active, nil
}

func (m *EntitlementManager) invalidate(entitlementName string) {
	m.mu.Lock()
	delete(m.cache, entitlementName)
	m.mu.Unlock()
}

// MemoryEntitlementStore is an in-memory EntitlementStore for testing.
type MemoryEntitlementStore struct {
	mu   sync.Mutex
	ents map[string]LocalEntitlement
}

// NewMemoryEntitlementStore creates a new in-memory entitlement store.
func NewMemoryEntitlementStore() *MemoryEntitlementStore {
	return &MemoryEntitlementStore{
		ents: make(map[string]LocalEntitlement),
	}
}

// Put stores a local entitlement.
func (s *MemoryEntitlementStore) Put(_ context.Context, ent LocalEntitlement) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ents[ent.Name] = ent
	return nil
}

// Get retrieves a local entitlement by name.
func (s *MemoryEntitlementStore) Get(_ context.Context, entitlementName string) (*LocalEntitlement, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ent, ok := s.ents[entitlementName]
	if !ok {
		return nil, nil
	}
	return &ent, nil
}

// List returns all entitlements for an account.
func (s *MemoryEntitlementStore) List(_ context.Context, accountName string) ([]LocalEntitlement, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []LocalEntitlement
	for _, ent := range s.ents {
		if ent.Account == accountName {
			out = append(out, ent)
		}
	}
	return out, nil
}

// Delete removes a local entitlement.
func (s *MemoryEntitlementStore) Delete(_ context.Context, entitlementName string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.ents, entitlementName)
	return nil
}
