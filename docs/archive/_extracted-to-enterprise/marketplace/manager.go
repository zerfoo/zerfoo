package marketplace

import (
	"context"
	"sync"
)

// MultiCloudManager routes marketplace operations to the correct cloud provider
// based on subscription source. It implements MeteringService, SubscriptionManager,
// and EntitlementChecker by delegating to registered providers.
type MultiCloudManager struct {
	mu        sync.RWMutex
	providers map[CloudProvider]Provider
}

// NewMultiCloudManager creates a manager with the given providers registered.
func NewMultiCloudManager(providers ...Provider) *MultiCloudManager {
	m := &MultiCloudManager{
		providers: make(map[CloudProvider]Provider, len(providers)),
	}
	for _, p := range providers {
		m.providers[p.Name()] = p
	}
	return m
}

// Register adds or replaces a provider in the manager.
func (m *MultiCloudManager) Register(p Provider) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.providers[p.Name()] = p
}

// Provider returns the registered provider for the given cloud, or an error
// if no provider is registered.
func (m *MultiCloudManager) Provider(cloud CloudProvider) (Provider, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	p, ok := m.providers[cloud]
	if !ok {
		return nil, ErrProviderNotRegistered
	}
	return p, nil
}

// SubmitUsage routes usage records to the specified cloud provider's metering service.
func (m *MultiCloudManager) SubmitUsage(ctx context.Context, cloud CloudProvider, records []UsageRecord) error {
	p, err := m.Provider(cloud)
	if err != nil {
		return err
	}
	return p.SubmitUsage(ctx, records)
}

// ResolveSubscription looks up a subscription by customer ID on the specified provider.
func (m *MultiCloudManager) ResolveSubscription(ctx context.Context, cloud CloudProvider, customerID string) (*Subscription, error) {
	p, err := m.Provider(cloud)
	if err != nil {
		return nil, err
	}
	return p.ResolveSubscription(ctx, customerID)
}

// IsActive checks whether a customer has an active subscription on the specified provider.
func (m *MultiCloudManager) IsActive(ctx context.Context, cloud CloudProvider, customerID string) (bool, error) {
	p, err := m.Provider(cloud)
	if err != nil {
		return false, err
	}
	return p.IsActive(ctx, customerID)
}

// CheckEntitlement verifies a customer's entitlement on the specified provider.
func (m *MultiCloudManager) CheckEntitlement(ctx context.Context, cloud CloudProvider, customerID string, feature string) error {
	p, err := m.Provider(cloud)
	if err != nil {
		return err
	}
	return p.CheckEntitlement(ctx, customerID, feature)
}
