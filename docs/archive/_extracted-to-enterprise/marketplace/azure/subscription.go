package azure

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SubscriptionStore persists Azure SaaS subscription records.
type SubscriptionStore interface {
	// Put stores or updates a subscription.
	Put(ctx context.Context, sub SaaSSubscription) error
	// Get retrieves a subscription by ID.
	Get(ctx context.Context, subscriptionID string) (*SaaSSubscription, error)
	// List returns all subscriptions with the given status.
	List(ctx context.Context, status SaaSStatus) ([]SaaSSubscription, error)
	// Delete removes a subscription by ID.
	Delete(ctx context.Context, subscriptionID string) error
}

// SubscriptionManager handles the Azure SaaS subscription lifecycle
// using the Fulfillment API and a local subscription store.
type SubscriptionManager struct {
	store       SubscriptionStore
	fulfillment FulfillmentAPI
}

// NewSubscriptionManager creates a SubscriptionManager backed by the given
// store and fulfillment API client.
func NewSubscriptionManager(store SubscriptionStore, fulfillment FulfillmentAPI) *SubscriptionManager {
	return &SubscriptionManager{
		store:       store,
		fulfillment: fulfillment,
	}
}

// ResolveAndActivate resolves a marketplace purchase token, activates the
// subscription with the specified plan, and stores the subscription locally.
func (m *SubscriptionManager) ResolveAndActivate(ctx context.Context, marketplaceToken string, plan PlanDetails) (*SaaSSubscription, error) {
	resolved, err := m.fulfillment.Resolve(ctx, marketplaceToken)
	if err != nil {
		return nil, fmt.Errorf("resolve subscription: %w", err)
	}

	if err := m.fulfillment.Activate(ctx, resolved.ID, plan); err != nil {
		return nil, fmt.Errorf("activate subscription %s: %w", resolved.ID, err)
	}

	sub := SaaSSubscription{
		ID:               resolved.ID,
		SubscriptionName: resolved.SubscriptionName,
		OfferID:          resolved.OfferID,
		PlanID:           plan.PlanID,
		Quantity:         plan.Quantity,
		Status:           SaaSStatusSubscribed,
		Created:          time.Now(),
	}

	if err := m.store.Put(ctx, sub); err != nil {
		return nil, fmt.Errorf("store subscription: %w", err)
	}

	return &sub, nil
}

// Suspend suspends a subscription both in Azure and in the local store.
func (m *SubscriptionManager) Suspend(ctx context.Context, subscriptionID string) error {
	sub, err := m.store.Get(ctx, subscriptionID)
	if err != nil {
		return fmt.Errorf("get subscription: %w", err)
	}
	if sub == nil {
		return fmt.Errorf("subscription not found: %s", subscriptionID)
	}

	sub.Status = SaaSStatusSuspended
	if err := m.store.Put(ctx, *sub); err != nil {
		return fmt.Errorf("update subscription: %w", err)
	}
	return nil
}

// Reinstate changes a suspended subscription back to subscribed.
func (m *SubscriptionManager) Reinstate(ctx context.Context, subscriptionID string) error {
	sub, err := m.store.Get(ctx, subscriptionID)
	if err != nil {
		return fmt.Errorf("get subscription: %w", err)
	}
	if sub == nil {
		return fmt.Errorf("subscription not found: %s", subscriptionID)
	}
	if sub.Status != SaaSStatusSuspended {
		return fmt.Errorf("subscription %s is not suspended (status: %s)", subscriptionID, sub.Status)
	}

	sub.Status = SaaSStatusSubscribed
	if err := m.store.Put(ctx, *sub); err != nil {
		return fmt.Errorf("update subscription: %w", err)
	}
	return nil
}

// Unsubscribe marks a subscription as unsubscribed and deletes it in Azure.
func (m *SubscriptionManager) Unsubscribe(ctx context.Context, subscriptionID string) error {
	sub, err := m.store.Get(ctx, subscriptionID)
	if err != nil {
		return fmt.Errorf("get subscription: %w", err)
	}
	if sub == nil {
		return fmt.Errorf("subscription not found: %s", subscriptionID)
	}

	sub.Status = SaaSStatusUnsubscribed
	if err := m.store.Put(ctx, *sub); err != nil {
		return fmt.Errorf("update subscription: %w", err)
	}
	return nil
}

// IsActive checks whether a subscription is in the Subscribed state.
func (m *SubscriptionManager) IsActive(ctx context.Context, subscriptionID string) (bool, error) {
	sub, err := m.store.Get(ctx, subscriptionID)
	if err != nil {
		return false, fmt.Errorf("get subscription: %w", err)
	}
	if sub == nil {
		return false, nil
	}
	return sub.Status == SaaSStatusSubscribed, nil
}

// ChangePlan updates the subscription's plan via the Fulfillment API.
func (m *SubscriptionManager) ChangePlan(ctx context.Context, subscriptionID, newPlanID string) (*OperationLocation, error) {
	op, err := m.fulfillment.UpdateSubscription(ctx, subscriptionID, SubscriptionUpdate{PlanID: newPlanID})
	if err != nil {
		return nil, fmt.Errorf("change plan: %w", err)
	}

	sub, err := m.store.Get(ctx, subscriptionID)
	if err != nil {
		return op, fmt.Errorf("get subscription for plan update: %w", err)
	}
	if sub != nil {
		sub.PlanID = newPlanID
		if err := m.store.Put(ctx, *sub); err != nil {
			return op, fmt.Errorf("update local plan: %w", err)
		}
	}

	return op, nil
}

// ChangeQuantity updates the subscription's seat quantity via the Fulfillment API.
func (m *SubscriptionManager) ChangeQuantity(ctx context.Context, subscriptionID string, quantity int) (*OperationLocation, error) {
	op, err := m.fulfillment.UpdateSubscription(ctx, subscriptionID, SubscriptionUpdate{Quantity: quantity})
	if err != nil {
		return nil, fmt.Errorf("change quantity: %w", err)
	}

	sub, err := m.store.Get(ctx, subscriptionID)
	if err != nil {
		return op, fmt.Errorf("get subscription for quantity update: %w", err)
	}
	if sub != nil {
		sub.Quantity = quantity
		if err := m.store.Put(ctx, *sub); err != nil {
			return op, fmt.Errorf("update local quantity: %w", err)
		}
	}

	return op, nil
}

// MemorySubscriptionStore is an in-memory SubscriptionStore for testing.
type MemorySubscriptionStore struct {
	mu   sync.Mutex
	subs map[string]SaaSSubscription
}

// NewMemorySubscriptionStore creates a new in-memory subscription store.
func NewMemorySubscriptionStore() *MemorySubscriptionStore {
	return &MemorySubscriptionStore{
		subs: make(map[string]SaaSSubscription),
	}
}

// Put stores a subscription.
func (s *MemorySubscriptionStore) Put(_ context.Context, sub SaaSSubscription) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.subs[sub.ID] = sub
	return nil
}

// Get retrieves a subscription by ID.
func (s *MemorySubscriptionStore) Get(_ context.Context, subscriptionID string) (*SaaSSubscription, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[subscriptionID]
	if !ok {
		return nil, nil
	}
	return &sub, nil
}

// List returns all subscriptions with the given status.
func (s *MemorySubscriptionStore) List(_ context.Context, status SaaSStatus) ([]SaaSSubscription, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []SaaSSubscription
	for _, sub := range s.subs {
		if sub.Status == status {
			out = append(out, sub)
		}
	}
	return out, nil
}

// Delete removes a subscription by ID.
func (s *MemorySubscriptionStore) Delete(_ context.Context, subscriptionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.subs, subscriptionID)
	return nil
}
