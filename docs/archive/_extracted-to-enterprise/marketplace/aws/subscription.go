package aws

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SubscriptionStatus represents the lifecycle state of a SaaS subscription.
type SubscriptionStatus string

const (
	SubscriptionActive      SubscriptionStatus = "active"
	SubscriptionPending     SubscriptionStatus = "pending"
	SubscriptionUnsubscribed SubscriptionStatus = "unsubscribed"
	SubscriptionExpired     SubscriptionStatus = "expired"
)

// Subscription represents an AWS Marketplace SaaS subscription.
type Subscription struct {
	CustomerIdentifier string             `json:"customerIdentifier"`
	ProductCode        string             `json:"productCode"`
	Status             SubscriptionStatus `json:"status"`
	SubscribedAt       time.Time          `json:"subscribedAt"`
	UnsubscribedAt     *time.Time         `json:"unsubscribedAt,omitempty"`
}

// SubscriptionStore persists subscription records.
type SubscriptionStore interface {
	// Put stores or updates a subscription.
	Put(ctx context.Context, sub Subscription) error
	// Get retrieves a subscription by customer identifier.
	Get(ctx context.Context, customerIdentifier string) (*Subscription, error)
	// List returns all subscriptions with the given status.
	List(ctx context.Context, status SubscriptionStatus) ([]Subscription, error)
	// Delete removes a subscription by customer identifier.
	Delete(ctx context.Context, customerIdentifier string) error
}

// SubscriptionManager handles the SaaS subscription lifecycle.
type SubscriptionManager struct {
	store    SubscriptionStore
	metering MeteringAPI
}

// NewSubscriptionManager creates a SubscriptionManager backed by the given store
// and metering API client.
func NewSubscriptionManager(store SubscriptionStore, metering MeteringAPI) *SubscriptionManager {
	return &SubscriptionManager{
		store:    store,
		metering: metering,
	}
}

// Subscribe processes a new subscription by resolving the registration token
// and storing the subscription record.
func (m *SubscriptionManager) Subscribe(ctx context.Context, registrationToken string) (*Subscription, error) {
	result, err := m.metering.ResolveCustomer(ctx, registrationToken)
	if err != nil {
		return nil, fmt.Errorf("resolve customer: %w", err)
	}

	sub := Subscription{
		CustomerIdentifier: result.CustomerIdentifier,
		ProductCode:        result.ProductCode,
		Status:             SubscriptionActive,
		SubscribedAt:       time.Now(),
	}

	if err := m.store.Put(ctx, sub); err != nil {
		return nil, fmt.Errorf("store subscription: %w", err)
	}

	return &sub, nil
}

// Unsubscribe marks a subscription as unsubscribed.
func (m *SubscriptionManager) Unsubscribe(ctx context.Context, customerIdentifier string) error {
	sub, err := m.store.Get(ctx, customerIdentifier)
	if err != nil {
		return fmt.Errorf("get subscription: %w", err)
	}
	if sub == nil {
		return fmt.Errorf("subscription not found: %s", customerIdentifier)
	}

	now := time.Now()
	sub.Status = SubscriptionUnsubscribed
	sub.UnsubscribedAt = &now

	if err := m.store.Put(ctx, *sub); err != nil {
		return fmt.Errorf("update subscription: %w", err)
	}

	return nil
}

// IsActive checks whether a customer has an active subscription.
func (m *SubscriptionManager) IsActive(ctx context.Context, customerIdentifier string) (bool, error) {
	sub, err := m.store.Get(ctx, customerIdentifier)
	if err != nil {
		return false, fmt.Errorf("get subscription: %w", err)
	}
	if sub == nil {
		return false, nil
	}
	return sub.Status == SubscriptionActive, nil
}

// MemorySubscriptionStore is an in-memory SubscriptionStore for testing.
type MemorySubscriptionStore struct {
	mu   sync.Mutex
	subs map[string]Subscription
}

// NewMemorySubscriptionStore creates a new in-memory subscription store.
func NewMemorySubscriptionStore() *MemorySubscriptionStore {
	return &MemorySubscriptionStore{
		subs: make(map[string]Subscription),
	}
}

// Put stores a subscription.
func (s *MemorySubscriptionStore) Put(_ context.Context, sub Subscription) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.subs[sub.CustomerIdentifier] = sub
	return nil
}

// Get retrieves a subscription by customer identifier.
func (s *MemorySubscriptionStore) Get(_ context.Context, customerIdentifier string) (*Subscription, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub, ok := s.subs[customerIdentifier]
	if !ok {
		return nil, nil
	}
	return &sub, nil
}

// List returns all subscriptions with the given status.
func (s *MemorySubscriptionStore) List(_ context.Context, status SubscriptionStatus) ([]Subscription, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Subscription
	for _, sub := range s.subs {
		if sub.Status == status {
			out = append(out, sub)
		}
	}
	return out, nil
}

// Delete removes a subscription by customer identifier.
func (s *MemorySubscriptionStore) Delete(_ context.Context, customerIdentifier string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.subs, customerIdentifier)
	return nil
}
