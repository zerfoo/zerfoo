// Package marketplace provides a unified abstraction layer for cloud marketplace
// integrations across AWS, GCP, and Azure.
package marketplace

import (
	"context"
	"errors"
	"time"
)

// CloudProvider identifies which cloud marketplace a subscription originates from.
type CloudProvider string

const (
	AWS   CloudProvider = "aws"
	GCP   CloudProvider = "gcp"
	Azure CloudProvider = "azure"
)

// Common errors returned by marketplace operations.
var (
	ErrProviderNotRegistered = errors.New("marketplace: provider not registered")
	ErrNoActiveSubscription  = errors.New("marketplace: no active subscription")
	ErrEntitlementDenied     = errors.New("marketplace: entitlement denied")
	ErrMeteringFailed        = errors.New("marketplace: metering submission failed")
)

// UsageRecord represents a single unit of metered consumption.
type UsageRecord struct {
	// CustomerID is the cloud-provider-specific customer identifier.
	CustomerID string

	// Dimension identifies what is being metered (e.g., "inference_tokens", "gpu_hours").
	Dimension string

	// Quantity is the amount consumed.
	Quantity int64

	// Timestamp is when the usage occurred.
	Timestamp time.Time
}

// Subscription represents a marketplace subscription.
type Subscription struct {
	// ID is the cloud-provider-specific subscription identifier.
	ID string

	// CustomerID is the cloud-provider-specific customer identifier.
	CustomerID string

	// Provider indicates which cloud marketplace this subscription belongs to.
	Provider CloudProvider

	// ProductCode is the marketplace product identifier.
	ProductCode string

	// Status is the current subscription state (e.g., "active", "cancelled", "expired").
	Status string

	// Entitlements lists the features or tiers the customer is entitled to.
	Entitlements []string

	// ExpiresAt is when the subscription expires, if applicable.
	ExpiresAt time.Time
}

// MeteringService submits consumption usage records to a cloud marketplace.
type MeteringService interface {
	// SubmitUsage sends a batch of usage records to the cloud provider's metering API.
	SubmitUsage(ctx context.Context, records []UsageRecord) error
}

// SubscriptionManager resolves and manages marketplace subscriptions.
type SubscriptionManager interface {
	// ResolveSubscription looks up a subscription by customer ID.
	ResolveSubscription(ctx context.Context, customerID string) (*Subscription, error)

	// IsActive returns whether the customer has an active subscription.
	IsActive(ctx context.Context, customerID string) (bool, error)
}

// EntitlementChecker verifies whether a customer is entitled to a specific feature.
type EntitlementChecker interface {
	// CheckEntitlement returns nil if the customer is entitled to the given feature,
	// or ErrEntitlementDenied otherwise.
	CheckEntitlement(ctx context.Context, customerID string, feature string) error
}

// Provider bundles all marketplace operations for a single cloud provider.
type Provider interface {
	MeteringService
	SubscriptionManager
	EntitlementChecker

	// Name returns the cloud provider identifier.
	Name() CloudProvider
}
