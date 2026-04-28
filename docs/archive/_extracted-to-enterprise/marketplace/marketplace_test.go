package marketplace

import (
	"context"
	"errors"
	"testing"
)

// mockProvider is a test double that implements Provider.
type mockProvider struct {
	name          CloudProvider
	usageRecords  []UsageRecord
	subscriptions map[string]*Subscription
	entitlements  map[string]map[string]bool // customerID -> feature -> allowed
	submitErr     error
}

func newMockProvider(name CloudProvider) *mockProvider {
	return &mockProvider{
		name:          name,
		subscriptions: make(map[string]*Subscription),
		entitlements:  make(map[string]map[string]bool),
	}
}

func (m *mockProvider) Name() CloudProvider { return m.name }

func (m *mockProvider) SubmitUsage(_ context.Context, records []UsageRecord) error {
	if m.submitErr != nil {
		return m.submitErr
	}
	m.usageRecords = append(m.usageRecords, records...)
	return nil
}

func (m *mockProvider) ResolveSubscription(_ context.Context, customerID string) (*Subscription, error) {
	sub, ok := m.subscriptions[customerID]
	if !ok {
		return nil, ErrNoActiveSubscription
	}
	return sub, nil
}

func (m *mockProvider) IsActive(_ context.Context, customerID string) (bool, error) {
	sub, ok := m.subscriptions[customerID]
	if !ok {
		return false, nil
	}
	return sub.Status == "active", nil
}

func (m *mockProvider) CheckEntitlement(_ context.Context, customerID string, feature string) error {
	feats, ok := m.entitlements[customerID]
	if !ok {
		return ErrEntitlementDenied
	}
	if !feats[feature] {
		return ErrEntitlementDenied
	}
	return nil
}

func TestMarketplace_AWSMetering(t *testing.T) {
	aws := newMockProvider(AWS)
	mgr := NewMultiCloudManager(aws)

	records := []UsageRecord{
		{CustomerID: "cust-001", Dimension: "inference_tokens", Quantity: 1000},
		{CustomerID: "cust-001", Dimension: "gpu_hours", Quantity: 2},
	}

	ctx := context.Background()
	if err := mgr.SubmitUsage(ctx, AWS, records); err != nil {
		t.Fatalf("SubmitUsage: unexpected error: %v", err)
	}

	if got := len(aws.usageRecords); got != 2 {
		t.Fatalf("expected 2 usage records, got %d", got)
	}
	if aws.usageRecords[0].Dimension != "inference_tokens" {
		t.Errorf("expected dimension inference_tokens, got %s", aws.usageRecords[0].Dimension)
	}
	if aws.usageRecords[1].Quantity != 2 {
		t.Errorf("expected quantity 2, got %d", aws.usageRecords[1].Quantity)
	}
}

func TestMultiCloudManager_ProviderNotRegistered(t *testing.T) {
	mgr := NewMultiCloudManager()
	ctx := context.Background()

	_, err := mgr.ResolveSubscription(ctx, GCP, "cust-001")
	if !errors.Is(err, ErrProviderNotRegistered) {
		t.Fatalf("expected ErrProviderNotRegistered, got %v", err)
	}
}

func TestMultiCloudManager_RouteToCorrectProvider(t *testing.T) {
	aws := newMockProvider(AWS)
	gcp := newMockProvider(GCP)

	aws.subscriptions["cust-aws"] = &Subscription{
		ID: "sub-1", CustomerID: "cust-aws", Provider: AWS, Status: "active",
	}
	gcp.subscriptions["cust-gcp"] = &Subscription{
		ID: "sub-2", CustomerID: "cust-gcp", Provider: GCP, Status: "active",
	}

	mgr := NewMultiCloudManager(aws, gcp)
	ctx := context.Background()

	sub, err := mgr.ResolveSubscription(ctx, AWS, "cust-aws")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sub.Provider != AWS {
		t.Errorf("expected AWS provider, got %s", sub.Provider)
	}

	sub, err = mgr.ResolveSubscription(ctx, GCP, "cust-gcp")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sub.Provider != GCP {
		t.Errorf("expected GCP provider, got %s", sub.Provider)
	}

	// Cross-provider should not find a subscription.
	_, err = mgr.ResolveSubscription(ctx, AWS, "cust-gcp")
	if !errors.Is(err, ErrNoActiveSubscription) {
		t.Errorf("expected ErrNoActiveSubscription for cross-provider lookup, got %v", err)
	}
}

func TestMultiCloudManager_Entitlements(t *testing.T) {
	azure := newMockProvider(Azure)
	azure.entitlements["cust-az"] = map[string]bool{
		"inference": true,
		"training":  false,
	}

	mgr := NewMultiCloudManager(azure)
	ctx := context.Background()

	if err := mgr.CheckEntitlement(ctx, Azure, "cust-az", "inference"); err != nil {
		t.Fatalf("expected entitlement granted, got error: %v", err)
	}

	if err := mgr.CheckEntitlement(ctx, Azure, "cust-az", "training"); !errors.Is(err, ErrEntitlementDenied) {
		t.Errorf("expected ErrEntitlementDenied for training, got %v", err)
	}
}

func TestMultiCloudManager_Register(t *testing.T) {
	mgr := NewMultiCloudManager()

	_, err := mgr.Provider(AWS)
	if !errors.Is(err, ErrProviderNotRegistered) {
		t.Fatalf("expected ErrProviderNotRegistered before register, got %v", err)
	}

	aws := newMockProvider(AWS)
	mgr.Register(aws)

	p, err := mgr.Provider(AWS)
	if err != nil {
		t.Fatalf("unexpected error after register: %v", err)
	}
	if p.Name() != AWS {
		t.Errorf("expected AWS, got %s", p.Name())
	}
}

func TestUsageTracker_RecordAndFlush(t *testing.T) {
	tracker := NewUsageTracker()
	tracker.Record("cust-001", "inference_tokens", 500)
	tracker.Record("cust-001", "inference_tokens", 300)
	tracker.Record("cust-002", "gpu_hours", 1)

	if got := tracker.Pending(); got != 3 {
		t.Fatalf("expected 3 pending, got %d", got)
	}

	// Check aggregation.
	agg := tracker.Aggregate()
	if agg["cust-001"]["inference_tokens"] != 800 {
		t.Errorf("expected 800 inference_tokens for cust-001, got %d", agg["cust-001"]["inference_tokens"])
	}
	if agg["cust-002"]["gpu_hours"] != 1 {
		t.Errorf("expected 1 gpu_hours for cust-002, got %d", agg["cust-002"]["gpu_hours"])
	}

	// Flush to a mock metering service.
	mock := newMockProvider(AWS)
	ctx := context.Background()
	n, err := tracker.Flush(ctx, mock)
	if err != nil {
		t.Fatalf("Flush error: %v", err)
	}
	if n != 3 {
		t.Errorf("expected 3 flushed, got %d", n)
	}
	if tracker.Pending() != 0 {
		t.Errorf("expected 0 pending after flush, got %d", tracker.Pending())
	}
}

func TestUsageTracker_FlushRequeuesOnError(t *testing.T) {
	tracker := NewUsageTracker()
	tracker.Record("cust-001", "tokens", 100)

	mock := newMockProvider(AWS)
	mock.submitErr = ErrMeteringFailed

	ctx := context.Background()
	n, err := tracker.Flush(ctx, mock)
	if !errors.Is(err, ErrMeteringFailed) {
		t.Fatalf("expected ErrMeteringFailed, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 flushed on error, got %d", n)
	}
	if tracker.Pending() != 1 {
		t.Errorf("expected 1 pending after failed flush, got %d", tracker.Pending())
	}
}

func TestUsageTracker_FlushEmpty(t *testing.T) {
	tracker := NewUsageTracker()
	mock := newMockProvider(AWS)

	n, err := tracker.Flush(context.Background(), mock)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 flushed, got %d", n)
	}
}
