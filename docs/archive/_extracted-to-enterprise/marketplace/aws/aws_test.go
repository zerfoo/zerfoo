package aws

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/marketplace"
)

// mockMeteringAPI is a test double for MeteringAPI.
type mockMeteringAPI struct {
	resolveFunc    func(ctx context.Context, token string) (*ResolveCustomerOutput, error)
	batchMeterFunc func(ctx context.Context, input *BatchMeterUsageInput) (*BatchMeterUsageOutput, error)
	meterFunc      func(ctx context.Context, input *MeterUsageInput) (*MeterUsageOutput, error)
}

func (m *mockMeteringAPI) ResolveCustomer(ctx context.Context, token string) (*ResolveCustomerOutput, error) {
	if m.resolveFunc != nil {
		return m.resolveFunc(ctx, token)
	}
	return &ResolveCustomerOutput{CustomerIdentifier: "cust-123", ProductCode: "prod-abc"}, nil
}

func (m *mockMeteringAPI) BatchMeterUsage(ctx context.Context, input *BatchMeterUsageInput) (*BatchMeterUsageOutput, error) {
	if m.batchMeterFunc != nil {
		return m.batchMeterFunc(ctx, input)
	}
	var results []UsageRecordResult
	for _, r := range input.UsageRecords {
		results = append(results, UsageRecordResult{
			UsageRecord:    r,
			MeteringStatus: "Success",
		})
	}
	return &BatchMeterUsageOutput{Results: results}, nil
}

func (m *mockMeteringAPI) MeterUsage(ctx context.Context, input *MeterUsageInput) (*MeterUsageOutput, error) {
	if m.meterFunc != nil {
		return m.meterFunc(ctx, input)
	}
	return &MeterUsageOutput{MeteringRecordID: "record-001"}, nil
}

func TestMeteringClient_ResolveCustomer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Amz-Target") != "AWSMPMeteringService.ResolveCustomer" {
			t.Errorf("unexpected target: %s", r.Header.Get("X-Amz-Target"))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ResolveCustomerOutput{
			CustomerIdentifier: "cust-456",
			ProductCode:        "prod-xyz",
		})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	out, err := client.ResolveCustomer(context.Background(), "token-abc")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.CustomerIdentifier != "cust-456" {
		t.Errorf("got customer %q, want %q", out.CustomerIdentifier, "cust-456")
	}
	if out.ProductCode != "prod-xyz" {
		t.Errorf("got product %q, want %q", out.ProductCode, "prod-xyz")
	}
}

func TestMeteringClient_BatchMeterUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Amz-Target") != "AWSMPMeteringService.BatchMeterUsage" {
			t.Errorf("unexpected target: %s", r.Header.Get("X-Amz-Target"))
		}
		var input BatchMeterUsageInput
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		var results []UsageRecordResult
		for _, rec := range input.UsageRecords {
			results = append(results, UsageRecordResult{
				UsageRecord:    rec,
				MeteringStatus: "Success",
			})
		}
		json.NewEncoder(w).Encode(BatchMeterUsageOutput{Results: results})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	out, err := client.BatchMeterUsage(context.Background(), &BatchMeterUsageInput{
		ProductCode: "prod-xyz",
		UsageRecords: []UsageRecord{
			{CustomerIdentifier: "cust-1", Dimension: "tokens_1m", Quantity: 5, Timestamp: time.Now()},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(out.Results))
	}
	if out.Results[0].MeteringStatus != "Success" {
		t.Errorf("got status %q, want %q", out.Results[0].MeteringStatus, "Success")
	}
}

func TestMeteringClient_MeterUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(MeterUsageOutput{MeteringRecordID: "rec-789"})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	out, err := client.MeterUsage(context.Background(), &MeterUsageInput{
		ProductCode: "prod-xyz",
		Dimension:   "tokens_1m",
		Quantity:    10,
		Timestamp:   time.Now(),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.MeteringRecordID != "rec-789" {
		t.Errorf("got record ID %q, want %q", out.MeteringRecordID, "rec-789")
	}
}

func TestMeteringClient_BatchMeterUsage_RetryOn429(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte("throttled"))
			return
		}
		var input BatchMeterUsageInput
		json.NewDecoder(r.Body).Decode(&input)
		var results []UsageRecordResult
		for _, rec := range input.UsageRecords {
			results = append(results, UsageRecordResult{
				UsageRecord:    rec,
				MeteringStatus: "Success",
			})
		}
		json.NewEncoder(w).Encode(BatchMeterUsageOutput{Results: results})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	client.Retry = marketplace.RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}

	out, err := client.BatchMeterUsage(context.Background(), &BatchMeterUsageInput{
		ProductCode:  "prod-xyz",
		UsageRecords: []UsageRecord{{CustomerIdentifier: "cust-1", Dimension: "tokens_1m", Quantity: 5, Timestamp: time.Now()}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(out.Results))
	}
	if got := attempts.Load(); got != 3 {
		t.Errorf("got %d attempts, want 3", got)
	}
}

func TestMeteringClient_MeterUsage_RetryOn429(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte("throttled"))
			return
		}
		json.NewEncoder(w).Encode(MeterUsageOutput{MeteringRecordID: "rec-retry"})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	client.Retry = marketplace.RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}

	out, err := client.MeterUsage(context.Background(), &MeterUsageInput{
		ProductCode: "prod-xyz",
		Dimension:   "tokens_1m",
		Quantity:    10,
		Timestamp:   time.Now(),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.MeteringRecordID != "rec-retry" {
		t.Errorf("got record ID %q, want %q", out.MeteringRecordID, "rec-retry")
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("got %d attempts, want 2", got)
	}
}

func TestSubscription_Lifecycle(t *testing.T) {
	store := NewMemorySubscriptionStore()
	metering := &mockMeteringAPI{}
	mgr := NewSubscriptionManager(store, metering)
	ctx := context.Background()

	// Subscribe.
	sub, err := mgr.Subscribe(ctx, "registration-token-1")
	if err != nil {
		t.Fatalf("subscribe: %v", err)
	}
	if sub.CustomerIdentifier != "cust-123" {
		t.Errorf("got customer %q, want %q", sub.CustomerIdentifier, "cust-123")
	}
	if sub.Status != SubscriptionActive {
		t.Errorf("got status %q, want %q", sub.Status, SubscriptionActive)
	}

	// Check active.
	active, err := mgr.IsActive(ctx, "cust-123")
	if err != nil {
		t.Fatalf("is active: %v", err)
	}
	if !active {
		t.Error("expected subscription to be active")
	}

	// Unsubscribe.
	if err := mgr.Unsubscribe(ctx, "cust-123"); err != nil {
		t.Fatalf("unsubscribe: %v", err)
	}

	active, err = mgr.IsActive(ctx, "cust-123")
	if err != nil {
		t.Fatalf("is active after unsubscribe: %v", err)
	}
	if active {
		t.Error("expected subscription to be inactive after unsubscribe")
	}
}

func TestSubscription_UnsubscribeNotFound(t *testing.T) {
	store := NewMemorySubscriptionStore()
	metering := &mockMeteringAPI{}
	mgr := NewSubscriptionManager(store, metering)

	err := mgr.Unsubscribe(context.Background(), "nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent subscription")
	}
}

func TestEntitlement_CheckAndCache(t *testing.T) {
	store := NewMemoryEntitlementStore()
	checker := NewEntitlementChecker(store, 5*time.Minute)
	ctx := context.Background()

	// No entitlement yet.
	ok, err := checker.IsEntitled(ctx, "cust-1", "tokens_1m")
	if err != nil {
		t.Fatalf("check entitlement: %v", err)
	}
	if ok {
		t.Error("expected not entitled before grant")
	}

	// Grant entitlement.
	err = checker.GrantEntitlement(ctx, Entitlement{
		CustomerIdentifier: "cust-1",
		ProductCode:        "prod-abc",
		Dimension:          "tokens_1m",
		Value:              1000,
		ExpiresAt:          time.Now().Add(24 * time.Hour),
	})
	if err != nil {
		t.Fatalf("grant entitlement: %v", err)
	}

	// Now entitled.
	ok, err = checker.IsEntitled(ctx, "cust-1", "tokens_1m")
	if err != nil {
		t.Fatalf("check entitlement: %v", err)
	}
	if !ok {
		t.Error("expected entitled after grant")
	}

	// Revoke.
	if err := checker.RevokeEntitlement(ctx, "cust-1", "tokens_1m"); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	ok, err = checker.IsEntitled(ctx, "cust-1", "tokens_1m")
	if err != nil {
		t.Fatalf("check after revoke: %v", err)
	}
	if ok {
		t.Error("expected not entitled after revoke")
	}
}

func TestEntitlement_Expired(t *testing.T) {
	store := NewMemoryEntitlementStore()
	checker := NewEntitlementChecker(store, 5*time.Minute)
	ctx := context.Background()

	// Grant expired entitlement.
	err := checker.GrantEntitlement(ctx, Entitlement{
		CustomerIdentifier: "cust-1",
		ProductCode:        "prod-abc",
		Dimension:          "tokens_1m",
		Value:              1000,
		ExpiresAt:          time.Now().Add(-1 * time.Hour),
	})
	if err != nil {
		t.Fatalf("grant: %v", err)
	}

	ok, err := checker.IsEntitled(ctx, "cust-1", "tokens_1m")
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if ok {
		t.Error("expected not entitled for expired entitlement")
	}
}

func TestTokenBilling_RecordAndFlush(t *testing.T) {
	metering := &mockMeteringAPI{}
	tracker := NewTokenBillingTracker("prod-abc", metering)

	// Record usage.
	tracker.RecordUsage("cust-1", 500_000, 500_000) // 1M total
	tracker.RecordUsage("cust-1", 300_000, 200_000) // +500K
	tracker.RecordUsage("cust-2", 2_000_000, 1_000_000) // 3M total

	// Check snapshot.
	snap := tracker.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("got %d snapshot records, want 2", len(snap))
	}

	// Flush.
	out, err := tracker.Flush(context.Background())
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	if len(out.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(out.Results))
	}

	// Verify quantities (cust-1: 1.5M => 2 units, cust-2: 3M => 3 units).
	quantities := make(map[string]int)
	for _, r := range out.Results {
		quantities[r.UsageRecord.CustomerIdentifier] = r.UsageRecord.Quantity
	}
	if quantities["cust-1"] != 2 {
		t.Errorf("cust-1 quantity: got %d, want 2", quantities["cust-1"])
	}
	if quantities["cust-2"] != 3 {
		t.Errorf("cust-2 quantity: got %d, want 3", quantities["cust-2"])
	}

	// After flush, snapshot should be empty.
	snap = tracker.Snapshot()
	if len(snap) != 0 {
		t.Errorf("got %d snapshot records after flush, want 0", len(snap))
	}
}

func TestTokenBilling_FlushEmpty(t *testing.T) {
	metering := &mockMeteringAPI{}
	tracker := NewTokenBillingTracker("prod-abc", metering)

	out, err := tracker.Flush(context.Background())
	if err != nil {
		t.Fatalf("flush empty: %v", err)
	}
	if len(out.Results) != 0 {
		t.Errorf("got %d results for empty flush, want 0", len(out.Results))
	}
}

func TestMemorySubscriptionStore_List(t *testing.T) {
	store := NewMemorySubscriptionStore()
	ctx := context.Background()

	store.Put(ctx, Subscription{CustomerIdentifier: "a", Status: SubscriptionActive})
	store.Put(ctx, Subscription{CustomerIdentifier: "b", Status: SubscriptionUnsubscribed})
	store.Put(ctx, Subscription{CustomerIdentifier: "c", Status: SubscriptionActive})

	active, err := store.List(ctx, SubscriptionActive)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(active) != 2 {
		t.Errorf("got %d active, want 2", len(active))
	}
}

func TestMemoryEntitlementStore_List(t *testing.T) {
	store := NewMemoryEntitlementStore()
	ctx := context.Background()

	store.Put(ctx, Entitlement{CustomerIdentifier: "cust-1", Dimension: "tokens_1m", Value: 100})
	store.Put(ctx, Entitlement{CustomerIdentifier: "cust-1", Dimension: "storage_gb", Value: 50})
	store.Put(ctx, Entitlement{CustomerIdentifier: "cust-2", Dimension: "tokens_1m", Value: 200})

	ents, err := store.List(ctx, "cust-1")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(ents) != 2 {
		t.Errorf("got %d entitlements, want 2", len(ents))
	}
}
