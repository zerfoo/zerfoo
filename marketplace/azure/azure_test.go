package azure

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/marketplace"
)

// mockFulfillmentAPI is a test double for FulfillmentAPI.
type mockFulfillmentAPI struct {
	resolveFunc      func(ctx context.Context, token string) (*ResolvedSubscription, error)
	activateFunc     func(ctx context.Context, id string, plan PlanDetails) error
	getSubFunc       func(ctx context.Context, id string) (*SaaSSubscription, error)
	updateSubFunc    func(ctx context.Context, id string, update SubscriptionUpdate) (*OperationLocation, error)
	suspendFunc      func(ctx context.Context, id string) error
	deleteFunc       func(ctx context.Context, id string) error
	listSubsFunc     func(ctx context.Context) ([]SaaSSubscription, error)
}

func (m *mockFulfillmentAPI) Resolve(ctx context.Context, token string) (*ResolvedSubscription, error) {
	if m.resolveFunc != nil {
		return m.resolveFunc(ctx, token)
	}
	return &ResolvedSubscription{
		ID:               "sub-123",
		SubscriptionName: "Test Subscription",
		OfferID:          "zerfoo-cloud",
		PlanID:           "basic",
	}, nil
}

func (m *mockFulfillmentAPI) Activate(ctx context.Context, id string, plan PlanDetails) error {
	if m.activateFunc != nil {
		return m.activateFunc(ctx, id, plan)
	}
	return nil
}

func (m *mockFulfillmentAPI) GetSubscription(ctx context.Context, id string) (*SaaSSubscription, error) {
	if m.getSubFunc != nil {
		return m.getSubFunc(ctx, id)
	}
	return &SaaSSubscription{ID: id, Status: SaaSStatusSubscribed}, nil
}

func (m *mockFulfillmentAPI) UpdateSubscription(ctx context.Context, id string, update SubscriptionUpdate) (*OperationLocation, error) {
	if m.updateSubFunc != nil {
		return m.updateSubFunc(ctx, id, update)
	}
	return &OperationLocation{Location: "https://example.com/operations/op-1"}, nil
}

func (m *mockFulfillmentAPI) Suspend(ctx context.Context, id string) error {
	if m.suspendFunc != nil {
		return m.suspendFunc(ctx, id)
	}
	return nil
}

func (m *mockFulfillmentAPI) Delete(ctx context.Context, id string) error {
	if m.deleteFunc != nil {
		return m.deleteFunc(ctx, id)
	}
	return nil
}

func (m *mockFulfillmentAPI) ListSubscriptions(ctx context.Context) ([]SaaSSubscription, error) {
	if m.listSubsFunc != nil {
		return m.listSubsFunc(ctx)
	}
	return nil, nil
}

// mockMeteringAPI is a test double for MeteringAPI.
type mockMeteringAPI struct {
	postUsageFunc      func(ctx context.Context, event UsageEvent) (*UsageEventResult, error)
	postBatchFunc      func(ctx context.Context, events []UsageEvent) (*BatchUsageResult, error)
}

func (m *mockMeteringAPI) PostUsageEvent(ctx context.Context, event UsageEvent) (*UsageEventResult, error) {
	if m.postUsageFunc != nil {
		return m.postUsageFunc(ctx, event)
	}
	return &UsageEventResult{
		UsageEventID: "evt-001",
		Status:       MeteringStatusAccepted,
		Quantity:     event.Quantity,
		Dimension:    event.Dimension,
	}, nil
}

func (m *mockMeteringAPI) PostBatchUsageEvent(ctx context.Context, events []UsageEvent) (*BatchUsageResult, error) {
	if m.postBatchFunc != nil {
		return m.postBatchFunc(ctx, events)
	}
	var results []UsageEventResult
	for _, e := range events {
		results = append(results, UsageEventResult{
			UsageEventID: "evt-batch",
			Status:       MeteringStatusAccepted,
			Quantity:     e.Quantity,
			Dimension:    e.Dimension,
		})
	}
	return &BatchUsageResult{Results: results, Count: len(results)}, nil
}

// --- Fulfillment Client Tests ---

func TestFulfillmentClient_Resolve(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.Contains(r.URL.Path, "/saas/subscriptions/resolve") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Header.Get("x-ms-marketplace-token") == "" {
			t.Error("missing marketplace token header")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ResolvedSubscription{
			ID:               "sub-456",
			SubscriptionName: "Resolved Sub",
			OfferID:          "zerfoo-cloud",
			PlanID:           "pro",
		})
	}))
	defer srv.Close()

	client := NewFulfillmentClient(srv.URL, nil)
	out, err := client.Resolve(context.Background(), "test-token")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.ID != "sub-456" {
		t.Errorf("got ID %q, want %q", out.ID, "sub-456")
	}
	if out.PlanID != "pro" {
		t.Errorf("got plan %q, want %q", out.PlanID, "pro")
	}
}

func TestFulfillmentClient_Activate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.Contains(r.URL.Path, "/activate") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := NewFulfillmentClient(srv.URL, nil)
	err := client.Activate(context.Background(), "sub-123", PlanDetails{PlanID: "basic"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFulfillmentClient_GetSubscription(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		json.NewEncoder(w).Encode(SaaSSubscription{
			ID:     "sub-789",
			Status: SaaSStatusSubscribed,
			PlanID: "enterprise",
		})
	}))
	defer srv.Close()

	client := NewFulfillmentClient(srv.URL, nil)
	out, err := client.GetSubscription(context.Background(), "sub-789")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Status != SaaSStatusSubscribed {
		t.Errorf("got status %q, want %q", out.Status, SaaSStatusSubscribed)
	}
}

func TestFulfillmentClient_Delete(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			t.Errorf("expected DELETE, got %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := NewFulfillmentClient(srv.URL, nil)
	err := client.Delete(context.Background(), "sub-123")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFulfillmentClient_ListSubscriptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(struct {
			Subscriptions []SaaSSubscription `json:"subscriptions"`
		}{
			Subscriptions: []SaaSSubscription{
				{ID: "sub-1", Status: SaaSStatusSubscribed},
				{ID: "sub-2", Status: SaaSStatusSuspended},
			},
		})
	}))
	defer srv.Close()

	client := NewFulfillmentClient(srv.URL, nil)
	subs, err := client.ListSubscriptions(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(subs) != 2 {
		t.Fatalf("got %d subscriptions, want 2", len(subs))
	}
}

func TestFulfillmentClient_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte("access denied"))
	}))
	defer srv.Close()

	client := NewFulfillmentClient(srv.URL, nil)
	_, err := client.Resolve(context.Background(), "bad-token")
	if err == nil {
		t.Fatal("expected error for 403 response")
	}
	if !strings.Contains(err.Error(), "403") {
		t.Errorf("error should mention status 403: %v", err)
	}
}

// --- Metering Client Tests ---

func TestMeteringClient_PostUsageEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.Contains(r.URL.Path, "/usageEvent") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		json.NewEncoder(w).Encode(UsageEventResult{
			UsageEventID: "evt-123",
			Status:       MeteringStatusAccepted,
			Dimension:    "tokens_1m",
			Quantity:     5,
		})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	out, err := client.PostUsageEvent(context.Background(), UsageEvent{
		ResourceID: "sub-1",
		Quantity:   5,
		Dimension:  "tokens_1m",
		PlanID:     "basic",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Status != MeteringStatusAccepted {
		t.Errorf("got status %q, want %q", out.Status, MeteringStatusAccepted)
	}
}

func TestMeteringClient_PostBatchUsageEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(BatchUsageResult{
			Results: []UsageEventResult{
				{UsageEventID: "evt-1", Status: MeteringStatusAccepted},
				{UsageEventID: "evt-2", Status: MeteringStatusAccepted},
			},
			Count: 2,
		})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	out, err := client.PostBatchUsageEvent(context.Background(), []UsageEvent{
		{ResourceID: "sub-1", Quantity: 3, Dimension: "tokens_1m"},
		{ResourceID: "sub-2", Quantity: 7, Dimension: "tokens_1m"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Count != 2 {
		t.Errorf("got count %d, want 2", out.Count)
	}
}

func TestMeteringClient_PostUsageEvent_RetryOn429(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte("throttled"))
			return
		}
		json.NewEncoder(w).Encode(UsageEventResult{
			UsageEventID: "evt-retry",
			Status:       MeteringStatusAccepted,
			Dimension:    "tokens_1m",
			Quantity:     5,
		})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	client.Retry = marketplace.RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}

	out, err := client.PostUsageEvent(context.Background(), UsageEvent{
		ResourceID: "sub-1",
		Quantity:   5,
		Dimension:  "tokens_1m",
		PlanID:     "basic",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.UsageEventID != "evt-retry" {
		t.Errorf("got event ID %q, want %q", out.UsageEventID, "evt-retry")
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("got %d attempts, want 2", got)
	}
}

func TestMeteringClient_PostBatchUsageEvent_RetryOn429(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte("throttled"))
			return
		}
		json.NewEncoder(w).Encode(BatchUsageResult{
			Results: []UsageEventResult{{UsageEventID: "evt-retry", Status: MeteringStatusAccepted}},
			Count:   1,
		})
	}))
	defer srv.Close()

	client := NewMeteringClient(srv.URL, nil)
	client.Retry = marketplace.RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}

	out, err := client.PostBatchUsageEvent(context.Background(), []UsageEvent{
		{ResourceID: "sub-1", Quantity: 3, Dimension: "tokens_1m"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Count != 1 {
		t.Errorf("got count %d, want 1", out.Count)
	}
	if got := attempts.Load(); got != 3 {
		t.Errorf("got %d attempts, want 3", got)
	}
}

// --- Subscription Manager Tests ---

func TestSubscription_ResolveAndActivate(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	sub, err := mgr.ResolveAndActivate(ctx, "marketplace-token-1", PlanDetails{PlanID: "basic"})
	if err != nil {
		t.Fatalf("resolve and activate: %v", err)
	}
	if sub.ID != "sub-123" {
		t.Errorf("got ID %q, want %q", sub.ID, "sub-123")
	}
	if sub.Status != SaaSStatusSubscribed {
		t.Errorf("got status %q, want %q", sub.Status, SaaSStatusSubscribed)
	}

	active, err := mgr.IsActive(ctx, "sub-123")
	if err != nil {
		t.Fatalf("is active: %v", err)
	}
	if !active {
		t.Error("expected subscription to be active")
	}
}

func TestSubscription_SuspendAndReinstate(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	// Create subscription.
	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	// Suspend.
	if err := mgr.Suspend(ctx, "sub-123"); err != nil {
		t.Fatalf("suspend: %v", err)
	}
	active, _ := mgr.IsActive(ctx, "sub-123")
	if active {
		t.Error("expected subscription to be suspended")
	}

	// Reinstate.
	if err := mgr.Reinstate(ctx, "sub-123"); err != nil {
		t.Fatalf("reinstate: %v", err)
	}
	active, _ = mgr.IsActive(ctx, "sub-123")
	if !active {
		t.Error("expected subscription to be active after reinstate")
	}
}

func TestSubscription_Unsubscribe(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	if err := mgr.Unsubscribe(ctx, "sub-123"); err != nil {
		t.Fatalf("unsubscribe: %v", err)
	}

	active, _ := mgr.IsActive(ctx, "sub-123")
	if active {
		t.Error("expected subscription to be inactive after unsubscribe")
	}
}

func TestSubscription_NotFound(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)

	err := mgr.Suspend(context.Background(), "nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent subscription")
	}
}

func TestSubscription_ReinstateNotSuspended(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	err := mgr.Reinstate(ctx, "sub-123")
	if err == nil {
		t.Fatal("expected error when reinstating non-suspended subscription")
	}
}

func TestSubscription_ChangePlan(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	op, err := mgr.ChangePlan(ctx, "sub-123", "enterprise")
	if err != nil {
		t.Fatalf("change plan: %v", err)
	}
	if op.Location == "" {
		t.Error("expected operation location")
	}

	// Verify plan updated in store.
	sub, _ := store.Get(ctx, "sub-123")
	if sub.PlanID != "enterprise" {
		t.Errorf("got plan %q, want %q", sub.PlanID, "enterprise")
	}
}

func TestSubscription_ChangeQuantity(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic", Quantity: 5})

	op, err := mgr.ChangeQuantity(ctx, "sub-123", 10)
	if err != nil {
		t.Fatalf("change quantity: %v", err)
	}
	if op.Location == "" {
		t.Error("expected operation location")
	}

	sub, _ := store.Get(ctx, "sub-123")
	if sub.Quantity != 10 {
		t.Errorf("got quantity %d, want 10", sub.Quantity)
	}
}

// --- Token Billing Tests ---

func TestTokenBilling_RecordAndFlush(t *testing.T) {
	metering := &mockMeteringAPI{}
	tracker := NewTokenBillingTracker("basic", metering)

	tracker.RecordUsage("sub-1", 500_000, 500_000) // 1M total
	tracker.RecordUsage("sub-1", 300_000, 200_000) // +500K
	tracker.RecordUsage("sub-2", 2_000_000, 1_000_000) // 3M total

	snap := tracker.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("got %d snapshot records, want 2", len(snap))
	}

	out, err := tracker.Flush(context.Background())
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	if out.Count != 2 {
		t.Fatalf("got count %d, want 2", out.Count)
	}

	// After flush, snapshot should be empty.
	snap = tracker.Snapshot()
	if len(snap) != 0 {
		t.Errorf("got %d snapshot records after flush, want 0", len(snap))
	}
}

func TestTokenBilling_FlushEmpty(t *testing.T) {
	metering := &mockMeteringAPI{}
	tracker := NewTokenBillingTracker("basic", metering)

	out, err := tracker.Flush(context.Background())
	if err != nil {
		t.Fatalf("flush empty: %v", err)
	}
	if len(out.Results) != 0 {
		t.Errorf("got %d results for empty flush, want 0", len(out.Results))
	}
}

// --- Webhook Tests ---

const testWebhookSecret = "test-secret-key"

// signWebhookBody computes the HMAC-SHA256 signature for a webhook body.
func signWebhookBody(secret string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}

// signedWebhookRequest creates a signed POST request for webhook testing.
func signedWebhookRequest(secret string, body []byte) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/webhook", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-ms-signature", signWebhookBody(secret, body))
	return req
}

func TestWebhook_SuspendEvent(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	handler := NewWebhookHandler(testWebhookSecret, mgr)

	payload := WebhookPayload{
		ID:             "evt-1",
		SubscriptionID: "sub-123",
		OperationID:    "op-suspend-1",
		Action:         ActionSuspend,
		Status:         WebhookStatusSuccess,
	}
	body, _ := json.Marshal(payload)

	req := signedWebhookRequest(testWebhookSecret, body)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("got status %d, want %d: %s", rec.Code, http.StatusOK, rec.Body.String())
	}

	active, _ := mgr.IsActive(ctx, "sub-123")
	if active {
		t.Error("expected subscription to be suspended after webhook")
	}
}

func TestWebhook_UnsubscribeEvent(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	handler := NewWebhookHandler(testWebhookSecret, mgr)

	payload := WebhookPayload{
		ID:             "evt-2",
		SubscriptionID: "sub-123",
		OperationID:    "op-unsub-1",
		Action:         ActionUnsubscribe,
		Status:         WebhookStatusSuccess,
	}
	body, _ := json.Marshal(payload)

	req := signedWebhookRequest(testWebhookSecret, body)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("got status %d, want %d", rec.Code, http.StatusOK)
	}

	active, _ := mgr.IsActive(ctx, "sub-123")
	if active {
		t.Error("expected subscription to be inactive after unsubscribe webhook")
	}
}

func TestWebhook_ReinstateEvent(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})
	mgr.Suspend(ctx, "sub-123")

	handler := NewWebhookHandler(testWebhookSecret, mgr)

	payload := WebhookPayload{
		ID:             "evt-3",
		SubscriptionID: "sub-123",
		OperationID:    "op-reinstate-1",
		Action:         ActionReinstate,
	}
	body, _ := json.Marshal(payload)

	req := signedWebhookRequest(testWebhookSecret, body)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("got status %d, want %d", rec.Code, http.StatusOK)
	}

	active, _ := mgr.IsActive(ctx, "sub-123")
	if !active {
		t.Error("expected subscription to be active after reinstate webhook")
	}
}

func TestWebhook_ChangePlanEvent(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	handler := NewWebhookHandler(testWebhookSecret, mgr)

	payload := WebhookPayload{
		ID:             "evt-4",
		SubscriptionID: "sub-123",
		OperationID:    "op-changeplan-1",
		Action:         ActionChangePlan,
		PlanID:         "enterprise",
	}
	body, _ := json.Marshal(payload)

	req := signedWebhookRequest(testWebhookSecret, body)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("got status %d, want %d", rec.Code, http.StatusOK)
	}

	sub, _ := store.Get(ctx, "sub-123")
	if sub.PlanID != "enterprise" {
		t.Errorf("got plan %q, want %q", sub.PlanID, "enterprise")
	}
}

func TestWebhook_SignatureValidation(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	secret := "test-secret-key"
	handler := NewWebhookHandler(secret, mgr)

	payload := WebhookPayload{
		ID:             "evt-5",
		SubscriptionID: "sub-123",
		OperationID:    "op-sig-1",
		Action:         ActionRenew,
	}
	body, _ := json.Marshal(payload)

	// Invalid signature.
	req := httptest.NewRequest(http.MethodPost, "/webhook", strings.NewReader(string(body)))
	req.Header.Set("x-ms-signature", "invalid")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("got status %d, want %d for invalid signature", rec.Code, http.StatusUnauthorized)
	}

	// Valid signature.
	req = signedWebhookRequest(secret, body)
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("got status %d, want %d for valid signature", rec.Code, http.StatusOK)
	}
}

func TestWebhook_MethodNotAllowed(t *testing.T) {
	handler := NewWebhookHandler("secret", nil)

	req := httptest.NewRequest(http.MethodGet, "/webhook", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("got status %d, want %d", rec.Code, http.StatusMethodNotAllowed)
	}
}

func TestWebhook_EmptySecretReturns500(t *testing.T) {
	handler := NewWebhookHandler("", nil)

	payload := WebhookPayload{
		ID:     "evt-nosecret",
		Action: ActionRenew,
	}
	body, _ := json.Marshal(payload)

	req := httptest.NewRequest(http.MethodPost, "/webhook", strings.NewReader(string(body)))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Errorf("got status %d, want %d for empty secret", rec.Code, http.StatusInternalServerError)
	}
	if !strings.Contains(rec.Body.String(), "webhook secret not configured") {
		t.Errorf("expected body to mention secret not configured, got: %s", rec.Body.String())
	}
}

func TestWebhook_ExpiredTimestampReturns400(t *testing.T) {
	handler := NewWebhookHandler(testWebhookSecret, nil)
	handler.Now = func() time.Time {
		return time.Date(2026, 3, 23, 12, 0, 0, 0, time.UTC)
	}

	payload := WebhookPayload{
		ID:          "evt-expired",
		OperationID: "op-expired-1",
		Action:      ActionRenew,
		Timestamp:   time.Date(2026, 3, 23, 11, 50, 0, 0, time.UTC), // 10 min old
	}
	body, _ := json.Marshal(payload)

	req := signedWebhookRequest(testWebhookSecret, body)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("got status %d, want %d for expired timestamp", rec.Code, http.StatusBadRequest)
	}
	if !strings.Contains(rec.Body.String(), "timestamp expired") {
		t.Errorf("expected body to mention timestamp expired, got: %s", rec.Body.String())
	}
}

func TestWebhook_ReplayedOperationIDReturns409(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()
	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	handler := NewWebhookHandler(testWebhookSecret, mgr)

	payload := WebhookPayload{
		ID:             "evt-replay",
		SubscriptionID: "sub-123",
		OperationID:    "op-replay-1",
		Action:         ActionRenew,
	}
	body, _ := json.Marshal(payload)

	// First request should succeed.
	req := signedWebhookRequest(testWebhookSecret, body)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("first request: got status %d, want %d", rec.Code, http.StatusOK)
	}

	// Replayed request should return 409.
	req = signedWebhookRequest(testWebhookSecret, body)
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusConflict {
		t.Errorf("replayed request: got status %d, want %d", rec.Code, http.StatusConflict)
	}
}

func TestWebhook_OnEventCallback(t *testing.T) {
	store := NewMemorySubscriptionStore()
	fulfillment := &mockFulfillmentAPI{}
	mgr := NewSubscriptionManager(store, fulfillment)
	ctx := context.Background()

	mgr.ResolveAndActivate(ctx, "token-1", PlanDetails{PlanID: "basic"})

	var callbackAction WebhookAction
	handler := NewWebhookHandler(testWebhookSecret, mgr)
	handler.OnEvent = func(p WebhookPayload, err error) {
		callbackAction = p.Action
	}

	payload := WebhookPayload{
		ID:             "evt-6",
		SubscriptionID: "sub-123",
		OperationID:    "op-callback-1",
		Action:         ActionRenew,
	}
	body, _ := json.Marshal(payload)

	req := signedWebhookRequest(testWebhookSecret, body)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if callbackAction != ActionRenew {
		t.Errorf("callback got action %q, want %q", callbackAction, ActionRenew)
	}
}

// --- Memory Store Tests ---

func TestMemorySubscriptionStore_List(t *testing.T) {
	store := NewMemorySubscriptionStore()
	ctx := context.Background()

	store.Put(ctx, SaaSSubscription{ID: "a", Status: SaaSStatusSubscribed})
	store.Put(ctx, SaaSSubscription{ID: "b", Status: SaaSStatusSuspended})
	store.Put(ctx, SaaSSubscription{ID: "c", Status: SaaSStatusSubscribed})

	active, err := store.List(ctx, SaaSStatusSubscribed)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(active) != 2 {
		t.Errorf("got %d active, want 2", len(active))
	}
}

func TestMemorySubscriptionStore_Delete(t *testing.T) {
	store := NewMemorySubscriptionStore()
	ctx := context.Background()

	store.Put(ctx, SaaSSubscription{ID: "a", Status: SaaSStatusSubscribed})
	store.Delete(ctx, "a")

	sub, err := store.Get(ctx, "a")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if sub != nil {
		t.Error("expected nil after delete")
	}
}

// Verify unused import guard — time is used in test structs above.
var _ = time.Now
