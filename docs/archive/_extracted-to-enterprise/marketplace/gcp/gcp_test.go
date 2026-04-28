package gcp

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

// mockProcurementAPI is a test double for ProcurementAPI.
type mockProcurementAPI struct {
	getAccountFunc        func(ctx context.Context, name string) (*Account, error)
	listAccountsFunc      func(ctx context.Context, parent string) ([]Account, error)
	getEntitlementFunc    func(ctx context.Context, name string) (*ProcurementEntitlement, error)
	listEntitlementsFunc  func(ctx context.Context, parent string) ([]ProcurementEntitlement, error)
	approveFunc           func(ctx context.Context, name string) error
	rejectFunc            func(ctx context.Context, name, reason string) error
	suspendFunc           func(ctx context.Context, name, reason string) error
	reinstateFunc         func(ctx context.Context, name string) error
}

func (m *mockProcurementAPI) GetAccount(ctx context.Context, name string) (*Account, error) {
	if m.getAccountFunc != nil {
		return m.getAccountFunc(ctx, name)
	}
	return &Account{Name: name, State: AccountActive}, nil
}

func (m *mockProcurementAPI) ListAccounts(ctx context.Context, parent string) ([]Account, error) {
	if m.listAccountsFunc != nil {
		return m.listAccountsFunc(ctx, parent)
	}
	return []Account{{Name: parent + "/accounts/acct-1", State: AccountActive}}, nil
}

func (m *mockProcurementAPI) GetEntitlement(ctx context.Context, name string) (*ProcurementEntitlement, error) {
	if m.getEntitlementFunc != nil {
		return m.getEntitlementFunc(ctx, name)
	}
	return &ProcurementEntitlement{
		Name:    name,
		Account: "providers/zerfoo/accounts/acct-1",
		Product: "zerfoo-ml",
		Plan:    "standard",
		State:   EntitlementActive,
	}, nil
}

func (m *mockProcurementAPI) ListEntitlements(ctx context.Context, parent string) ([]ProcurementEntitlement, error) {
	if m.listEntitlementsFunc != nil {
		return m.listEntitlementsFunc(ctx, parent)
	}
	return []ProcurementEntitlement{
		{Name: parent + "/entitlements/ent-1", State: EntitlementActive},
	}, nil
}

func (m *mockProcurementAPI) ApproveEntitlement(ctx context.Context, name string) error {
	if m.approveFunc != nil {
		return m.approveFunc(ctx, name)
	}
	return nil
}

func (m *mockProcurementAPI) RejectEntitlement(ctx context.Context, name, reason string) error {
	if m.rejectFunc != nil {
		return m.rejectFunc(ctx, name, reason)
	}
	return nil
}

func (m *mockProcurementAPI) SuspendEntitlement(ctx context.Context, name, reason string) error {
	if m.suspendFunc != nil {
		return m.suspendFunc(ctx, name, reason)
	}
	return nil
}

func (m *mockProcurementAPI) ReinstateEntitlement(ctx context.Context, name string) error {
	if m.reinstateFunc != nil {
		return m.reinstateFunc(ctx, name)
	}
	return nil
}

// mockServiceControlAPI is a test double for ServiceControlAPI.
type mockServiceControlAPI struct {
	reportFunc func(ctx context.Context, serviceName string, ops []Operation) error
	reported   []Operation
}

func (m *mockServiceControlAPI) Report(ctx context.Context, serviceName string, ops []Operation) error {
	m.reported = append(m.reported, ops...)
	if m.reportFunc != nil {
		return m.reportFunc(ctx, serviceName, ops)
	}
	return nil
}

func TestProcurementClient_GetAccount(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("unexpected method: %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(Account{
			Name:  "providers/zerfoo/accounts/acct-123",
			State: AccountActive,
		})
	}))
	defer srv.Close()

	client := NewProcurementClient(nil)
	client.Endpoint = srv.URL

	acct, err := client.GetAccount(context.Background(), "providers/zerfoo/accounts/acct-123")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if acct.Name != "providers/zerfoo/accounts/acct-123" {
		t.Errorf("got name %q, want %q", acct.Name, "providers/zerfoo/accounts/acct-123")
	}
	if acct.State != AccountActive {
		t.Errorf("got state %q, want %q", acct.State, AccountActive)
	}
}

func TestProcurementClient_ListEntitlements(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(struct {
			Entitlements []ProcurementEntitlement `json:"entitlements"`
		}{
			Entitlements: []ProcurementEntitlement{
				{Name: "providers/zerfoo/entitlements/ent-1", State: EntitlementActive},
				{Name: "providers/zerfoo/entitlements/ent-2", State: EntitlementSuspended},
			},
		})
	}))
	defer srv.Close()

	client := NewProcurementClient(nil)
	client.Endpoint = srv.URL

	ents, err := client.ListEntitlements(context.Background(), "providers/zerfoo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ents) != 2 {
		t.Fatalf("got %d entitlements, want 2", len(ents))
	}
	if ents[0].State != EntitlementActive {
		t.Errorf("got state %q, want %q", ents[0].State, EntitlementActive)
	}
}

func TestProcurementClient_ApproveEntitlement(t *testing.T) {
	var calledPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calledPath = r.URL.Path
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("{}"))
	}))
	defer srv.Close()

	client := NewProcurementClient(nil)
	client.Endpoint = srv.URL

	err := client.ApproveEntitlement(context.Background(), "providers/zerfoo/entitlements/ent-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "/providers/zerfoo/entitlements/ent-1:approve"
	if calledPath != want {
		t.Errorf("got path %q, want %q", calledPath, want)
	}
}

func TestProcurementClient_RejectEntitlement(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]string
		json.NewDecoder(r.Body).Decode(&body)
		if body["reason"] != "policy violation" {
			t.Errorf("got reason %q, want %q", body["reason"], "policy violation")
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("{}"))
	}))
	defer srv.Close()

	client := NewProcurementClient(nil)
	client.Endpoint = srv.URL

	err := client.RejectEntitlement(context.Background(), "providers/zerfoo/entitlements/ent-1", "policy violation")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestServiceControlClient_Report(t *testing.T) {
	var receivedOps []Operation
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		var req ReportRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		receivedOps = req.Operations
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ReportResponse{})
	}))
	defer srv.Close()

	client := NewServiceControlClient(nil)
	client.Endpoint = srv.URL

	quantity := int64(5)
	ops := []Operation{
		{
			OperationID:   "op-1",
			OperationName: "zerfoo.usage.report",
			ConsumerID:    "project:my-project",
			StartTime:     time.Now(),
			EndTime:       time.Now(),
			MetricValues: []MetricValueSet{
				{
					MetricName: DimensionTokens,
					MetricValues: []MetricValue{
						{Int64Value: &quantity},
					},
				},
			},
		},
	}

	err := client.Report(context.Background(), "zerfoo-marketplace.googleapis.com", ops)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(receivedOps) != 1 {
		t.Fatalf("got %d operations, want 1", len(receivedOps))
	}
	if receivedOps[0].OperationID != "op-1" {
		t.Errorf("got operation ID %q, want %q", receivedOps[0].OperationID, "op-1")
	}
}

func TestServiceControlClient_ReportErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ReportResponse{
			ReportErrors: []ReportError{
				{OperationID: "op-1", Status: Status{Code: 7, Message: "permission denied"}},
			},
		})
	}))
	defer srv.Close()

	client := NewServiceControlClient(nil)
	client.Endpoint = srv.URL

	err := client.Report(context.Background(), "svc", []Operation{{OperationID: "op-1"}})
	if err == nil {
		t.Fatal("expected error for report with errors")
	}
}

func TestServiceControlClient_Report_RetryOn429(t *testing.T) {
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte("throttled"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ReportResponse{})
	}))
	defer srv.Close()

	client := NewServiceControlClient(nil)
	client.Endpoint = srv.URL
	client.Retry = marketplace.RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}

	quantity := int64(5)
	ops := []Operation{
		{
			OperationID: "op-retry",
			ConsumerID:  "project:my-project",
			MetricValues: []MetricValueSet{
				{
					MetricName:   DimensionTokens,
					MetricValues: []MetricValue{{Int64Value: &quantity}},
				},
			},
		},
	}

	err := client.Report(context.Background(), "zerfoo-marketplace.googleapis.com", ops)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := attempts.Load(); got != 3 {
		t.Errorf("got %d attempts, want 3", got)
	}
}

func TestEntitlementManager_ApproveAndCheck(t *testing.T) {
	procurement := &mockProcurementAPI{}
	store := NewMemoryEntitlementStore()
	mgr := NewEntitlementManager(procurement, store, 5*time.Minute)
	ctx := context.Background()

	entName := "providers/zerfoo/entitlements/ent-1"

	// Approve.
	if err := mgr.Approve(ctx, entName); err != nil {
		t.Fatalf("approve: %v", err)
	}

	// Check active.
	active, err := mgr.IsActive(ctx, entName)
	if err != nil {
		t.Fatalf("is active: %v", err)
	}
	if !active {
		t.Error("expected entitlement to be active after approve")
	}
}

func TestEntitlementManager_RejectEntitlement(t *testing.T) {
	procurement := &mockProcurementAPI{}
	store := NewMemoryEntitlementStore()
	mgr := NewEntitlementManager(procurement, store, 5*time.Minute)
	ctx := context.Background()

	entName := "providers/zerfoo/entitlements/ent-2"

	if err := mgr.Reject(ctx, entName, "invalid account"); err != nil {
		t.Fatalf("reject: %v", err)
	}

	active, err := mgr.IsActive(ctx, entName)
	if err != nil {
		t.Fatalf("is active: %v", err)
	}
	if active {
		t.Error("expected entitlement to be inactive after reject")
	}
}

func TestEntitlementManager_SuspendAndReinstate(t *testing.T) {
	procurement := &mockProcurementAPI{}
	store := NewMemoryEntitlementStore()
	mgr := NewEntitlementManager(procurement, store, 5*time.Minute)
	ctx := context.Background()

	entName := "providers/zerfoo/entitlements/ent-3"

	// First approve it.
	if err := mgr.Approve(ctx, entName); err != nil {
		t.Fatalf("approve: %v", err)
	}

	// Suspend.
	if err := mgr.Suspend(ctx, entName, "payment overdue"); err != nil {
		t.Fatalf("suspend: %v", err)
	}

	active, err := mgr.IsActive(ctx, entName)
	if err != nil {
		t.Fatalf("is active after suspend: %v", err)
	}
	if active {
		t.Error("expected entitlement to be inactive after suspend")
	}

	// Reinstate.
	if err := mgr.Reinstate(ctx, entName); err != nil {
		t.Fatalf("reinstate: %v", err)
	}

	active, err = mgr.IsActive(ctx, entName)
	if err != nil {
		t.Fatalf("is active after reinstate: %v", err)
	}
	if !active {
		t.Error("expected entitlement to be active after reinstate")
	}
}

func TestTokenBilling_RecordAndFlush(t *testing.T) {
	metering := &mockServiceControlAPI{}
	tracker := NewTokenBillingTracker("zerfoo-marketplace.googleapis.com", metering)

	// Record usage.
	tracker.RecordUsage("project:proj-1", 500_000, 500_000) // 1M total
	tracker.RecordUsage("project:proj-1", 300_000, 200_000) // +500K
	tracker.RecordUsage("project:proj-2", 2_000_000, 1_000_000) // 3M total

	// Check snapshot.
	snap := tracker.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("got %d snapshot records, want 2", len(snap))
	}

	// Flush.
	count, err := tracker.Flush(context.Background())
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	if count != 2 {
		t.Fatalf("got %d flushed operations, want 2", count)
	}
	if len(metering.reported) != 2 {
		t.Fatalf("got %d reported operations, want 2", len(metering.reported))
	}

	// Verify quantities.
	quantities := make(map[string]int64)
	for _, op := range metering.reported {
		if len(op.MetricValues) > 0 && len(op.MetricValues[0].MetricValues) > 0 {
			quantities[op.ConsumerID] = *op.MetricValues[0].MetricValues[0].Int64Value
		}
	}
	// proj-1: 1.5M => 2 units, proj-2: 3M => 3 units.
	if quantities["project:proj-1"] != 2 {
		t.Errorf("proj-1 quantity: got %d, want 2", quantities["project:proj-1"])
	}
	if quantities["project:proj-2"] != 3 {
		t.Errorf("proj-2 quantity: got %d, want 3", quantities["project:proj-2"])
	}

	// After flush, snapshot should be empty.
	snap = tracker.Snapshot()
	if len(snap) != 0 {
		t.Errorf("got %d snapshot records after flush, want 0", len(snap))
	}
}

func TestTokenBilling_FlushEmpty(t *testing.T) {
	metering := &mockServiceControlAPI{}
	tracker := NewTokenBillingTracker("svc", metering)

	count, err := tracker.Flush(context.Background())
	if err != nil {
		t.Fatalf("flush empty: %v", err)
	}
	if count != 0 {
		t.Errorf("got %d flushed for empty, want 0", count)
	}
}

func TestMemoryEntitlementStore_List(t *testing.T) {
	store := NewMemoryEntitlementStore()
	ctx := context.Background()

	store.Put(ctx, LocalEntitlement{Name: "ent-1", Account: "acct-1", State: EntitlementActive})
	store.Put(ctx, LocalEntitlement{Name: "ent-2", Account: "acct-1", State: EntitlementSuspended})
	store.Put(ctx, LocalEntitlement{Name: "ent-3", Account: "acct-2", State: EntitlementActive})

	ents, err := store.List(ctx, "acct-1")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(ents) != 2 {
		t.Errorf("got %d entitlements, want 2", len(ents))
	}
}

func TestMemoryEntitlementStore_Delete(t *testing.T) {
	store := NewMemoryEntitlementStore()
	ctx := context.Background()

	store.Put(ctx, LocalEntitlement{Name: "ent-1", Account: "acct-1"})
	store.Delete(ctx, "ent-1")

	got, err := store.Get(ctx, "ent-1")
	if err != nil {
		t.Fatalf("get after delete: %v", err)
	}
	if got != nil {
		t.Error("expected nil after delete")
	}
}
