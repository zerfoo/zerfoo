package cloud

import (
	"path/filepath"
	"testing"
)

func TestBboltTenantStoreBackend(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "tenants.db")
	backend, err := NewBboltTenantStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("NewBboltTenantStoreBackend: %v", err)
	}
	defer backend.Close()

	tm := NewTenantManager(WithTenantBackend(backend))

	// Create a tenant.
	cfg := TenantConfig{
		ID:                    "t-1",
		APIKey:                "key-abc",
		RateLimit:             100,
		TokenBudget:           5000,
		MaxConcurrentRequests: 10,
		ModelAllowList:        []string{"llama3"},
	}
	if err := tm.Create(cfg); err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Get by ID.
	tenant, err := tm.Get("t-1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if tenant.ID != "t-1" {
		t.Fatalf("got ID %q, want %q", tenant.ID, "t-1")
	}

	// Update rate limits.
	if err := tm.Update("t-1", 200, 10000); err != nil {
		t.Fatalf("Update: %v", err)
	}

	// Verify update in backend.
	stored, ok := backend.LoadTenant("t-1")
	if !ok {
		t.Fatal("LoadTenant after update: not found")
	}
	if stored.RateLimit != 200 {
		t.Fatalf("stored RateLimit = %d, want 200", stored.RateLimit)
	}
	if stored.TokenBudget != 10000 {
		t.Fatalf("stored TokenBudget = %d, want 10000", stored.TokenBudget)
	}

	// List tenants.
	list := tm.List()
	if len(list) != 1 {
		t.Fatalf("List len = %d, want 1", len(list))
	}
	if list[0].ID != "t-1" {
		t.Fatalf("List[0].ID = %q, want %q", list[0].ID, "t-1")
	}

	// Delete tenant.
	if err := tm.Delete("t-1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := tm.Get("t-1"); err == nil {
		t.Fatal("Get after Delete: expected error")
	}
	if items := backend.ListTenants(); len(items) != 0 {
		t.Fatalf("ListTenants after Delete: got %d, want 0", len(items))
	}

	// Round-trip: reopen the database and verify persistence.
	cfg2 := TenantConfig{
		ID:          "t-2",
		APIKey:      "key-xyz",
		RateLimit:   50,
		TokenBudget: 2000,
	}
	if err := tm.Create(cfg2); err != nil {
		t.Fatalf("Create t-2: %v", err)
	}
	backend.Close()

	backend2, err := NewBboltTenantStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer backend2.Close()

	tm2 := NewTenantManager(WithTenantBackend(backend2))
	tenant2, err := tm2.Get("t-2")
	if err != nil {
		t.Fatalf("Get t-2 after reopen: %v", err)
	}
	if tenant2.rateLimit.Load() != 50 {
		t.Fatalf("reloaded RateLimit = %d, want 50", tenant2.rateLimit.Load())
	}
	if tenant2.tokenBudget.Load() != 2000 {
		t.Fatalf("reloaded TokenBudget = %d, want 2000", tenant2.tokenBudget.Load())
	}
}
