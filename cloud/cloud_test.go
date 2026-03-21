package cloud

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// fakeInferenceHandler returns a handler that echoes a fixed usage response.
func fakeInferenceHandler(promptTokens, completionTokens int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "test-model",
			"choices": []interface{}{},
			"usage": map[string]int{
				"prompt_tokens":     promptTokens,
				"completion_tokens": completionTokens,
				"total_tokens":      promptTokens + completionTokens,
			},
		})
	})
}

func setupCloudServer(handler http.Handler) (*CloudServer, *MemoryBillingStore) {
	tm := NewTenantManager()
	store := NewMemoryBillingStore()
	meter := NewTokenMeter(store)
	cs := NewCloudServer(handler, tm, meter)
	return cs, store
}

func TestCloud_MultiTenant(t *testing.T) {
	tests := []struct {
		name       string
		setup      func(tm *TenantManager)
		apiKey     string
		wantStatus int
	}{
		{
			name: "valid tenant gets 200",
			setup: func(tm *TenantManager) {
				tm.Create(TenantConfig{ID: "t1", APIKey: "key-a", RateLimit: 100, TokenBudget: 10000})
			},
			apiKey:     "key-a",
			wantStatus: http.StatusOK,
		},
		{
			name:       "unknown key gets 401",
			setup:      func(tm *TenantManager) {},
			apiKey:     "unknown",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "missing auth gets 401",
			setup:      func(tm *TenantManager) {},
			apiKey:     "",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name: "tenant isolation - each has own key",
			setup: func(tm *TenantManager) {
				tm.Create(TenantConfig{ID: "t1", APIKey: "key-a", RateLimit: 100, TokenBudget: 10000})
				tm.Create(TenantConfig{ID: "t2", APIKey: "key-b", RateLimit: 100, TokenBudget: 10000})
			},
			apiKey:     "key-b",
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs, _ := setupCloudServer(fakeInferenceHandler(10, 5))
			tt.setup(cs.Tenants())
			handler := cs.Handler()

			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			if tt.apiKey != "" {
				req.Header.Set("Authorization", "Bearer "+tt.apiKey)
			}
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d", w.Code, tt.wantStatus)
			}
		})
	}

	// Verify tenant-specific data isolation.
	t.Run("tenant data isolation", func(t *testing.T) {
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tenant := tenantFromContext(r.Context())
			tenantID := ""
			if tenant != nil {
				tenantID = tenant.ID
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"tenant_id": tenantID,
				"usage": map[string]int{
					"prompt_tokens":     10,
					"completion_tokens": 5,
					"total_tokens":      15,
				},
			})
		})

		cs, _ := setupCloudServer(inner)
		cs.Tenants().Create(TenantConfig{ID: "alpha", APIKey: "key-alpha", RateLimit: 100, TokenBudget: 10000})
		cs.Tenants().Create(TenantConfig{ID: "beta", APIKey: "key-beta", RateLimit: 100, TokenBudget: 10000})
		handler := cs.Handler()

		// Request as alpha
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer key-alpha")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		var resp struct {
			TenantID string `json:"tenant_id"`
		}
		json.Unmarshal(w.Body.Bytes(), &resp)
		if resp.TenantID != "alpha" {
			t.Errorf("tenant_id = %q, want %q", resp.TenantID, "alpha")
		}

		// Request as beta
		req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req2.Header.Set("Authorization", "Bearer key-beta")
		w2 := httptest.NewRecorder()
		handler.ServeHTTP(w2, req2)

		json.Unmarshal(w2.Body.Bytes(), &resp)
		if resp.TenantID != "beta" {
			t.Errorf("tenant_id = %q, want %q", resp.TenantID, "beta")
		}
	})
}

func TestCloud_Billing(t *testing.T) {
	tests := []struct {
		name             string
		promptTokens     int
		completionTokens int
		wantRecords      int
	}{
		{
			name:             "records usage",
			promptTokens:     100,
			completionTokens: 50,
			wantRecords:      1,
		},
		{
			name:             "skips zero usage",
			promptTokens:     0,
			completionTokens: 0,
			wantRecords:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs, store := setupCloudServer(fakeInferenceHandler(tt.promptTokens, tt.completionTokens))
			cs.Tenants().Create(TenantConfig{ID: "t1", APIKey: "key-a", RateLimit: 1000, TokenBudget: 100000})
			handler := cs.Handler()

			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			req.Header.Set("Authorization", "Bearer key-a")
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)

			records := store.All()
			if len(records) != tt.wantRecords {
				t.Fatalf("got %d billing records, want %d", len(records), tt.wantRecords)
			}
			if tt.wantRecords > 0 {
				r := records[0]
				if r.TenantID != "t1" {
					t.Errorf("TenantID = %q, want %q", r.TenantID, "t1")
				}
				if r.InputTokens != tt.promptTokens {
					t.Errorf("InputTokens = %d, want %d", r.InputTokens, tt.promptTokens)
				}
				if r.OutputTokens != tt.completionTokens {
					t.Errorf("OutputTokens = %d, want %d", r.OutputTokens, tt.completionTokens)
				}
				if r.Timestamp.IsZero() {
					t.Error("Timestamp should not be zero")
				}
			}
		})
	}

	t.Run("billing query by tenant and time range", func(t *testing.T) {
		store := NewMemoryBillingStore()
		meter := NewTokenMeter(store)

		meter.Record("t1", 10, 5)
		meter.Record("t2", 20, 10)
		meter.Record("t1", 30, 15)

		now := time.Now()
		records, err := meter.Query("t1", now.Add(-time.Minute), now.Add(time.Minute))
		if err != nil {
			t.Fatalf("Query: %v", err)
		}
		if len(records) != 2 {
			t.Fatalf("got %d records for t1, want 2", len(records))
		}

		// t2 should have its own records
		records, err = meter.Query("t2", now.Add(-time.Minute), now.Add(time.Minute))
		if err != nil {
			t.Fatalf("Query: %v", err)
		}
		if len(records) != 1 {
			t.Fatalf("got %d records for t2, want 1", len(records))
		}
	})
}

func TestCloud_RateLimit(t *testing.T) {
	tests := []struct {
		name        string
		rateLimit   int64
		tokenBudget int64
		requests    int
		wantReject  bool
	}{
		{
			name:        "under rate limit",
			rateLimit:   10,
			tokenBudget: 100000,
			requests:    5,
			wantReject:  false,
		},
		{
			name:        "exceeds rate limit",
			rateLimit:   3,
			tokenBudget: 100000,
			requests:    5,
			wantReject:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs, _ := setupCloudServer(fakeInferenceHandler(10, 5))
			cs.Tenants().Create(TenantConfig{
				ID: "t1", APIKey: "key-a",
				RateLimit: tt.rateLimit, TokenBudget: tt.tokenBudget,
			})
			handler := cs.Handler()

			var rejected int
			for i := 0; i < tt.requests; i++ {
				req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
				req.Header.Set("Authorization", "Bearer key-a")
				w := httptest.NewRecorder()
				handler.ServeHTTP(w, req)
				if w.Code == http.StatusTooManyRequests {
					rejected++
				}
			}

			if tt.wantReject && rejected == 0 {
				t.Error("expected at least one 429 response")
			}
			if !tt.wantReject && rejected > 0 {
				t.Errorf("expected no rejections, got %d", rejected)
			}
		})
	}

	t.Run("retry-after header on 429", func(t *testing.T) {
		cs, _ := setupCloudServer(fakeInferenceHandler(10, 5))
		cs.Tenants().Create(TenantConfig{ID: "t1", APIKey: "key-a", RateLimit: 1, TokenBudget: 100000})
		handler := cs.Handler()

		// First request succeeds.
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer key-a")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("first request status = %d, want 200", w.Code)
		}

		// Second request should get 429 with Retry-After.
		req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req2.Header.Set("Authorization", "Bearer key-a")
		w2 := httptest.NewRecorder()
		handler.ServeHTTP(w2, req2)
		if w2.Code != http.StatusTooManyRequests {
			t.Fatalf("second request status = %d, want 429", w2.Code)
		}

		retryAfter := w2.Header().Get("Retry-After")
		if retryAfter == "" {
			t.Error("expected Retry-After header on 429 response")
		}
		secs := retryAfterSeconds(w2.Header())
		if secs != 60 {
			t.Errorf("Retry-After = %d seconds, want 60", secs)
		}
	})

	t.Run("health check", func(t *testing.T) {
		cs, _ := setupCloudServer(fakeInferenceHandler(10, 5))
		handler := cs.Handler()

		// Healthy by default.
		req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("healthz status = %d, want 200", w.Code)
		}

		// Mark unhealthy.
		cs.SetHealthy(false)
		w2 := httptest.NewRecorder()
		handler.ServeHTTP(w2, httptest.NewRequest(http.MethodGet, "/healthz", nil))
		if w2.Code != http.StatusServiceUnavailable {
			t.Errorf("healthz degraded status = %d, want 503", w2.Code)
		}

		// Restore.
		cs.SetHealthy(true)
		w3 := httptest.NewRecorder()
		handler.ServeHTTP(w3, httptest.NewRequest(http.MethodGet, "/healthz", nil))
		if w3.Code != http.StatusOK {
			t.Errorf("healthz restored status = %d, want 200", w3.Code)
		}
	})
}

func TestCloud_TenantContextPropagation(t *testing.T) {
	t.Run("tenant available in context through middleware chain", func(t *testing.T) {
		var capturedTenant *Tenant
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capturedTenant = tenantFromContext(r.Context())
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"usage": map[string]int{
					"prompt_tokens":     5,
					"completion_tokens": 3,
					"total_tokens":      8,
				},
			})
		})

		cs, _ := setupCloudServer(inner)
		cs.Tenants().Create(TenantConfig{ID: "ctx-t1", APIKey: "ctx-key-1", RateLimit: 100, TokenBudget: 10000})
		handler := cs.Handler()

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer ctx-key-1")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		if capturedTenant == nil {
			t.Fatal("tenant was nil in context, expected non-nil")
		}
		if capturedTenant.ID != "ctx-t1" {
			t.Errorf("tenant.ID = %q, want %q", capturedTenant.ID, "ctx-t1")
		}
	})

	t.Run("no X-Tenant-ID header leaked to inner handler", func(t *testing.T) {
		var headerVal string
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			headerVal = r.Header.Get("X-Tenant-ID")
			w.WriteHeader(http.StatusOK)
		})

		cs, _ := setupCloudServer(inner)
		cs.Tenants().Create(TenantConfig{ID: "leak-t1", APIKey: "leak-key-1", RateLimit: 100, TokenBudget: 10000})
		handler := cs.Handler()

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer leak-key-1")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if headerVal != "" {
			t.Errorf("X-Tenant-ID header = %q, want empty (should use context instead)", headerVal)
		}
	})

	t.Run("tenantFromContext returns nil for empty context", func(t *testing.T) {
		ctx := context.Background()
		if tenant := tenantFromContext(ctx); tenant != nil {
			t.Errorf("tenantFromContext(empty) = %v, want nil", tenant)
		}
	})

	t.Run("client-supplied X-Tenant-ID header cannot spoof tenant", func(t *testing.T) {
		var capturedTenant *Tenant
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capturedTenant = tenantFromContext(r.Context())
			w.WriteHeader(http.StatusOK)
		})

		cs, _ := setupCloudServer(inner)
		cs.Tenants().Create(TenantConfig{ID: "real", APIKey: "real-key", RateLimit: 100, TokenBudget: 10000})
		cs.Tenants().Create(TenantConfig{ID: "victim", APIKey: "victim-key", RateLimit: 100, TokenBudget: 10000})
		handler := cs.Handler()

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer real-key")
		req.Header.Set("X-Tenant-ID", "victim") // attacker tries to spoof
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if capturedTenant == nil {
			t.Fatal("tenant was nil")
		}
		if capturedTenant.ID != "real" {
			t.Errorf("tenant.ID = %q, want %q (spoofed header should be ignored)", capturedTenant.ID, "real")
		}
	})
}

func TestTenantManager_CRUD(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TenantConfig
		wantErr bool
	}{
		{
			name:    "valid tenant",
			cfg:     TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000},
			wantErr: false,
		},
		{
			name:    "empty ID",
			cfg:     TenantConfig{ID: "", APIKey: "key-2", RateLimit: 10, TokenBudget: 1000},
			wantErr: true,
		},
		{
			name:    "empty API key",
			cfg:     TenantConfig{ID: "t2", APIKey: "", RateLimit: 10, TokenBudget: 1000},
			wantErr: true,
		},
		{
			name:    "zero rate limit",
			cfg:     TenantConfig{ID: "t3", APIKey: "key-3", RateLimit: 0, TokenBudget: 1000},
			wantErr: true,
		},
		{
			name:    "zero token budget",
			cfg:     TenantConfig{ID: "t4", APIKey: "key-4", RateLimit: 10, TokenBudget: 0},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tm := NewTenantManager()
			err := tm.Create(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("Create() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}

	t.Run("duplicate ID rejected", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		err := tm.Create(TenantConfig{ID: "t1", APIKey: "key-2", RateLimit: 10, TokenBudget: 1000})
		if err == nil {
			t.Error("expected error for duplicate tenant ID")
		}
	})

	t.Run("duplicate API key rejected", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		err := tm.Create(TenantConfig{ID: "t2", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		if err == nil {
			t.Error("expected error for duplicate API key")
		}
	})

	t.Run("get by ID", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		tenant, err := tm.Get("t1")
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		if tenant.ID != "t1" {
			t.Errorf("ID = %q, want %q", tenant.ID, "t1")
		}
	})

	t.Run("get by API key", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		tenant, err := tm.GetByAPIKey("key-1")
		if err != nil {
			t.Fatalf("GetByAPIKey: %v", err)
		}
		if tenant.ID != "t1" {
			t.Errorf("ID = %q, want %q", tenant.ID, "t1")
		}
	})

	t.Run("update", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		if err := tm.Update("t1", 20, 2000); err != nil {
			t.Fatalf("Update: %v", err)
		}
		tenant, _ := tm.Get("t1")
		cfg := tenant.Config()
		if cfg.RateLimit != 20 {
			t.Errorf("RateLimit = %d, want 20", cfg.RateLimit)
		}
		if cfg.TokenBudget != 2000 {
			t.Errorf("TokenBudget = %d, want 2000", cfg.TokenBudget)
		}
	})

	t.Run("delete", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		if err := tm.Delete("t1"); err != nil {
			t.Fatalf("Delete: %v", err)
		}
		_, err := tm.Get("t1")
		if err == nil {
			t.Error("expected error after delete")
		}
		_, err = tm.GetByAPIKey("key-1")
		if err == nil {
			t.Error("expected error for deleted API key")
		}
	})

	t.Run("list", func(t *testing.T) {
		tm := NewTenantManager()
		tm.Create(TenantConfig{ID: "t1", APIKey: "key-1", RateLimit: 10, TokenBudget: 1000})
		tm.Create(TenantConfig{ID: "t2", APIKey: "key-2", RateLimit: 20, TokenBudget: 2000})
		tenants := tm.List()
		if len(tenants) != 2 {
			t.Fatalf("List() len = %d, want 2", len(tenants))
		}
	})
}

func TestTenant_APIKeyRedaction(t *testing.T) {
	rawKeys := []string{"secret-key-alpha", "secret-key-beta"}

	tm := NewTenantManager()
	tm.Create(TenantConfig{ID: "t1", APIKey: rawKeys[0], RateLimit: 10, TokenBudget: 1000})
	tm.Create(TenantConfig{ID: "t2", APIKey: rawKeys[1], RateLimit: 20, TokenBudget: 2000})

	t.Run("Config redacts API key", func(t *testing.T) {
		tenant, err := tm.Get("t1")
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		cfg := tenant.Config()
		if cfg.APIKey == rawKeys[0] {
			t.Errorf("Config().APIKey returned raw key %q, want redacted", cfg.APIKey)
		}
		if cfg.APIKey != redactedAPIKey {
			t.Errorf("Config().APIKey = %q, want %q", cfg.APIKey, redactedAPIKey)
		}
	})

	t.Run("List redacts all API keys", func(t *testing.T) {
		configs := tm.List()
		if len(configs) != 2 {
			t.Fatalf("List() returned %d configs, want 2", len(configs))
		}
		for _, cfg := range configs {
			for _, raw := range rawKeys {
				if cfg.APIKey == raw {
					t.Errorf("List() returned raw key %q for tenant %q", raw, cfg.ID)
				}
			}
			if cfg.APIKey != redactedAPIKey {
				t.Errorf("List() tenant %q APIKey = %q, want %q", cfg.ID, cfg.APIKey, redactedAPIKey)
			}
		}
	})

	t.Run("raw key still works for authentication", func(t *testing.T) {
		tenant, err := tm.GetByAPIKey(rawKeys[0])
		if err != nil {
			t.Fatalf("GetByAPIKey should still work with raw key: %v", err)
		}
		if tenant.ID != "t1" {
			t.Errorf("GetByAPIKey returned tenant %q, want %q", tenant.ID, "t1")
		}
	})
}
