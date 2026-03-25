package cloud

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestTenantConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  TenantConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: TenantConfig{
				MaxConcurrentRequests: 10,
				MaxTokensPerMinute:   1000,
				ModelAllowList:        []string{"llama3-8b"},
			},
		},
		{
			name: "zero concurrency is invalid",
			config: TenantConfig{
				MaxConcurrentRequests: 0,
				MaxTokensPerMinute:   1000,
			},
			wantErr: true,
		},
		{
			name: "zero tokens per minute is invalid",
			config: TenantConfig{
				MaxConcurrentRequests: 10,
				MaxTokensPerMinute:   0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestTenantRegistryRegisterAndGet(t *testing.T) {
	reg := NewTenantRegistry()

	cfg := TenantConfig{
		MaxConcurrentRequests: 5,
		MaxTokensPerMinute:   500,
		ModelAllowList:        []string{"llama3-8b", "gemma-2b"},
	}

	if err := reg.Register("key-abc", cfg); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	tenant, err := reg.Get("key-abc")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if tenant.Config.MaxConcurrentRequests != 5 {
		t.Errorf("MaxConcurrentRequests = %d, want 5", tenant.Config.MaxConcurrentRequests)
	}
	if tenant.Config.MaxTokensPerMinute != 500 {
		t.Errorf("MaxTokensPerMinute = %d, want 500", tenant.Config.MaxTokensPerMinute)
	}
	if len(tenant.Config.ModelAllowList) != 2 {
		t.Errorf("ModelAllowList len = %d, want 2", len(tenant.Config.ModelAllowList))
	}
}

func TestTenantRegistryGetNotFound(t *testing.T) {
	reg := NewTenantRegistry()

	_, err := reg.Get("nonexistent")
	if err == nil {
		t.Fatal("Get() expected error for nonexistent key")
	}
}

func TestTenantRegistryDuplicateRegister(t *testing.T) {
	reg := NewTenantRegistry()
	cfg := TenantConfig{
		MaxConcurrentRequests: 5,
		MaxTokensPerMinute:   500,
	}

	if err := reg.Register("key-abc", cfg); err != nil {
		t.Fatalf("first Register() error = %v", err)
	}
	if err := reg.Register("key-abc", cfg); err == nil {
		t.Fatal("second Register() expected error for duplicate key")
	}
}

func TestTenantRegistryRemove(t *testing.T) {
	reg := NewTenantRegistry()
	cfg := TenantConfig{
		MaxConcurrentRequests: 5,
		MaxTokensPerMinute:   500,
	}

	if err := reg.Register("key-abc", cfg); err != nil {
		t.Fatalf("Register() error = %v", err)
	}
	if err := reg.Remove("key-abc"); err != nil {
		t.Fatalf("Remove() error = %v", err)
	}
	_, err := reg.Get("key-abc")
	if err == nil {
		t.Fatal("Get() expected error after Remove()")
	}
}

func TestTenantRegistryConcurrentAccess(t *testing.T) {
	reg := NewTenantRegistry()
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			key := "key-" + string(rune('A'+n%26))
			cfg := TenantConfig{
				MaxConcurrentRequests: 5,
				MaxTokensPerMinute:   500,
			}
			_ = reg.Register(key, cfg)
			_, _ = reg.Get(key)
		}(i)
	}
	wg.Wait()
}

func TestConcurrencyLimiter(t *testing.T) {
	reg := NewTenantRegistry()
	cfg := TenantConfig{
		MaxConcurrentRequests: 2,
		MaxTokensPerMinute:   100000,
	}
	if err := reg.Register("key-abc", cfg); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	var inflight atomic.Int32
	var maxSeen atomic.Int32
	handler := reg.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := inflight.Add(1)
		defer inflight.Add(-1)
		for {
			old := maxSeen.Load()
			if cur <= old || maxSeen.CompareAndSwap(old, cur) {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))

	var wg sync.WaitGroup
	results := make([]int, 5)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			req.Header.Set("Authorization", "Bearer key-abc")
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)
			results[idx] = w.Code
		}(i)
	}
	wg.Wait()

	// At least some requests should have been rejected (429)
	rejected := 0
	for _, code := range results {
		if code == http.StatusTooManyRequests {
			rejected++
		}
	}
	// With concurrency limit of 2 and 5 concurrent requests each taking 50ms,
	// we expect some rejections
	if rejected == 0 {
		t.Error("expected at least one request to be rejected with 429")
	}
	if maxSeen.Load() > 2 {
		t.Errorf("max concurrent = %d, want <= 2", maxSeen.Load())
	}
}

func TestTokenRateLimiter(t *testing.T) {
	reg := NewTenantRegistry()
	cfg := TenantConfig{
		MaxConcurrentRequests: 100,
		MaxTokensPerMinute:   10,
	}
	if err := reg.Register("key-abc", cfg); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	tenant, _ := reg.Get("key-abc")

	// Consume all tokens
	if !tenant.ConsumeTokens(10) {
		t.Fatal("ConsumeTokens(10) should succeed when bucket is full")
	}

	// Next request should be rejected
	if tenant.ConsumeTokens(1) {
		t.Fatal("ConsumeTokens(1) should fail when bucket is empty")
	}
}

func TestMiddlewareNoAuthHeader(t *testing.T) {
	reg := NewTenantRegistry()
	handler := reg.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", w.Code, http.StatusUnauthorized)
	}
}

func TestMiddlewareUnknownKey(t *testing.T) {
	reg := NewTenantRegistry()
	handler := reg.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer unknown-key")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", w.Code, http.StatusUnauthorized)
	}
}

func TestMiddlewareTokenRateLimit(t *testing.T) {
	reg := NewTenantRegistry()
	cfg := TenantConfig{
		MaxConcurrentRequests: 100,
		MaxTokensPerMinute:   5,
	}
	if err := reg.Register("key-abc", cfg); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	handler := reg.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate token consumption via the tenant stored in context
		tenant := TenantFromContext(r.Context())
		if tenant == nil {
			t.Fatal("tenant not in context")
		}
		tenant.ConsumeTokens(5)
		w.WriteHeader(http.StatusOK)
	}))

	// First request: succeeds and consumes all tokens
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer key-abc")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("first request status = %d, want %d", w.Code, http.StatusOK)
	}

	// Second request: should be rate limited (tokens exhausted)
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req2.Header.Set("Authorization", "Bearer key-abc")
	w2 := httptest.NewRecorder()
	handler.ServeHTTP(w2, req2)
	if w2.Code != http.StatusTooManyRequests {
		t.Errorf("second request status = %d, want %d", w2.Code, http.StatusTooManyRequests)
	}
}

func TestTenantIsolation(t *testing.T) {
	reg := NewTenantRegistry()

	// Register two tenants with different limits
	if err := reg.Register("tenant-a", TenantConfig{
		MaxConcurrentRequests: 1,
		MaxTokensPerMinute:   100,
		ModelAllowList:        []string{"llama3-8b"},
	}); err != nil {
		t.Fatalf("Register tenant-a: %v", err)
	}

	if err := reg.Register("tenant-b", TenantConfig{
		MaxConcurrentRequests: 10,
		MaxTokensPerMinute:   1000,
		ModelAllowList:        []string{"gemma-2b"},
	}); err != nil {
		t.Fatalf("Register tenant-b: %v", err)
	}

	// Exhaust tenant-a's token budget
	tenantA, _ := reg.Get("tenant-a")
	if !tenantA.ConsumeTokens(100) {
		t.Fatal("tenant-a ConsumeTokens(100) should succeed")
	}

	// Tenant-a should be rate limited
	if tenantA.ConsumeTokens(1) {
		t.Fatal("tenant-a should be rate limited after exhausting tokens")
	}

	// Tenant-b should still have budget
	tenantB, _ := reg.Get("tenant-b")
	if !tenantB.ConsumeTokens(500) {
		t.Fatal("tenant-b ConsumeTokens(500) should succeed (independent budget)")
	}

	// Verify tenant configs are independent
	if tenantA.Config.MaxConcurrentRequests != 1 {
		t.Errorf("tenant-a MaxConcurrent = %d, want 1", tenantA.Config.MaxConcurrentRequests)
	}
	if tenantB.Config.MaxConcurrentRequests != 10 {
		t.Errorf("tenant-b MaxConcurrent = %d, want 10", tenantB.Config.MaxConcurrentRequests)
	}

	// Verify model allow lists are separate
	if !tenantA.ModelAllowed("llama3-8b") {
		t.Error("tenant-a should allow llama3-8b")
	}
	if tenantA.ModelAllowed("gemma-2b") {
		t.Error("tenant-a should not allow gemma-2b")
	}
	if !tenantB.ModelAllowed("gemma-2b") {
		t.Error("tenant-b should allow gemma-2b")
	}
	if tenantB.ModelAllowed("llama3-8b") {
		t.Error("tenant-b should not allow llama3-8b")
	}
}

func TestMiddlewareModelAllowList(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	t.Run("allowed model passes", func(t *testing.T) {
		reg := NewTenantRegistry()
		if err := reg.Register("key-allow", TenantConfig{
			MaxConcurrentRequests: 5,
			MaxTokensPerMinute:   10000,
			ModelAllowList:        []string{"llama3"},
		}); err != nil {
			t.Fatal(err)
		}
		handler := reg.Middleware(ok)

		body := `{"model":"llama3","messages":[]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-allow")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want %d", w.Code, http.StatusOK)
		}
	})

	t.Run("disallowed model rejected", func(t *testing.T) {
		reg := NewTenantRegistry()
		if err := reg.Register("key-allow", TenantConfig{
			MaxConcurrentRequests: 5,
			MaxTokensPerMinute:   10000,
			ModelAllowList:        []string{"llama3"},
		}); err != nil {
			t.Fatal(err)
		}
		handler := reg.Middleware(ok)

		body := `{"model":"gemma3","messages":[]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-allow")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusForbidden {
			t.Errorf("status = %d, want %d", w.Code, http.StatusForbidden)
		}
	})

	t.Run("empty allow list permits any model", func(t *testing.T) {
		reg := NewTenantRegistry()
		if err := reg.Register("key-open", TenantConfig{
			MaxConcurrentRequests: 5,
			MaxTokensPerMinute:   10000,
			ModelAllowList:        nil,
		}); err != nil {
			t.Fatal(err)
		}
		handler := reg.Middleware(ok)

		body := `{"model":"any-model","messages":[]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-open")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want %d", w.Code, http.StatusOK)
		}
	})
}

func TestModelAllowedEmptyList(t *testing.T) {
	reg := NewTenantRegistry()
	if err := reg.Register("key-open", TenantConfig{
		MaxConcurrentRequests: 5,
		MaxTokensPerMinute:   500,
		ModelAllowList:        nil,
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	tenant, _ := reg.Get("key-open")
	// Empty allow list means all models are allowed
	if !tenant.ModelAllowed("any-model") {
		t.Error("empty allow list should allow any model")
	}
}
