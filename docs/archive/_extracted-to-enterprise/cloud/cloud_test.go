package cloud

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/generate"
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

	t.Run("streaming SSE response produces billing via context", func(t *testing.T) {
		// Simulate a streaming handler that writes SSE chunks (not valid JSON)
		// but records token usage via the context-based TokenUsage.
		sseHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Write SSE chunks — not parseable as a single JSON object.
			w.Header().Set("Content-Type", "text/event-stream")
			for i := range 3 {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok%d\"}}]}\n\n", i)
			}
			fmt.Fprint(w, "data: [DONE]\n\n")

			// The generation layer records usage via context.
			if usage := generate.TokenUsageFromContext(r.Context()); usage != nil {
				usage.SetPromptTokens(42)
				usage.SetCompletionTokens(3)
			}
		})

		cs, store := setupCloudServer(sseHandler)
		cs.Tenants().Create(TenantConfig{ID: "t-stream", APIKey: "key-stream", RateLimit: 1000, TokenBudget: 100000})
		handler := cs.Handler()

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer key-stream")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		records := store.All()
		if len(records) != 1 {
			t.Fatalf("got %d billing records, want 1", len(records))
		}
		r := records[0]
		if r.TenantID != "t-stream" {
			t.Errorf("TenantID = %q, want %q", r.TenantID, "t-stream")
		}
		if r.InputTokens != 42 {
			t.Errorf("InputTokens = %d, want 42", r.InputTokens)
		}
		if r.OutputTokens != 3 {
			t.Errorf("OutputTokens = %d, want 3", r.OutputTokens)
		}
	})

	t.Run("non-streaming JSON fallback still works", func(t *testing.T) {
		cs, store := setupCloudServer(fakeInferenceHandler(77, 33))
		cs.Tenants().Create(TenantConfig{ID: "t-json", APIKey: "key-json", RateLimit: 1000, TokenBudget: 100000})
		handler := cs.Handler()

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer key-json")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		records := store.All()
		if len(records) != 1 {
			t.Fatalf("got %d billing records, want 1", len(records))
		}
		if records[0].InputTokens != 77 {
			t.Errorf("InputTokens = %d, want 77", records[0].InputTokens)
		}
		if records[0].OutputTokens != 33 {
			t.Errorf("OutputTokens = %d, want 33", records[0].OutputTokens)
		}
	})
}

func TestCloud_TokenBudgetPreAuth(t *testing.T) {
	t.Run("exhausted budget returns 429 before inference", func(t *testing.T) {
		var called bool
		inner := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			called = true
			w.WriteHeader(http.StatusOK)
		})

		cs, _ := setupCloudServer(inner)
		// Token budget of 10 — too small for defaultMaxTokens (4096).
		cs.Tenants().Create(TenantConfig{ID: "t-budget", APIKey: "key-budget", RateLimit: 100, TokenBudget: 10})
		handler := cs.Handler()

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer key-budget")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusTooManyRequests {
			t.Errorf("status = %d, want 429", w.Code)
		}
		if called {
			t.Error("inference handler should not have been called")
		}
		if w.Header().Get("Retry-After") == "" {
			t.Error("expected Retry-After header on 429")
		}
	})

	t.Run("request-specified max_tokens used for pre-auth", func(t *testing.T) {
		var called bool
		inner := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			called = true
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"usage": map[string]int{
					"prompt_tokens":     5,
					"completion_tokens": 10,
					"total_tokens":      15,
				},
			})
		})

		cs, _ := setupCloudServer(inner)
		// Budget of 100 — enough for max_tokens=50 but not for defaultMaxTokens.
		cs.Tenants().Create(TenantConfig{ID: "t-mt", APIKey: "key-mt", RateLimit: 100, TokenBudget: 100})
		handler := cs.Handler()

		body := `{"max_tokens": 50}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-mt")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
		if !called {
			t.Error("inference handler should have been called")
		}
	})

	t.Run("reconciles estimated vs actual tokens", func(t *testing.T) {
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Report actual usage via context (15 tokens total).
			if usage := generate.TokenUsageFromContext(r.Context()); usage != nil {
				usage.SetPromptTokens(5)
				usage.SetCompletionTokens(10)
			}
			w.WriteHeader(http.StatusOK)
		})

		cs, _ := setupCloudServer(inner)
		// Budget of 200.
		cs.Tenants().Create(TenantConfig{ID: "t-recon", APIKey: "key-recon", RateLimit: 100, TokenBudget: 200})
		handler := cs.Handler()

		// First request with max_tokens=100 — pre-authorizes 100, actual=15, refund 85.
		body := `{"max_tokens": 100}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-recon")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("first request status = %d, want 200", w.Code)
		}

		// After reconciliation, budget used = 15 (not 100).
		// A second request with max_tokens=100 should still succeed (15+100 <= 200).
		req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req2.Header.Set("Authorization", "Bearer key-recon")
		w2 := httptest.NewRecorder()
		handler.ServeHTTP(w2, req2)

		if w2.Code != http.StatusOK {
			t.Errorf("second request status = %d, want 200 (reconciliation should have refunded tokens)", w2.Code)
		}
	})

	t.Run("deducts excess tokens when actual exceeds estimate", func(t *testing.T) {
		// Simulate a request with max_tokens=1 but the model generates 50 tokens.
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if usage := generate.TokenUsageFromContext(r.Context()); usage != nil {
				usage.SetPromptTokens(10)
				usage.SetCompletionTokens(40)
			}
			w.WriteHeader(http.StatusOK)
		})

		cs, _ := setupCloudServer(inner)
		// Budget of 100 tokens.
		cs.Tenants().Create(TenantConfig{ID: "t-excess", APIKey: "key-excess", RateLimit: 100, TokenBudget: 100})
		handler := cs.Handler()

		// Request with max_tokens=1 — pre-authorizes only 1 token, but actual=50.
		body := `{"max_tokens": 1}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-excess")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("first request status = %d, want 200", w.Code)
		}

		// After reconciliation the full 50 tokens should be consumed (not just 1).
		// A second identical request should succeed only if budget has 50 left
		// (100 - 50 = 50 remaining, pre-auth 1 OK, then 50 more deducted → 100 used).
		req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req2.Header.Set("Authorization", "Bearer key-excess")
		w2 := httptest.NewRecorder()
		handler.ServeHTTP(w2, req2)

		if w2.Code != http.StatusOK {
			t.Fatalf("second request status = %d, want 200 (50 tokens remaining)", w2.Code)
		}

		// After two requests, 100 tokens consumed. A third should be rejected.
		req3 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req3.Header.Set("Authorization", "Bearer key-excess")
		w3 := httptest.NewRecorder()
		handler.ServeHTTP(w3, req3)

		if w3.Code != http.StatusTooManyRequests {
			t.Errorf("third request status = %d, want 429 (budget exhausted after excess deductions)", w3.Code)
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

func TestCloud_SSEStreamingFlusher(t *testing.T) {
	// Verify that responseCapture implements http.Flusher so that SSE
	// streaming works through the cloud middleware chain.
	t.Run("responseCapture exposes Flusher", func(t *testing.T) {
		inner := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			flusher, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, "streaming not supported", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.WriteHeader(http.StatusOK)

			chunks := []string{
				"data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
				"data: {\"id\":\"2\",\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
				"data: [DONE]\n\n",
			}
			for _, chunk := range chunks {
				w.Write([]byte(chunk)) //nolint:errcheck
				flusher.Flush()
			}
		})

		cs, _ := setupCloudServer(inner)
		cs.Tenants().Create(TenantConfig{ID: "stream-t1", APIKey: "stream-key", RateLimit: 100, TokenBudget: 100000})
		handler := cs.Handler()

		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer stream-key")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body: %s", w.Code, w.Body.String())
		}

		body := w.Body.String()
		if !strings.Contains(body, "data: ") {
			t.Errorf("expected SSE data chunks in body, got: %s", body)
		}
		if !strings.Contains(body, "[DONE]") {
			t.Errorf("expected [DONE] sentinel in body, got: %s", body)
		}
	})

	t.Run("Flush is no-op when inner writer lacks Flusher", func(t *testing.T) {
		// Ensure Flush does not panic when the underlying writer
		// does not implement http.Flusher.
		rc := &responseCapture{
			ResponseWriter: noFlushWriter{},
			statusCode:     http.StatusOK,
		}
		// Must not panic.
		rc.Flush()
	})
}

// noFlushWriter is an http.ResponseWriter that does NOT implement http.Flusher,
// used to verify that responseCapture.Flush degrades gracefully.
type noFlushWriter struct{}

func (noFlushWriter) Header() http.Header        { return http.Header{} }
func (noFlushWriter) Write(b []byte) (int, error) { return len(b), nil }
func (noFlushWriter) WriteHeader(int)             {}

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

func TestCloud_BillingBodyLimit(t *testing.T) {
	t.Run("oversized body is truncated at 10MB", func(t *testing.T) {
		var bodyLen int
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			bodyLen = len(b)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"usage": map[string]int{
					"prompt_tokens":     10,
					"completion_tokens": 5,
					"total_tokens":      15,
				},
			})
		})

		cs, _ := setupCloudServer(inner)
		cs.Tenants().Create(TenantConfig{ID: "t-big", APIKey: "key-big", RateLimit: 1000, TokenBudget: 100000})
		handler := cs.Handler()

		// Create a body larger than 10MB.
		bigBody := strings.NewReader(strings.Repeat("x", 11<<20))
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bigBody)
		req.Header.Set("Authorization", "Bearer key-big")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		// The billing middleware should have limited the read to 10MB.
		if bodyLen > 10<<20 {
			t.Errorf("downstream saw %d bytes, want <= %d", bodyLen, 10<<20)
		}
	})

	t.Run("normal request body still works", func(t *testing.T) {
		var sawMaxTokens bool
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"usage": map[string]int{
					"prompt_tokens":     10,
					"completion_tokens": 5,
					"total_tokens":      15,
				},
			})
		})

		cs, _ := setupCloudServer(inner)
		cs.Tenants().Create(TenantConfig{ID: "t-norm", APIKey: "key-norm", RateLimit: 1000, TokenBudget: 100000})
		handler := cs.Handler()

		body := `{"max_tokens": 42}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer key-norm")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		_ = sawMaxTokens
	})
}

func TestTenant_ModelAllowed(t *testing.T) {
	tm := NewTenantManager()
	if err := tm.Create(TenantConfig{
		ID: "t-model", APIKey: "key-model", RateLimit: 100, TokenBudget: 10000,
		ModelAllowList: []string{"llama3-8b", "gemma-2b"},
	}); err != nil {
		t.Fatalf("Create: %v", err)
	}

	tenant, _ := tm.Get("t-model")

	if !tenant.ModelAllowed("llama3-8b") {
		t.Error("ModelAllowed(llama3-8b) = false, want true")
	}
	if !tenant.ModelAllowed("gemma-2b") {
		t.Error("ModelAllowed(gemma-2b) = false, want true")
	}
	if tenant.ModelAllowed("mistral-7b") {
		t.Error("ModelAllowed(mistral-7b) = true, want false")
	}
}

func TestTenant_ModelAllowedEmptyList(t *testing.T) {
	tm := NewTenantManager()
	if err := tm.Create(TenantConfig{
		ID: "t-open", APIKey: "key-open", RateLimit: 100, TokenBudget: 10000,
	}); err != nil {
		t.Fatalf("Create: %v", err)
	}

	tenant, _ := tm.Get("t-open")
	if !tenant.ModelAllowed("any-model") {
		t.Error("empty allow list should permit any model")
	}
}

func TestTenant_AllowConcurrent(t *testing.T) {
	tm := NewTenantManager()
	if err := tm.Create(TenantConfig{
		ID: "t-conc", APIKey: "key-conc", RateLimit: 100, TokenBudget: 10000,
		MaxConcurrentRequests: 2,
	}); err != nil {
		t.Fatalf("Create: %v", err)
	}

	tenant, _ := tm.Get("t-conc")

	// First two should succeed
	if !tenant.AllowConcurrent() {
		t.Fatal("first AllowConcurrent should succeed")
	}
	if !tenant.AllowConcurrent() {
		t.Fatal("second AllowConcurrent should succeed")
	}

	// Third should be rejected
	if tenant.AllowConcurrent() {
		t.Fatal("third AllowConcurrent should fail (limit=2)")
	}

	// Release one, then it should succeed again
	tenant.ReleaseConcurrent()
	if !tenant.AllowConcurrent() {
		t.Fatal("AllowConcurrent should succeed after release")
	}
}

func TestTenant_AllowConcurrentUnlimited(t *testing.T) {
	tm := NewTenantManager()
	if err := tm.Create(TenantConfig{
		ID: "t-unlim", APIKey: "key-unlim", RateLimit: 100, TokenBudget: 10000,
		// MaxConcurrentRequests: 0 (default, unlimited)
	}); err != nil {
		t.Fatalf("Create: %v", err)
	}

	tenant, _ := tm.Get("t-unlim")

	// Should always succeed when unlimited
	for i := 0; i < 100; i++ {
		if !tenant.AllowConcurrent() {
			t.Fatalf("AllowConcurrent should always succeed when unlimited (iteration %d)", i)
		}
	}
}

func TestTenant_ConfigIncludesNewFields(t *testing.T) {
	tm := NewTenantManager()
	if err := tm.Create(TenantConfig{
		ID: "t-cfg", APIKey: "key-cfg", RateLimit: 50, TokenBudget: 5000,
		MaxConcurrentRequests: 10,
		ModelAllowList:        []string{"llama3", "gemma3"},
	}); err != nil {
		t.Fatalf("Create: %v", err)
	}

	tenant, _ := tm.Get("t-cfg")
	cfg := tenant.Config()

	if cfg.MaxConcurrentRequests != 10 {
		t.Errorf("Config().MaxConcurrentRequests = %d, want 10", cfg.MaxConcurrentRequests)
	}
	if len(cfg.ModelAllowList) != 2 {
		t.Errorf("Config().ModelAllowList len = %d, want 2", len(cfg.ModelAllowList))
	}
	if cfg.ModelAllowList[0] != "llama3" || cfg.ModelAllowList[1] != "gemma3" {
		t.Errorf("Config().ModelAllowList = %v, want [llama3 gemma3]", cfg.ModelAllowList)
	}

	// Ensure the returned slice is a copy (not shared with internal state)
	cfg.ModelAllowList[0] = "mutated"
	if tenant.modelAllowList[0] == "mutated" {
		t.Error("Config().ModelAllowList should be a defensive copy")
	}
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
