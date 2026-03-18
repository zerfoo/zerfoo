package cloud

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"sync"
	"testing"
	"time"
)

func TestCloudLoadTest(t *testing.T) {
	const (
		numTenants         = 100
		requestsPerTenant  = 10
		maxP99Latency      = 2 * time.Second
	)

	// Create registry and register 100 tenants with unique API keys.
	registry := NewTenantRegistry()
	apiKeys := make([]string, numTenants)
	for i := range numTenants {
		key := fmt.Sprintf("tenant-key-%04d", i)
		apiKeys[i] = key
		err := registry.Register(key, TenantConfig{
			MaxConcurrentRequests: requestsPerTenant,
			MaxTokensPerMinute:   100000,
		})
		if err != nil {
			t.Fatalf("failed to register tenant %d: %v", i, err)
		}
	}

	// Handler that returns the tenant's API key as tenant_id in JSON.
	handler := registry.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tenantID := extractBearerToken(r)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"tenant_id": tenantID})
	}))

	srv := httptest.NewServer(handler)
	defer srv.Close()

	type result struct {
		tenantIdx int
		reqIdx    int
		latency   time.Duration
		tenantID  string
		err       error
	}

	results := make([]result, numTenants*requestsPerTenant)

	var wg sync.WaitGroup
	wg.Add(numTenants)

	for i := range numTenants {
		go func(idx int) {
			defer wg.Done()
			client := &http.Client{Timeout: 5 * time.Second}
			for j := range requestsPerTenant {
				start := time.Now()

				req, err := http.NewRequest(http.MethodGet, srv.URL+"/v1/chat/completions", nil)
				if err != nil {
					results[idx*requestsPerTenant+j] = result{tenantIdx: idx, reqIdx: j, err: err}
					continue
				}
				req.Header.Set("Authorization", "Bearer "+apiKeys[idx])

				resp, err := client.Do(req)
				elapsed := time.Since(start)
				if err != nil {
					results[idx*requestsPerTenant+j] = result{tenantIdx: idx, reqIdx: j, err: err}
					continue
				}

				var body struct {
					TenantID string `json:"tenant_id"`
				}
				json.NewDecoder(resp.Body).Decode(&body)
				resp.Body.Close()

				results[idx*requestsPerTenant+j] = result{
					tenantIdx: idx,
					reqIdx:    j,
					latency:   elapsed,
					tenantID:  body.TenantID,
				}
			}
		}(i)
	}

	wg.Wait()

	// Collect latencies and verify correctness.
	var latencies []time.Duration
	for _, r := range results {
		if r.err != nil {
			t.Errorf("tenant %d request %d failed: %v", r.tenantIdx, r.reqIdx, r.err)
			continue
		}

		// Verify no cross-tenant data leakage.
		expectedKey := apiKeys[r.tenantIdx]
		if r.tenantID != expectedKey {
			t.Errorf("cross-tenant leakage: tenant %d got tenant_id=%q, want %q",
				r.tenantIdx, r.tenantID, expectedKey)
		}

		latencies = append(latencies, r.latency)
	}

	if len(latencies) == 0 {
		t.Fatal("no successful requests")
	}

	// Compute P99 latency.
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	p99Index := int(float64(len(latencies)) * 0.99)
	if p99Index >= len(latencies) {
		p99Index = len(latencies) - 1
	}
	p99 := latencies[p99Index]

	t.Logf("total requests: %d, P50: %v, P99: %v, max: %v",
		len(latencies), latencies[len(latencies)/2], p99, latencies[len(latencies)-1])

	if p99 > maxP99Latency {
		t.Errorf("P99 latency %v exceeds threshold %v", p99, maxP99Latency)
	}
}
