package serve

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

func percentile(sorted []time.Duration, p float64) time.Duration {
	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func TestLoadTest_ConcurrentRequests(t *testing.T) {
	// Each request needs its own model because fixedLogitsNode has mutable
	// callCount state that isn't safe for concurrent use. We create separate
	// servers per request to exercise concurrent HTTP handling without
	// triggering a data race on the mock node.
	const numRequests = 8
	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`

	servers := make([]*httptest.Server, numRequests)
	for i := range numRequests {
		mdl := buildTestModel(t)
		srv := NewServer(mdl)
		servers[i] = httptest.NewServer(srv.Handler())
	}
	defer func() {
		for _, s := range servers {
			s.Close()
		}
	}()

	var wg sync.WaitGroup
	wg.Add(numRequests)

	type result struct {
		status  int
		latency time.Duration
		err     error
	}
	results := make([]result, numRequests)

	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	for i := range numRequests {
		go func(idx int) {
			defer wg.Done()
			start := time.Now()
			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, servers[idx].URL+"/v1/chat/completions", strings.NewReader(body))
			if err != nil {
				results[idx] = result{err: err}
				return
			}
			req.Header.Set("Content-Type", "application/json")
			resp, err := http.DefaultClient.Do(req)
			elapsed := time.Since(start)
			if err != nil {
				results[idx] = result{err: err, latency: elapsed}
				return
			}
			defer func() { _ = resp.Body.Close() }()

			var chatResp ChatCompletionResponse
			_ = json.NewDecoder(resp.Body).Decode(&chatResp)
			results[idx] = result{status: resp.StatusCode, latency: elapsed}
		}(i)
	}

	wg.Wait()

	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	latencies := make([]time.Duration, 0, numRequests)
	for i, r := range results {
		if r.err != nil {
			t.Fatalf("request %d failed: %v", i, r.err)
		}
		if r.status != http.StatusOK {
			t.Errorf("request %d: status = %d, want 200", i, r.status)
		}
		latencies = append(latencies, r.latency)
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)

	t.Logf("Concurrent requests: %d", numRequests)
	t.Logf("Latency p50=%v  p95=%v  p99=%v", p50, p95, p99)
	t.Logf("Latency min=%v  max=%v", latencies[0], latencies[len(latencies)-1])
	t.Logf("Peak heap alloc delta: %d KB", (memAfter.HeapAlloc-memBefore.HeapAlloc)/1024)
	t.Logf("Total alloc during test: %d KB", (memAfter.TotalAlloc-memBefore.TotalAlloc)/1024)
}
