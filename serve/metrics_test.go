package serve

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/metrics/runtime"
)

func TestServerMetrics_RecordRequest(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	m.RecordRequest(10, 100*time.Millisecond)

	snap := c.Snapshot()

	if got := snap.Counters["requests_total"]; got != 1 {
		t.Errorf("requests_total = %d, want 1", got)
	}
	if got := snap.Counters["tokens_generated_total"]; got != 10 {
		t.Errorf("tokens_generated_total = %d, want 10", got)
	}
	if got := snap.Gauges["tokens_per_second"]; got <= 0 {
		t.Errorf("tokens_per_second = %f, want > 0", got)
	}

	h, ok := snap.Histograms["request_latency_ms"]
	if !ok {
		t.Fatal("request_latency_ms histogram not found")
	}
	if h.Count != 1 {
		t.Errorf("histogram count = %d, want 1", h.Count)
	}
}

func TestServerMetrics_MultipleRequests(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	m.RecordRequest(5, 50*time.Millisecond)
	m.RecordRequest(15, 200*time.Millisecond)
	m.RecordRequest(3, 10*time.Millisecond)

	snap := c.Snapshot()

	if got := snap.Counters["requests_total"]; got != 3 {
		t.Errorf("requests_total = %d, want 3", got)
	}
	if got := snap.Counters["tokens_generated_total"]; got != 23 {
		t.Errorf("tokens_generated_total = %d, want 23", got)
	}
}

func TestServerMetrics_ZeroTokens(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	m.RecordRequest(0, 50*time.Millisecond)

	snap := c.Snapshot()

	if got := snap.Counters["requests_total"]; got != 1 {
		t.Errorf("requests_total = %d, want 1", got)
	}
	if got := snap.Counters["tokens_generated_total"]; got != 0 {
		t.Errorf("tokens_generated_total = %d, want 0", got)
	}
	// tokens_per_second should not be updated for zero tokens.
	if got := snap.Gauges["tokens_per_second"]; got != 0 {
		t.Errorf("tokens_per_second = %f, want 0", got)
	}
}

func TestWithMetrics_NilDefaultsToNop(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)

	// Should not panic when no metrics collector is set.
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":3}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

func TestWithMetrics_RecordsOnCompletion(t *testing.T) {
	c := runtime.NewInMemory()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithMetrics(c))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	snap := c.Snapshot()
	if got := snap.Counters["requests_total"]; got != 1 {
		t.Errorf("requests_total = %d, want 1", got)
	}
	if got := snap.Counters["tokens_generated_total"]; got < 1 {
		t.Errorf("tokens_generated_total = %d, want >= 1", got)
	}
}

func TestWithMetrics_RecordsOnChatCompletion(t *testing.T) {
	c := runtime.NewInMemory()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithMetrics(c))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	snap := c.Snapshot()
	if got := snap.Counters["requests_total"]; got != 1 {
		t.Errorf("requests_total = %d, want 1", got)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	c := runtime.NewInMemory()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithMetrics(c))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Make a completion request first to generate some metrics.
	body := `{"prompt":"hello","max_tokens":3}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	_, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	// Now hit /metrics.
	metricsResp := doGet(t, ts.URL+"/metrics")
	defer func() { _ = metricsResp.Body.Close() }()

	if metricsResp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", metricsResp.StatusCode)
	}

	metricsBody, err := io.ReadAll(metricsResp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	text := string(metricsBody)

	checks := []struct {
		name    string
		pattern string
	}{
		{"requests_total header", "# TYPE requests_total counter"},
		{"requests_total value", "requests_total "},
		{"tokens_generated_total header", "# TYPE tokens_generated_total counter"},
		{"tokens_per_second header", "# TYPE tokens_per_second gauge"},
		{"request_latency_ms header", "# TYPE request_latency_ms histogram"},
		{"request_latency_ms bucket", "request_latency_ms_bucket"},
		{"request_latency_ms count", "request_latency_ms_count"},
		{"request_latency_ms sum", "request_latency_ms_sum"},
	}
	for _, tc := range checks {
		if !strings.Contains(text, tc.pattern) {
			t.Errorf("%s: expected %q in metrics output", tc.name, tc.pattern)
		}
	}
}

func TestMetricsEndpoint_NopCollector(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl) // No WithMetrics - defaults to Nop
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/metrics")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

func TestMetricsEndpoint_HistogramBuckets(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	// Record requests at various latencies.
	m.RecordRequest(1, 5*time.Millisecond)    // <= 10ms bucket
	m.RecordRequest(1, 75*time.Millisecond)   // <= 100ms bucket
	m.RecordRequest(1, 3000*time.Millisecond) // <= 5000ms bucket

	snap := c.Snapshot()
	h, ok := snap.Histograms["request_latency_ms"]
	if !ok {
		t.Fatal("histogram not found")
	}

	if h.Count != 3 {
		t.Errorf("count = %d, want 3", h.Count)
	}

	// The 10ms bucket should have exactly 1 observation.
	if got := h.Buckets[10]; got != 1 {
		t.Errorf("bucket le=10: %d, want 1", got)
	}
	// The 100ms bucket should have 2 (cumulative: <=10 + <=100).
	if got := h.Buckets[100]; got != 2 {
		t.Errorf("bucket le=100: %d, want 2", got)
	}
	// The 10000ms bucket should have 3 (all observations).
	if got := h.Buckets[10000]; got != 3 {
		t.Errorf("bucket le=10000: %d, want 3", got)
	}
}

func TestServerMetrics_RecordError(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	m.RecordError("/v1/completions", 400)
	m.RecordError("/v1/completions", 400)
	m.RecordError("/v1/chat/completions", 500)

	snap := c.Snapshot()

	key400 := `errors_total{endpoint="/v1/completions",status_code="400"}`
	if got := snap.Counters[key400]; got != 2 {
		t.Errorf("%s = %d, want 2", key400, got)
	}

	key500 := `errors_total{endpoint="/v1/chat/completions",status_code="500"}`
	if got := snap.Counters[key500]; got != 1 {
		t.Errorf("%s = %d, want 1", key500, got)
	}
}

func TestServerMetrics_ActiveRequests(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	if got := m.ActiveRequests(); got != 0 {
		t.Fatalf("initial active_requests = %d, want 0", got)
	}

	m.IncActiveRequests()
	if got := m.ActiveRequests(); got != 1 {
		t.Errorf("after inc: active_requests = %d, want 1", got)
	}

	snap := c.Snapshot()
	if got := snap.Gauges["active_requests"]; got != 1 {
		t.Errorf("gauge active_requests = %f, want 1", got)
	}

	m.DecActiveRequests()
	if got := m.ActiveRequests(); got != 0 {
		t.Errorf("after dec: active_requests = %d, want 0", got)
	}

	snap = c.Snapshot()
	if got := snap.Gauges["active_requests"]; got != 0 {
		t.Errorf("gauge active_requests = %f, want 0", got)
	}
}

func TestErrorsTotal_IncrementedOnFailedRequest(t *testing.T) {
	c := runtime.NewInMemory()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithMetrics(c))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Send an invalid request body to trigger a 400 error.
	body := `not-json`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	_, _ = io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}

	snap := c.Snapshot()
	key := `errors_total{endpoint="/v1/completions",status_code="400"}`
	if got := snap.Counters[key]; got != 1 {
		t.Errorf("%s = %d, want 1", key, got)
	}
}

func TestActiveRequests_DuringInflight(t *testing.T) {
	c := runtime.NewInMemory()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithMetrics(c))

	// Use a channel to block the handler so we can observe active_requests = 1.
	gate := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		srv.metrics.IncActiveRequests()
		gate <- struct{}{} // signal that we're in-flight
		<-gate             // wait for test to check metrics
		srv.metrics.DecActiveRequests()
		w.WriteHeader(http.StatusOK)
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// Start a request in a goroutine.
	done := make(chan struct{})
	go func() {
		defer close(done)
		resp, err := http.Get(ts.URL + "/test")
		if err != nil {
			return
		}
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}()

	// Wait for the handler to signal it's in-flight.
	<-gate

	// While the request is in-flight, active_requests should be 1.
	if got := srv.metrics.ActiveRequests(); got != 1 {
		t.Errorf("active_requests during inflight = %d, want 1", got)
	}

	// Release the handler.
	gate <- struct{}{}
	<-done

	// After completion, active_requests should be 0.
	if got := srv.metrics.ActiveRequests(); got != 0 {
		t.Errorf("active_requests after completion = %d, want 0", got)
	}
}

func TestEWMA_ConvergesToMean(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	// Simulate 10 requests each producing 100 tokens in 1 second = 100 tok/s.
	for i := 0; i < 10; i++ {
		m.RecordRequest(100, 1*time.Second)
	}

	snap := c.Snapshot()
	ewmaVal := snap.Gauges["tokens_per_second_ewma"]

	// With constant 100 tok/s input and alpha=0.1, EWMA should equal 100.
	if diff := ewmaVal - 100.0; diff < -0.01 || diff > 0.01 {
		t.Errorf("tokens_per_second_ewma = %f, want ~100.0", ewmaVal)
	}
}

func TestEWMA_ConvergesFromVaryingRates(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	// Feed 10 requests alternating between 80 and 120 tok/s (mean = 100).
	for i := 0; i < 10; i++ {
		if i%2 == 0 {
			// 80 tokens in 1s = 80 tok/s
			m.RecordRequest(80, 1*time.Second)
		} else {
			// 120 tokens in 1s = 120 tok/s
			m.RecordRequest(120, 1*time.Second)
		}
	}

	snap := c.Snapshot()
	ewmaVal := snap.Gauges["tokens_per_second_ewma"]

	// EWMA with alpha=0.1 and alternating 80/120 should converge near 100.
	// After 10 steps it won't be exact, but should be within 15 of 100.
	if ewmaVal < 85 || ewmaVal > 115 {
		t.Errorf("tokens_per_second_ewma = %f, want within [85, 115]", ewmaVal)
	}

	// Per-request gauge should still reflect the last observation.
	tps := snap.Gauges["tokens_per_second"]
	if diff := tps - 120.0; diff < -0.01 || diff > 0.01 {
		t.Errorf("tokens_per_second (last) = %f, want ~120.0", tps)
	}
}

func TestEWMA_ThreadSafety(t *testing.T) {
	c := runtime.NewInMemory()
	m := NewServerMetrics(c)

	// Concurrent updates should not race.
	done := make(chan struct{})
	for g := 0; g < 4; g++ {
		go func() {
			for i := 0; i < 100; i++ {
				m.RecordRequest(50, 500*time.Millisecond)
			}
			done <- struct{}{}
		}()
	}
	for g := 0; g < 4; g++ {
		<-done
	}

	snap := c.Snapshot()
	ewmaVal := snap.Gauges["tokens_per_second_ewma"]
	// All requests produce 100 tok/s, so EWMA should be ~100.
	if ewmaVal < 90 || ewmaVal > 110 {
		t.Errorf("tokens_per_second_ewma = %f after concurrent updates, want ~100", ewmaVal)
	}
}
