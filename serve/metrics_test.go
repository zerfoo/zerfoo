package serve

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/metrics/runtime"
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
