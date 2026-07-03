package serve

import (
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/zerfoo/ztensor/metrics/runtime"
)

// latencyBuckets are the histogram bucket boundaries for request latency in ms.
var latencyBuckets = []float64{10, 50, 100, 250, 500, 1000, 2500, 5000, 10000}

// ewma is a thread-safe exponentially weighted moving average.
type ewma struct {
	alpha float64
	bits  atomic.Uint64
	init  atomic.Bool
}

// update applies a new observation to the EWMA and returns the new value.
func (e *ewma) update(v float64) float64 {
	if !e.init.Load() {
		// First observation seeds the EWMA.
		e.bits.Store(math.Float64bits(v))
		e.init.Store(true)
		return v
	}
	for {
		oldBits := e.bits.Load()
		oldVal := math.Float64frombits(oldBits)
		newVal := e.alpha*v + (1-e.alpha)*oldVal
		if e.bits.CompareAndSwap(oldBits, math.Float64bits(newVal)) {
			return newVal
		}
	}
}

// value returns the current EWMA value.
func (e *ewma) value() float64 {
	return math.Float64frombits(e.bits.Load())
}

// ServerMetrics records serving metrics using a runtime.Collector.
type ServerMetrics struct {
	collector        runtime.Collector
	requestsTotal    runtime.CounterMetric
	tokensTotal      runtime.CounterMetric
	tokensPerSecond  runtime.GaugeMetric
	requestLatencyMs runtime.HistogramMetric
	tpsEWMA          ewma
	tpsEWMAGauge     runtime.GaugeMetric
	activeRequests   int64 // atomic; tracks in-flight requests
	activeReqGauge   runtime.GaugeMetric
}

// NewServerMetrics creates a ServerMetrics backed by the given collector.
func NewServerMetrics(c runtime.Collector) *ServerMetrics {
	return &ServerMetrics{
		collector:        c,
		requestsTotal:    c.Counter("requests_total"),
		tokensTotal:      c.Counter("tokens_generated_total"),
		tokensPerSecond:  c.Gauge("tokens_per_second"),
		requestLatencyMs: c.Histogram("request_latency_ms", latencyBuckets),
		tpsEWMA:          ewma{alpha: 0.1},
		tpsEWMAGauge:     c.Gauge("tokens_per_second_ewma"),
		activeReqGauge:   c.Gauge("active_requests"),
	}
}

// RecordRequest records a completed request's metrics.
func (m *ServerMetrics) RecordRequest(tokens int, latency time.Duration) {
	m.requestsTotal.Inc()
	m.tokensTotal.Add(int64(tokens))
	ms := float64(latency.Microseconds()) / 1000.0
	m.requestLatencyMs.Observe(ms)
	if latency > 0 && tokens > 0 {
		tps := float64(tokens) / latency.Seconds()
		m.tokensPerSecond.Set(tps)
		avg := m.tpsEWMA.update(tps)
		m.tpsEWMAGauge.Set(avg)
	}
}

// RecordError increments the errors_total counter for the given endpoint and
// HTTP status code. Labels are encoded in the counter name so that the
// Prometheus exposition can emit them as {endpoint="...",status_code="..."}.
//
// Callers MUST pass an already-bounded endpoint label (e.g. the output of
// normalizeRoute), never a raw, attacker-controlled request path: RecordError
// is invoked from logMiddleware, which runs before authMiddleware, so an
// unauthenticated caller could otherwise mint one permanent counter entry per
// distinct path it sends (SERVE-1: metric label cardinality DoS).
func (m *ServerMetrics) RecordError(endpoint string, statusCode int) {
	name := "errors_total{endpoint=\"" + endpoint + "\",status_code=\"" + strconv.Itoa(statusCode) + "\"}"
	m.collector.Counter(name).Inc()
}

// knownRoutes is the fixed set of registered route paths that get their own
// metric label. It MUST be kept in sync with the routes registered on s.mux
// in server.go's newServer setup. Anything not in this set (including
// nonexistent paths probed pre-auth) collapses to the single "other" label.
var knownRoutes = map[string]struct{}{
	"/v1/chat/completions":     {},
	"/v1/completions":          {},
	"/v1/embeddings":           {},
	"/v1/audio/transcriptions": {},
	"/v1/classify":             {},
	"/v1/guard":                {},
	"/v1/guard/batch":          {},
	"/v1/guard/scan":           {},
	"/v1/models":               {},
	"/healthz":                 {},
	"/readyz":                  {},
	"/openapi.yaml":            {},
	"/metrics":                 {},
}

// normalizeRoute maps a request path to a bounded, fixed set of metric
// labels. This prevents pre-auth requests to arbitrary or nonexistent paths
// from creating unbounded permanent counter entries (SERVE-1): every path
// that is not one of the server's registered routes collapses to "other",
// and the parameterized /v1/models/{id...} route collapses to a single
// "/v1/models/{id}" label rather than echoing the attacker-chosen id.
func normalizeRoute(p string) string {
	if _, ok := knownRoutes[p]; ok {
		return p
	}
	if strings.HasPrefix(p, "/v1/models/") {
		return "/v1/models/{id}"
	}
	return "other"
}

// IncActiveRequests increments the active request count and updates the gauge.
func (m *ServerMetrics) IncActiveRequests() {
	n := atomic.AddInt64(&m.activeRequests, 1)
	m.activeReqGauge.Set(float64(n))
}

// DecActiveRequests decrements the active request count and updates the gauge.
func (m *ServerMetrics) DecActiveRequests() {
	n := atomic.AddInt64(&m.activeRequests, -1)
	m.activeReqGauge.Set(float64(n))
}

// ActiveRequests returns the current number of in-flight requests.
func (m *ServerMetrics) ActiveRequests() int64 {
	return atomic.LoadInt64(&m.activeRequests)
}

// handleMetrics writes metrics in Prometheus text exposition format.
// It requires the collector to be an *runtime.InMemoryCollector to access Snapshot().
func handleMetrics(c runtime.Collector) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		imc, ok := c.(*runtime.InMemoryCollector)
		if !ok {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			w.WriteHeader(http.StatusOK)
			return
		}

		snap := imc.Snapshot()
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		w.WriteHeader(http.StatusOK)

		// Counters.
		writeCounter(w, "requests_total", "Total number of requests", snap.Counters)
		writeCounter(w, "tokens_generated_total", "Total tokens generated", snap.Counters)

		// Labeled error counters.
		writeLabeledCounters(w, "errors_total", "Total number of errors by endpoint and status code", snap.Counters)

		// Gauges.
		writeGauge(w, "tokens_per_second", "Last request tokens per second", snap.Gauges)
		writeGauge(w, "tokens_per_second_ewma", "EWMA tokens per second", snap.Gauges)
		writeGauge(w, "active_requests", "Number of in-flight requests", snap.Gauges)
		writeGauge(w, "speculative_acceptance_rate", "Speculative decoding acceptance rate", snap.Gauges)

		// Histograms.
		writeHistogram(w, "request_latency_ms", "Request latency histogram", snap.Histograms)
	}
}

func writeCounter(w http.ResponseWriter, name, help string, counters map[string]int64) {
	v, ok := counters[name]
	if !ok {
		return
	}
	_, _ = fmt.Fprintf(w, "# HELP %s %s\n", name, help)
	_, _ = fmt.Fprintf(w, "# TYPE %s counter\n", name)
	_, _ = fmt.Fprintf(w, "%s %d\n", name, v)
}

func writeGauge(w http.ResponseWriter, name, help string, gauges map[string]float64) {
	v, ok := gauges[name]
	if !ok {
		return
	}
	_, _ = fmt.Fprintf(w, "# HELP %s %s\n", name, help)
	_, _ = fmt.Fprintf(w, "# TYPE %s gauge\n", name)
	_, _ = fmt.Fprintf(w, "%s %g\n", name, v)
}

// writeLabeledCounters emits all counters whose names start with prefix + "{"
// as a single metric family. The counter names contain embedded Prometheus
// labels, e.g. errors_total{endpoint="/v1/completions",status_code="400"}.
func writeLabeledCounters(w http.ResponseWriter, prefix, help string, counters map[string]int64) {
	headerWritten := false
	for name, v := range counters {
		if !strings.HasPrefix(name, prefix+"{") {
			continue
		}
		if !headerWritten {
			_, _ = fmt.Fprintf(w, "# HELP %s %s\n", prefix, help)
			_, _ = fmt.Fprintf(w, "# TYPE %s counter\n", prefix)
			headerWritten = true
		}
		_, _ = fmt.Fprintf(w, "%s %d\n", name, v)
	}
}

func writeHistogram(w http.ResponseWriter, name, help string, histograms map[string]runtime.HistogramSnapshot) {
	h, ok := histograms[name]
	if !ok {
		return
	}
	_, _ = fmt.Fprintf(w, "# HELP %s %s\n", name, help)
	_, _ = fmt.Fprintf(w, "# TYPE %s histogram\n", name)

	// Sort bucket boundaries.
	bounds := make([]float64, 0, len(h.Buckets))
	for b := range h.Buckets {
		bounds = append(bounds, b)
	}
	sort.Float64s(bounds)

	for _, b := range bounds {
		_, _ = fmt.Fprintf(w, "%s_bucket{le=\"%g\"} %d\n", name, b, h.Buckets[b])
	}
	_, _ = fmt.Fprintf(w, "%s_bucket{le=\"+Inf\"} %d\n", name, h.Count)
	_, _ = fmt.Fprintf(w, "%s_sum %g\n", name, h.Sum)
	_, _ = fmt.Fprintf(w, "%s_count %d\n", name, h.Count)
}
