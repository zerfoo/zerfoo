package serve

import (
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/zerfoo/ztensor/metrics/runtime"
)

// latencyBuckets are the histogram bucket boundaries for request latency in ms.
var latencyBuckets = []float64{10, 50, 100, 250, 500, 1000, 2500, 5000, 10000}

// ServerMetrics records serving metrics using a runtime.Collector.
type ServerMetrics struct {
	collector        runtime.Collector
	requestsTotal    runtime.CounterMetric
	tokensTotal      runtime.CounterMetric
	tokensPerSecond  runtime.GaugeMetric
	requestLatencyMs runtime.HistogramMetric
}

// NewServerMetrics creates a ServerMetrics backed by the given collector.
func NewServerMetrics(c runtime.Collector) *ServerMetrics {
	return &ServerMetrics{
		collector:        c,
		requestsTotal:    c.Counter("requests_total"),
		tokensTotal:      c.Counter("tokens_generated_total"),
		tokensPerSecond:  c.Gauge("tokens_per_second"),
		requestLatencyMs: c.Histogram("request_latency_ms", latencyBuckets),
	}
}

// RecordRequest records a completed request's metrics.
func (m *ServerMetrics) RecordRequest(tokens int, latency time.Duration) {
	m.requestsTotal.Inc()
	for range tokens {
		m.tokensTotal.Inc()
	}
	ms := float64(latency.Microseconds()) / 1000.0
	m.requestLatencyMs.Observe(ms)
	if latency > 0 && tokens > 0 {
		tps := float64(tokens) / latency.Seconds()
		m.tokensPerSecond.Set(tps)
	}
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

		// Gauges.
		writeGauge(w, "tokens_per_second", "Rolling average tokens per second", snap.Gauges)
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
