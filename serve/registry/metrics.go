package registry

import (
	"math"
	"sort"
	"sync"
	"time"
)

// LatencyStats holds latency percentiles in milliseconds.
type LatencyStats struct {
	P50 float64
	P95 float64
	P99 float64
}

// VersionMetrics holds performance metrics for a model version.
type VersionMetrics struct {
	ModelID      string
	RequestCount int64
	ErrorCount   int64
	ErrorRate    float64
	Latency      LatencyStats
	LastUpdated  time.Time
}

// metricsEntry is the internal per-model accumulator.
type metricsEntry struct {
	requestCount int64
	errorCount   int64
	latencies    []float64
	lastUpdated  time.Time
}

// MetricsStore is an in-memory, thread-safe store of per-model performance metrics.
type MetricsStore struct {
	mu      sync.Mutex
	entries map[string]*metricsEntry
}

// NewMetricsStore returns a ready-to-use MetricsStore.
func NewMetricsStore() *MetricsStore {
	return &MetricsStore{
		entries: make(map[string]*metricsEntry),
	}
}

// Record appends a latency sample and increments counters for the given model.
func (s *MetricsStore) Record(modelID string, latencyMs float64, isError bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	e, ok := s.entries[modelID]
	if !ok {
		e = &metricsEntry{}
		s.entries[modelID] = e
	}
	e.requestCount++
	if isError {
		e.errorCount++
	}
	e.latencies = append(e.latencies, latencyMs)
	e.lastUpdated = time.Now()
}

// GetMetrics computes and returns the current metrics for the given model.
func (s *MetricsStore) GetMetrics(modelID string) (VersionMetrics, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	e, ok := s.entries[modelID]
	if !ok {
		return VersionMetrics{}, false
	}

	latency := computePercentiles(e.latencies)
	errorRate := 0.0
	if e.requestCount > 0 {
		errorRate = float64(e.errorCount) / float64(e.requestCount)
	}

	return VersionMetrics{
		ModelID:      modelID,
		RequestCount: e.requestCount,
		ErrorCount:   e.errorCount,
		ErrorRate:    errorRate,
		Latency:      latency,
		LastUpdated:  e.lastUpdated,
	}, true
}

// Reset clears all recorded data for the given model.
func (s *MetricsStore) Reset(modelID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.entries, modelID)
}

// All returns metrics for every registered model.
func (s *MetricsStore) All() []VersionMetrics {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]VersionMetrics, 0, len(s.entries))
	for id, e := range s.entries {
		latency := computePercentiles(e.latencies)
		errorRate := 0.0
		if e.requestCount > 0 {
			errorRate = float64(e.errorCount) / float64(e.requestCount)
		}
		out = append(out, VersionMetrics{
			ModelID:      id,
			RequestCount: e.requestCount,
			ErrorCount:   e.errorCount,
			ErrorRate:    errorRate,
			Latency:      latency,
			LastUpdated:  e.lastUpdated,
		})
	}
	return out
}

// computePercentiles returns P50, P95, P99 from a latency sample using
// nearest-rank on a sorted copy.
func computePercentiles(samples []float64) LatencyStats {
	if len(samples) == 0 {
		return LatencyStats{}
	}
	sorted := make([]float64, len(samples))
	copy(sorted, samples)
	sort.Float64s(sorted)

	return LatencyStats{
		P50: percentile(sorted, 50),
		P95: percentile(sorted, 95),
		P99: percentile(sorted, 99),
	}
}

// percentile returns the p-th percentile (0–100) from a pre-sorted slice
// using nearest-rank.
func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	rank := int(math.Ceil(p/100*float64(n))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= n {
		rank = n - 1
	}
	return sorted[rank]
}
