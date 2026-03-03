// Package runtime provides a backend-agnostic metrics collection abstraction
// for runtime observability. It includes an in-memory implementation for
// testing and a no-op implementation for zero-overhead production use.
package runtime

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"
)

// Collector is the interface for recording runtime metrics.
type Collector interface {
	// Counter returns a named counter (creates if needed).
	Counter(name string) CounterMetric
	// Gauge returns a named gauge (creates if needed).
	Gauge(name string) GaugeMetric
	// Histogram returns a named histogram with the given bucket boundaries.
	Histogram(name string, buckets []float64) HistogramMetric
}

// CounterMetric is a monotonically increasing counter.
type CounterMetric interface {
	Inc()
}

// GaugeMetric is a metric that can be set to any value.
type GaugeMetric interface {
	Set(value float64)
}

// HistogramMetric records observations into pre-defined buckets.
type HistogramMetric interface {
	Observe(value float64)
}

// --- InMemory implementation ---

// InMemoryCollector stores metrics in memory for testing and local use.
type InMemoryCollector struct {
	mu         sync.RWMutex
	counters   map[string]*inMemCounter
	gauges     map[string]*inMemGauge
	histograms map[string]*inMemHistogram
}

// NewInMemory creates a new InMemoryCollector.
func NewInMemory() *InMemoryCollector {
	return &InMemoryCollector{
		counters:   make(map[string]*inMemCounter),
		gauges:     make(map[string]*inMemGauge),
		histograms: make(map[string]*inMemHistogram),
	}
}

// Counter returns or creates a named counter.
func (c *InMemoryCollector) Counter(name string) CounterMetric {
	c.mu.Lock()
	defer c.mu.Unlock()

	if m, ok := c.counters[name]; ok {
		return m
	}

	m := &inMemCounter{}
	c.counters[name] = m

	return m
}

// Gauge returns or creates a named gauge.
func (c *InMemoryCollector) Gauge(name string) GaugeMetric {
	c.mu.Lock()
	defer c.mu.Unlock()

	if m, ok := c.gauges[name]; ok {
		return m
	}

	m := &inMemGauge{}
	c.gauges[name] = m

	return m
}

// Histogram returns or creates a named histogram with bucket boundaries.
func (c *InMemoryCollector) Histogram(name string, buckets []float64) HistogramMetric {
	c.mu.Lock()
	defer c.mu.Unlock()

	if m, ok := c.histograms[name]; ok {
		return m
	}

	sorted := make([]float64, len(buckets))
	copy(sorted, buckets)
	sort.Float64s(sorted)

	m := &inMemHistogram{
		buckets:      sorted,
		bucketCounts: make([]int64, len(sorted)),
	}
	c.histograms[name] = m

	return m
}

// Snapshot is a point-in-time copy of all metric values.
type Snapshot struct {
	Counters   map[string]int64
	Gauges     map[string]float64
	Histograms map[string]HistogramSnapshot
}

// HistogramSnapshot is a point-in-time copy of a histogram.
type HistogramSnapshot struct {
	Count   int64
	Sum     float64
	Buckets map[float64]int64 // upper bound -> cumulative count
}

// Snapshot returns a point-in-time copy of all metrics.
func (c *InMemoryCollector) Snapshot() Snapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()

	snap := Snapshot{
		Counters:   make(map[string]int64, len(c.counters)),
		Gauges:     make(map[string]float64, len(c.gauges)),
		Histograms: make(map[string]HistogramSnapshot, len(c.histograms)),
	}

	for name, m := range c.counters {
		snap.Counters[name] = m.val.Load()
	}

	for name, m := range c.gauges {
		snap.Gauges[name] = m.load()
	}

	for name, m := range c.histograms {
		hs := HistogramSnapshot{
			Count:   m.count.Load(),
			Sum:     m.loadSum(),
			Buckets: make(map[float64]int64, len(m.buckets)),
		}
		for i, b := range m.buckets {
			hs.Buckets[b] = atomic.LoadInt64(&m.bucketCounts[i])
		}
		snap.Histograms[name] = hs
	}

	return snap
}

// --- inMem types ---

type inMemCounter struct {
	val atomic.Int64
}

func (c *inMemCounter) Inc() {
	c.val.Add(1)
}

type inMemGauge struct {
	bits atomic.Uint64
}

func (g *inMemGauge) Set(value float64) {
	g.bits.Store(float64bits(value))
}

func (g *inMemGauge) load() float64 {
	return float64frombits(g.bits.Load())
}

type inMemHistogram struct {
	buckets      []float64
	bucketCounts []int64 // atomic
	count        atomic.Int64
	sumBits      atomic.Uint64
}

func (h *inMemHistogram) Observe(value float64) {
	h.count.Add(1)

	// Atomic float add via CAS loop.
	for {
		old := h.sumBits.Load()
		newVal := float64frombits(old) + value
		if h.sumBits.CompareAndSwap(old, float64bits(newVal)) {
			break
		}
	}

	for i, b := range h.buckets {
		if value <= b {
			atomic.AddInt64(&h.bucketCounts[i], 1)
		}
	}
}

func (h *inMemHistogram) loadSum() float64 {
	return float64frombits(h.sumBits.Load())
}

// --- Nop implementation ---

type nopCollector struct{}

// Nop returns a Collector that discards all metrics with zero allocation.
func Nop() Collector {
	return nopCollector{}
}

func (nopCollector) Counter(_ string) CounterMetric                   { return nopMetric{} }
func (nopCollector) Gauge(_ string) GaugeMetric                      { return nopMetric{} }
func (nopCollector) Histogram(_ string, _ []float64) HistogramMetric { return nopMetric{} }

type nopMetric struct{}

func (nopMetric) Inc()              {}
func (nopMetric) Set(_ float64)     {}
func (nopMetric) Observe(_ float64) {}

// --- float64 bit helpers ---

func float64bits(f float64) uint64    { return math.Float64bits(f) }
func float64frombits(b uint64) float64 { return math.Float64frombits(b) }
