package runtime

import (
	"sync"
	"testing"
)

func TestNopCollector(t *testing.T) {
	c := Nop()

	// Must not panic.
	c.Counter("ops").Inc()
	c.Gauge("mem").Set(100)
	c.Histogram("latency", []float64{0.1, 0.5, 1.0}).Observe(0.3)
}

func TestInMemoryCollector_Counter(t *testing.T) {
	c := NewInMemory()

	counter := c.Counter("requests")
	counter.Inc()
	counter.Inc()
	counter.Inc()

	snap := c.Snapshot()
	if snap.Counters["requests"] != 3 {
		t.Errorf("requests = %d, want 3", snap.Counters["requests"])
	}
}

func TestInMemoryCollector_CounterSameName(t *testing.T) {
	c := NewInMemory()

	c1 := c.Counter("ops")
	c2 := c.Counter("ops")

	c1.Inc()
	c2.Inc()

	snap := c.Snapshot()
	if snap.Counters["ops"] != 2 {
		t.Errorf("ops = %d, want 2", snap.Counters["ops"])
	}
}

func TestInMemoryCollector_Gauge(t *testing.T) {
	c := NewInMemory()

	gauge := c.Gauge("memory_mb")
	gauge.Set(512)

	snap := c.Snapshot()
	if snap.Gauges["memory_mb"] != 512 {
		t.Errorf("memory_mb = %f, want 512", snap.Gauges["memory_mb"])
	}

	gauge.Set(1024)
	snap = c.Snapshot()
	if snap.Gauges["memory_mb"] != 1024 {
		t.Errorf("memory_mb = %f, want 1024", snap.Gauges["memory_mb"])
	}
}

func TestInMemoryCollector_Histogram(t *testing.T) {
	c := NewInMemory()

	hist := c.Histogram("latency_ms", []float64{10, 50, 100})
	hist.Observe(5)
	hist.Observe(25)
	hist.Observe(75)
	hist.Observe(150)

	snap := c.Snapshot()
	h, ok := snap.Histograms["latency_ms"]
	if !ok {
		t.Fatal("expected histogram latency_ms in snapshot")
	}
	if h.Count != 4 {
		t.Errorf("count = %d, want 4", h.Count)
	}
	if h.Sum != 255 {
		t.Errorf("sum = %f, want 255", h.Sum)
	}

	// Bucket checks: <=10, <=50, <=100
	if h.Buckets[10] != 1 {
		t.Errorf("bucket[10] = %d, want 1", h.Buckets[10])
	}
	if h.Buckets[50] != 2 {
		t.Errorf("bucket[50] = %d, want 2", h.Buckets[50])
	}
	if h.Buckets[100] != 3 {
		t.Errorf("bucket[100] = %d, want 3", h.Buckets[100])
	}
}

func TestInMemoryCollector_ConcurrentAccess(t *testing.T) {
	c := NewInMemory()
	counter := c.Counter("concurrent")
	gauge := c.Gauge("concurrent_gauge")
	hist := c.Histogram("concurrent_hist", []float64{1, 10})

	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counter.Inc()
			gauge.Set(42)
			hist.Observe(5)
		}()
	}
	wg.Wait()

	snap := c.Snapshot()
	if snap.Counters["concurrent"] != 100 {
		t.Errorf("concurrent = %d, want 100", snap.Counters["concurrent"])
	}
}

func TestInMemoryCollector_MultipleMetrics(t *testing.T) {
	c := NewInMemory()

	c.Counter("a").Inc()
	c.Counter("b").Inc()
	c.Counter("b").Inc()
	c.Gauge("x").Set(1)
	c.Gauge("y").Set(2)

	snap := c.Snapshot()
	if len(snap.Counters) != 2 {
		t.Errorf("expected 2 counters, got %d", len(snap.Counters))
	}
	if len(snap.Gauges) != 2 {
		t.Errorf("expected 2 gauges, got %d", len(snap.Gauges))
	}
}

func TestInMemoryCollector_SnapshotIsolation(t *testing.T) {
	c := NewInMemory()
	c.Counter("ops").Inc()

	snap := c.Snapshot()

	// Mutating after snapshot should not affect the snapshot.
	c.Counter("ops").Inc()

	if snap.Counters["ops"] != 1 {
		t.Errorf("snapshot should be isolated, got %d", snap.Counters["ops"])
	}
}

func TestHistogram_EmptyBuckets(t *testing.T) {
	c := NewInMemory()
	hist := c.Histogram("empty", nil)
	hist.Observe(42)

	snap := c.Snapshot()
	h := snap.Histograms["empty"]
	if h.Count != 1 {
		t.Errorf("count = %d, want 1", h.Count)
	}
	if h.Sum != 42 {
		t.Errorf("sum = %f, want 42", h.Sum)
	}
}
