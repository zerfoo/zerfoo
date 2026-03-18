package registry

import (
	"sync"
	"testing"
)

func TestRecordAndGet(t *testing.T) {
	s := NewMetricsStore()
	for i := 0; i < 100; i++ {
		s.Record("model-a", float64(i), false)
	}
	m, ok := s.GetMetrics("model-a")
	if !ok {
		t.Fatal("expected metrics for model-a")
	}
	if m.RequestCount != 100 {
		t.Fatalf("RequestCount = %d, want 100", m.RequestCount)
	}
	if m.ErrorCount != 0 {
		t.Fatalf("ErrorCount = %d, want 0", m.ErrorCount)
	}
	if m.ModelID != "model-a" {
		t.Fatalf("ModelID = %q, want %q", m.ModelID, "model-a")
	}
}

func TestLatencyPercentiles(t *testing.T) {
	s := NewMetricsStore()
	// Record latencies 1..100 so percentiles are deterministic.
	for i := 1; i <= 100; i++ {
		s.Record("model-b", float64(i), false)
	}
	m, ok := s.GetMetrics("model-b")
	if !ok {
		t.Fatal("expected metrics for model-b")
	}
	if m.Latency.P50 != 50 {
		t.Fatalf("P50 = %v, want 50", m.Latency.P50)
	}
	if m.Latency.P95 != 95 {
		t.Fatalf("P95 = %v, want 95", m.Latency.P95)
	}
	if m.Latency.P99 != 99 {
		t.Fatalf("P99 = %v, want 99", m.Latency.P99)
	}
}

func TestErrorRate(t *testing.T) {
	s := NewMetricsStore()
	for i := 0; i < 10; i++ {
		s.Record("model-c", 1.0, i < 3) // first 3 are errors
	}
	m, ok := s.GetMetrics("model-c")
	if !ok {
		t.Fatal("expected metrics for model-c")
	}
	if m.ErrorCount != 3 {
		t.Fatalf("ErrorCount = %d, want 3", m.ErrorCount)
	}
	want := 0.3
	if m.ErrorRate < want-0.001 || m.ErrorRate > want+0.001 {
		t.Fatalf("ErrorRate = %v, want %v", m.ErrorRate, want)
	}
}

func TestReset(t *testing.T) {
	s := NewMetricsStore()
	s.Record("model-d", 5.0, false)
	s.Reset("model-d")
	_, ok := s.GetMetrics("model-d")
	if ok {
		t.Fatal("expected no metrics after reset")
	}
}

func TestConcurrentRecord(t *testing.T) {
	s := NewMetricsStore()
	var wg sync.WaitGroup
	for g := 0; g < 10; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				s.Record("model-e", float64(i), false)
			}
		}()
	}
	wg.Wait()

	m, ok := s.GetMetrics("model-e")
	if !ok {
		t.Fatal("expected metrics for model-e")
	}
	if m.RequestCount != 1000 {
		t.Fatalf("RequestCount = %d, want 1000", m.RequestCount)
	}
}

func TestAll(t *testing.T) {
	s := NewMetricsStore()
	s.Record("model-x", 1.0, false)
	s.Record("model-y", 2.0, true)
	all := s.All()
	if len(all) != 2 {
		t.Fatalf("All() returned %d entries, want 2", len(all))
	}
}
