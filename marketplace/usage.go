package marketplace

import (
	"context"
	"sync"
	"time"
)

// UsageTracker aggregates usage records in memory and flushes them to a
// MeteringService on demand or at a configured interval.
type UsageTracker struct {
	mu      sync.Mutex
	pending []UsageRecord
}

// NewUsageTracker creates an empty usage tracker.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{}
}

// Record adds a usage record to the pending buffer.
func (t *UsageTracker) Record(customerID, dimension string, quantity int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.pending = append(t.pending, UsageRecord{
		CustomerID: customerID,
		Dimension:  dimension,
		Quantity:    quantity,
		Timestamp:  time.Now(),
	})
}

// Pending returns the number of buffered records.
func (t *UsageTracker) Pending() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.pending)
}

// Flush drains all pending records and submits them to the given metering service.
// Returns the number of records flushed and any error from submission.
func (t *UsageTracker) Flush(ctx context.Context, svc MeteringService) (int, error) {
	t.mu.Lock()
	records := t.pending
	t.pending = nil
	t.mu.Unlock()

	if len(records) == 0 {
		return 0, nil
	}

	if err := svc.SubmitUsage(ctx, records); err != nil {
		// Re-enqueue on failure so records are not lost.
		t.mu.Lock()
		t.pending = append(records, t.pending...)
		t.mu.Unlock()
		return 0, err
	}
	return len(records), nil
}

// Aggregate returns the total quantity per (customerID, dimension) pair
// across all pending records without draining them.
func (t *UsageTracker) Aggregate() map[string]map[string]int64 {
	t.mu.Lock()
	defer t.mu.Unlock()

	result := make(map[string]map[string]int64)
	for _, r := range t.pending {
		dims, ok := result[r.CustomerID]
		if !ok {
			dims = make(map[string]int64)
			result[r.CustomerID] = dims
		}
		dims[r.Dimension] += r.Quantity
	}
	return result
}
