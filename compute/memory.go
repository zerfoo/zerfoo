package compute

import (
	"errors"
	"sync/atomic"
)

// ErrMemoryLimitExceeded is returned when a tensor allocation would exceed
// the configured memory limit.
var ErrMemoryLimitExceeded = errors.New("memory limit exceeded")

// MemoryTracker tracks total allocated bytes with an optional upper limit.
// All methods are safe for concurrent use.
type MemoryTracker struct {
	allocated atomic.Int64
	limit     int64 // 0 means unlimited
}

// NewMemoryTracker creates a tracker with the given byte limit.
// A limit of 0 disables enforcement (unlimited).
func NewMemoryTracker(limit int64) *MemoryTracker {
	return &MemoryTracker{limit: limit}
}

// Alloc reserves bytes. If the allocation would exceed the limit, it returns
// ErrMemoryLimitExceeded without modifying the counter.
func (m *MemoryTracker) Alloc(bytes int64) error {
	if m.limit <= 0 {
		m.allocated.Add(bytes)
		return nil
	}
	for {
		cur := m.allocated.Load()
		next := cur + bytes
		if next > m.limit {
			return ErrMemoryLimitExceeded
		}
		if m.allocated.CompareAndSwap(cur, next) {
			return nil
		}
	}
}

// Free releases previously allocated bytes.
func (m *MemoryTracker) Free(bytes int64) {
	m.allocated.Add(-bytes)
}

// Allocated returns the current total allocated bytes.
func (m *MemoryTracker) Allocated() int64 {
	return m.allocated.Load()
}

// Limit returns the configured byte limit (0 means unlimited).
func (m *MemoryTracker) Limit() int64 {
	return m.limit
}
