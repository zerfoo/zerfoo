package rl

import (
	"errors"
	"math/rand/v2"
)

// ReplayBuffer stores experience tuples for off-policy learning.
// When the buffer is full the oldest entry is overwritten (FIFO eviction).
type ReplayBuffer struct {
	buf      []Experience
	capacity int
	head     int // index of the next write position
	size     int // number of valid entries
}

// NewReplayBuffer returns a ReplayBuffer with the given capacity.
// capacity must be > 0; otherwise a non-nil error is returned.
func NewReplayBuffer(capacity int) (*ReplayBuffer, error) {
	if capacity <= 0 {
		return nil, errors.New("rl: ReplayBuffer capacity must be > 0")
	}
	return &ReplayBuffer{
		buf:      make([]Experience, capacity),
		capacity: capacity,
	}, nil
}

// Len returns the number of experiences currently stored.
func (rb *ReplayBuffer) Len() int { return rb.size }

// Add appends an experience, overwriting the oldest entry when full.
func (rb *ReplayBuffer) Add(exp Experience) {
	rb.buf[rb.head] = exp
	rb.head = (rb.head + 1) % rb.capacity
	if rb.size < rb.capacity {
		rb.size++
	}
}

// Sample returns batchSize experiences chosen uniformly at random (with replacement).
func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
	if rb.size == 0 || batchSize <= 0 {
		return nil
	}
	out := make([]Experience, batchSize)
	for i := range out {
		out[i] = rb.buf[rand.IntN(rb.size)]
	}
	return out
}

// SamplePrioritized returns batchSize experiences sampled proportionally to the
// provided priorities slice (one weight per stored experience, index 0 = oldest).
// priorities must have length equal to rb.Len(); any negative value is treated as 0.
func (rb *ReplayBuffer) SamplePrioritized(batchSize int, priorities []float64) ([]Experience, error) {
	if rb.size == 0 || batchSize <= 0 {
		return nil, nil
	}
	if len(priorities) != rb.size {
		return nil, errors.New("rl: priorities length must equal ReplayBuffer.Len()")
	}

	// Build cumulative weight array over the logical order of stored entries.
	// Logical index 0 is the oldest entry; physical index = (head - size + capacity) % capacity.
	total := 0.0
	cum := make([]float64, rb.size)
	for i, p := range priorities {
		if p < 0 {
			p = 0
		}
		total += p
		cum[i] = total
	}

	out := make([]Experience, batchSize)
	for i := range out {
		var logicalIdx int
		if total <= 0 {
			// All weights zero — fall back to uniform.
			logicalIdx = rand.IntN(rb.size)
		} else {
			r := rand.Float64() * total
			// Binary search for the bucket.
			lo, hi := 0, rb.size-1
			for lo < hi {
				mid := (lo + hi) / 2
				if cum[mid] < r {
					lo = mid + 1
				} else {
					hi = mid
				}
			}
			logicalIdx = lo
		}
		// Convert logical index to physical index in the ring buffer.
		oldest := (rb.head - rb.size + rb.capacity) % rb.capacity
		physIdx := (oldest + logicalIdx) % rb.capacity
		out[i] = rb.buf[physIdx]
	}
	return out, nil
}
